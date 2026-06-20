// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// Gated Delta Rule prefill kernel — processes all tokens in a sequence.
///
/// State is kept in global memory (no shared memory dependency).
/// Each thread handles one (v_dim) element, accessing column v of the
/// 2D state matrix S [head_k_dim × head_v_dim] directly in global memory.
///
/// Reference recurrence:
///   g[t][h] = -exp(-A_log[h]) * softplus(a[t][h] + dt_bias[h])
///   beta[t][h] = sigmoid(b[t][h])
///   S *= exp(g[t][h])
///   kv_mem = S @ k[t][h]
///   delta = beta * (v[t][h] - kv_mem)
///   S += k[t][h] ⊗ delta
///   y[t][h] = S @ q[t][h]

#include "common.cuh"

extern "C" {

/// Gated Delta Rule prefill kernel — no shared memory, state in global memory.
///
/// Each thread handles one v-dim element. State matrix is indexed as
/// `state[h * K * V + k * V + v]` — each thread v accesses all k elements
/// of its column (contiguous in k).
///
/// # Arguments
/// * query   — [seq_len, num_heads, head_k_dim] BF16
/// * key     — [seq_len, num_heads, head_k_dim] BF16
/// * value   — [seq_len, num_heads, head_v_dim] BF16
/// * a_proj  — [seq_len, num_heads] BF16
/// * b_proj  — [seq_len, num_heads] BF16
/// * A_log   — [num_heads] float32
/// * dt_bias — [num_heads] float32
/// * state   — [num_heads, head_k_dim, head_v_dim] float32 mutable
/// * output  — [seq_len, num_heads, head_v_dim] BF16
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_gated_delta_prefill_bf16(
    const __nv_bfloat16* __restrict__ query,   // [S, H, K]
    const __nv_bfloat16* __restrict__ key,     // [S, H, K]
    const __nv_bfloat16* __restrict__ value,   // [S, H, V]
    const __nv_bfloat16* __restrict__ a_proj,  // [S, H]
    const __nv_bfloat16* __restrict__ b_proj,  // [S, H]
    const float* __restrict__ A_log,            // [H]
    const float* __restrict__ dt_bias,          // [H]
    float* __restrict__ state,                  // [H, K, V]
    __nv_bfloat16* __restrict__ output,         // [S, H, V]
    int seq_len,
    int num_heads,
    int head_k_dim,
    int head_v_dim
) {
    int total = num_heads * head_v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int h = idx / head_v_dim;      // head index
    int v = idx % head_v_dim;      // v-dim element for this thread

    int K = head_k_dim;
    int V = head_v_dim;
    float rcp_sqrt_k = rsqrtf((float)K);

    // g = -exp(A_log) * softplus(a + dt_bias) — HF reference formula
    float decay_rate_h = expf(A_log[h]);    // A = exp(A_log)

    for (int t = 0; t < seq_len; t++) {
        // Compute g[t][h] and beta[t][h]
        float a_val = __bfloat162float(a_proj[t * num_heads + h]);
        float sp_val = a_val + dt_bias[h];
        float softplus_val;
        if (sp_val > 20.0f) {
            softplus_val = sp_val;
        } else if (sp_val < -20.0f) {
            softplus_val = 0.0f;
        } else {
            softplus_val = logf(1.0f + expf(sp_val));
        }
        float g_val = -decay_rate_h * softplus_val;
        float b_val = __bfloat162float(b_proj[t * num_heads + h]);
        float beta_val = 1.0f / (1.0f + expf(-b_val));
        float decay = expf(g_val);

        // L2-normalize query and key (HF: use_qk_l2norm_in_kernel=True)
        float k_l2 = 0.0f, q_l2 = 0.0f;
        for (int k = 0; k < K; k++) {
            float kv = __bfloat162float(key[t * num_heads * K + h * K + k]);
            float qv = __bfloat162float(query[t * num_heads * K + h * K + k]);
            k_l2 += kv * kv;
            q_l2 += qv * qv;
        }
        k_l2 = rsqrtf(k_l2 + 1e-6f);   // 1 / ||k||
        q_l2 = rsqrtf(q_l2 + 1e-6f);   // 1 / ||q||

        int state_base = h * K * V + v;

        // Step 1: S *= exp(g)
        for (int k = 0; k < K; k++) {
            float s = state[state_base + k * V];
            state[state_base + k * V] = isfinite(s) ? s * decay : 0.0f;
        }

        // Step 2: kv_mem = sum_k S[k][v] * key[t][h][k] (key L2-normalized)
        float kv_mem = 0.0f;
        for (int k = 0; k < K; k++) {
            float s_val = state[state_base + k * V];
            float k_val = __bfloat162float(key[t * num_heads * K + h * K + k]) * k_l2;
            if (isfinite(k_val)) kv_mem += s_val * k_val;
        }

        // Step 3: delta = beta * (value - kv_mem)
        float v_val = __bfloat162float(value[t * num_heads * V + h * V + v]);
        v_val = isfinite(v_val) ? v_val : 0.0f;
        float delta = isfinite(beta_val) ? beta_val * (v_val - kv_mem) : 0.0f;

        // Step 4: State update: S[k][v] += key[t][h][k] * delta (key L2-normalized)
        if (isfinite(delta)) {
            for (int k = 0; k < K; k++) {
                float k_val = __bfloat162float(key[t * num_heads * K + h * K + k]) * k_l2;
                if (isfinite(k_val)) {
                    state[state_base + k * V] += k_val * delta;
                }
            }
        }

        // Step 5: y[v] = sum_k S[k][v] * (query[t][h][k] * q_l2 * rcp_sqrt_k)
        // HF: query = l2norm(query) * 1/sqrt(K), then y = S @ query
        float y_val = 0.0f;
        for (int k = 0; k < K; k++) {
            float s_val = state[state_base + k * V];
            float q_val = __bfloat162float(query[t * num_heads * K + h * K + k]) * q_l2 * rcp_sqrt_k;
            if (isfinite(s_val) && isfinite(q_val)) {
                y_val += s_val * q_val;
            }
        }

        // Write output
        output[t * num_heads * V + h * V + v] = __float2bfloat16(y_val);
    }
}

} // extern "C"
