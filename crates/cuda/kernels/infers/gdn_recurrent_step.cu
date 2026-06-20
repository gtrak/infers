// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// Single-step Gated Delta Rule recurrent kernel.
///
/// Processes exactly ONE token of the recurrence (all heads in parallel).
/// Each thread handles one (head_idx, v_idx) element.
///
/// This is a simplified version of gdn_gated_delta_prefill that:
///  - Operates on a single token instead of the full sequence
///  - Removes all isfinite() guards (pure computation, no clamping)
///  - Computes L2 normalization and Q scaling inline (matching PyTorch reference)
///  - Computes g_decay and beta from a_proj, b_proj, A_log, dt_bias internally
///
/// Reference: torch_recurrent_gated_delta_rule in modeling_qwen3_5.py

#include "common.cuh"

extern "C" {

/// Single-token Gated Delta Rule step (bf16 inputs, fp32 state).
///
/// # Arguments
/// * query   — [H, K] BF16 — raw (un-normalized) query for this token
/// * key     — [H, K] BF16 — raw (un-normalized) key for this token
/// * value   — [H, V] BF16 — raw value for this token
/// * a_proj  — [H] BF16 — decay projection for this token
/// * b_proj  — [H] BF16 — beta projection for this token
/// * A_log   — [H] float32 — log-A parameter (constant)
/// * dt_bias — [H] float32 — dt bias (constant)
/// * state   — [H, K, V] float32 — mutable recurrent state
/// * output  — [H, V] BF16 — output for this token
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_recurrent_step_bf16(
    const __nv_bfloat16* __restrict__ query,   // [H, K]
    const __nv_bfloat16* __restrict__ key,     // [H, K]
    const __nv_bfloat16* __restrict__ value,   // [H, V]
    const __nv_bfloat16* __restrict__ a_proj,  // [H]
    const __nv_bfloat16* __restrict__ b_proj,  // [H]
    const float* __restrict__ A_log,            // [H]
    const float* __restrict__ dt_bias,          // [H]
    float* __restrict__ state,                  // [H, K, V]
    __nv_bfloat16* __restrict__ output,         // [H, V]
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

    // ── Compute g[h] and beta[h] from projections ──
    // g = -exp(A_log[h]) * softplus(a_proj[h] + dt_bias[h])
    float decay_rate_h = expf(A_log[h]);   // A = exp(A_log)
    float a_val = __bfloat162float(a_proj[h]);
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
    float decay = expf(g_val);

    // beta[h] = sigmoid(b_proj[h])
    float b_val = __bfloat162float(b_proj[h]);
    float beta_val = 1.0f / (1.0f + expf(-b_val));

    // ── L2-normalize key and query, scale query by 1/sqrt(K) ──
    // Matching torch_recurrent_gated_delta_rule exactly:
    //   query = l2norm(query, dim=-1, eps=1e-6) * (1/sqrt(K))
    //   key   = l2norm(key, dim=-1, eps=1e-6)
    float k_l2_sq = 0.0f;
    float q_l2_sq = 0.0f;
    for (int k = 0; k < K; k++) {
        float kv = __bfloat162float(key[h * K + k]);
        float qv = __bfloat162float(query[h * K + k]);
        k_l2_sq += kv * kv;
        q_l2_sq += qv * qv;
    }

    // Reciprocal norms: 1 / ||k|| and 1 / ||q|| (with epsilon)
    float k_rcp = rsqrtf(k_l2_sq + 1e-6f);
    float q_rcp = rsqrtf(q_l2_sq + 1e-6f);

    // ── Base index into state matrix: state[h][k][v] = state[h*K*V + k*V + v] ──
    int state_base = h * K * V + v;

    // ── Step 1: State decay — S[h][k][v] *= exp(g) for all k ──
    for (int k = 0; k < K; k++) {
        state[state_base + k * V] *= decay;
    }

    // ── Step 2: kv_mem = sum_k S[h][k][v] * key_normed[h][k] ──
    float kv_mem = 0.0f;
    for (int k = 0; k < K; k++) {
        float s_val = state[state_base + k * V];
        float k_val = __bfloat162float(key[h * K + k]) * k_rcp;
        kv_mem += s_val * k_val;
    }

    // ── Step 3: delta = beta * (value[h][v] - kv_mem) ──
    float v_val = __bfloat162float(value[h * V + v]);
    float delta = beta_val * (v_val - kv_mem);

    // ── Step 4: State update — S[h][k][v] += key_normed[h][k] * delta for all k ──
    for (int k = 0; k < K; k++) {
        float k_val = __bfloat162float(key[h * K + k]) * k_rcp;
        state[state_base + k * V] += k_val * delta;
    }

    // ── Step 5: Output — y[h][v] = sum_k S[h][k][v] * query_normed_scaled[h][k] ──
    float y_val = 0.0f;
    for (int k = 0; k < K; k++) {
        float s_val = state[state_base + k * V];
        // query is L2-normalized and scaled by 1/sqrt(K)
        float q_val = (__bfloat162float(query[h * K + k]) * q_rcp) * rcp_sqrt_k;
        y_val += s_val * q_val;
    }

    // Write output
    output[h * V + v] = __float2bfloat16(y_val);
}

} // extern "C"
