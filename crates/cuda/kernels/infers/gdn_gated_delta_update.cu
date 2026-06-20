// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// Gated Delta Rule single-token update kernel for decode.
///
/// No shared memory — state is kept and modified in global memory.
/// Each thread handles one (h, v) element.

#include "common.cuh"

extern "C" {

/// Gated Delta Rule update kernel — no shared memory.
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_gated_delta_update_bf16(
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

    int h = idx / head_v_dim;
    int v = idx % head_v_dim;

    int K = head_k_dim;
    int V = head_v_dim;
    float rcp_sqrt_k = rsqrtf((float)K);

    // Compute g[h] and beta[h]
    // g = -exp(A_log) * softplus(a + dt_bias) — HF reference formula
    float decay_rate_h = expf(A_log[h]);  // A = exp(A_log)
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
    float b_val = __bfloat162float(b_proj[h]);
    float beta_val = 1.0f / (1.0f + expf(-b_val));
    float decay = expf(g_val);

    // L2-normalize key and query
    float k_l2 = 0.0f, q_l2 = 0.0f;
    for (int k = 0; k < K; k++) {
        float kv = __bfloat162float(key[h * K + k]);
        float qv = __bfloat162float(query[h * K + k]);
        k_l2 += kv * kv;
        q_l2 += qv * qv;
    }
    k_l2 = rsqrtf(k_l2 + 1e-6f);
    q_l2 = rsqrtf(q_l2 + 1e-6f);

    int state_base = h * K * V + v;

    // Step 1: S *= exp(g)
    for (int k = 0; k < K; k++) {
        state[state_base + k * V] *= decay;
    }

    // Step 2: kv_mem = sum_k S[k][v] * key[h][k] (key L2-normalized)
    float kv_mem = 0.0f;
    for (int k = 0; k < K; k++) {
        float s_val = state[state_base + k * V];
        float k_val = __bfloat162float(key[h * K + k]) * k_l2;
        kv_mem += s_val * k_val;
    }

    // Step 3: delta = beta * (value - kv_mem)
    float v_val = __bfloat162float(value[h * V + v]);
    float delta = beta_val * (v_val - kv_mem);

    // Step 4: State update (key L2-normalized)
    for (int k = 0; k < K; k++) {
        float k_val = __bfloat162float(key[h * K + k]) * k_l2;
        state[state_base + k * V] += k_val * delta;
    }

    // Step 5: y[v] = sum_k S[k][v] * (query[h][k] * q_l2 * rcp_sqrt_k)
    float y_val = 0.0f;
    for (int k = 0; k < K; k++) {
        float s_val = state[state_base + k * V];
        float q_val = __bfloat162float(query[h * K + k]) * q_l2 * rcp_sqrt_k;
        y_val += s_val * q_val;
    }

    output[h * V + v] = __float2bfloat16(y_val);
}

} // extern "C"
