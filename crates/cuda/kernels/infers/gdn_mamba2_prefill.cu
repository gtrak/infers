// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Mamba2 SSM prefill kernel for Qwen3.6-27B GDN layers.
///
/// Implements the Mamba2-style SSM recurrence on pre-computed BF16 projections.
/// No INT4 dequantization inside the kernel — all inputs are BF16.
///
/// Each thread processes one element of the total state vector
/// (total_dim = num_heads * head_dim). Per-head signals (x_proj, b_proj,
/// A_log, dt_bias) broadcast across all head_dim elements of each head.
///
/// Per-element recurrence for token t and element idx (head = idx / head_dim):
///   decay = sigmoid(A_log[head])
///   delta = softplus(dt_proj[t*total_dim + idx] + dt_bias[head])
///   state[idx] = decay * state[idx] + delta * b_proj[t*num_heads + head]
///   ssm_out = state[idx] * x_proj[t*num_heads + head]
///   z = z_gate[t*total_dim + idx]
///   silu_z = z / (1 + exp(-z))
///   output[t*total_dim + idx] = ssm_out * silu_z
///
/// After the kernel, the host-side applies output projection (GEMM) to expand
/// total_dim → hidden_size, plus residual addition. The kernel output buffer is
/// [seq_len × total_dim].
///
/// Each thread processes one element of the state vector through all tokens.

#include "common.cuh"

extern "C" {

/// Mamba2 SSM prefill kernel (BF16).
///
/// Each thread handles one SSM state element, iterating sequentially through
/// all tokens. Per-head signals broadcast across head_dim elements.
/// State is a flat [total_dim] vector.
///
/// # Launch configuration
/// * grid: `(total_dim + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
/// * shared: none
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_mamba2_prefill_bf16(
    const __nv_bfloat16* __restrict__ x_proj,    // [seq_len, num_heads]
    const __nv_bfloat16* __restrict__ b_proj,    // [seq_len, num_heads]
    const __nv_bfloat16* __restrict__ dt_proj,   // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ z_gate,    // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ A_log,      // [num_heads]
    const __nv_bfloat16* __restrict__ dt_bias,    // [num_heads]
    __nv_bfloat16* __restrict__ state,            // [total_dim]
    __nv_bfloat16* __restrict__ output,           // [seq_len, total_dim]
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = num_heads * head_dim;
    if (idx >= total_dim) return;

    int head = idx / head_dim;

    // Pre-compute per-head constants
    float a_val = __bfloat162float(A_log[head]);
    // sigmoid maps A_log to (0, 1), giving stable SSM decay factors
    float decay = 1.0f / (1.0f + expf(-a_val));
    float bias_val = __bfloat162float(dt_bias[head]);

    // Accumulate state in float precision for numerical stability
    float s = __bfloat162float(state[idx]);

    for (int t = 0; t < seq_len; t++) {
        int offset = t * total_dim + idx;
        int head_offset = t * num_heads + head;

        // delta = softplus(dt_proj + dt_bias)
        // Use piecewise approximation for numerical stability
        float dt_val = __bfloat162float(dt_proj[offset]) + bias_val;
        float delta;
        if (dt_val > 2.0f) {
            delta = dt_val;
        } else if (dt_val < -20.0f) {
            delta = 0.0f;
        } else {
            delta = logf(1.0f + expf(dt_val));
        }

        // b contribution (per-head, broadcast across head_dim)
        float b_val = __bfloat162float(b_proj[head_offset]);

        // State update: s = decay * s + delta * b
        s = decay * s + delta * b_val;

        // Output: state * x_proj * silu(z)
        float x_val = __bfloat162float(x_proj[head_offset]);
        float z_val = __bfloat162float(z_gate[offset]);

        // SiLU: z * sigmoid(z) = z / (1 + exp(-z))
        // Use numerically stable formulation
        float silu_z;
        if (z_val > 0.0f) {
            silu_z = z_val / (1.0f + expf(-z_val));
        } else {
            float exp_z = expf(z_val);
            silu_z = z_val * exp_z / (1.0f + exp_z);
        }

        output[offset] = __float2bfloat16(s * x_val * silu_z);
    }

    // Store final state back
    state[idx] = __float2bfloat16(s);
}

} // extern "C"
