// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Mamba2 SSM prefill kernel for Qwen3.6-27B GDN layers.
///
/// Implements the Mamba2-style SSM recurrence on pre-computed BF16 projections.
/// No INT4 dequantization inside the kernel — all inputs are BF16.
///
/// Per-element recurrence for each token t and state element i:
///   decay = sigmoid(A_log[i])
///   delta = softplus(dt_proj[t*ssm_dim + i] + dt_bias[i])
///   state[i] = decay * state[i] + delta * b_proj[t*ssm_dim + i]
///   ssm_out = state[i] * x_proj[t*ssm_dim + i]
///   z = z_gate[t*ssm_dim + i]
///   silu_z = z / (1 + exp(-z))
///   output[t*ssm_dim + i] = ssm_out * silu_z
///
/// After the kernel, the host-side applies output projection (GEMM) to expand
/// ssm_dim → hidden_size, plus residual addition. The kernel output buffer is
/// [seq_len × ssm_dim].
///
/// Each thread processes one element of the state vector through all tokens.

#include "common.cuh"

extern "C" {

/// Mamba2 SSM prefill kernel (BF16).
///
/// Each thread handles one SSM state element, iterating sequentially through
/// all tokens. State is a flat [ssm_dim] vector. Decay is constant per element.
///
/// The kernel handles both the SSM recurrence and the SiLU gating in a single
/// pass, avoiding intermediate buffers. Output is [seq_len × ssm_dim].
///
/// # Launch configuration
/// * grid: `(ssm_dim + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
/// * shared: none
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_mamba2_prefill_bf16(
    const __nv_bfloat16* __restrict__ x_proj,
    const __nv_bfloat16* __restrict__ b_proj,
    const __nv_bfloat16* __restrict__ dt_proj,
    const __nv_bfloat16* __restrict__ z_gate,
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    int seq_len,
    int ssm_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ssm_dim) return;

    // Pre-compute per-element constants
    float a_val = __bfloat162float(A_log[idx]);
    // sigmoid maps A_log to (0, 1), giving stable SSM decay factors
    float decay = 1.0f / (1.0f + expf(-a_val));
    float bias_val = __bfloat162float(dt_bias[idx]);

    // Accumulate state in float precision for numerical stability
    float s = __bfloat162float(state[idx]);

    for (int t = 0; t < seq_len; t++) {
        int offset = t * ssm_dim + idx;

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

        // b contribution
        float b_val = __bfloat162float(b_proj[offset]);

        // State update: s = decay * s + delta * b
        s = decay * s + delta * b_val;

        // Output: state * x_proj * silu(z)
        float x_val = __bfloat162float(x_proj[offset]);
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
