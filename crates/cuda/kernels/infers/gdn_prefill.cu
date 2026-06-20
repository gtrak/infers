// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// Gated DeltaNet (GDN) prefill kernel — processes all tokens in a sequence.
///
/// Processes the entire token sequence in a single kernel launch. Each block
/// handles one row of the state matrix, iterating through all tokens sequentially.
///
/// Math per token t for row i of state (size hidden_size):
///   beta_i = sum_j(state[i][j] * b[t][j])
///   state[i][j] += b[t][j] * (x[t][i] - dt[t][i] * a[t][i] * beta_i)
///   output[t][i] = sum_j(updated_state[i][j] * a[t][j])
///
/// Tokens are processed sequentially within each block because the state
/// carries forward between tokens (recurrent dependency).

#include "common.cuh"

extern "C" {

/// GDN prefill kernel (BF16).
///
/// Each block handles one row of the H×H state matrix. Iterates over all tokens
/// sequentially, performing three phases per token:
///   Phase 1 — compute beta_i = dot(state_row, b[t])
///   Phase 2 — update state row with token t
///   Phase 3 — compute output[t][i] = dot(updated_state_row, a[t])
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_prefill_bf16(
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ dt,
    const __nv_bfloat16* __restrict__ x,
    int hidden_size,
    int seq_len
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    float beta = 0.0f;
    float out_val = 0.0f;
    float update_coeff = 0.0f;

    float* sdata = shared_mem;

    for (int t = 0; t < seq_len; t++) {
        int ta = t * hidden_size + row;
        int tb = t * hidden_size;
        int tdt = t * hidden_size + row;
        int tx = t * hidden_size + row;

        float x_val = __bfloat162float(x[tx]);
        float dt_val = __bfloat162float(dt[tdt]);
        float a_row_val = __bfloat162float(a[ta]);

        // --- Phase 1: Compute beta = dot(state_row, b[t]) ---
        beta = 0.0f;
        for (int j = tid; j < hidden_size; j += total_threads) {
            float s_val = __bfloat162float(state[row * hidden_size + j]);
            float b_val = __bfloat162float(b[tb + j]);
            beta += s_val * b_val;
        }

        sdata[tid] = beta;
        __syncthreads();

        for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride && tid + stride < blockDim.x) {
                sdata[tid] += sdata[tid + stride];
            }
        }
        __syncthreads();

        float beta_all = sdata[0];

        // --- Phase 2: Update state row ---
        // state[i][j] += b[t][j] * (x[t][i] - dt[t][i]*a[t][i]*beta_i)
        update_coeff = x_val - dt_val * a_row_val * beta_all;
        for (int j = tid; j < hidden_size; j += total_threads) {
            float s_val = __bfloat162float(state[row * hidden_size + j]);
            float b_val = __bfloat162float(b[tb + j]);
            state[row * hidden_size + j] = __float2bfloat16(s_val + b_val * update_coeff);
        }
        __syncthreads();

        // --- Phase 3: Compute output[t][i] = dot(updated_state_row, a[t]) ---
        out_val = 0.0f;
        for (int j = tid; j < hidden_size; j += total_threads) {
            float s_val = __bfloat162float(state[row * hidden_size + j]);
            float a_val = __bfloat162float(a[t * hidden_size + j]);
            out_val += s_val * a_val;
        }

        sdata[tid] = out_val;
        __syncthreads();

        for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride && tid + stride < blockDim.x) {
                sdata[tid] += sdata[tid + stride];
            }
        }
        __syncthreads();

        output[t * hidden_size + row] = __float2bfloat16(sdata[0]);
    }
}

} // extern "C"
