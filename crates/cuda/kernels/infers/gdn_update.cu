// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Gated DeltaNet (GDN) decode kernel — recurrent state update for a single token.
///
/// One block per state row. Each block cooperatively computes the dot product
/// of its row with the b-vector (beta), updates the state row, then computes
/// the output dot product with the a-vector.
///
/// Math for row i of state (size hidden_size):
///   beta_i = sum_j(state[i][j] * b[j])
///   state[i][j] += b[j] * (x[i] - dt[i] * a[i] * beta_i)
///   output[i] = sum_j(state[i][j] * a[j])

#include "common.cuh"

extern "C" {

/// GDN single-token update kernel (BF16).
///
/// Each block handles one row of the H×H state matrix. Three phases:
///   Phase 1 — compute beta_i = dot(state_row, b) via shared memory reduction
///   Phase 2 — update state row: state[i][j] += b[j] * (x[i] - dt[i]*a[i]*beta_i)
///   Phase 3 — compute output[i] = dot(updated_state_row, a) via reduction
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_update_bf16(
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ dt,
    const __nv_bfloat16* __restrict__ x,
    int hidden_size
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Load per-row scalars in float for precision
    float x_val = __bfloat162float(x[row]);
    float dt_val = __bfloat162float(dt[row]);
    float a_row_val = __bfloat162float(a[row]);

    // --- Phase 1: Compute beta = sum_j(state[row*H + j] * b[j]) ---
    float beta = 0.0f;
    for (int j = tid; j < hidden_size; j += total_threads) {
        float s_val = __bfloat162float(state[row * hidden_size + j]);
        float b_val = __bfloat162float(b[j]);
        beta += s_val * b_val;
    }

    float* sdata = shared_mem;
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
    // state[i][j] += b[j] * (x[i] - dt[i]*a[i]*beta_i)
    float update_coeff = x_val - dt_val * a_row_val * beta_all;
    for (int j = tid; j < hidden_size; j += total_threads) {
        float s_val = __bfloat162float(state[row * hidden_size + j]);
        float b_val = __bfloat162float(b[j]);
        state[row * hidden_size + j] = __float2bfloat16(s_val + b_val * update_coeff);
    }
    __syncthreads();

    // --- Phase 3: Compute output[i] = sum_j(updated_state_row[j] * a[j]) ---
    float out_val = 0.0f;
    for (int j = tid; j < hidden_size; j += total_threads) {
        float s_val = __bfloat162float(state[row * hidden_size + j]);
        float a_val = __bfloat162float(a[j]);
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

    output[row] = __float2bfloat16(sdata[0]);
}

} // extern "C"
