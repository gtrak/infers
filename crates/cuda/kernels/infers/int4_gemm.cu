// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Custom INT4 GEMM kernel with per-group dequantization in registers.
///
/// Computes: output = dequant(weight) @ input
/// where weight is packed INT4 with per-group scale and zero-point.
///
/// Weights stay in INT4-packed format in GPU memory — no dequantized copy.
/// Dequantization happens in registers during the inner loop.
///
/// Group size: 128 (fixed by AutoRound format).
/// Output: BF16 — feeds directly into next layer (RMSNorm, activation).
///
/// Kernel launch:
///   grid: (ceil(N/blockDim.x), ceil(M/blockDim.y), 1)
///   block: (16, 16, 1) — each thread computes one output element

#include "common.cuh"
#include <stdint.h>

/// INT4 GEMM kernel: output[M][N] = dequant(weight[N][K]) @ input[M][K]
///
/// Each thread computes one output element output[row][col].
///
/// # Arguments
/// * `output` — [M, N] output in BF16
/// * `weight` — [N, K/8] packed INT4 (8 weights per uint32_t)
/// * `scales` — [N, K/group_size] FP16 group scales
/// * `zeros`  — [N, K/group_size/8] packed INT4 zero points (8 per uint32_t)
/// * `input`  — [M, K] activation in BF16
/// * `M` — rows of input and output
/// * `N` — columns of output (rows of weight)
/// * `K` — inner dimension (columns of input and weight)
/// * `group_size` — quantization group size (typically 128)
extern "C" {
__global__ void int4_gemm_kernel(
    __nv_bfloat16* __restrict__ output,
    const uint32_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const uint32_t* __restrict__ zeros,
    const __nv_bfloat16* __restrict__ input,
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k += group_size) {
        // Load scale and zero point for this group
        int group_idx = k / group_size;
        float scale = __bfloat162float(scales[col * (K / group_size) + group_idx]);

        // Unpack zero point (8 per uint32_t)
        int zero_packed_idx = (col * (K / group_size) + group_idx) / 8;
        int zero_shift = ((col * (K / group_size) + group_idx) % 8) * 4;
        uint32_t zero_packed = zeros[zero_packed_idx];
        int8_t zero = (int8_t)((zero_packed >> zero_shift) & 0xF);

        for (int kk = 0; kk < group_size; kk += 8) {
            // Load 8 INT4 weights from one uint32_t
            int weight_idx = (col * K + k + kk) / 8;
            uint32_t packed = weight[weight_idx];

            // Unpack each of 8 weights and compute
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                int shift = w * 4;
                int8_t w_int4 = (int8_t)((packed >> shift) & 0xF);

                // Dequantize on-the-fly in registers: (w_int4 - zero) * scale
                float w_fp32 = ((float)(w_int4 - zero)) * scale;

                // Load activation (BF16 → float)
                float a_val = __bfloat162float(input[row * K + k + kk + w]);

                // Multiply and accumulate
                acc += w_fp32 * a_val;
            }
        }
    }

    // Write output in BF16
    output[row * N + col] = __float2bfloat16(acc);
}
} // extern "C"
