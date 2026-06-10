/// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Common utilities for infers CUDA kernels — BF16 helpers, block size constants,
/// and shared launch parameters used across all kernel implementations.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

/// Default thread block size for 1D kernels.
/// 256 threads per block gives good occupancy across most architectures.
#define INFERS_BLOCK_SIZE 256

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Block size for vectorized BF16 kernels using __nv_bfloat162.
/// 128 threads × 2 elements = 256 elements per block.
#define INFERS_BLOCK_SIZE_VEC 128

/// Thread-local macro for kernel indexing — global thread ID from 1D grid.
#define INFERS_THREAD_IDX blockDim.x * blockIdx.x + threadIdx.x

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Thread-local macro for vectorized kernels — each thread processes 2 elements.
#define INFERS_VEC_THREAD_IDX (blockDim.x * blockIdx.x + threadIdx.x) * 2

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Maximum shared memory per block (conservative estimate for compatibility).
#define INFERS_SHARED_MAX 48000

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Clamp macro for boundary-safe indexing.
#define INFERS_CLAMP(val, max) ((val) < (max) ? (val) : (max))

/// Inline BF16 → float conversion.
__device__ inline float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

/// Inline float → BF16 conversion.
__device__ inline __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Inline BF16 addition.
__device__ inline __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return a + b;
}

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Inline BF16 multiplication.
__device__ inline __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return a * b;
}

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Vectorized BF16-to-float2 conversion.
__device__ inline float2 bf162_to_float2(__nv_bfloat162 val) {
    return __bfloat1622float2(val);
}

/// Reserved for future vectorized BF16 kernels using __nv_bfloat162.
/// Vectorized float2-to-BF16 conversion.
__device__ inline __nv_bfloat162 float2_to_bf162(float2 val) {
    return __float22bfloat162_rn(val);
}
