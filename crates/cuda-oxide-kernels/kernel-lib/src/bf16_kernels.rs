//! BF16 tiled GEMM kernel.

use cuda_device::{cuda_module, kernel, launch_bounds, thread};

#[cuda_module]
pub mod bf16 {
    use super::*;

    /// Tiled bf16 GEMM: C[M,N] = A[M,K] @ B[N,K]^T
    ///
    /// A (input): [M, K] bf16, row-major
    /// B (weight): [N, K] bf16, row-major (each row is one output feature)
    /// C (output): [M, N] bf16, row-major
    ///
    /// Tile config: BM=64, BN=64
    /// Thread tile: TM=4, TN=4
    /// Block: 256 threads (16×16 thread mapping within tile)
    #[kernel]
    #[launch_bounds(256)]
    pub fn bf16_gemm_tiled(
        output: &mut [u16],               // [M, N] bf16
        input: &[u16],                    // [M, K] bf16  
        weight: &[u16],                   // [N, K] bf16
        m: u32,
        n: u32,
        k: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        const TM: usize = 4;
        const TN: usize = 4;

        // Use LOCAL thread index (within block) to map to 2D tile.
        // thread::index_1d() gives global index (blockIdx.x * blockDim.x + threadIdx.x)
        // which would cause ty > 15 for blocks beyond the first.
        let tid_local = thread::threadIdx_x() as usize;
        let tx = (tid_local % 16) as usize;  // 0-15, maps to columns within tile
        let ty = (tid_local / 16) as usize;  // 0-15, maps to rows within tile

        let block_m = (thread::blockIdx_y() * 64u32) as usize;
        let block_n = (thread::blockIdx_x() * 64u32) as usize;
        let m_usize = m as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;

        // Each thread computes a TM×TN = 4×4 sub-tile of the output
        let mut acc: [f32; TM * TN] = [0.0f32; TM * TN];

        // Load input values for 4 rows at column ki, with bounds checking
        macro_rules! load_input {
            ($row:expr, $ki:expr) => {
                if $row < m_usize {
                    f32::from_bits((input[$row * k_usize + $ki] as u32) << 16)
                } else {
                    0.0f32
                }
            };
        }

        // Load weight values for row at column ki, with bounds checking
        macro_rules! load_weight {
            ($row:expr, $ki:expr) => {
                if $row < n_usize {
                    f32::from_bits((weight[$row * k_usize + $ki] as u32) << 16)
                } else {
                    0.0f32
                }
            };
        }

        for ki in 0..k_usize {
            let r0 = block_m + ty * TM + 0;
            let r1 = block_m + ty * TM + 1;
            let r2 = block_m + ty * TM + 2;
            let r3 = block_m + ty * TM + 3;

            let c0 = block_n + tx * TN + 0;
            let c1 = block_n + tx * TN + 1;
            let c2 = block_n + tx * TN + 2;
            let c3 = block_n + tx * TN + 3;

            let a0 = load_input!(r0, ki);
            let a1 = load_input!(r1, ki);
            let a2 = load_input!(r2, ki);
            let a3 = load_input!(r3, ki);

            let w0 = load_weight!(c0, ki);
            let w1 = load_weight!(c1, ki);
            let w2 = load_weight!(c2, ki);
            let w3 = load_weight!(c3, ki);

            acc[0] += a0 * w0;
            acc[1] += a0 * w1;
            acc[2] += a0 * w2;
            acc[3] += a0 * w3;
            acc[4] += a1 * w0;
            acc[5] += a1 * w1;
            acc[6] += a1 * w2;
            acc[7] += a1 * w3;
            acc[8] += a2 * w0;
            acc[9] += a2 * w1;
            acc[10] += a2 * w2;
            acc[11] += a2 * w3;
            acc[12] += a3 * w0;
            acc[13] += a3 * w1;
            acc[14] += a3 * w2;
            acc[15] += a3 * w3;
        }

        // Write output - each thread writes its 4×4 sub-tile
        for i in 0..TM {
            for j in 0..TN {
                let g_row = block_m + ty * TM + i;
                let g_col = block_n + tx * TN + j;
                if g_row < m_usize && g_col < n_usize {
                    let idx = g_row * n_usize + g_col;
                    output[idx] = f32_to_bf16(acc[(i * TN + j) as usize]);
                }
            }
        }
    }
}
