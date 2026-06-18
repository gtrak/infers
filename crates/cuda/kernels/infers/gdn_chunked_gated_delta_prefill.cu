// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Chunked parallel Gated Delta Rule prefill kernel.
///
/// Implements the chunked parallel GDN attention algorithm matching HF's
/// torch_chunk_gated_delta_rule. Processes all chunks for one head per thread block.
///
/// Key differences from sequential version:
///   - L2-normalizes query and key internally (raw inputs expected)
///   - Uses intra-chunk WY representation via attn matrix + forward substitution
///   - Inter-chunk state recurrence (sequential across chunks)
///
/// Thread layout: one block per head, 256 threads per block.
/// Shared memory: ~80KB for C=64, K=128 (k_normed 32KB + k_beta 32KB + attn 16KB).

#include "common.cuh"

extern "C" {

__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_gdn_chunked_gated_delta_prefill_bf16(
    const __nv_bfloat16* __restrict__ query,   // [S, H, K]
    const __nv_bfloat16* __restrict__ key,     // [S, H, K]
    const __nv_bfloat16* __restrict__ value,   // [S, H, V]
    const __nv_bfloat16* __restrict__ a_proj,  // [S, H]
    const __nv_bfloat16* __restrict__ b_proj,  // [S, H]
    const float* __restrict__ A_log,            // [H]
    const float* __restrict__ dt_bias,          // [H]
    float* __restrict__ state,                  // [H, K, V] mutable
    __nv_bfloat16* __restrict__ output,         // [S, H, V]
    int seq_len,
    int num_heads,
    int head_k_dim,
    int head_v_dim,
    int chunk_size
) {
    int h          = blockIdx.x;       // head index — one block per head
    int C          = chunk_size;       // chunk size (typically 64)
    int K          = head_k_dim;       // key dimension
    int V          = head_v_dim;       // value dimension
    int num_chunks = (seq_len + C - 1) / C;

    float rcp_sqrt_k = rsqrtf((float)K);
    float decay_rate = expf(A_log[h]);   // A = exp(A_log[h])

    // ── Shared memory layout ────────────────────────────────────────────────
    // Dynamic: k_normed[C][K], k_beta[C][K], attn[C][C]
    // Plus: g_cs[C], beta_arr[C], row_buf[C] (Phase 3 temp)
    extern __shared__ char smem[];

    // k_normed[C][K]  — L2-normalized key vectors (fp32), used in GEMM and Phase 5
    float* k_normed   = (float*)(void*)(smem + 0);
    // k_beta[C][K]   — weighted keys = k_normed * beta (fp32)
    float* k_beta_sm  = (float*)(void*)(smem + C * K * sizeof(float));
    // attn[C][C]      — attention matrix with forward substitution (fp32)
    float* attn_sm    = (float*)(void*)(smem + 2 * C * K * sizeof(float));
    // g_cs[C]         — cumulative sum of g over chunk positions (fp32)
    float* g_cs       = (float*)(void*)(smem + 2 * C * K * sizeof(float) + C * C * sizeof(float));
    // beta_arr[C]     — beta per position (fp32)
    float* beta_arr   = (float*)(void*)(smem + 2 * C * K * sizeof(float) + C * C * sizeof(float) + C * sizeof(float));
    // row_buf[C]      — Phase 3 temporary (fp32)
    float* row_buf    = (float*)(void*)(smem + 2 * C * K * sizeof(float) + C * C * sizeof(float) + 2 * C * sizeof(float));

    int state_base = h * K * V;        // base index into state[h][k][v]

    // ── Per-chunk loop: sequential across chunks (state recurrence) ──────────
    for (int c = 0; c < num_chunks; c++) {
        int chunk_start = c * C;
        int actual_len  = min(C, seq_len - chunk_start);

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Compute g/beta/g_cumsum, L2-normalize key, compute k_beta
        // ═══════════════════════════════════════════════════════════════════

        {
            int tid = threadIdx.x;

            // ── Compute g[i] and beta[i] for each position in the chunk ─────
            for (int offset = 0; offset < C; offset += blockDim.x) {
                int i = tid + offset;
                if (i >= actual_len) continue;
                int seq_pos = chunk_start + i;

                // g = -exp(A_log[h]) * softplus(a_proj[seq_pos][h] + dt_bias[h])
                float a_val  = bf16_to_float(a_proj[(int)(seq_pos * num_heads) + h]);
                float sp_val = a_val + dt_bias[h];
                float softplus;
                if (sp_val > 20.0f)        softplus = sp_val;
                else if (sp_val < -20.0f)  softplus = 0.0f;
                else                      softplus = logf(1.0f + expf(sp_val));

                float g_val = -decay_rate * softplus;

                // beta = sigmoid(b_proj[seq_pos][h])
                float b_val   = bf16_to_float(b_proj[(int)(seq_pos * num_heads) + h]);
                float beta_v  = 1.0f / (1.0f + expf(-b_val));

                g_cs[i]   = g_val;
                beta_arr[i]  = beta_v;
            }
        }
        __syncthreads();

        // ── Zero-init padded positions ────────────────────────────────────
        {
            int tid = threadIdx.x;
            for (int i = tid; i < C; i += blockDim.x) {
                if (i >= actual_len) {
                    g_cs[i]  = 0.0f;
                    beta_arr[i] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ── Cumulative sum of g over chunk positions (sequential scan) ───────
        if (threadIdx.x == 0) {
            float running = 0.0f;
            for (int i = 0; i < C; i++) {
                running      += g_cs[i];
                g_cs[i] = running;
            }
        }
        __syncthreads();

        // ── L2-normalize key and compute k_beta ─────────────────────────────
        {
            int tid = threadIdx.x;

            // Step 1: load key into k_normed shared buffer
            for (int offset = 0; offset < C * K; offset += blockDim.x) {
                int idx   = tid + offset;
                int row   = idx / K;
                int col   = idx % K;
                if (row >= actual_len) {
                    k_normed[row * K + col] = 0.0f;
                    continue;
                }
                float seq_pos = chunk_start + row;
                int k_idx     = (int)(seq_pos * num_heads * K) + h * K + col;
                k_normed[row * K + col] = bf16_to_float(key[k_idx]);
            }

            // Step 2: per-row L2 norm and normalize (tid < C threads do work)
            if (tid < actual_len) {
                float sum_sq = 0.0f;
                for (int d = 0; d < K; d++) {
                    float kv = k_normed[tid * K + d];
                    sum_sq += kv * kv;
                }
                float rcp_norm = rsqrtf(sum_sq + 1e-6f);
                for (int d = 0; d < K; d++) {
                    k_normed[tid * K + d] *= rcp_norm;
                }
            }

            __syncthreads();

            // Step 3: compute k_beta = k_normed * beta, zero padded rows
            for (int offset = 0; offset < C * K; offset += blockDim.x) {
                int idx   = tid + offset;
                int row   = idx / K;
                int col   = idx % K;

                float kv  = k_normed[row * K + col];
                float btv = (row < actual_len) ? beta_arr[row] : 0.0f;
                k_beta_sm[row * K + col] = kv * btv;
            }
        }
        __syncthreads();

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: Intra-chunk GEMM — attn = -(k_beta @ k_normed^T) with decay
        //           Only lower triangular (row >= col) is non-zero.
        // ═══════════════════════════════════════════════════════════════════

        {
            int tid        = threadIdx.x;
            int total      = C * C;
            int per_thread = (total + blockDim.x - 1) / blockDim.x;

            for (int flat = tid * per_thread;
                 flat < min((tid + 1) * per_thread, total);
                 flat++) {
                int row = flat / C;
                int col = flat % C;

                // Dot product: k_beta[row] · k_normed[col]
                float sum = 0.0f;
                for (int d = 0; d < K; d++) {
                    sum += k_beta_sm[row * K + d] * k_normed[col * K + d];
                }

                // Apply decay mask: exp(g_cs[row] - g_cs[col]) for row >= col, else 0
                float attn_val = 0.0f;
                if (row >= col) {
                    float g_diff = g_cs[row] - g_cs[col];
                    attn_val     = -sum * expf(g_diff);
                }

                attn_sm[row * C + col] = attn_val;
            }
        }
        __syncthreads();

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Forward substitution + identity addition
        //           Sequential over rows (row i depends on rows 0..i-1).
        // ═══════════════════════════════════════════════════════════════════

        {
            if (threadIdx.x == 0) {
                // row_buf is in shared memory (see layout above)

                // Forward substitution: for each row i, update attn[i][j] using
                // original row values and the submatrix of previous rows.
                for (int i = 1; i < actual_len; i++) {
                    // Save original row values before update
                    for (int j = 0; j < i; j++) {
                        row_buf[j] = attn_sm[i * C + j];
                    }

                    // Update: attn[i][j] = row[j] + sum_m(row[m] * attn[m][j]) for m < i
                    // Python: attn[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
                    //   where row = attn_old[i, :i], sub = attn_updated[:i, :i]
                    for (int j = 0; j < i; j++) {
                        float accum = 0.0f;
                        for (int m = 0; m < i; m++) {
                            accum += row_buf[m] * attn_sm[m * C + j];
                        }
                        attn_sm[i * C + j] = row_buf[j] + accum;
                    }
                }

                // Add identity (only for valid positions, not padded)
                for (int i = 0; i < actual_len; i++) {
                    attn_sm[i * C + i] += 1.0f;
                }
            }
        }
        __syncthreads();

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: Compute core_attn_out per (row, col_v), write to output
        //
        // For each element, compute:
        //   attn_inter = q_scaled @ S       (inter-chunk contribution)
        //   v_nc = v_new - v_prime          (corrected values)
        //   output = attn_inter + attn_qk @ v_nc  (combined)
        //
        // Note: for c=0 with S=0, this simplifies to v_new since both
        // attn_inter and the qk contribution vanish. Phase 4 handles
        // all chunks uniformly.
        // ═══════════════════════════════════════════════════════════════════

        {
            int tid        = threadIdx.x;
            int total      = C * V;
            int per_thread = (total + blockDim.x - 1) / blockDim.x;

            for (int flat_v = tid * per_thread;
                 flat_v < min((tid + 1) * per_thread, total);
                 flat_v++) {

                int row    = flat_v / V;
                int col_v  = flat_v % V;

                if (row >= actual_len || col_v >= V) continue;

                float seq_row   = chunk_start + row;
                int q_base      = (int)(seq_row * num_heads * K) + h * K;

                // ── Load and L2-normalize query for this row ────────────────
                // q_reg is a per-thread register array (K typically 128)
                float q_reg[128];
                float q_l2_sq    = 0.0f;

                for (int d = 0; d < K; d++) {
                    float qv = bf16_to_float(query[q_base + d]);
                    q_reg[d] = qv;
                    q_l2_sq += qv * qv;
                }

                float q_rcp     = rsqrtf(q_l2_sq + 1e-6f) * rcp_sqrt_k;
                float exp_g_row = expf(g_cs[row]);

                // ── attn_inter[row][col_v] = (q_normed * exp(g_cs) @ S)[row][col_v] ──
                float attn_inter_val = 0.0f;
                for (int d = 0; d < K; d++) {
                    float q_scl = q_reg[d] * q_rcp * exp_g_row;
                    attn_inter_val += q_scl * state[state_base + d * V + col_v];
                }

                // ── Compute output contribution from attn_qk ────────────────
                // For j <= row (lower triangle including diagonal):
                //   output[row] += attn_qk[row][j] * v_nc[j]
                // attn_qk is masked to 0 for j > row (upper triangle)
                float output_from_qk = 0.0f;

                for (int j = 0; j <= row; j++) {
                    // ── attn_qk[row][j]: query-key dot product with decay ───
                    float qk_dot_j = 0.0f;
                    for (int d = 0; d < K; d++) {
                        qk_dot_j += q_reg[d] * q_rcp * k_normed[j * K + d];
                    }

                    float g_diff      = g_cs[row] - g_cs[j];
                    float attn_qk_val = qk_dot_j * expf(g_diff);

                    // ── v_nc[j][col_v]: corrected value for row j ───────────
                    // v_new[j] = attn[j][:] · (value * beta)[:, col_v]
                    float v_new_j = 0.0f;
                    for (int ii = 0; ii < actual_len; ii++) {
                        float seq_pos   = chunk_start + ii;
                        int v_idx       = (int)(seq_pos * num_heads * V) + h * V + col_v;
                        float v_val     = bf16_to_float(value[v_idx]) * beta_arr[ii];
                        v_new_j        += attn_sm[j * C + ii] * v_val;
                    }

                    // v_prime[j] = k_cumdecay @ S, computed on-the-fly
                    float v_prime_j = 0.0f;
                    for (int d = 0; d < K; d++) {
                        float k_cd_j = 0.0f;   // k_cumdecay[j][d]
                        for (int ii = 0; ii < actual_len; ii++) {
                            k_cd_j += attn_sm[j * C + ii]
                                     * k_beta_sm[ii * K + d]
                                     * expf(g_cs[ii]);
                        }
                        v_prime_j += k_cd_j * state[state_base + d * V + col_v];
                    }

                    float v_nc_j = v_new_j - v_prime_j;

                    output_from_qk += attn_qk_val * v_nc_j;
                }

                // ── Write final output to global memory ────────────────────
                float out_val = attn_inter_val + output_from_qk;
                int out_idx   = (int)(seq_row * num_heads * V) + h * V + col_v;
                output[out_idx] = float_to_bf16(out_val);

            }  // end flat_v loop
        }

        // ═══════════════════════════════════════════════════════════════════
        // PHASE 5: State update — S_new = S * exp(g_cs[-1]) + correction
        //           Correction = sum_j exp_diff[j] * k_normed[j]^T ⊗ v_nc[j]
        // ═══════════════════════════════════════════════════════════════════

        {
            int tid        = threadIdx.x;
            int total      = K * V;
            int per_thread = (total + blockDim.x - 1) / blockDim.x;

            for (int flat_s = tid * per_thread;
                 flat_s < min((tid + 1) * per_thread, total);
                 flat_s++) {
                int d     = flat_s / V;
                int col_v = flat_s % V;

                float exp_g_last = expf(g_cs[actual_len - 1]);

                // S[d][col_v] *= exp(g_cs[-1])
                float s_val = state[state_base + d * V + col_v] * exp_g_last;

                // Add: sum_j exp_diff[j] * k_normed[j][d] * v_nc[j][col_v]
                for (int j = 0; j < actual_len; j++) {
                    float exp_diff_j = expf(g_cs[actual_len - 1] - g_cs[j]);

                    // v_new[j][col_v] = attn @ (v * beta)
                    float v_new_j = 0.0f;
                    for (int ii = 0; ii < actual_len; ii++) {
                        float seq_pos   = chunk_start + ii;
                        int v_idx       = (int)(seq_pos * num_heads * V) + h * V + col_v;
                        float v_val     = bf16_to_float(value[v_idx]) * beta_arr[ii];
                        v_new_j        += attn_sm[j * C + ii] * v_val;
                    }

                    // v_prime[j][col_v] = k_cumdecay @ S (on-the-fly)
                    float v_prime_j = 0.0f;
                    for (int dd = 0; dd < K; dd++) {
                        float k_cd_j = 0.0f;   // k_cumdecay[j][dd]
                        for (int ii = 0; ii < actual_len; ii++) {
                            k_cd_j += attn_sm[j * C + ii]
                                     * k_beta_sm[ii * K + dd]
                                     * expf(g_cs[ii]);
                        }
                        v_prime_j += k_cd_j * state[state_base + dd * V + col_v];
                    }

                    float v_nc_j = v_new_j - v_prime_j;

                    s_val += exp_diff_j * k_normed[j * K + d] * v_nc_j;
                }

                state[state_base + d * V + col_v] = s_val;
            }
        }

    }  // end chunk loop

}

} // extern "C"
