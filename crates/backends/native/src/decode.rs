use std::sync::{Arc, Mutex};

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
use cuda_async::device_operation::DeviceOperation;
use infers_model::LayerType;

use crate::engine::ForwardEngine;
use crate::resources::{GpuResources, DecodeState};
use crate::probe;
use crate::sync;
use crate::sample::Xoshiro256PlusPlus;

/// Raw-pointer wrapper for crossing `Send` boundaries in cuda-async closures.
/// Sound because `and_then::execute()` calls closures sequentially on the same thread — no aliasing possible.
#[allow(clippy::missing_safety_doc)]
pub struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

impl ForwardEngine {
    /// Run paged single-token decode — zero CPU round-trips.
    ///
    /// Reads K/V from the paged cache, computes attention, and returns
    /// the sampled token. Unlike the legacy `decode`, this path uses
    /// paged attention kernels that operate entirely on GPU.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    /// * `seq_id` — Sequence ID from PagedKvManager
    ///
    /// # Returns
    /// The sampled token ID for the next generated token.
    // @lat: [[lat.md/lat#Forward Engine#Paged Decode Path]]
    pub fn decode_paged(
        &mut self,
        _stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
        sampling_config: &infers_scheduler::SamplingConfig,
        token_history: &[u32],
        num_prompt_tokens: usize,
        rng: &mut Xoshiro256PlusPlus,
        step: usize,
    ) -> Result<u32> {
        let res = &self.resources;
        let span = tracing::info_span!("decode");
        let _enter = span.enter();
        let num_gpus = res.metadata.len();

        // GPU timing: create events on each GPU's context
        let gpu_start_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = res.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU start event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_end_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = res.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU start event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record start events
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            gpu_start_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record start event on GPU {gpu_idx}: {:?}", e))?;
        }

        let config = &res.config;
        let head_dim = config.head_dim;

        // Probe instrumentation (cached at engine init — avoids per-step env::var calls)
        let probe_cfg = &res.probe_config;

        // Dynamically allocate pages as needed for the target position,
        // then read the (possibly updated) block table and cached-token count.
        // Use a scope to drop the mutable borrow before using `self` again.
        let (page_size, num_cached_tokens, block_table_i32): (usize, i32, Vec<i32>) = {
            let mut mgr = self.paged_kv_manager.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?
                .lock()
                .unwrap();
            let ps = mgr.page_size();

            // Allocate pages up to the page index that `position` falls in.
            let needed_pages = (position as usize / ps) + 1;
            let current_pages = mgr.block_table(seq_id)?.len();
            for _ in current_pages..needed_pages {
                mgr.append_page(seq_id)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate KV page for decode: {:?}", e))?;
            }

            let cached = mgr.num_tokens(seq_id)? as i32 + 1;  // +1 for current decode token
            let bt: Vec<i32> = mgr.block_table(seq_id)?.iter().map(|p| *p as i32).collect();
            (ps, cached, bt)
        };
        let position_i32 = [position as i32];

        // Write block table and position into pre-allocated staging buffers on each GPU (zero-alloc)
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let ws = &mut self.workspaces[gpu_idx];

            // Write block table into staging buffer via memcpy_htod
            gpu_stream.memcpy_htod(&block_table_i32, &mut ws.block_table_staging)
                .map_err(|e| anyhow::anyhow!("Failed to copy block table to staging: {e}"))?;

            // Write position into staging buffer via memcpy_htod
            gpu_stream.memcpy_htod(&position_i32, &mut ws.position_staging)
                .map_err(|e| anyhow::anyhow!("Failed to copy position to staging: {e}"))?;

            // Write num_cached_tokens into staging buffer via memcpy_htod (CUDA graph compatible)
            let num_cached_tokens_u32 = [num_cached_tokens as u32];
            gpu_stream.memcpy_htod(&num_cached_tokens_u32, &mut ws.num_cached_tokens_staging)
                .map_err(|e| anyhow::anyhow!("Failed to copy num_cached_tokens to staging: {e}"))?;
        }
        // Ensure page pools allocated on each GPU
        for gpu_idx in 0..num_gpus {
            for cache in &mut self.paged_kv_caches[gpu_idx] {
                cache.ensure_allocated(res.streams.get(gpu_idx).unwrap())?;
            }
        }

        // Embed single token on each GPU — using pre-allocated staging + output buffers (zero-alloc)
        let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let w = &res.metadata[gpu_idx];
            let embed_weight = w.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = res.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

            // Convert token_id to i32 and write into staging buffer, then embed into ws.embed_out
            let token_ids_i32 = [token_id as i32];
            {
                let ws = &mut self.workspaces[gpu_idx];
                crate::embedding::embed_tokens_into(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide, &embed_table,
                    &token_ids_i32, &mut ws.token_ids_staging, &mut ws.embed_out,
                    1, config.hidden_size,
                )?;
            }
            probe::dump(&gpu_stream, probe_cfg, usize::MAX, gpu_idx, "embed.output", &self.workspaces[gpu_idx].embed_out, &[1, config.hidden_size], "decode");
            hidden_states.push(self.workspaces[gpu_idx].embed_out.clone());
        }

        // Per-GPU sharded head counts
        let num_kv_heads_per_gpu = config.num_key_value_heads / num_gpus;
        let num_heads_per_gpu = config.num_attention_heads / num_gpus;
        let sharded_intermediate = config.intermediate_size / num_gpus;

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            let layer_type = config.get_layer_type(layer_idx);
            let stage_prefix = match layer_type {
                LayerType::FullAttention => "attn",
                LayerType::GatedDeltaNet => "gdn",
            };
            let layer_span = tracing::info_span!(
                "layer",
                layer_idx,
                layer_type = match layer_type {
                    LayerType::FullAttention => "full_attn",
                    LayerType::GatedDeltaNet => "gdn",
                }
            );
            let _layer_enter = layer_span.enter();

            // Dump hidden input at start of layer
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

           // Phase A: Attention/GDN on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let w = &res.metadata[gpu_idx];
                let layer = &w.layers[layer_idx];

                // Norm1
                let norm1_weight = res.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
                crate::norm::rms_norm_into(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut self.workspaces[gpu_idx].norm1_out,
                    &hidden_states[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &self.workspaces[gpu_idx].norm1_out, &[1, config.hidden_size], "decode");

                // Attention or GDN (decode versions)
                match config.get_layer_type(layer_idx) {
                    LayerType::GatedDeltaNet => {
                        let gdn_weights = layer.gdn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                        {
                            let ws = &mut self.workspaces[gpu_idx];
                            let mut ps = Some(&mut ws.partial_sums);
                            crate::gdn::decode_forward(
                                gemm, &gpu_stream,
                                &res.per_gpu_kernels[gpu_idx].oxide,
                                gdn_weights, &ws.norm1_out,
                                &mut self.gdn_states[gpu_idx][layer_idx],
                                config.hidden_size, config.as_ref(), res.group_size,
                                &res.weight_caches[gpu_idx],
                                layer_idx,
                                gpu_idx,
                                probe_cfg,
                                &mut ws.gdn, &mut ws.attn_out,
                                &mut ps,
                            )?;
                        }
                    }
                    LayerType::FullAttention => {
                        let attn_weights = layer.attn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                        {
                            let ws = &mut self.workspaces[gpu_idx];
                            let mut ps = Some(&mut ws.partial_sums);
                            crate::attention::decode_forward_paged(
                                gemm, &gpu_stream,
                                &res.per_gpu_kernels[gpu_idx].oxide,
                                attn_weights, &ws.norm1_out,
                                &mut self.paged_kv_caches[gpu_idx][layer_idx],
                                &ws.block_table_staging, &ws.position_staging,
                                position,
                                &ws.num_cached_tokens_staging,
                                head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                                config.rope_theta, config.partial_rotary_factor,
                                config.rms_norm_eps, res.group_size, &res.weight_caches[gpu_idx],
                                config.hidden_size,
                                config.attn_output_gate,
                                layer_idx,
                                gpu_idx,
                                probe_cfg,
                                res.rope_cos.as_ref().map(|v| &v[gpu_idx]),
                                res.rope_sin.as_ref().map(|v| &v[gpu_idx]),
                                &mut ws.attn, &mut ws.attn_out,
                                &mut ps,
                                &mut ws.rope_position_staging,
                                &position_i32,
                            )?;
                        }
                    }
                };

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.o_proj", stage_prefix), &self.workspaces[gpu_idx].attn_out, &[1, config.hidden_size], "decode");
            }

            // All-reduce attention outputs across GPUs
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(
                    &res.nccl, &gpu_stream, &mut self.workspaces[gpu_idx].attn_out,
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.after_ar", stage_prefix), &self.workspaces[gpu_idx].attn_out, &[1, config.hidden_size], "decode");
            }

            // Phase B: Residual add on each GPU (zero-alloc via workspace + swap)
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let ws = &mut self.workspaces[gpu_idx];
                crate::add::add_into(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.residual_buf,
                    &hidden_states[gpu_idx], &ws.attn_out,
                )?;
                std::mem::swap(&mut hidden_states[gpu_idx], &mut ws.residual_buf);
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "residual.attn", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase C: MLP on each GPU (column-parallel gate/up, row-parallel down)
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let w = &res.metadata[gpu_idx];
                let mlp_weights = &w.layers[layer_idx].mlp;
                let ws = &mut self.workspaces[gpu_idx];

                // Norm2 — into workspace
                let norm2_weight = res.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
                crate::norm::rms_norm_into(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.norm2_out,
                    &hidden_states[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.norm2", &ws.norm2_out, &[1, config.hidden_size], "decode");

                // Gate projection — into workspace.mlp_gate
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name,
                    &ws.norm2_out, &mut ws.mlp_gate,
                    1, sharded_intermediate, config.hidden_size,
                    res.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.gate_proj", &ws.mlp_gate, &[1, config.intermediate_size / num_gpus], "decode");

                // Up projection — into workspace.mlp_up
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name,
                    &ws.norm2_out, &mut ws.mlp_up,
                    1, sharded_intermediate, config.hidden_size,
                    res.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.up_proj", &ws.mlp_up, &[1, config.intermediate_size / num_gpus], "decode");

                // up * SiLU(gate) = SwiGLU — into workspace.mlp_silu
                res.per_gpu_kernels[gpu_idx].oxide.launch_silu_glu_bf16(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide.cc_stream(), &ws.mlp_up, &ws.mlp_gate, &mut ws.mlp_silu, sharded_intermediate as u32,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.silu", &ws.mlp_silu, &[1, config.intermediate_size / num_gpus], "decode");

                // Down projection — into workspace.mlp_out
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name,
                    &ws.mlp_silu, &mut ws.mlp_out,
                    1, config.hidden_size, sharded_intermediate,
                    res.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.down_raw", &ws.mlp_out, &[1, config.hidden_size], "decode");
            }

            // All-reduce MLP outputs across GPUs
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(
                    &res.nccl, &gpu_stream, &mut self.workspaces[gpu_idx].mlp_out,
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "mlp.down_ar", &self.workspaces[gpu_idx].mlp_out, &[1, config.hidden_size], "decode");
            }

            // Phase D: Residual add on each GPU (zero-alloc via workspace + swap)
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let ws = &mut self.workspaces[gpu_idx];
                crate::add::add_into(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.residual_buf,
                    &hidden_states[gpu_idx], &ws.mlp_out,
                )?;
                std::mem::swap(&mut hidden_states[gpu_idx], &mut ws.residual_buf);
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, "residual.mlp", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }
        }

        // ================================================================
        // Final norm + LM head + sample on GPU 0
        // ================================================================
        let final_stream = res.streams.get(0).unwrap().clone();
        let final_weights = &res.metadata[0];
        let final_hidden = hidden_states.into_iter().next().unwrap();

        // Final norm — write into workspace.norm1_out (reusing buffer since layer loop is done)
        let final_norm_weight = final_weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = res.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
        crate::norm::rms_norm_into(
            &final_stream, &res.per_gpu_kernels[0].oxide,
            &mut self.workspaces[0].norm1_out,
            &final_hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        probe::dump(&final_stream, probe_cfg, config.num_hidden_layers - 1, 0, "final.norm", &self.workspaces[0].norm1_out, &[1, config.hidden_size], "decode");

        // LM head — write into workspace.logits
        let lm_head_weight = final_weights.lm_head.as_ref()
            .or_else(|| final_weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        {
            let ws = &mut self.workspaces[0];
            let mut lm_head_ps = Some(&mut ws.lm_head_partial_sums);
            crate::gemm_dispatch::gemm_projection_cached(
                &res.gemm_engines[0], &res.per_gpu_kernels[0].oxide, &final_stream,
                &res.weight_caches[0],
                &lm_head_weight.name,
                &ws.norm1_out, &mut ws.logits,
                1, config.vocab_size, config.hidden_size,
                res.group_size,
                &mut lm_head_ps,
            )?;
        }

        probe::dump(&final_stream, probe_cfg, config.num_hidden_layers - 1, 0, "final.logits", &self.workspaces[0].logits, &[1, config.vocab_size], "decode");

        // Debug logit dump (only when INFERS_DUMP_LOGITS is set)
        // @lat: [[lat.md/lat#Forward Engine#Logit Dump Debug Tool]]
        if std::env::var("INFERS_DUMP_LOGITS").is_ok() {
            let logits_bf16: Vec<bf16> = final_stream.clone_dtoh(&self.workspaces[0].logits)
                .map_err(|e| anyhow::anyhow!("Failed to download logits for dump: {:?}", e))?;

            let logits_f32: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();

            // Find top-5 by sorting descending
            let mut indexed: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top5: Vec<_> = indexed.iter().take(5).map(|&(idx, val)| (idx as u32, val)).collect();
            let max_logit = indexed[0].1;
            let min_logit = indexed.last().unwrap().1;

            // Standard deviation
            let mean: f32 = logits_f32.iter().sum::<f32>() / logits_f32.len() as f32;
            let variance: f32 = logits_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / logits_f32.len() as f32;
            let std_logit = variance.sqrt();

            eprintln!(
                "[LOGIT-DUMP] step={} top5=[{:?}] max_logit={:.4} min_logit={:.4} logit_std={:.4}",
                step, top5, max_logit, min_logit, std_logit,
            );
        }

        // Sample (BF16 argmax)
        let sampled = crate::sample::sample_with_config(
            &final_stream, &self.workspaces[0].logits.as_view(), &res.per_gpu_kernels[0].oxide,
            sampling_config, token_history, num_prompt_tokens, rng,
        )?;

        // Record the new token in the KV manager so the next decode step
        // sees the correct block table and cached-token count.
        if let Some(m) = &self.paged_kv_manager {
            m.lock().unwrap().add_token(seq_id)
                .map_err(|e| anyhow::anyhow!("Failed to record decode token: {:?}", e))?;
        }

        // Record end events and report GPU timing
        let mut max_gpu_ms: f32 = 0.0;
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            gpu_end_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record end event on GPU {gpu_idx}: {:?}", e))?;
            gpu_end_events[gpu_idx].synchronize()
                .map_err(|e| anyhow::anyhow!("Failed to synchronize end event on GPU {gpu_idx}: {:?}", e))?;
            let gpu_ms = gpu_start_events[gpu_idx].elapsed_ms(&gpu_end_events[gpu_idx])
                .unwrap_or(0.0);
            max_gpu_ms = max_gpu_ms.max(gpu_ms);
        }
        tracing::info!(gpu_time_ms = max_gpu_ms as f64, phase = "decode", "GPU execution complete");

        Ok(sampled)
    }

    /// Run paged single-token decode as a cuda-async `DeviceOperation` pipeline.
    ///
    /// This is the async-friendly counterpart to [[Self::decode_paged]]. It wraps
    /// the decode logic in a `with_context` closure that captures `Arc<GpuResources>`
    /// (immutable) and `Arc<Mutex<DecodeState>>` (mutable per-sequence state).
    // @lat: [[lat.md/lat#Forward Engine#Paged Decode Pipeline]]
    pub fn decode_paged_async(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
        sampling_config: &infers_scheduler::SamplingConfig,
        token_history: &[u32],
        num_prompt_tokens: usize,
        rng: &mut Xoshiro256PlusPlus,
        step: usize,
    ) -> Result<u32> {
        use cuda_async::device_operation::{DeviceOperation, with_context, value};

        // Clone Arc<GpuResources> — cheap reference-count increment.
        let res = self.resources.clone();

        // Build DecodeState from current engine state for the closure to capture.
        // The mutable state is temporarily moved out of ForwardEngine and restored after.
        let decode_state = DecodeState {
            workspaces: std::mem::take(&mut self.workspaces),
            paged_kv_caches: std::mem::take(&mut self.paged_kv_caches),
            gdn_states: std::mem::take(&mut self.gdn_states),
            paged_kv_manager: self.paged_kv_manager.clone(),
        };

        // Wrap mutable state in Arc<Mutex<>> for sharing across closure.
        let state = Arc::new(Mutex::new(decode_state));

        let res_clone = res.clone();
        let state_clone = state.clone();

        // Build the with_context closure that runs the full decode.
        // This closure captures Arc<GpuResources> (Send+Sync) and Arc<Mutex<DecodeState>> (Send).
        let pipeline = with_context(move |_ctx| {
            // Lock the mutable state — only held during sequential execution, no concurrent access.
            let mut st = state_clone.lock().unwrap();

            // Call decode_paged_async_inner directly instead of going through ForwardEngine.
            let result = decode_paged_async_inner(
                &res_clone,
                &mut st,
                stream,
                token_id,
                position,
                seq_id,
                sampling_config,
                token_history,
                num_prompt_tokens,
                rng,
                step,
            );

            // Convert anyhow::Result<u32> to DeviceOperation-compatible result.
            match result {
                Ok(sampled) => value(sampled),
                Err(e) => value::<u32>(
                    panic!("decode error: {:?}", e)  // unwrap inside closure — panics propagate through with_context
                ),
            }
        });

        let sampled = pipeline.sync()?;

        // Restore mutable state from Arc<Mutex<>> back to ForwardEngine.
        let st = Arc::try_unwrap(state)
            .expect("DecodeState should be the only Arc reference at this point")
            .into_inner()
            .expect("Mutex should not be poisoned");
        self.workspaces = st.workspaces;
        self.paged_kv_caches = st.paged_kv_caches;
        self.gdn_states = st.gdn_states;

        Ok(sampled)
    }

    /// Decode a token using a standalone [[DecodeState]].

    /// Takes the engine's `&mut self` to access the shared PagedKvManager and
    /// PagedKvCaches (swapped temporarily into state), plus a per-sequence
    /// DecodeState with workspaces and GDN states.

    /// The returned DecodeState can be reused for subsequent decode steps on the
    /// same sequence, or dropped when the sequence is finished.

    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    /// * `seq_id` — Sequence ID from PagedKvManager
    /// * `state` — Per-sequence DecodeState (workspaces + GDN states)
    /// * `sampling_config` — Sampling configuration
    /// * `token_history` — Full token history for sampling
    /// * `num_prompt_tokens` — Number of prompt tokens (for position offset)
    /// * `rng` — RNG for sampling
    /// * `step` — Step counter for debugging

    /// # Returns
    /// Tuple of (sampled_token, DecodeState) — state is returned for reuse.
    // @lat: [[lat.md/lat#Forward Engine#GpuResources and DecodeState Architecture#Per-Sequence DecodeState Management]]
    pub fn decode_with_state(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
        state: DecodeState,
        sampling_config: &infers_scheduler::SamplingConfig,
        token_history: &[u32],
        num_prompt_tokens: usize,
        rng: &mut Xoshiro256PlusPlus,
        step: usize,
    ) -> Result<(u32, DecodeState)> {
        use cuda_async::device_operation::{DeviceOperation, with_context, value};

        let res = self.resources.clone();

        // Swap engine's shared state into the per-sequence DecodeState.
        // paged_kv_manager is CPU-side bookkeeping (page pool + block tables).
        // paged_kv_caches are GPU page pools shared across sequences.
        let mgr = self.paged_kv_manager.clone();
        let caches = std::mem::replace(&mut self.paged_kv_caches, Vec::new());

        let mut st = state;
        st.paged_kv_manager = mgr;
        st.paged_kv_caches = caches;

        let state_arc = Arc::new(Mutex::new(st));
        let res_clone = res.clone();
        let state_clone = state_arc.clone();

        let pipeline = with_context(move |_ctx| {
            let mut st = state_clone.lock().unwrap();
            let result = decode_paged_async_inner(
                &res_clone,
                &mut st,
                stream,
                token_id,
                position,
                seq_id,
                sampling_config,
                token_history,
                num_prompt_tokens,
                rng,
                step,
            );
            match result {
                Ok(sampled) => value(sampled),
                Err(e) => value::<u32>(panic!("decode error: {:?}", e)),
            }
        });

        let sampled = pipeline.sync()?;

        // Extract state and restore engine's shared state.
        let mut st = Arc::try_unwrap(state_arc)
            .expect("DecodeState should be the only Arc reference")
            .into_inner()
            .expect("Mutex should not be poisoned");

        // Restore engine's shared state from the DecodeState (via swap to avoid move).
        self.paged_kv_caches = std::mem::replace(&mut st.paged_kv_caches, Vec::new());

        Ok((sampled, st))
    }

    /// Build a spawnable `DeviceOperation` for decode.
    ///
    /// All captures are owned (`'static + Send`), enabling
    /// `tokio::spawn(pipeline.into_future())` for continuous batching.
    ///
    /// The caller provides owned values — no references to the engine are captured.
    /// The `DecodeState` is moved into the closure and returned alongside the
    /// sampled token, allowing it to be reused across decode steps.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches (owned Arc)
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    /// * `seq_id` — Sequence ID from PagedKvManager
    /// * `state` — Per-sequence DecodeState (owned, with paged_kv_manager set from engine)
    /// * `sampling_config` — Sampling configuration (owned)
    /// * `token_history` — Full token history for sampling (owned)
    /// * `num_prompt_tokens` — Number of prompt tokens (for position offset)
    /// * `rng` — RNG for sampling (owned)
    /// * `step` — Step counter for debugging
    ///
    /// # Returns
    /// A pipeline producing `(sampled_token, DecodeState)` via `.sync()` or `.await`.
    pub fn decode_spawnable(
        &self,
        stream: Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
        mut state: DecodeState,
        sampling_config: infers_scheduler::SamplingConfig,
        token_history: Vec<u32>,
        num_prompt_tokens: usize,
        mut rng: Xoshiro256PlusPlus,
        step: usize,
    ) -> impl DeviceOperation<Output = Result<(u32, DecodeState), String>> {
        use cuda_async::device_operation::{with_context, value};

        let res = self.resources.clone();

        with_context(move |_ctx| {
            let result = decode_paged_async_inner(
                &res,
                &mut state,
                &stream,
                token_id,
                position,
                seq_id,
                &sampling_config,
                &token_history,
                num_prompt_tokens,
                &mut rng,
                step,
            );
            match result {
                Ok(sampled) => value(Ok((sampled, state))),
                Err(e) => value(Err(format!("{:?}", e))),
            }
        })
    }

    /// Batched decode for M=2 sequences. Amortizes INT4 GEMM weight bandwidth
    /// by processing 2 tokens simultaneously through all GEMMs.
    // @lat: [[lat.md/lat#Forward Engine#Batched Decode]]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batched(
        &mut self,
        tokens: &[u32],             // M token IDs
        positions: &[u32],         // M positions
        seq_ids: &[infers_kv::SequenceId], // M sequence IDs
        states: &mut [DecodeState],       // M states
        sampling_configs: &[infers_scheduler::SamplingConfig], // M configs
        token_histories: &[Vec<u32>],     // M histories
        num_prompt_tokens: &[usize],      // M counts
        rngs: &mut [Xoshiro256PlusPlus],  // M RNGs
        _step: usize,
    ) -> Result<Vec<u32>> {
        const M: usize = 2;
        assert_eq!(tokens.len(), M);
        assert_eq!(positions.len(), M);
        assert_eq!(seq_ids.len(), M);
        assert_eq!(states.len(), M);

        let res = &self.resources;
        let config = &res.config;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let sharded_intermediate = config.intermediate_size / res.metadata.len();
        let num_gpus = res.metadata.len();
        let group_size = res.group_size;

        // Allocate batched buffers per GPU: [M, hidden_size] or [M, intermediate]
        let mut batched_hidden: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_residual_buf: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_norm1_out: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_norm2_out: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_attn_out: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_mlp_gate: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_mlp_up: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_mlp_silu: Vec<CudaSlice<bf16>> = Vec::new();
        let mut batched_mlp_out: Vec<CudaSlice<bf16>> = Vec::new();

        // Partial sums per GPU — sized for largest GEMM (lm_head with M=2)
        const K_SPLIT: u32 = 20;
        let max_ps = K_SPLIT as usize * vocab_size * M;
        let mut batched_partial_sums: Vec<CudaSlice<f32>> = Vec::new();

        for gpu_idx in 0..num_gpus {
            let stream = res.streams.get(gpu_idx).unwrap().clone();
            batched_hidden.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_residual_buf.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_norm1_out.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_norm2_out.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_attn_out.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_mlp_gate.push(unsafe { stream.alloc::<bf16>(M * sharded_intermediate)? });
            batched_mlp_up.push(unsafe { stream.alloc::<bf16>(M * sharded_intermediate)? });
            batched_mlp_silu.push(unsafe { stream.alloc::<bf16>(M * sharded_intermediate)? });
            batched_mlp_out.push(unsafe { stream.alloc::<bf16>(M * hidden_size)? });
            batched_partial_sums.push(unsafe { stream.alloc::<f32>(max_ps)? });
        }

        // Embed M tokens on each GPU into batched_hidden [M, hidden_size]
        for gpu_idx in 0..num_gpus {
            let stream = res.streams.get(gpu_idx).unwrap().clone();
            let embed_weight = res.metadata[gpu_idx].embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = res.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight not in cache"))?;
            let token_ids_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            // Allocate M-sized staging buffer (workspace's token_ids_staging is sized for M=1)
            let mut token_ids_staging = stream.alloc_zeros::<i32>(M)?;
            crate::embedding::embed_tokens_into(
                &stream, &res.per_gpu_kernels[gpu_idx].oxide, &embed_table,
                &token_ids_i32, &mut token_ids_staging,
                &mut batched_hidden[gpu_idx], M, hidden_size,
            )?;
        }

        // Per-GPU sharded head counts
        let num_kv_heads_per_gpu = config.num_key_value_heads / num_gpus;
        let num_heads_per_gpu = config.num_attention_heads / num_gpus;

        // Allocate pages for each sequence
        for seq_i in 0..M {
            let mut mgr = states[seq_i].paged_kv_manager.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?
                .lock().unwrap();
            let ps = mgr.page_size();
            let needed_pages = (positions[seq_i] as usize / ps) + 1;
            let current_pages = mgr.block_table(seq_ids[seq_i])?.len();
            for _ in current_pages..needed_pages {
                mgr.append_page(seq_ids[seq_i])
                    .map_err(|e| anyhow::anyhow!("Failed to allocate KV page: {:?}", e))?;
            }
            for gpu_idx in 0..num_gpus {
                for cache in &mut states[seq_i].paged_kv_caches[gpu_idx] {
                    cache.ensure_allocated(res.streams.get(gpu_idx).unwrap())?;
                }
            }
        }

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            let layer_type = config.get_layer_type(layer_idx);

            // Phase A: Norm1 + attention/GDN on each GPU
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let oxide = &res.per_gpu_kernels[gpu_idx].oxide;
                let layer = &res.metadata[gpu_idx].layers[layer_idx];

                // Norm1 — batched RMSNorm on [M, hidden_size]
                let norm1_weight = res.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight not in cache"))?;
                crate::norm::rms_norm_into(
                    &stream, oxide, &mut batched_norm1_out[gpu_idx],
                    &batched_hidden[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, hidden_size,
                )?;

                if layer_type == LayerType::GatedDeltaNet {
                    // GDN: call existing decode_forward per-sequence (M=2 calls)
                    let gdn_weights = layer.gdn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GDN weights not found"))?;
                    for seq_i in 0..M {
                        let ws = &mut states[seq_i].workspaces[gpu_idx];
                        let gdn_state = &mut states[seq_i].gdn_states[gpu_idx][layer_idx];

                        // Copy row seq_i from batched_norm1_out into workspace
                        let src = batched_norm1_out[gpu_idx].slice(seq_i * hidden_size..(seq_i + 1) * hidden_size);
                        stream.memcpy_dtod(&src, &mut ws.norm1_out)
                            .map_err(|e| anyhow::anyhow!("Copy norm1 row: {e}"))?;

                        // Run per-sequence GDN decode_forward (takes [1,hidden] input/output)
                        let mut ps = Some(&mut batched_partial_sums[gpu_idx]);
                        crate::gdn::decode_forward(
                            gemm, &stream, oxide, gdn_weights,
                            &ws.norm1_out, gdn_state, hidden_size, config.as_ref(),
                            group_size, &res.weight_caches[gpu_idx], layer_idx, gpu_idx,
                            &res.probe_config, &mut ws.gdn, &mut ws.attn_out,
                            &mut ps,
                        )?;

                        // Copy from ws.attn_out (row seq_i) into batched_attn_out
                        let mut dst = batched_attn_out[gpu_idx].slice_mut(seq_i * hidden_size..(seq_i + 1) * hidden_size);
                        stream.memcpy_dtod(&ws.attn_out, &mut dst)
                            .map_err(|e| anyhow::anyhow!("Copy GDN output: {e}"))?;
                    }
                } else {
                    // FullAttention: call existing decode_forward_paged per-sequence (M=2 calls)
                    let attn_weights = layer.attn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Attention weights not found"))?;

                    for seq_i in 0..M {
                        let ws = &mut states[seq_i].workspaces[gpu_idx];
                        let position = positions[seq_i];
                        let seq_id = seq_ids[seq_i];

                        // Copy row seq_i from batched_norm1_out into workspace
                        let src = batched_norm1_out[gpu_idx].slice(seq_i * hidden_size..(seq_i + 1) * hidden_size);
                        stream.memcpy_dtod(&src, &mut ws.norm1_out)
                            .map_err(|e| anyhow::anyhow!("Copy norm1 row: {e}"))?;

                        // Get block table and position for this sequence
                        let mut mgr = states[seq_i].paged_kv_manager.as_ref()
                            .unwrap().lock().unwrap();
                        let bt = mgr.block_table(seq_id)
                            .map_err(|e| anyhow::anyhow!("Failed to get block table: {:?}", e))?;
                        let bt_i32: Vec<i32> = bt.iter().map(|p| *p as i32).collect();
                        stream.memcpy_htod(&bt_i32, &mut ws.block_table_staging)
                            .map_err(|e| anyhow::anyhow!("Copy block table: {e}"))?;

                        let num_cached = mgr.num_tokens(seq_id)? as i32 + 1;
                        stream.memcpy_htod(&[num_cached as u32], &mut ws.num_cached_tokens_staging)
                            .map_err(|e| anyhow::anyhow!("Copy num_cached: {e}"))?;
                        stream.memcpy_htod(&[position as i32], &mut ws.position_staging)
                            .map_err(|e| anyhow::anyhow!("Copy position: {e}"))?;

                        let page_size = mgr.page_size();
                        let mut ps = Some(&mut batched_partial_sums[gpu_idx]);
                        crate::attention::decode_forward_paged(
                            gemm, &stream, oxide, attn_weights,
                            &ws.norm1_out,
                            &mut states[seq_i].paged_kv_caches[gpu_idx][layer_idx],
                            &ws.block_table_staging, &ws.position_staging, position,
                            &ws.num_cached_tokens_staging,
                            config.head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                            config.rope_theta, config.partial_rotary_factor,
                            config.rms_norm_eps, group_size, &res.weight_caches[gpu_idx],
                            hidden_size, config.attn_output_gate, layer_idx, gpu_idx,
                            &res.probe_config,
                            res.rope_cos.as_ref().map(|v| &v[gpu_idx]),
                            res.rope_sin.as_ref().map(|v| &v[gpu_idx]),
                            &mut ws.attn, &mut ws.attn_out,
                            &mut ps, &mut ws.rope_position_staging,
                            &[position as i32],
                        )?;

                        // Copy from ws.attn_out (row seq_i) into batched_attn_out
                        let mut dst = batched_attn_out[gpu_idx].slice_mut(seq_i * hidden_size..(seq_i + 1) * hidden_size);
                        stream.memcpy_dtod(&ws.attn_out, &mut dst)
                            .map_err(|e| anyhow::anyhow!("Copy attn output: {e}"))?;
                    }
                }

            }

            // All-reduce attention outputs across GPUs — batched buffer [M, hidden]
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(&res.nccl, &stream, &mut batched_attn_out[gpu_idx])?;
            }

            // Phase B: Residual add — batched element-wise add into [M, hidden]
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                crate::add::add_into(
                    &stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut batched_residual_buf[gpu_idx],
                    &batched_hidden[gpu_idx], &batched_attn_out[gpu_idx],
                )?;
                std::mem::swap(&mut batched_hidden[gpu_idx], &mut batched_residual_buf[gpu_idx]);
            }

            // Phase C: MLP on each GPU — batched GEMMs [M, hidden] → [M, intermediate]
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let oxide = &res.per_gpu_kernels[gpu_idx].oxide;
                let mlp_weights = &res.metadata[gpu_idx].layers[layer_idx].mlp;

                // Norm2 — batched RMSNorm on [M, hidden_size]
                let norm2_weight = res.weight_caches[gpu_idx].get_bf16(
                    &res.metadata[gpu_idx].layers[layer_idx].norm2.name,
                )
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight not in cache"))?;
                crate::norm::rms_norm_into(
                    &stream, oxide, &mut batched_norm2_out[gpu_idx],
                    &batched_hidden[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, hidden_size,
                )?;

                // Gate projection — batched [M, sharded_intermediate]
                let mut ps = Some(&mut batched_partial_sums[gpu_idx]);
                crate::gemm_dispatch::gemm_projection_cached_m(
                    gemm, oxide, &stream, &res.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name, &batched_norm2_out[gpu_idx],
                    &mut batched_mlp_gate[gpu_idx], M, sharded_intermediate, hidden_size,
                    group_size, &mut ps,
                )?;

                // Up projection — batched [M, sharded_intermediate]
                let mut ps = Some(&mut batched_partial_sums[gpu_idx]);
                crate::gemm_dispatch::gemm_projection_cached_m(
                    gemm, oxide, &stream, &res.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name, &batched_norm2_out[gpu_idx],
                    &mut batched_mlp_up[gpu_idx], M, sharded_intermediate, hidden_size,
                    group_size, &mut ps,
                )?;

                // SiLU+GLU — batched over [M * sharded_intermediate]
                oxide.launch_silu_glu_bf16(
                    &stream, &oxide.cc_stream(),
                    &batched_mlp_up[gpu_idx], &batched_mlp_gate[gpu_idx],
                    &mut batched_mlp_silu[gpu_idx], (M * sharded_intermediate) as u32,
                )?;

                // Down projection — batched [M, hidden_size]
                let mut ps = Some(&mut batched_partial_sums[gpu_idx]);
                crate::gemm_dispatch::gemm_projection_cached_m(
                    gemm, oxide, &stream, &res.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name, &batched_mlp_silu[gpu_idx],
                    &mut batched_mlp_out[gpu_idx], M, hidden_size, sharded_intermediate,
                    group_size, &mut ps,
                )?;
            }

            // All-reduce MLP outputs across GPUs — batched buffer [M, hidden]
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(&res.nccl, &stream, &mut batched_mlp_out[gpu_idx])?;
            }

            // Phase D: Residual add — batched element-wise add into [M, hidden]
            for gpu_idx in 0..num_gpus {
                let stream = res.streams.get(gpu_idx).unwrap().clone();
                crate::add::add_into(
                    &stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &mut batched_residual_buf[gpu_idx],
                    &batched_hidden[gpu_idx], &batched_mlp_out[gpu_idx],
                )?;
                std::mem::swap(&mut batched_hidden[gpu_idx], &mut batched_residual_buf[gpu_idx]);
            }
        }

        // ================================================================
        // Final norm + LM head + sample on GPU 0
        // ================================================================

        // Record new tokens in KV manager (once per sequence, not per layer)
        for seq_i in 0..M {
            if let Some(m) = &states[seq_i].paged_kv_manager {
                m.lock().unwrap().add_token(seq_ids[seq_i])
                    .map_err(|e| anyhow::anyhow!("Failed to record decode token: {:?}", e))?;
            }
        }

        let final_stream = res.streams.get(0).unwrap().clone();
        let final_norm_weight = res.metadata[0].norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = res.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight not in cache"))?;
        crate::norm::rms_norm_into(
            &final_stream, &res.per_gpu_kernels[0].oxide,
            &mut batched_norm1_out[0], &batched_hidden[0],
            &final_norm_gpu, config.rms_norm_eps, hidden_size,
        )?;

        let mut batched_logits = unsafe { final_stream.alloc::<bf16>(M * vocab_size)? };
        let lm_head_weight = res.metadata[0].lm_head.as_ref()
            .or_else(|| res.metadata[0].embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding found"))?;

        let mut lm_head_ps = Some(&mut batched_partial_sums[0]);
        crate::gemm_dispatch::gemm_projection_cached_m(
            &res.gemm_engines[0], &res.per_gpu_kernels[0].oxide, &final_stream,
            &res.weight_caches[0], &lm_head_weight.name,
            &batched_norm1_out[0], &mut batched_logits,
            M, vocab_size, hidden_size, group_size, &mut lm_head_ps,
        )?;

        // Sample per sequence from [M, vocab] logits
        let mut sampled: Vec<u32> = Vec::with_capacity(M);
        for seq_i in 0..M {
            let logit_view = batched_logits.slice(seq_i * vocab_size..(seq_i + 1) * vocab_size);
            let token = crate::sample::sample_with_config(
                &final_stream, &logit_view, &res.per_gpu_kernels[0].oxide,
                &sampling_configs[seq_i], &token_histories[seq_i],
                num_prompt_tokens[seq_i], &mut rngs[seq_i],
            )?;
            sampled.push(token);
        }

        Ok(sampled)
    }
}

/// Inner decode logic for the async pipeline.
/// Takes `&GpuResources` (immutable shared state) and `&mut DecodeState` (mutable per-sequence state).
#[allow(clippy::too_many_arguments)]
fn decode_paged_async_inner(
    res: &GpuResources,
    state: &mut DecodeState,
    stream: &Arc<CudaStream>,
    token_id: u32,
    position: u32,
    seq_id: infers_kv::SequenceId,
    sampling_config: &infers_scheduler::SamplingConfig,
    token_history: &[u32],
    num_prompt_tokens: usize,
    rng: &mut Xoshiro256PlusPlus,
    step: usize,
) -> Result<u32> {
    let config = &res.config;
    let head_dim = config.head_dim;
    let probe_cfg = &res.probe_config;
    let num_gpus = res.metadata.len();

    // Dynamically allocate pages as needed for the target position.
    let (page_size, num_cached_tokens, block_table_i32): (usize, i32, Vec<i32>) = {
        let mut mgr = state.paged_kv_manager.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?
            .lock()
            .unwrap();
        let ps = mgr.page_size();

        let needed_pages = (position as usize / ps) + 1;
        let current_pages = mgr.block_table(seq_id)?.len();
        for _ in current_pages..needed_pages {
            mgr.append_page(seq_id)
                .map_err(|e| anyhow::anyhow!("Failed to allocate KV page for decode: {:?}", e))?;
        }

        let cached = mgr.num_tokens(seq_id)? as i32 + 1;
        let bt: Vec<i32> = mgr.block_table(seq_id)?.iter().map(|p| *p as i32).collect();
        (ps, cached, bt)
    };
    let position_i32 = [position as i32];

    // Write block table and position into pre-allocated staging buffers on each GPU.
    for gpu_idx in 0..num_gpus {
        let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
        let ws = &mut state.workspaces[gpu_idx];

        gpu_stream.memcpy_htod(&block_table_i32, &mut ws.block_table_staging)
            .map_err(|e| anyhow::anyhow!("Failed to copy block table to staging: {e}"))?;

        gpu_stream.memcpy_htod(&position_i32, &mut ws.position_staging)
            .map_err(|e| anyhow::anyhow!("Failed to copy position to staging: {e}"))?;

        let num_cached_tokens_u32 = [num_cached_tokens as u32];
        gpu_stream.memcpy_htod(&num_cached_tokens_u32, &mut ws.num_cached_tokens_staging)
            .map_err(|e| anyhow::anyhow!("Failed to copy num_cached_tokens to staging: {e}"))?;
    }

    // Ensure page pools allocated on each GPU.
    for gpu_idx in 0..num_gpus {
        for cache in &mut state.paged_kv_caches[gpu_idx] {
            cache.ensure_allocated(res.streams.get(gpu_idx).unwrap())?;
        }
    }

    // Embed single token on each GPU.
    let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
    for gpu_idx in 0..num_gpus {
        let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
        let w = &res.metadata[gpu_idx];
        let embed_weight = w.embedding.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
        let embed_table = res.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

        let token_ids_i32 = [token_id as i32];
        {
            let ws = &mut state.workspaces[gpu_idx];
            crate::embedding::embed_tokens_into(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide, &embed_table,
                &token_ids_i32, &mut ws.token_ids_staging, &mut ws.embed_out,
                1, config.hidden_size,
            )?;
        }
        probe::dump(&gpu_stream, probe_cfg, usize::MAX, gpu_idx, "embed.output", &state.workspaces[gpu_idx].embed_out, &[1, config.hidden_size], "decode");
        hidden_states.push(state.workspaces[gpu_idx].embed_out.clone());
    }

    let num_kv_heads_per_gpu = config.num_key_value_heads / num_gpus;
    let num_heads_per_gpu = config.num_attention_heads / num_gpus;
    let sharded_intermediate = config.intermediate_size / num_gpus;

    // Layer loop
    for layer_idx in 0..config.num_hidden_layers {
        let layer_type = config.get_layer_type(layer_idx);
        let stage_prefix = match layer_type {
            LayerType::FullAttention => "attn",
            LayerType::GatedDeltaNet => "gdn",
        };

        // Dump hidden input at start of layer.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
        }

        // Phase A: Attention/GDN on each GPU.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let gemm = &res.gemm_engines[gpu_idx];
            let w = &res.metadata[gpu_idx];
            let layer = &w.layers[layer_idx];

            let norm1_weight = res.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
            crate::norm::rms_norm_into(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                &mut state.workspaces[gpu_idx].norm1_out,
                &hidden_states[gpu_idx], &norm1_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &state.workspaces[gpu_idx].norm1_out, &[1, config.hidden_size], "decode");

            match config.get_layer_type(layer_idx) {
                LayerType::GatedDeltaNet => {
                    let gdn_weights = layer.gdn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                    {
                        let ws = &mut state.workspaces[gpu_idx];
                        let mut ps = Some(&mut ws.partial_sums);
                        crate::gdn::decode_forward(
                            gemm, &gpu_stream,
                            &res.per_gpu_kernels[gpu_idx].oxide,
                            gdn_weights, &ws.norm1_out,
                            &mut state.gdn_states[gpu_idx][layer_idx],
                            config.hidden_size, config.as_ref(), res.group_size,
                            &res.weight_caches[gpu_idx],
                            layer_idx,
                            gpu_idx,
                            probe_cfg,
                            &mut ws.gdn, &mut ws.attn_out,
                            &mut ps,
                        )?;
                    }
                }
                LayerType::FullAttention => {
                    let attn_weights = layer.attn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                    {
                        let ws = &mut state.workspaces[gpu_idx];
                        let mut ps = Some(&mut ws.partial_sums);
                        crate::attention::decode_forward_paged(
                            gemm, &gpu_stream,
                            &res.per_gpu_kernels[gpu_idx].oxide,
                            attn_weights, &ws.norm1_out,
                            &mut state.paged_kv_caches[gpu_idx][layer_idx],
                            &ws.block_table_staging, &ws.position_staging,
                            position,
                            &ws.num_cached_tokens_staging,
                            head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                            config.rope_theta, config.partial_rotary_factor,
                            config.rms_norm_eps, res.group_size, &res.weight_caches[gpu_idx],
                            config.hidden_size,
                            config.attn_output_gate,
                            layer_idx,
                            gpu_idx,
                            probe_cfg,
                            res.rope_cos.as_ref().map(|v| &v[gpu_idx]),
                            res.rope_sin.as_ref().map(|v| &v[gpu_idx]),
                            &mut ws.attn, &mut ws.attn_out,
                            &mut ps,
                            &mut ws.rope_position_staging,
                            &position_i32,
                        )?;
                    }
                }
            };

            probe::dump(&gpu_stream, probe_cfg, layer_idx, gpu_idx, &format!("{}.o_proj", stage_prefix), &state.workspaces[gpu_idx].attn_out, &[1, config.hidden_size], "decode");
        }

        // All-reduce attention outputs across GPUs.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            sync::all_reduce_attention(
                &res.nccl, &gpu_stream, &mut state.workspaces[gpu_idx].attn_out,
            )?;
        }

        // Phase B: Residual add on each GPU.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let ws = &mut state.workspaces[gpu_idx];
            crate::add::add_into(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                &mut ws.residual_buf,
                &hidden_states[gpu_idx], &ws.attn_out,
            )?;
            std::mem::swap(&mut hidden_states[gpu_idx], &mut ws.residual_buf);
        }

        // Phase C: MLP on each GPU.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let gemm = &res.gemm_engines[gpu_idx];
            let w = &res.metadata[gpu_idx];
            let mlp_weights = &w.layers[layer_idx].mlp;
            let ws = &mut state.workspaces[gpu_idx];

            let norm2_weight = res.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
            crate::norm::rms_norm_into(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                &mut ws.norm2_out,
                &hidden_states[gpu_idx], &norm2_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            let mut ps = Some(&mut ws.partial_sums);
            crate::gemm_dispatch::gemm_projection_cached(
                gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                &res.weight_caches[gpu_idx],
                &mlp_weights.gate_proj.name,
                &ws.norm2_out, &mut ws.mlp_gate,
                1, sharded_intermediate, config.hidden_size,
                res.group_size,
                &mut ps,
            )?;

            let mut ps = Some(&mut ws.partial_sums);
            crate::gemm_dispatch::gemm_projection_cached(
                gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                &res.weight_caches[gpu_idx],
                &mlp_weights.up_proj.name,
                &ws.norm2_out, &mut ws.mlp_up,
                1, sharded_intermediate, config.hidden_size,
                res.group_size,
                &mut ps,
            )?;

            res.per_gpu_kernels[gpu_idx].oxide.launch_silu_glu_bf16(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide.cc_stream(), &ws.mlp_up, &ws.mlp_gate, &mut ws.mlp_silu, sharded_intermediate as u32,
            )?;

            let mut ps = Some(&mut ws.partial_sums);
            crate::gemm_dispatch::gemm_projection_cached(
                gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                &res.weight_caches[gpu_idx],
                &mlp_weights.down_proj.name,
                &ws.mlp_silu, &mut ws.mlp_out,
                1, config.hidden_size, sharded_intermediate,
                res.group_size,
                &mut ps,
            )?;
        }

        // All-reduce MLP outputs across GPUs.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            sync::all_reduce_mlp(
                &res.nccl, &gpu_stream, &mut state.workspaces[gpu_idx].mlp_out,
            )?;
        }

        // Phase D: Residual add on each GPU.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let ws = &mut state.workspaces[gpu_idx];
            crate::add::add_into(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                &mut ws.residual_buf,
                &hidden_states[gpu_idx], &ws.mlp_out,
            )?;
            std::mem::swap(&mut hidden_states[gpu_idx], &mut ws.residual_buf);
        }
    }

    // ================================================================
    // Final norm + LM head + sample on GPU 0
    // ================================================================
    let final_stream = res.streams.get(0).unwrap().clone();
    let final_weights = &res.metadata[0];
    let final_hidden = hidden_states.into_iter().next().unwrap();

    let final_norm_weight = final_weights.norm.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
    let final_norm_gpu = res.weight_caches[0].get_bf16(&final_norm_weight.name)
        .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
    crate::norm::rms_norm_into(
        &final_stream, &res.per_gpu_kernels[0].oxide,
        &mut state.workspaces[0].norm1_out,
        &final_hidden, &final_norm_gpu,
        config.rms_norm_eps, config.hidden_size,
    )?;

    probe::dump(&final_stream, probe_cfg, config.num_hidden_layers - 1, 0, "final.norm", &state.workspaces[0].norm1_out, &[1, config.hidden_size], "decode");

    let lm_head_weight = final_weights.lm_head.as_ref()
        .or_else(|| final_weights.embedding.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
    {
        let ws = &mut state.workspaces[0];
        let mut lm_head_ps = Some(&mut ws.lm_head_partial_sums);
        crate::gemm_dispatch::gemm_projection_cached(
            &res.gemm_engines[0], &res.per_gpu_kernels[0].oxide, &final_stream,
            &res.weight_caches[0],
            &lm_head_weight.name,
            &ws.norm1_out, &mut ws.logits,
            1, config.vocab_size, config.hidden_size,
            res.group_size,
            &mut lm_head_ps,
        )?;
    }

    probe::dump(&final_stream, probe_cfg, config.num_hidden_layers - 1, 0, "final.logits", &state.workspaces[0].logits, &[1, config.vocab_size], "decode");

    // Sample (BF16 argmax)
    let sampled = crate::sample::sample_with_config(
        &final_stream, &state.workspaces[0].logits.as_view(), &res.per_gpu_kernels[0].oxide,
        sampling_config, token_history, num_prompt_tokens, rng,
    )?;

    // Record the new token in the KV manager.
    if let Some(m) = &state.paged_kv_manager {
        m.lock().unwrap().add_token(seq_id)
            .map_err(|e| anyhow::anyhow!("Failed to record decode token: {:?}", e))?;
    }

    Ok(sampled)
}
