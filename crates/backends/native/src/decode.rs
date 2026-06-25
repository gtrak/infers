use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_cuda::{CudaContext, CudaGraph, CudaSlice, CudaStream};
use infers_cuda::{group_end, group_start};
use infers_model::LayerType;

use crate::engine::{ForwardEngine, PerGpuKernels};
use crate::gpu_cache::GpuWeightCache;
use crate::workspace::DecodeWorkspace;
use crate::gdn::GdnState;
use crate::attention::{KvCache, PagedKvCache};
use crate::probe;
use crate::sync;
use crate::sample::Xoshiro256PlusPlus;

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
        let span = tracing::info_span!("decode");
        let _enter = span.enter();
        let num_gpus = self.metadata.len();

        let config = &self.config;
        let head_dim = config.head_dim;

        // Probe instrumentation
        let probe = probe::ProbeConfig::from_env();
        probe::dump_config(&self.config, num_gpus, self.group_size);

        // Dynamically allocate pages as needed for the target position,
        // then read the (possibly updated) block table and cached-token count.
        // Use a scope to drop the mutable borrow before using `self` again.
        let (page_size, num_cached_tokens, block_table_i32): (usize, i32, Vec<i32>) = {
            let mgr = self.paged_kv_manager.as_mut()
                .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;
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
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
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
                cache.ensure_allocated(self.streams.get(gpu_idx).unwrap())?;
            }
        }

        // Pre-define final_stream (GPU 0) — needed by both replay and normal paths below.
        let final_stream = self.streams.get(0).unwrap().clone();

        // Determine CUDA graph mode:
        //   step 0 = warm-up (normal execute, no capture),
        //   step 1 = capture (record kernel topology into graph),
        //   step 2+ = replay (launch captured graph, skip compute loop).
        let is_replay = self.decode_step_count >= 2 && self.decode_graphs[0].is_some();
        let is_capture = self.decode_step_count == 1;

        if is_replay {
            // Replay mode: launch captured graph instead of executing kernels individually.
            tracing::info!("CUDA graph replay mode (step {})", step);
            for gpu_idx in 0..num_gpus {
                if let Some(ref graph) = self.decode_graphs[gpu_idx] {
                    graph.launch()
                        .map_err(|e| anyhow::anyhow!("Graph launch failed on GPU {}: {:?}", gpu_idx, e))?;
                }
            }
            // Graph replayed — skip to sampling (final norm + LM head already in graph).
        } else {
            // GPU timing events are NOT created during warm-up or capture steps.
            // cudaEventSynchronize (called by elapsed_ms) creates an implicit sync on the stream
            // that prevents CUDA graph capture on subsequent steps. Timing can be added back
            // later using a separate async approach.

            // Begin capture if this is the capture step
            if is_capture {
                tracing::info!("CUDA graph capture mode (step {})", step);

                // Check stream capture status before starting (eprintln for test visibility)
                for gpu_idx in 0..num_gpus {
                    let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                    match gpu_stream.capture_status() {
                        Ok(status) => eprintln!("[DEBUG] GPU{} capture status BEFORE begin_capture: {:?}", gpu_idx, status),
                        Err(e) => eprintln!("[ERROR] GPU{} failed to check capture status: {:?}", gpu_idx, e),
                    }
                }

                // NOTE: Do NOT synchronize streams before begin_capture — NVIDIA explicitly
                // prohibits this as it puts the stream in a state incompatible with capture.
                let capture_mode = cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL;
                for gpu_idx in 0..num_gpus {
                    let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                    gpu_stream.begin_capture(capture_mode)
                        .map_err(|e| anyhow::anyhow!("Begin capture failed on GPU {}: {:?}", gpu_idx, e))?;
                }
            }

            // Embed single token on each GPU — using pre-allocated staging + output buffers (zero-alloc)
        let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            let w = &self.metadata[gpu_idx];
            let embed_weight = w.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = self.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

            // Convert token_id to i32 and write into staging buffer, then embed into ws.embed_out
            let token_ids_i32 = [token_id as i32];
            {
                let ws = &mut self.workspaces[gpu_idx];
                crate::embedding::embed_tokens_into(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].oxide, &embed_table,
                    &token_ids_i32, &mut ws.token_ids_staging, &mut ws.embed_out,
                    1, config.hidden_size,
                )?;
            }
            probe::dump(&gpu_stream, &probe, usize::MAX, gpu_idx, "embed.output", &self.workspaces[gpu_idx].embed_out, &[1, config.hidden_size], "decode");
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
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

           // Phase A: Attention/GDN on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.metadata[gpu_idx];
                let layer = &w.layers[layer_idx];

                // Norm1
                let norm1_weight = self.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
                crate::norm::rms_norm_into(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].oxide,
                    &mut self.workspaces[gpu_idx].norm1_out,
                    &hidden_states[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &self.workspaces[gpu_idx].norm1_out, &[1, config.hidden_size], "decode");

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
                                &self.per_gpu_kernels[gpu_idx].oxide,
                                gdn_weights, &ws.norm1_out,
                                &mut self.gdn_states[gpu_idx][layer_idx],
                                config.hidden_size, config.as_ref(), self.group_size,
                                &self.weight_caches[gpu_idx],
                                layer_idx,
                                gpu_idx,
                                &probe,
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
                                &self.per_gpu_kernels[gpu_idx].oxide,
                                attn_weights, &ws.norm1_out,
                                &mut self.paged_kv_caches[gpu_idx][layer_idx],
                                &ws.block_table_staging, &ws.position_staging,
                                position,
                                &ws.num_cached_tokens_staging,
                                head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                                config.rope_theta, config.partial_rotary_factor,
                                config.rms_norm_eps, self.group_size, &self.weight_caches[gpu_idx],
                                config.hidden_size,
                                config.attn_output_gate,
                                layer_idx,
                                gpu_idx,
                                &probe,
                                self.rope_cos.as_ref().map(|v| &v[gpu_idx]),
                                self.rope_sin.as_ref().map(|v| &v[gpu_idx]),
                                &mut ws.attn, &mut ws.attn_out,
                                &mut ps,
                                &mut ws.rope_position_staging,
                                &position_i32,
                            )?;
                        }
                    }
                };

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.o_proj", stage_prefix), &self.workspaces[gpu_idx].attn_out, &[1, config.hidden_size], "decode");
            }

            // All-reduce attention outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(
                    &self.nccl, &gpu_stream, &mut self.workspaces[gpu_idx].attn_out,
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.after_ar", stage_prefix), &self.workspaces[gpu_idx].attn_out, &[1, config.hidden_size], "decode");
            }

            // Phase B: Residual add on each GPU (zero-alloc via workspace + copy)
            // NOTE: memcpy_dtod instead of swap — CUDA graph requires fixed device addresses.
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let ws = &mut self.workspaces[gpu_idx];
                crate::add::add_into(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.residual_buf,
                    &hidden_states[gpu_idx], &ws.attn_out,
                )?;
                // Copy result from residual_buf back to embed_out so that hidden_states[gpu_idx]
                // keeps the same device address (required for CUDA graph capture).
                gpu_stream.memcpy_dtod(&ws.residual_buf, &mut hidden_states[gpu_idx])
                    .map_err(|e| anyhow::anyhow!("Failed to copy residual result: {e}"))?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.attn", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase C: MLP on each GPU (column-parallel gate/up, row-parallel down)
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.metadata[gpu_idx];
                let mlp_weights = &w.layers[layer_idx].mlp;
                let ws = &mut self.workspaces[gpu_idx];

                // Norm2 — into workspace
                let norm2_weight = self.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
                crate::norm::rms_norm_into(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.norm2_out,
                    &hidden_states[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.norm2", &ws.norm2_out, &[1, config.hidden_size], "decode");

                // Gate projection — into workspace.mlp_gate
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name,
                    &ws.norm2_out, &mut ws.mlp_gate,
                    1, sharded_intermediate, config.hidden_size,
                    self.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.gate_proj", &ws.mlp_gate, &[1, config.intermediate_size / num_gpus], "decode");

                // Up projection — into workspace.mlp_up
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name,
                    &ws.norm2_out, &mut ws.mlp_up,
                    1, sharded_intermediate, config.hidden_size,
                    self.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.up_proj", &ws.mlp_up, &[1, config.intermediate_size / num_gpus], "decode");

                // up * SiLU(gate) = SwiGLU — into workspace.mlp_silu
                self.per_gpu_kernels[gpu_idx].oxide.launch_silu_glu_bf16(
                    &gpu_stream, &ws.mlp_up, &ws.mlp_gate, &mut ws.mlp_silu, sharded_intermediate as u32,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.silu", &ws.mlp_silu, &[1, config.intermediate_size / num_gpus], "decode");

                // Down projection — into workspace.mlp_out
                let mut ps = Some(&mut ws.partial_sums);
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name,
                    &ws.mlp_silu, &mut ws.mlp_out,
                    1, config.hidden_size, sharded_intermediate,
                    self.group_size,
                    &mut ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_raw", &ws.mlp_out, &[1, config.hidden_size], "decode");
            }

            // All-reduce MLP outputs across GPUs (grouped) — directly on workspace buffers
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(
                    &self.nccl, &gpu_stream, &mut self.workspaces[gpu_idx].mlp_out,
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_ar", &self.workspaces[gpu_idx].mlp_out, &[1, config.hidden_size], "decode");
            }

            // Phase D: Residual add on each GPU (zero-alloc via workspace + copy)
            // NOTE: memcpy_dtod instead of swap — CUDA graph requires fixed device addresses.
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let ws = &mut self.workspaces[gpu_idx];
                crate::add::add_into(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].oxide,
                    &mut ws.residual_buf,
                    &hidden_states[gpu_idx], &ws.mlp_out,
                )?;
                // Copy result from residual_buf back to embed_out so that hidden_states[gpu_idx]
                // keeps the same device address (required for CUDA graph capture).
                gpu_stream.memcpy_dtod(&ws.residual_buf, &mut hidden_states[gpu_idx])
                    .map_err(|e| anyhow::anyhow!("Failed to copy residual result: {e}"))?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.mlp", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }
        }

        // ================================================================
        // Final norm + LM head + sample on GPU 0
        // ================================================================
        // (final_stream was pre-defined earlier for both replay and normal paths)
        let final_weights = &self.metadata[0];
        let final_hidden = hidden_states.into_iter().next().unwrap();

        // Final norm — write into workspace.norm1_out (reusing buffer since layer loop is done)
        let final_norm_weight = final_weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = self.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
        crate::norm::rms_norm_into(
            &final_stream, &self.per_gpu_kernels[0].oxide,
            &mut self.workspaces[0].norm1_out,
            &final_hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.norm", &self.workspaces[0].norm1_out, &[1, config.hidden_size], "decode");

        // LM head — write into workspace.logits
        let lm_head_weight = final_weights.lm_head.as_ref()
            .or_else(|| final_weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        {
            let ws = &mut self.workspaces[0];
            let mut lm_head_ps = Some(&mut ws.lm_head_partial_sums);
            crate::gemm_dispatch::gemm_projection_cached(
                &mut self.gemm_engines[0], &self.per_gpu_kernels[0].oxide, &final_stream,
                &self.weight_caches[0],
                &lm_head_weight.name,
                &ws.norm1_out, &mut ws.logits,
                1, config.vocab_size, config.hidden_size,
                self.group_size,
                &mut lm_head_ps,
            )?;
        }

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.logits", &self.workspaces[0].logits, &[1, config.vocab_size], "decode");

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

        // GPU timing: disabled during warm-up and capture to avoid implicit stream synchronization
        // (cudaEventSynchronize prevents CUDA graph capture on subsequent steps).
        tracing::info!(phase = "decode", "GPU execution complete");

        // Post-compute capture status check (eprintln for test visibility) — helps diagnose
        // whether warm-up left the stream in a capturing state incompatible with begin_capture.
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            match gpu_stream.capture_status() {
                Ok(status) => eprintln!("[DEBUG] GPU{} capture status AFTER compute (step={}): {:?}", gpu_idx, step, status),
                Err(e) => eprintln!("[ERROR] GPU{} failed to check capture status after compute: {:?}", gpu_idx, e),
            }
        }

        // End capture if this was the capture step (inside else block for end_capture API)
        if is_capture {
            tracing::info!("Ending CUDA graph capture");
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                // Pass 0 as CUgraphInstantiate_flags (no flags — "auto free on launch" disabled)
                let flags: cudarc::driver::sys::CUgraphInstantiate_flags_enum =
                    unsafe { std::mem::transmute(0u32) };
                let graph = gpu_stream.end_capture(flags)
                    .map_err(|e| anyhow::anyhow!("End capture failed on GPU {}: {:?}", gpu_idx, e))?;
                if let Some(g) = graph {
                    g.upload().ok(); // pre-upload for faster first replay
                    self.decode_graphs[gpu_idx] = Some(g);
                    tracing::info!("CUDA graph captured and uploaded for GPU {}", gpu_idx);
                }
            }
        }

        // Close the else block — replay path skipped embedding through LM head.
        }

        // Sample (BF16 argmax) — runs in both replay and normal modes.
        let sampled = crate::sample::sample_with_config(
            &final_stream, &self.workspaces[0].logits.as_view(), &self.per_gpu_kernels[0].oxide,
            sampling_config, token_history, num_prompt_tokens, rng,
        )?;

        // Increment decode step counter for graph capture scheduling
        self.decode_step_count += 1;

        // Record the new token in the KV manager so the next decode step
        // sees the correct block table and cached-token count.
        if let Some(mgr) = self.paged_kv_manager.as_mut() {
            mgr.add_token(seq_id)
                .map_err(|e| anyhow::anyhow!("Failed to record decode token: {:?}", e))?;
        }

        Ok(sampled)
    }
}
