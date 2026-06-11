//! Inference orchestrator — wires scheduler, engine, and response channels.
//!
//! The `InferenceOrchestrator` owns the scheduling loop, inference engine,
//! eviction store, and token response channels. It runs a continuous
//! schedule→prefill→decode→respond loop.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;

use infers_backend_native::{BackendEvictionStore, ForwardEngine};
use infers_cuda::CudaStream;
use infers_kv::SequenceId;
use infers_scheduler::{
    lifecycle, Request, RoundRobinScheduler, SamplingConfig,
};

/// Central orchestrator connecting HTTP server, scheduler, and GPU inference engine.
pub struct InferenceOrchestrator {
    /// Round-robin continuous batching scheduler.
    scheduler: RoundRobinScheduler,
    /// GPU inference engine (forward pass).
    engine: ForwardEngine,
    /// Per-layer evicted page data store.
    #[allow(dead_code)]
    eviction_store: BackendEvictionStore,
    /// CUDA stream for GPU kernel launches.
    stream: Arc<CudaStream>,
    /// Response channels for active sessions (SequenceId → Sender<u32>).
    response_tx: HashMap<SequenceId, mpsc::Sender<u32>>,
    /// Pending response channels for requests not yet admitted (routing_id → Sender<u32>).
    pending_tx: HashMap<usize, mpsc::Sender<u32>>,
    /// Counter for assigning unique routing IDs.
    next_routing_id: usize,
    /// Number of transformer layers (for eviction store).
    #[allow(dead_code)]
    num_layers: usize,
    /// Whether MTP speculative decoding is enabled.
    #[allow(dead_code)]
    enable_mtp: bool,
    /// MTP speculative decoding engine (optional).
    #[allow(dead_code)]
    mtp: Option<infers_mtp::MtpEngine>,
    /// MTP metrics tracker (optional).
    #[allow(dead_code)]
    mtp_metrics: Option<infers_mtp::MtpMetrics>,
}

impl InferenceOrchestrator {
    /// Create a new orchestrator.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scheduler: RoundRobinScheduler,
        engine: ForwardEngine,
        eviction_store: BackendEvictionStore,
        stream: Arc<CudaStream>,
        num_layers: usize,
        enable_mtp: bool,
        mtp: Option<infers_mtp::MtpEngine>,
        mtp_metrics: Option<infers_mtp::MtpMetrics>,
    ) -> Self {
        Self {
            scheduler,
            engine,
            eviction_store,
            stream,
            response_tx: HashMap::new(),
            pending_tx: HashMap::new(),
            next_routing_id: 1,
            num_layers,
            enable_mtp,
            mtp,
            mtp_metrics,
        }
    }

    /// Enqueue a tokenized request for inference.
    ///
    /// Returns a `routing_id` that the caller uses to register a response channel
    /// via [`register_response_channel`].
    pub fn enqueue_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: SamplingConfig,
    ) -> usize {
        let routing_id = self.next_routing_id;
        self.next_routing_id = self.next_routing_id.wrapping_add(1);

        let mut request = Request::new(routing_id, prompt_tokens, config);
        request.routing_id = Some(routing_id);
        self.scheduler.enqueue_request(request);

        routing_id
    }

    /// Register an `mpsc` sender to receive generated tokens for a request.
    ///
    /// The `routing_id` must match the value returned by [`enqueue_request`].
    pub fn register_response_channel(
        &mut self,
        routing_id: usize,
        tx: mpsc::Sender<u32>,
    ) {
        self.pending_tx.insert(routing_id, tx);
    }

    /// Run one scheduling iteration.
    ///
    /// 1. Calls [`RoundRobinScheduler::schedule`] to admit requests and build batches.
    /// 2. Maps newly created sessions to their pending response channels.
    /// 3. Handles eviction if memory pressure triggered it.
    /// 4. Runs prefill for newly admitted sessions.
    /// 5. Runs decode for active sessions.
    /// 6. Sends generated tokens through response channels.
    /// 7. Cleans up completed sessions' response channels.
    pub fn step(&mut self) -> Result<()> {
        // Step 1: Schedule — admit requests, build decode/prefill batches
        let work = self.scheduler.schedule()?;

        // Step 2: Map newly created sessions to pending response channels.
        // Sessions created in this step have their routing_id set from the request.
        let num_before = self.response_tx.len();
        for session in &self.scheduler.active_sessions {
            if let Some(routing_id) = session.routing_id {
                if !self.response_tx.contains_key(&session.id) {
                    if let Some(tx) = self.pending_tx.remove(&routing_id) {
                        tracing::debug!(
                            "Mapped routing_id={} to seq_id={}",
                            routing_id,
                            session.id
                        );
                        self.response_tx.insert(session.id, tx);
                    }
                }
            }
        }
        if self.response_tx.len() > num_before {
            tracing::info!(
                "New sessions mapped: {} total active",
                self.response_tx.len()
            );
        }

        // Step 3: Handle eviction
        if let Some(evicted_id) = work.evicted_session {
            tracing::info!("Evicting session {} due to memory pressure", evicted_id);
            // For now, use direct cleanup (full paged eviction is deferred)
            let _ = self.scheduler.kv_manager.delete_sequence(evicted_id);
            self.response_tx.remove(&evicted_id);
        }

        // Step 4: Handle prefill batch
        if let Some(prefill_batch) = &work.prefill_batch {
            for &seq_id in &prefill_batch.sessions {
                // Get the prompt tokens from the batch
                let tokens = &prefill_batch.input_tokens;
                tracing::debug!(
                    "Prefilling session {} with {} tokens",
                    seq_id,
                    tokens.len()
                );

                let sampled = self.engine.prefill(&self.stream, tokens)?;

                // Update session state and push generated token
                if let Some(session) = self
                    .scheduler
                    .active_sessions
                    .iter_mut()
                    .find(|s| s.id == seq_id)
                {
                    let _ = lifecycle::finish_prefill(session);
                    session.tokens.push(sampled);
                    session.num_generated_tokens = session.num_generated_tokens.saturating_add(1);
                }

                // Send generated token through response channel
                if let Some(tx) = self.response_tx.get(&seq_id) {
                    let _ = tx.try_send(sampled);
                }
            }
        }

        // Step 5: Handle decode batch
        for (i, &seq_id) in work.decode_batch.sessions.iter().enumerate() {
            let token_id = work.decode_batch.input_tokens[i];

            // Calculate position: index of the current input token in the sequence
            let position = self
                .scheduler
                .active_sessions
                .iter()
                .find(|s| s.id == seq_id)
                .map(|s| s.tokens.len().saturating_sub(1) as u32)
                .unwrap_or(0);

            tracing::debug!(
                "Decoding session {}: token_id={}, position={}",
                seq_id,
                token_id,
                position
            );

            let sampled = self.engine.decode(&self.stream, token_id, position)?;

            // Update session
            if let Some(session) = self
                .scheduler
                .active_sessions
                .iter_mut()
                .find(|s| s.id == seq_id)
            {
                session.tokens.push(sampled);
                session.num_generated_tokens = session.num_generated_tokens.saturating_add(1);

                // Check if session completed
                if session.is_complete() {
                    tracing::info!(
                        "Session {} complete: {} generated tokens",
                        seq_id,
                        session.num_generated_tokens
                    );
                    let _ = lifecycle::complete_session(session);
                    // Close response channel
                    self.response_tx.remove(&seq_id);
                }
            }

            // Send token through response channel
            if let Some(tx) = self.response_tx.get(&seq_id) {
                let _ = tx.try_send(sampled);
            }
        }

        Ok(())
    }

    /// Number of active sessions.
    #[allow(dead_code)]
    pub fn active_count(&self) -> usize {
        self.scheduler.active_count()
    }

    /// Number of pending (not yet admitted) requests.
    #[allow(dead_code)]
    pub fn pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    /// Whether the orchestrator has any work pending or in progress.
    #[allow(dead_code)]
    pub fn is_busy(&self) -> bool {
        self.scheduler.is_busy()
    }
}
