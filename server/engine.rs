//! The single boundary between the HTTP server and the model.
//!
//! Everything model-specific (chat templating, tokenization, the decode
//! loop, sampling, detokenization, stop strings, reasoning separation) lives
//! behind [`Engine`]. The production implementation, [`GeneratorEngine`],
//! delegates all of it to `xor_net::generation::Generator`.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use anyhow::anyhow;
use serde::Serialize;
use serde_json::{Map, Value};
use xor_net::batch_session::SessionShared;
use xor_net::generation::{
    FinishReason as GenFinish, GenerationParams, Generator, ReasoningFormat, StreamEvent, Timings,
    common_prefix,
};
use xor_net::models::batched::BatchedCausalLM;
use xor_net::prompt_cache::PromptCacheStats;

use crate::batcher::{BatchOptions, BatchRunner};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Reasoning of a previous assistant turn (templates such as Qwen3's
    /// read it when re-rendering history).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[cfg(test)]
impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat(Vec<ChatMessage>),
}

/// Sampling parameters as validated by the API layer.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = disabled.
    pub top_k: usize,
    /// 0 = disabled.
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 1 = disabled.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    /// The core sampler's parameters (vLLM penalty semantics: repetition
    /// over prompt + output, presence / frequency over the output only).
    pub fn to_core(&self) -> xor_net::SamplingParams {
        xor_net::SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: None,
            seed: Some(self.seed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    /// Split `<think>…</think>` off into reasoning events.
    pub reasoning: bool,
    /// Prefix boundaries of the prompt (see [`Engine::prefix_boundaries`]).
    pub boundaries: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token.
    Eos,
    /// A stop string matched.
    StopString,
    /// `max_tokens` reached.
    Length,
    /// The callback asked to stop (client went away).
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct GenResult {
    pub finish: FinishReason,
    /// Number of sampled tokens (including a final EOS).
    pub completion_tokens: usize,
    pub stop_word: Option<String>,
    pub timings: Option<Timings>,
    /// Prompt tokens reused instead of recomputed (slot state or prompt
    /// cache).
    pub cached_tokens: usize,
}

/// The model as seen by the server.
pub trait Engine: Send + Sync {
    fn model_id(&self) -> &str;
    /// Maximum prompt + generated tokens.
    fn context_length(&self) -> usize;
    /// Whether the chat template knows `<think>` blocks (decides the
    /// default reasoning format).
    fn chat_template_has_thinking(&self) -> bool {
        false
    }
    /// Template (for chat, with `template_kwargs`) and tokenize a prompt.
    /// An `Err` is a client error (e.g. out-of-vocabulary token ids, a
    /// template `raise_exception`).
    fn encode(
        &self,
        prompt: &Prompt,
        template_kwargs: &Map<String, Value>,
    ) -> Result<Vec<u32>, String>;
    /// Prompt positions where later requests are likely to diverge (chat:
    /// end of the system prompt, end of the history); prompt-cache
    /// insertion points. Empty by default.
    fn prefix_boundaries(
        &self,
        prompt: &Prompt,
        template_kwargs: &Map<String, Value>,
        ids: &[u32],
    ) -> Vec<usize> {
        let _ = (prompt, template_kwargs, ids);
        Vec::new()
    }
    /// How many leading tokens of `ids` slot `slot`'s current state already
    /// holds (slot affinity for scheduling). Must not block: 0 if unknown.
    fn slot_affinity(&self, slot: usize, ids: &[u32]) -> usize {
        let _ = (slot, ids);
        0
    }
    /// Shared prompt-cache counters, if a cache is enabled.
    fn prompt_cache_stats(&self) -> Option<PromptCacheStats> {
        None
    }
    /// Run one generation on worker slot `slot`. `on_event` receives text
    /// and reasoning deltas; returning `ControlFlow::Break` aborts promptly.
    fn generate(
        &self,
        slot: usize,
        req: &GenRequest,
        on_event: &mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<GenResult>;
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------
// Generator-backed engine
// ---------------------------------------------------------------------------

/// One shared model, served by one of two schedulers:
/// - slots: one [`Generator`] fork (own state, KV cache, sampler) per worker
///   slot, each decoding independently. Slot `i` is only ever used by worker
///   `i`, so the slot mutexes are uncontended; they exist to make the engine
///   `Sync`.
/// - batch: a [`BatchRunner`] engine thread decodes every active request
///   together (continuous batching); worker `i` relays request `i`.
pub struct GeneratorEngine {
    /// Used for tokenization / templating only (never generates).
    base: Mutex<Generator>,
    backend: Backend,
    model_id: String,
    ctx: usize,
    vocab: usize,
    has_thinking: bool,
    /// Speculative decoding for the slot scheduler (`--draft`).
    spec: Option<xor_net::speculative::SpecConfig>,
}

enum Backend {
    Slots(Vec<Mutex<Generator>>),
    Batch(BatchRunner),
}

impl GeneratorEngine {
    /// Slot scheduler with `slots` independent generators.
    pub fn new(generator: Generator, slots: usize, model_id: String) -> Self {
        let slots = (0..slots.max(1))
            .map(|_| Mutex::new(generator.fork()))
            .collect();
        Self::with_backend(generator, Backend::Slots(slots), model_id)
    }

    /// Continuous-batching scheduler over `opts.max_batch` slots. `batched`
    /// is the model's batched forward (the same model as `generator`'s);
    /// `None` falls back to one forward per sequence and step.
    pub fn new_batched(
        generator: Generator,
        batched: Option<Arc<dyn BatchedCausalLM>>,
        opts: BatchOptions,
        model_id: String,
    ) -> anyhow::Result<Self> {
        let shared = SessionShared::from_generator(&generator);
        let runner = BatchRunner::spawn(shared, batched, opts)?;
        Ok(Self::with_backend(
            generator,
            Backend::Batch(runner),
            model_id,
        ))
    }

    fn with_backend(generator: Generator, backend: Backend, model_id: String) -> Self {
        let model = generator.model();
        let ctx = model.max_seq_len();
        let vocab = model.vocab_size();
        let has_thinking = generator.chat_template().is_some_and(|t| t.has_thinking());
        Self {
            base: Mutex::new(generator),
            backend,
            model_id,
            ctx,
            vocab,
            has_thinking,
            spec: None,
        }
    }

    /// Decode speculatively (slot scheduler only; the batch scheduler
    /// ignores it).
    pub fn with_speculative(mut self, spec: Option<xor_net::speculative::SpecConfig>) -> Self {
        self.spec = spec;
        self
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        lock(&self.base).eos_token_ids().to_vec()
    }

    pub fn has_chat_template(&self) -> bool {
        lock(&self.base).chat_template().is_some()
    }

    fn check_ids(&self, ids: &[u32]) -> Result<(), String> {
        match ids.iter().find(|&&i| i as usize >= self.vocab) {
            Some(bad) => Err(format!(
                "token id {bad} is out of range (vocab size {})",
                self.vocab
            )),
            None => Ok(()),
        }
    }
}

impl Engine for GeneratorEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn context_length(&self) -> usize {
        self.ctx
    }

    fn chat_template_has_thinking(&self) -> bool {
        self.has_thinking
    }

    fn encode(
        &self,
        prompt: &Prompt,
        template_kwargs: &Map<String, Value>,
    ) -> Result<Vec<u32>, String> {
        let g = lock(&self.base);
        let ids = match prompt {
            Prompt::Tokens(ids) => ids.clone(),
            Prompt::Text(s) => g.encode(s, true).map_err(|e| format!("{e:#}"))?,
            Prompt::Chat(msgs) => {
                let text = g
                    .apply_chat_template_with_kwargs(msgs, true, template_kwargs)
                    .map_err(|e| format!("{e:#}"))?;
                g.encode(&text, false).map_err(|e| format!("{e:#}"))?
            }
        };
        drop(g);
        self.check_ids(&ids)?;
        Ok(ids)
    }

    fn prefix_boundaries(
        &self,
        prompt: &Prompt,
        template_kwargs: &Map<String, Value>,
        ids: &[u32],
    ) -> Vec<usize> {
        match prompt {
            Prompt::Chat(msgs) => {
                lock(&self.base).chat_prefix_boundaries(msgs, template_kwargs, ids)
            }
            _ => Vec::new(),
        }
    }

    fn slot_affinity(&self, slot: usize, ids: &[u32]) -> usize {
        let slots = match &self.backend {
            Backend::Slots(s) => s,
            Backend::Batch(r) => return r.affinity(slot, ids),
        };
        match slots.get(slot).map(Mutex::try_lock) {
            Some(Ok(g)) => common_prefix(g.processed_tokens(), ids),
            Some(Err(TryLockError::Poisoned(p))) => {
                common_prefix(p.into_inner().processed_tokens(), ids)
            }
            _ => 0,
        }
    }

    fn prompt_cache_stats(&self) -> Option<PromptCacheStats> {
        lock(&self.base).prompt_cache().map(|c| c.stats())
    }

    fn generate(
        &self,
        slot: usize,
        req: &GenRequest,
        on_event: &mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<GenResult> {
        // Defense in depth: the API layer already checked these.
        if req.prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        if req.max_tokens == 0 || req.prompt_ids.len() + req.max_tokens > self.ctx {
            return Err(anyhow!("prompt + max_tokens exceeds the context length"));
        }
        self.check_ids(&req.prompt_ids).map_err(|e| anyhow!(e))?;

        let params = GenerationParams::default()
            .with_max_tokens(req.max_tokens)
            .with_sampling(req.sampling.to_core())
            .with_stop(req.stop.iter().cloned())
            .with_reasoning_format(if req.reasoning {
                ReasoningFormat::Separate
            } else {
                ReasoningFormat::None
            });
        let finish = |f| match f {
            GenFinish::Stop => FinishReason::Eos,
            GenFinish::StopString => FinishReason::StopString,
            GenFinish::Length => FinishReason::Length,
            GenFinish::Aborted => FinishReason::Cancelled,
        };
        let slots = match &self.backend {
            Backend::Slots(s) => s,
            Backend::Batch(r) => {
                let out = r.generate(slot, &req.prompt_ids, &params, &req.boundaries, on_event)?;
                return Ok(GenResult {
                    finish: finish(out.finish),
                    completion_tokens: out.usage.completion_tokens,
                    stop_word: out.stop_string,
                    timings: Some(out.timings),
                    cached_tokens: out.usage.cached_prompt_tokens,
                });
            }
        };
        let params = match &self.spec {
            Some(spec) => params.with_speculative(spec.clone()),
            None => params,
        };
        let slot = slots
            .get(slot)
            .ok_or_else(|| anyhow!("no generator for slot {slot}"))?;
        let mut g = lock(slot);
        let out = g.generate_from_tokens_with_boundaries(
            &req.prompt_ids,
            &params,
            &req.boundaries,
            on_event,
        )?;
        Ok(GenResult {
            finish: finish(out.finish_reason),
            completion_tokens: out.usage.completion_tokens,
            stop_word: out.stop_string,
            timings: Some(out.timings),
            cached_tokens: out.usage.cached_prompt_tokens,
        })
    }
}

#[cfg(test)]
pub mod mock {
    //! A tiny deterministic engine for end-to-end server tests: a byte-level
    //! "tokenizer" and a model that replays a fixed script of token bytes.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use xor_net::generation::{ReasoningSplitter, StopMatcher};

    pub struct MockEngine {
        pub script: Vec<u8>,
        pub ctx_len: usize,
        /// Sleep per token (to test disconnects / queueing).
        pub token_delay: Duration,
        /// Tokens actually produced across all requests.
        pub produced: AtomicUsize,
        pub eos: u32,
        pub has_thinking: bool,
        /// Template kwargs of the last chat request.
        pub last_kwargs: Mutex<Map<String, Value>>,
    }

    impl MockEngine {
        pub fn new(script: &str) -> Self {
            Self {
                script: script.as_bytes().to_vec(),
                ctx_len: 64,
                token_delay: Duration::ZERO,
                produced: AtomicUsize::new(0),
                eos: 0,
                has_thinking: false,
                last_kwargs: Mutex::new(Map::new()),
            }
        }
    }

    /// The mock's chat "template".
    pub fn render_chat(messages: &[ChatMessage]) -> String {
        let mut p = String::new();
        for m in messages {
            p.push_str(&m.role);
            p.push_str(": ");
            p.push_str(&m.content);
            p.push('\n');
        }
        p.push_str("assistant: ");
        p
    }

    /// Bytes → text (complete UTF-8 only) → stop strings → reasoning split.
    struct Pipeline {
        bytes: Vec<u8>,
        stops: StopMatcher,
        split: Option<ReasoningSplitter>,
        out: Vec<(bool, String)>,
    }

    impl Pipeline {
        /// Returns `(events, stop_hit)`.
        fn push(&mut self, b: Option<u8>) -> bool {
            let text = match b {
                Some(b) => {
                    self.bytes.push(b);
                    match std::str::from_utf8(&self.bytes) {
                        Ok(s) => {
                            let s = s.to_owned();
                            self.bytes.clear();
                            s
                        }
                        Err(_) => return false,
                    }
                }
                None => String::from_utf8_lossy(&std::mem::take(&mut self.bytes)).into_owned(),
            };
            let (mut safe, hit) = self.stops.push(&text);
            if b.is_none() && !hit {
                safe.push_str(&self.stops.flush());
            }
            match &mut self.split {
                Some(sp) => {
                    sp.push(&safe, &mut self.out);
                    if b.is_none() || hit {
                        sp.flush(&mut self.out);
                    }
                }
                None if !safe.is_empty() => self.out.push((false, safe)),
                None => {}
            }
            hit
        }

        fn drain(
            &mut self,
            on_event: &mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
        ) -> ControlFlow<()> {
            let mut flow = ControlFlow::Continue(());
            // Always call at least once per token so cancellation is prompt.
            if self.out.is_empty() {
                return on_event(StreamEvent::Text(""));
            }
            for (r, s) in self.out.drain(..) {
                let ev = if r {
                    StreamEvent::Reasoning(&s)
                } else {
                    StreamEvent::Text(&s)
                };
                if flow.is_continue() {
                    flow = on_event(ev);
                }
            }
            flow
        }
    }

    impl Engine for MockEngine {
        fn model_id(&self) -> &str {
            "mock-model"
        }
        fn context_length(&self) -> usize {
            self.ctx_len
        }
        fn chat_template_has_thinking(&self) -> bool {
            self.has_thinking
        }
        fn encode(
            &self,
            prompt: &Prompt,
            template_kwargs: &Map<String, Value>,
        ) -> Result<Vec<u32>, String> {
            let text = match prompt {
                Prompt::Text(s) => s.clone(),
                Prompt::Chat(m) => {
                    *lock(&self.last_kwargs) = template_kwargs.clone();
                    render_chat(m)
                }
                Prompt::Tokens(t) => {
                    if t.iter().any(|&i| i > 255) {
                        return Err("token id out of range".into());
                    }
                    return Ok(t.clone());
                }
            };
            // BOS(1) + bytes
            Ok(std::iter::once(1)
                .chain(text.bytes().map(u32::from))
                .collect())
        }
        fn generate(
            &self,
            _slot: usize,
            req: &GenRequest,
            on_event: &mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
        ) -> anyhow::Result<GenResult> {
            assert!(req.prompt_ids.len() + req.max_tokens <= self.ctx_len);
            let mut p = Pipeline {
                bytes: Vec::new(),
                stops: StopMatcher::new(&req.stop),
                split: req.reasoning.then(|| ReasoningSplitter::new(false)),
                out: Vec::new(),
            };
            let result = |finish, n, stops: &StopMatcher| GenResult {
                finish,
                completion_tokens: n,
                stop_word: stops.matched().map(str::to_owned),
                timings: None,
                cached_tokens: 0,
            };
            let mut n = 0;
            for &b in self.script.iter().take(req.max_tokens) {
                if !self.token_delay.is_zero() {
                    std::thread::sleep(self.token_delay);
                }
                n += 1;
                self.produced.fetch_add(1, Ordering::SeqCst);
                if u32::from(b) == self.eos {
                    let hit = p.push(None);
                    let _ = p.drain(on_event);
                    let finish = if hit {
                        FinishReason::StopString
                    } else {
                        FinishReason::Eos
                    };
                    return Ok(result(finish, n, &p.stops));
                }
                let hit = p.push(Some(b));
                if p.drain(on_event).is_break() {
                    return Ok(result(FinishReason::Cancelled, n, &p.stops));
                }
                if hit {
                    return Ok(result(FinishReason::StopString, n, &p.stops));
                }
            }
            let hit = p.push(None);
            let _ = p.drain(on_event);
            let finish = if hit {
                FinishReason::StopString
            } else {
                FinishReason::Length
            };
            Ok(result(finish, n, &p.stops))
        }
    }

    #[test]
    fn chat_render() {
        let m = vec![ChatMessage::new("user", "hi")];
        assert_eq!(render_chat(&m), "user: hi\nassistant: ");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_maps_to_core() {
        let sp = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            presence_penalty: 0.5,
            frequency_penalty: -0.5,
            repetition_penalty: 1.1,
            seed: 42,
        };
        let c = sp.to_core();
        assert_eq!(
            (c.temperature, c.top_p, c.top_k, c.min_p),
            (0.7, 0.9, 40, 0.05)
        );
        assert_eq!(
            (
                c.presence_penalty,
                c.frequency_penalty,
                c.repetition_penalty
            ),
            (0.5, -0.5, 1.1)
        );
        assert_eq!((c.seed, c.penalty_last_n), (Some(42), None));
        assert!(
            SamplingParams {
                temperature: 0.0,
                ..Default::default()
            }
            .to_core()
            .is_greedy()
        );
    }

    #[test]
    fn chat_message_serializes_for_templates() {
        let mut m = ChatMessage::new("assistant", "4");
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({"role": "assistant", "content": "4"})
        );
        m.reasoning_content = Some("2+2".into());
        assert_eq!(
            serde_json::to_value(&m).unwrap()["reasoning_content"],
            "2+2"
        );
    }
}
