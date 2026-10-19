//! The single boundary between the HTTP server and the model.
//!
//! Everything model-specific (prompt templating, tokenization, the decode
//! loop, sampling, detokenization, stop strings) lives behind [`Engine`].
//! Swapping the current hand-written loop for `xor_net::generation::Generator`
//! only requires a new `Engine` impl in this file.

use std::ops::ControlFlow;
use std::path::Path;

use anyhow::anyhow;
use tokenizers::Tokenizer;
use xor_net::models::llama::{Cache, Config, Llama, LlamaEosToks};
use xor_net::sampler::Sampler;

use crate::text::{IncrementalDetok, StopBuffer, StopOutcome};

#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat(Vec<ChatMessage>),
}

/// Sampling parameters. Validated by the API layer; engines may ignore
/// knobs they don't support yet.
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

#[derive(Clone, Debug)]
pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
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
}

/// The model as seen by the server.
pub trait Engine: Send + Sync {
    fn model_id(&self) -> &str;
    /// Maximum prompt + generated tokens.
    fn context_length(&self) -> usize;
    /// Template (for chat) and tokenize a prompt. An `Err` is a client error
    /// (e.g. out-of-vocabulary token ids).
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, String>;
    /// Run one generation. `on_delta` is called at least once per sampled
    /// token with the newly stable text (possibly empty); returning
    /// `ControlFlow::Break` aborts generation promptly.
    fn generate(
        &self,
        req: &GenRequest,
        on_delta: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<GenResult>;
}

// ---------------------------------------------------------------------------
// Llama implementation (current engine; to be replaced by Generator)
// ---------------------------------------------------------------------------

pub struct LlamaEngine {
    model: Llama,
    config: Config,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    model_id: String,
}

impl LlamaEngine {
    pub fn new(
        model: Llama,
        config: Config,
        tokenizer: Tokenizer,
        model_dir: Option<&Path>,
        model_id: String,
    ) -> Self {
        let eos = collect_eos(&config, &tokenizer, model_dir);
        Self {
            model,
            config,
            tokenizer,
            eos,
            model_id,
        }
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer.decode(ids, true).map_err(|e| e.to_string())
    }
}

fn eos_from_value(v: &serde_json::Value, out: &mut Vec<u32>) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(id) = n.as_u64().and_then(|x| u32::try_from(x).ok()) {
                out.push(id);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| eos_from_value(x, out)),
        _ => {}
    }
}

/// EOS ids from config.json, plus generation_config.json (which for Llama-3
/// style chat models usually adds `<|eot_id|>`), falling back to the
/// tokenizer's well-known end tokens when the config has none.
fn collect_eos(config: &Config, tokenizer: &Tokenizer, model_dir: Option<&Path>) -> Vec<u32> {
    let mut eos = Vec::new();
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(id)) => eos.push(*id),
        Some(LlamaEosToks::Multiple(ids)) => eos.extend_from_slice(ids),
        None => {}
    }
    if let Some(dir) = model_dir
        && let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json"))
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(&s)
        && let Some(e) = v.get("eos_token_id")
    {
        eos_from_value(e, &mut eos);
    }
    if eos.is_empty() {
        for t in [
            "</s>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "<|endoftext|>",
            "<|im_end|>",
        ] {
            if let Some(id) = tokenizer.token_to_id(t) {
                eos.push(id);
            }
        }
    }
    eos.sort_unstable();
    eos.dedup();
    eos
}

/// Minimal chat template (the real one from tokenizer_config.json will come
/// with `Generator`).
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

impl Engine for LlamaEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn context_length(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, String> {
        let ids = match prompt {
            Prompt::Tokens(ids) => ids.clone(),
            Prompt::Text(s) => self
                .tokenizer
                .encode(s.as_str(), true)
                .map_err(|e| format!("tokenize: {e}"))?
                .get_ids()
                .to_vec(),
            Prompt::Chat(msgs) => self
                .tokenizer
                .encode(render_chat(msgs).as_str(), true)
                .map_err(|e| format!("tokenize: {e}"))?
                .get_ids()
                .to_vec(),
        };
        let vocab = self.config.vocab_size;
        if let Some(bad) = ids.iter().find(|&&i| i as usize >= vocab) {
            return Err(format!(
                "token id {bad} is out of range (vocab size {vocab})"
            ));
        }
        Ok(ids)
    }

    fn generate(
        &self,
        req: &GenRequest,
        on_delta: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<GenResult> {
        // Defense in depth: the API layer already checked these, but an
        // out-of-range position would index past the RoPE table / KV cache.
        if req.prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt"));
        }
        if req.max_tokens == 0 || req.prompt_ids.len() + req.max_tokens > self.context_length() {
            return Err(anyhow!("prompt + max_tokens exceeds the context length"));
        }
        let vocab = self.config.vocab_size;
        if req.prompt_ids.iter().any(|&i| i as usize >= vocab) {
            return Err(anyhow!("token id out of range"));
        }

        let mut cache = Cache::new(true, &self.config)?;
        let sp = &req.sampling;
        let temperature = (sp.temperature > 0.0).then_some(sp.temperature);
        let top_p = (sp.top_p > 0.0 && sp.top_p < 1.0).then_some(sp.top_p);
        // Sampler treats seed 0 as "random"; keep explicit seeds deterministic.
        let seed = if sp.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            sp.seed
        };
        let mut sampler = Sampler::new(seed, temperature, top_p, sp.repetition_penalty);

        let decode = |ids: &[u32]| self.decode(ids);
        let ctx_start = req.prompt_ids.len().saturating_sub(4);
        let mut detok = IncrementalDetok::new(&req.prompt_ids[ctx_start..]);
        let mut stops = StopBuffer::new(&req.stop);
        let mut counts: Vec<(u32, u32)> = Vec::new(); // generated token counts

        let mut toks = req.prompt_ids.clone();
        let mut index_pos = 0usize;
        let mut generated = 0usize;

        macro_rules! emit {
            ($text:expr) => {
                if on_delta($text).is_break() {
                    return Ok(GenResult {
                        finish: FinishReason::Cancelled,
                        completion_tokens: generated,
                        stop_word: None,
                    });
                }
            };
        }

        while generated < req.max_tokens {
            let ctx = if index_pos == 0 { toks.len() } else { 1 };
            let start = toks.len() - ctx;
            let logits = self.model.forward(&toks[start..], index_pos, &mut cache)?;
            let mut ld = logits.into_data();
            if ld.len() != vocab {
                return Err(anyhow!("logits size {} != vocab {}", ld.len(), vocab));
            }
            apply_logit_processors(&mut ld, sp, &counts);
            let nt = sampler.sample(&mut ld, &toks)?;
            toks.push(nt);
            index_pos += ctx;
            generated += 1;
            match counts.iter_mut().find(|(t, _)| *t == nt) {
                Some((_, c)) => *c += 1,
                None => counts.push((nt, 1)),
            }

            if self.eos.contains(&nt) {
                let tail = detok.finish(&decode).map_err(|e| anyhow!(e))?;
                let out = match stops.push(&tail) {
                    StopOutcome::Emit(e) => e + &stops.finish(),
                    StopOutcome::Stopped { emit, word } => {
                        emit!(&emit);
                        return Ok(GenResult {
                            finish: FinishReason::StopString,
                            completion_tokens: generated,
                            stop_word: Some(word),
                        });
                    }
                };
                emit!(&out);
                return Ok(GenResult {
                    finish: FinishReason::Eos,
                    completion_tokens: generated,
                    stop_word: None,
                });
            }

            let delta = detok.push(nt, &decode).map_err(|e| anyhow!(e))?;
            match stops.push(&delta) {
                StopOutcome::Emit(e) => emit!(&e),
                StopOutcome::Stopped { emit, word } => {
                    emit!(&emit);
                    return Ok(GenResult {
                        finish: FinishReason::StopString,
                        completion_tokens: generated,
                        stop_word: Some(word),
                    });
                }
            }
        }

        let tail = detok.finish(&decode).map_err(|e| anyhow!(e))?;
        match stops.push(&tail) {
            StopOutcome::Emit(e) => {
                let out = e + &stops.finish();
                emit!(&out);
                Ok(GenResult {
                    finish: FinishReason::Length,
                    completion_tokens: generated,
                    stop_word: None,
                })
            }
            StopOutcome::Stopped { emit, word } => {
                emit!(&emit);
                Ok(GenResult {
                    finish: FinishReason::StopString,
                    completion_tokens: generated,
                    stop_word: Some(word),
                })
            }
        }
    }
}

/// Logit processors the core `Sampler` doesn't implement yet: presence /
/// frequency penalties (over generated tokens), top-k and min-p. Masked
/// logits are set to -inf, which the sampler handles (exp(-inf) = 0).
pub fn apply_logit_processors(logits: &mut [f32], sp: &SamplingParams, counts: &[(u32, u32)]) {
    if sp.presence_penalty != 0.0 || sp.frequency_penalty != 0.0 {
        for &(t, c) in counts {
            if let Some(l) = logits.get_mut(t as usize) {
                *l -= sp.presence_penalty + sp.frequency_penalty * c as f32;
            }
        }
    }
    if sp.temperature <= 0.0 || logits.is_empty() {
        return; // greedy: top-k / min-p cannot change the argmax
    }
    if sp.top_k > 0 && sp.top_k < logits.len() {
        let mut v: Vec<f32> = logits.to_vec();
        let k = sp.top_k;
        let idx = v.len() - k;
        v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
        let threshold = v[idx];
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    if sp.min_p > 0.0 {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max.is_finite() {
            // p_i / p_max = exp((l_i - max) / T) < min_p
            let threshold = max + sp.temperature * sp.min_p.ln();
            for l in logits.iter_mut() {
                if *l < threshold {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }
}

#[cfg(test)]
pub mod mock {
    //! A tiny deterministic engine for end-to-end server tests: a byte-level
    //! "tokenizer" and a model that replays a fixed script of token bytes.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    pub struct MockEngine {
        pub script: Vec<u8>,
        pub ctx_len: usize,
        /// Sleep per token (to test disconnects / queueing).
        pub token_delay: Duration,
        /// Tokens actually produced across all requests.
        pub produced: AtomicUsize,
        pub eos: u32,
    }

    impl MockEngine {
        pub fn new(script: &str) -> Self {
            Self {
                script: script.as_bytes().to_vec(),
                ctx_len: 64,
                token_delay: Duration::ZERO,
                produced: AtomicUsize::new(0),
                eos: 0,
            }
        }
    }

    fn decode(ids: &[u32]) -> Result<String, ()> {
        let b: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
        Ok(String::from_utf8_lossy(&b).into_owned())
    }

    impl Engine for MockEngine {
        fn model_id(&self) -> &str {
            "mock-model"
        }
        fn context_length(&self) -> usize {
            self.ctx_len
        }
        fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, String> {
            let text = match prompt {
                Prompt::Text(s) => s.clone(),
                Prompt::Chat(m) => render_chat(m),
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
            req: &GenRequest,
            on_delta: &mut dyn FnMut(&str) -> ControlFlow<()>,
        ) -> anyhow::Result<GenResult> {
            assert!(req.prompt_ids.len() + req.max_tokens <= self.ctx_len);
            let mut detok = IncrementalDetok::new(&[]);
            let mut stops = StopBuffer::new(&req.stop);
            let mut n = 0;
            for &b in self.script.iter().take(req.max_tokens) {
                if !self.token_delay.is_zero() {
                    std::thread::sleep(self.token_delay);
                }
                n += 1;
                self.produced.fetch_add(1, Ordering::SeqCst);
                if u32::from(b) == self.eos {
                    let out = detok.finish(&decode).unwrap_or_default();
                    let out = match stops.push(&out) {
                        StopOutcome::Emit(e) => e + &stops.finish(),
                        StopOutcome::Stopped { emit, word } => {
                            let _ = on_delta(&emit);
                            return Ok(GenResult {
                                finish: FinishReason::StopString,
                                completion_tokens: n,
                                stop_word: Some(word),
                            });
                        }
                    };
                    let _ = on_delta(&out);
                    return Ok(GenResult {
                        finish: FinishReason::Eos,
                        completion_tokens: n,
                        stop_word: None,
                    });
                }
                let d = detok.push(u32::from(b), &decode).unwrap_or_default();
                let (text, stop) = match stops.push(&d) {
                    StopOutcome::Emit(e) => (e, None),
                    StopOutcome::Stopped { emit, word } => (emit, Some(word)),
                };
                if on_delta(&text).is_break() {
                    return Ok(GenResult {
                        finish: FinishReason::Cancelled,
                        completion_tokens: n,
                        stop_word: None,
                    });
                }
                if let Some(word) = stop {
                    return Ok(GenResult {
                        finish: FinishReason::StopString,
                        completion_tokens: n,
                        stop_word: Some(word),
                    });
                }
            }
            let tail = detok.finish(&decode).unwrap_or_default() + &stops.finish();
            let _ = on_delta(&tail);
            Ok(GenResult {
                finish: FinishReason::Length,
                completion_tokens: n,
                stop_word: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_and_min_p_mask() {
        let sp = SamplingParams {
            top_k: 2,
            ..Default::default()
        };
        let mut l = vec![1.0, 5.0, 3.0, 4.0];
        apply_logit_processors(&mut l, &sp, &[]);
        assert_eq!(l[1], 5.0);
        assert_eq!(l[3], 4.0);
        assert!(l[0].is_infinite() && l[2].is_infinite());

        let sp = SamplingParams {
            min_p: 0.5,
            ..Default::default()
        };
        // ln(0.5) = -0.69: keeps logits >= max - 0.69
        let mut l = vec![10.0, 9.5, 9.0];
        apply_logit_processors(&mut l, &sp, &[]);
        assert_eq!(&l[..2], &[10.0, 9.5]);
        assert!(l[2].is_infinite());
    }

    #[test]
    fn penalties() {
        let sp = SamplingParams {
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            temperature: 0.0,
            ..Default::default()
        };
        let mut l = vec![0.0; 4];
        apply_logit_processors(&mut l, &sp, &[(2, 3), (9, 1)]);
        assert_eq!(l, vec![0.0, 0.0, -2.5, 0.0]);
    }

    #[test]
    fn chat_render() {
        let m = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        assert_eq!(render_chat(&m), "user: hi\nassistant: ");
    }

    /// Real-tokenizer check of incremental detokenization (needs the model
    /// directory's tokenizer.json; not loaded by default).
    #[test]
    #[ignore]
    fn real_tokenizer_multibyte() {
        let dir = std::env::var("XORNET_MODEL").unwrap_or_else(|_| "models/bitnet-2b".into());
        let tok = Tokenizer::from_file(Path::new(&dir).join("tokenizer.json")).unwrap();
        let text = "Grüße 🦀 日本語 — ok";
        let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
        let decode = |ids: &[u32]| tok.decode(ids, true);
        let mut d = IncrementalDetok::new(&[]);
        let mut out = String::new();
        for id in ids {
            let delta = d.push(id, &decode).unwrap();
            assert!(!delta.contains('\u{FFFD}'));
            out.push_str(&delta);
        }
        out.push_str(&d.finish(&decode).unwrap());
        assert_eq!(out, text);
    }
}
