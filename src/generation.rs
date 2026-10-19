//! High-level text generation on top of any [`CausalLM`].
//!
//! [`Generator`] owns a shared model, a tokenizer, an optional chat template
//! and one reusable per-sequence state. It implements the full decode loop
//! (chunked prefill, sampling, EOS / stop-token / stop-string handling,
//! streaming UTF-8-safe detokenisation) so callers don't have to.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use xor_net::generation::{Generator, GenerationParams};
//! # use xor_net::sampler::SamplingParams;
//! # fn demo(model: Arc<dyn xor_net::models::traits::CausalLM>) -> anyhow::Result<()> {
//! let mut generator = Generator::from_dir(model, "models/bitnet-2b")?;
//! let params = GenerationParams::default()
//!     .with_max_tokens(64)
//!     .with_sampling(SamplingParams::default().with_temperature(0.7).with_top_p(0.9));
//! let out = generator.generate_stream("The capital of France is", &params, |delta| {
//!     print!("{delta}");
//!     std::ops::ControlFlow::Continue(())
//! })?;
//! println!("\n{:?} {:.1} tok/s", out.finish_reason, out.timings.decode_tok_per_s());
//! # Ok(()) }
//! ```

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::models::traits::{CausalLM, ModelState};
use crate::sampler::{Sampler, SamplingParams};

// ---------------------------------------------------------------------------
// Parameters & results
// ---------------------------------------------------------------------------

/// Per-request generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParams {
    /// Maximum number of tokens to sample (including a final EOS).
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop strings, matched on the decoded text. The matched string and
    /// anything after it are not emitted.
    pub stop: Vec<String>,
    /// Extra token ids that end generation (in addition to EOS).
    pub stop_token_ids: Vec<u32>,
    /// Do not stop on the model's / tokenizer's EOS ids.
    pub ignore_eos: bool,
    /// Let the tokenizer add special tokens (e.g. BOS) when encoding a raw
    /// prompt. Chat prompts never do: the template is responsible.
    pub add_special_tokens: bool,
    /// Prompts longer than this are prefilled in chunks of this size.
    pub prefill_chunk_size: usize,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampling: SamplingParams::default(),
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            add_special_tokens: true,
            prefill_chunk_size: 512,
        }
    }
}

impl GenerationParams {
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }
    pub fn with_sampling(mut self, s: SamplingParams) -> Self {
        self.sampling = s;
        self
    }
    pub fn with_stop<S: Into<String>>(mut self, stop: impl IntoIterator<Item = S>) -> Self {
        self.stop = stop.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_stop_token_ids(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.stop_token_ids = ids.into_iter().collect();
        self
    }
    pub fn with_ignore_eos(mut self, v: bool) -> Self {
        self.ignore_eos = v;
        self
    }
    pub fn with_add_special_tokens(mut self, v: bool) -> Self {
        self.add_special_tokens = v;
        self
    }
    pub fn with_prefill_chunk_size(mut self, n: usize) -> Self {
        self.prefill_chunk_size = n;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS or a stop token id was sampled.
    Stop,
    /// `max_tokens` reached.
    Length,
    /// A stop string was matched.
    StopString,
    /// The streaming callback returned `ControlFlow::Break`.
    Aborted,
}

impl FinishReason {
    /// OpenAI `finish_reason` string.
    pub fn as_openai_str(self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::StopString | FinishReason::Aborted => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Number of sampled tokens (a final EOS / stop token counts).
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Prompt tokens whose state was reused from the previous request
    /// (prefix caching) and therefore not recomputed.
    pub cached_prompt_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// Wall time of the prompt forward passes (up to the first logits).
    pub prefill: Duration,
    /// Prompt tokens actually run through the model.
    pub prefill_tokens: usize,
    /// Wall time from the first sampled token to the end.
    pub decode: Duration,
    /// Tokens sampled during decode.
    pub decode_tokens: usize,
}

impl Timings {
    pub fn prefill_ms(&self) -> f64 {
        self.prefill.as_secs_f64() * 1e3
    }
    pub fn prefill_tok_per_s(&self) -> f64 {
        rate(self.prefill_tokens, self.prefill)
    }
    pub fn decode_tok_per_s(&self) -> f64 {
        rate(self.decode_tokens, self.decode)
    }
}

fn rate(n: usize, d: Duration) -> f64 {
    let s = d.as_secs_f64();
    if s > 0.0 { n as f64 / s } else { 0.0 }
}

/// Result of one generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// Completion text only (never the prompt); identical to the
    /// concatenation of all streamed deltas.
    pub text: String,
    /// Completion token ids, excluding a terminating EOS / stop token.
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub timings: Timings,
}

/// Typed request errors (returned inside `anyhow::Error`; use
/// `err.downcast_ref::<GenerationError>()` to inspect).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    /// `prompt_tokens + max_tokens > max_seq_len`.
    ContextOverflow {
        prompt_tokens: usize,
        max_tokens: usize,
        max_seq_len: usize,
    },
    /// The prompt encodes to no tokens and the model has no BOS token.
    EmptyPrompt,
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                max_seq_len,
            } => write!(
                f,
                "prompt ({prompt_tokens} tokens) + max_tokens ({max_tokens}) exceeds the \
                 model context of {max_seq_len} tokens"
            ),
            GenerationError::EmptyPrompt => {
                write!(f, "empty prompt and the model defines no BOS token")
            }
        }
    }
}

impl std::error::Error for GenerationError {}

// ---------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------

/// One chat turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// A HF-style Jinja chat template rendered with minijinja (jinja2 with
/// `trim_blocks` / `lstrip_blocks`, Python string methods, and the
/// `raise_exception` helper, as in `transformers`).
pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    source: String,
    bos_token: String,
    eos_token: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("source", &self.source)
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .finish()
    }
}

/// Used when the tokenizer config has no `chat_template`.
pub const FALLBACK_CHAT_TEMPLATE: &str = "{{ bos_token }}\
{% for message in messages %}\
{% if message.role == 'system' %}System: {{ message.content }}\n\
{% elif message.role == 'user' %}User: {{ message.content }}\n\
{% else %}Assistant: {{ message.content }}{{ eos_token }}\n{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}Assistant:{% endif %}";

impl ChatTemplate {
    pub fn new(
        source: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let source = source.into();
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<minijinja::Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template_owned("chat", source.clone())
            .map_err(|e| anyhow::anyhow!("invalid chat template: {e}"))?;
        Ok(Self {
            env,
            source,
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        })
    }

    /// The generic `System:/User:/Assistant:` template. Pair it with
    /// `stop = ["\nUser:"]` for base models.
    pub fn fallback(bos_token: impl Into<String>, eos_token: impl Into<String>) -> Self {
        Self::new(FALLBACK_CHAT_TEMPLATE, bos_token, eos_token)
            .expect("fallback chat template is valid")
    }

    /// Build from a parsed `tokenizer_config.json`. Returns `Ok(None)` when it
    /// has no `chat_template`. Supports the string form and the list form
    /// (`[{"name": "default", "template": ...}, ...]`, picks "default").
    pub fn from_tokenizer_config(cfg: &serde_json::Value) -> anyhow::Result<Option<Self>> {
        let Some(src) = cfg.get("chat_template").and_then(template_source) else {
            return Ok(None);
        };
        let bos = special_token_str(cfg.get("bos_token")).unwrap_or_default();
        let eos = special_token_str(cfg.get("eos_token")).unwrap_or_default();
        Self::new(src, bos, eos).map(Some)
    }

    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    /// Render `messages` (anything serialisable with `role` / `content`
    /// fields, e.g. [`ChatMessage`] or raw JSON objects).
    pub fn render<M: Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.as_str(),
            eos_token => self.eos_token.as_str(),
        })
        .map_err(|e| anyhow::anyhow!("chat template render failed: {e:#}"))
    }
}

fn template_source(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(list) => {
            let named = |name: &str| {
                list.iter().find_map(|e| {
                    (e.get("name")?.as_str()? == name)
                        .then(|| e.get("template")?.as_str().map(str::to_owned))
                        .flatten()
                })
            };
            named("default").or_else(|| {
                list.first()
                    .and_then(|e| e.get("template")?.as_str().map(str::to_owned))
            })
        }
        _ => None,
    }
}

/// `"<s>"` or `{"content": "<s>", ...}`.
fn special_token_str(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

fn eos_ids_from_json(v: Option<&serde_json::Value>) -> Vec<u32> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Streaming detokenizer and stop-string matcher
// ---------------------------------------------------------------------------

/// Incremental detokenizer: emits only complete UTF-8 text, handles
/// byte-fallback tokens (a character split over several tokens) and
/// SentencePiece / byte-level leading-space merges by decoding a small
/// window of tokens and diffing (`prefix_offset` / `read_offset`).
#[derive(Debug, Clone)]
pub struct StreamDecoder {
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
    skip_special_tokens: bool,
}

impl StreamDecoder {
    /// `context`: the last few prompt tokens, so the first generated token
    /// is decoded in context (keeps its leading space).
    pub fn new(context: &[u32], skip_special_tokens: bool) -> Self {
        const CTX: usize = 6;
        let ctx = &context[context.len().saturating_sub(CTX)..];
        Self {
            tokens: ctx.to_vec(),
            prefix_offset: 0,
            read_offset: ctx.len(),
            skip_special_tokens,
        }
    }

    fn decode(&self, tok: &Tokenizer, ids: &[u32]) -> anyhow::Result<String> {
        tok.decode(ids, self.skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }

    /// Feed one token; returns newly completed text, if any.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> anyhow::Result<Option<String>> {
        self.tokens.push(id);
        let prefix = self.decode(tok, &self.tokens[self.prefix_offset..self.read_offset])?;
        let full = self.decode(tok, &self.tokens[self.prefix_offset..])?;
        if full.len() <= prefix.len() || full.ends_with('\u{FFFD}') {
            // Nothing new yet, or an incomplete UTF-8 sequence: wait.
            return Ok(None);
        }
        let delta = diff_suffix(&prefix, &full).to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        if self.prefix_offset > 64 {
            self.tokens.drain(..self.prefix_offset);
            self.read_offset -= self.prefix_offset;
            self.prefix_offset = 0;
        }
        Ok(Some(delta))
    }

    /// Emit whatever is still held back (possibly with U+FFFD for a
    /// truncated UTF-8 sequence).
    pub fn flush(&mut self, tok: &Tokenizer) -> anyhow::Result<Option<String>> {
        if self.read_offset == self.tokens.len() {
            return Ok(None);
        }
        let prefix = self.decode(tok, &self.tokens[self.prefix_offset..self.read_offset])?;
        let full = self.decode(tok, &self.tokens[self.prefix_offset..])?;
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        if full.len() <= prefix.len() {
            return Ok(None);
        }
        Ok(Some(diff_suffix(&prefix, &full).to_owned()))
    }
}

/// The part of `full` after its common prefix with `prefix` (normally
/// `full` starts with `prefix`; fall back to the longest common prefix on a
/// char boundary if a normaliser rewrote it).
fn diff_suffix<'a>(prefix: &str, full: &'a str) -> &'a str {
    if let Some(rest) = full.strip_prefix(prefix) {
        return rest;
    }
    let mut common = 0;
    for ((i, a), b) in full.char_indices().zip(prefix.chars()) {
        if a != b {
            break;
        }
        common = i + a.len_utf8();
    }
    &full[common..]
}

/// Holds back text that could be the start of a stop string.
#[derive(Debug, Clone, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
        }
    }

    /// Append decoded text. Returns `(safe_to_emit, stop_matched)`. On a
    /// match, the text before the stop string is returned and the rest is
    /// discarded.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_owned(), false);
        }
        self.pending.push_str(text);
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(at) = hit {
            let out = self.pending[..at].to_owned();
            self.pending.clear();
            return (out, true);
        }
        let mut hold = 0;
        for s in &self.stops {
            let max_k = (s.len() - 1).min(self.pending.len());
            for k in (hold + 1..=max_k).rev() {
                if s.is_char_boundary(k) && self.pending.ends_with(&s[..k]) {
                    hold = k;
                    break;
                }
            }
        }
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_owned();
        self.pending.drain(..cut);
        (out, false)
    }

    /// Release held-back text (generation ended for another reason).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Text generation front-end for one sequence at a time. Cheap to
/// [`fork`](Self::fork) per worker thread: the model, tokenizer and template
/// are shared; each fork owns its own model state.
pub struct Generator {
    model: Arc<dyn CausalLM>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Option<Arc<ChatTemplate>>,
    /// EOS ids from generation_config.json / tokenizer_config.json, merged
    /// with the model's own.
    eos_ids: Vec<u32>,
    sampler: Sampler,
    state: Option<Box<dyn ModelState>>,
    /// Tokens already fed into `state` (empty if the state is unusable).
    fed: Vec<u32>,
    prefix_caching: bool,
}

impl Generator {
    pub fn new(model: Arc<dyn CausalLM>, tokenizer: impl Into<Arc<Tokenizer>>) -> Self {
        let eos_ids = model.eos_token_ids().to_vec();
        Self {
            model,
            tokenizer: tokenizer.into(),
            chat_template: None,
            eos_ids,
            sampler: Sampler::from_params(SamplingParams::greedy().with_seed(Some(0))),
            state: None,
            fed: Vec::new(),
            prefix_caching: true,
        }
    }

    /// Load `tokenizer.json` (required), plus `tokenizer_config.json`
    /// (chat template, bos/eos strings), `chat_template.jinja` and
    /// `generation_config.json` (extra EOS ids) when present.
    pub fn from_dir(model: Arc<dyn CausalLM>, dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| {
            anyhow::anyhow!("loading {}: {e}", dir.join("tokenizer.json").display())
        })?;
        let mut g = Self::new(model, tokenizer);

        let read_json = |name: &str| -> anyhow::Result<Option<serde_json::Value>> {
            let p = dir.join(name);
            if !p.exists() {
                return Ok(None);
            }
            let s = std::fs::read_to_string(&p)?;
            Ok(Some(serde_json::from_str(&s).map_err(|e| {
                anyhow::anyhow!("parsing {}: {e}", p.display())
            })?))
        };

        let tok_cfg = read_json("tokenizer_config.json")?;
        let mut template = match &tok_cfg {
            Some(cfg) => ChatTemplate::from_tokenizer_config(cfg)?,
            None => None,
        };
        if template.is_none() {
            let jinja = dir.join("chat_template.jinja");
            if jinja.exists() {
                let cfg = tok_cfg.clone().unwrap_or(serde_json::Value::Null);
                template = Some(ChatTemplate::new(
                    std::fs::read_to_string(jinja)?,
                    special_token_str(cfg.get("bos_token")).unwrap_or_else(|| g.default_bos_str()),
                    special_token_str(cfg.get("eos_token")).unwrap_or_else(|| g.default_eos_str()),
                )?);
            }
        }
        if let Some(t) = template {
            g.chat_template = Some(Arc::new(t));
        }

        let mut extra = Vec::new();
        if let Some(eos) = tok_cfg
            .as_ref()
            .and_then(|cfg| special_token_str(cfg.get("eos_token")))
        {
            extra.extend(g.tokenizer.token_to_id(&eos));
        }
        if let Some(gen_cfg) = read_json("generation_config.json")? {
            extra.extend(eos_ids_from_json(gen_cfg.get("eos_token_id")));
        }
        g.add_eos_token_ids(extra);
        Ok(g)
    }

    /// A new generator sharing model, tokenizer and template, with its own
    /// (lazily allocated) state.
    pub fn fork(&self) -> Self {
        Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            chat_template: self.chat_template.clone(),
            eos_ids: self.eos_ids.clone(),
            sampler: Sampler::from_params(SamplingParams::greedy().with_seed(Some(0))),
            state: None,
            fed: Vec::new(),
            prefix_caching: self.prefix_caching,
        }
    }

    pub fn with_chat_template(mut self, t: ChatTemplate) -> Self {
        self.chat_template = Some(Arc::new(t));
        self
    }

    pub fn add_eos_token_ids(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            if !self.eos_ids.contains(&id) {
                self.eos_ids.push(id);
            }
        }
    }

    /// Reuse the model state across requests when the new prompt extends
    /// exactly the tokens already processed (multi-turn chat). On by default.
    pub fn set_prefix_caching(&mut self, on: bool) {
        self.prefix_caching = on;
    }

    /// Drop the model state (frees KV-cache memory until the next request).
    pub fn release_state(&mut self) {
        self.state = None;
        self.fed.clear();
    }

    pub fn model(&self) -> &Arc<dyn CausalLM> {
        &self.model
    }
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.chat_template.as_deref()
    }
    /// All ids treated as EOS (model config + tokenizer/generation config).
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    fn default_bos_str(&self) -> String {
        self.model
            .bos_token_id()
            .and_then(|id| self.tokenizer.id_to_token(id))
            .unwrap_or_default()
    }

    fn default_eos_str(&self) -> String {
        self.model
            .eos_token_ids()
            .first()
            .and_then(|&id| self.tokenizer.id_to_token(id))
            .unwrap_or_default()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }

    /// Render a conversation with the model's chat template (or
    /// [`FALLBACK_CHAT_TEMPLATE`] if it has none).
    pub fn apply_chat_template<M: Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        match &self.chat_template {
            Some(t) => t.render(messages, add_generation_prompt),
            None => ChatTemplate::fallback(self.default_bos_str(), self.default_eos_str())
                .render(messages, add_generation_prompt),
        }
    }

    /// Generate a completion for a raw text prompt.
    pub fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
    ) -> anyhow::Result<Generation> {
        self.generate_stream(prompt, params, |_| ControlFlow::Continue(()))
    }

    /// Like [`generate`](Self::generate), calling `on_text` with each new
    /// piece of text. Returning `ControlFlow::Break(())` stops generation
    /// (finish reason [`FinishReason::Aborted`]).
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        let ids = self.encode(prompt, params.add_special_tokens)?;
        self.generate_from_tokens(&ids, params, on_text)
    }

    /// Chat completion: renders `messages` with `add_generation_prompt`
    /// and encodes without adding special tokens (the template owns BOS).
    pub fn chat<M: Serialize>(
        &mut self,
        messages: &[M],
        params: &GenerationParams,
    ) -> anyhow::Result<Generation> {
        self.chat_stream(messages, params, |_| ControlFlow::Continue(()))
    }

    pub fn chat_stream<M: Serialize>(
        &mut self,
        messages: &[M],
        params: &GenerationParams,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        let prompt = self.apply_chat_template(messages, true)?;
        let ids = self.encode(&prompt, false)?;
        self.generate_from_tokens(&ids, params, on_text)
    }

    /// Core decode loop on an already tokenised prompt.
    pub fn generate_from_tokens(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        mut on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        // --- validation ------------------------------------------------------
        let bos_only;
        let prompt = if prompt.is_empty() {
            match self.model.bos_token_id() {
                Some(bos) => {
                    bos_only = [bos];
                    &bos_only[..]
                }
                None => return Err(GenerationError::EmptyPrompt.into()),
            }
        } else {
            prompt
        };
        let max_seq_len = self.model.max_seq_len();
        if prompt.len() + params.max_tokens > max_seq_len {
            return Err(GenerationError::ContextOverflow {
                prompt_tokens: prompt.len(),
                max_tokens: params.max_tokens,
                max_seq_len,
            }
            .into());
        }
        let vocab = self.model.vocab_size();
        let mut usage = Usage {
            prompt_tokens: prompt.len(),
            ..Usage::default()
        };
        let mut timings = Timings::default();
        if params.max_tokens == 0 {
            return Ok(Generation {
                text: String::new(),
                tokens: Vec::new(),
                finish_reason: FinishReason::Length,
                usage: Usage {
                    total_tokens: prompt.len(),
                    ..usage
                },
                timings,
            });
        }

        // --- state: reuse the processed prefix when possible ---------------
        let mut fed = std::mem::take(&mut self.fed); // empty on any early exit
        let reuse = self.prefix_caching
            && self.state.is_some()
            && !fed.is_empty()
            && prompt.len() > fed.len()
            && prompt.starts_with(&fed);
        let mut state = match self.state.take() {
            Some(mut s) => {
                if !reuse {
                    s.reset()?;
                }
                s
            }
            None => self.model.new_state()?,
        };
        if !reuse {
            fed.clear();
        }
        usage.cached_prompt_tokens = fed.len();

        let result = self.run(
            prompt,
            params,
            &mut *state,
            &mut fed,
            &mut usage,
            &mut timings,
            vocab,
            &mut on_text,
        );
        // The state is valid for `fed` only if every forward succeeded.
        self.state = Some(state);
        match result {
            Ok((text, tokens, finish_reason)) => {
                self.fed = fed;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
                Ok(Generation {
                    text,
                    tokens,
                    finish_reason,
                    usage,
                    timings,
                })
            }
            Err(e) => Err(e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        state: &mut dyn ModelState,
        fed: &mut Vec<u32>,
        usage: &mut Usage,
        timings: &mut Timings,
        vocab: usize,
        on_text: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<(String, Vec<u32>, FinishReason)> {
        let model = self.model.clone();
        let check = |l: &Vec<f32>| -> anyhow::Result<()> {
            anyhow::ensure!(
                l.len() == vocab,
                "model returned {} logits, expected vocab_size {vocab}",
                l.len()
            );
            Ok(())
        };

        // --- prefill ---------------------------------------------------------
        let t0 = Instant::now();
        let chunk = params.prefill_chunk_size.max(1);
        let mut logits = Vec::new();
        for piece in prompt[fed.len()..].chunks(chunk) {
            logits = model.forward(piece, fed.len(), state)?;
            fed.extend_from_slice(piece);
        }
        check(&logits)?;
        timings.prefill = t0.elapsed();
        timings.prefill_tokens = prompt.len() - usage.cached_prompt_tokens;

        // --- decode ----------------------------------------------------------
        let t1 = Instant::now();
        self.sampler.set_params(params.sampling.clone());
        let mut detok = StreamDecoder::new(prompt, true);
        let mut stops = StopMatcher::new(&params.stop);
        let mut text = String::new();
        let mut out: Vec<u32> = Vec::with_capacity(params.max_tokens);
        let is_stop_id = |t: u32| {
            params.stop_token_ids.contains(&t) || (!params.ignore_eos && self.eos_ids.contains(&t))
        };
        let mut emit = |s: &str, text: &mut String| -> ControlFlow<()> {
            if s.is_empty() {
                return ControlFlow::Continue(());
            }
            text.push_str(s);
            on_text(s)
        };

        let finish = loop {
            let tok = self.sampler.sample_split(&mut logits, prompt, &out)?;
            usage.completion_tokens += 1;
            if is_stop_id(tok) {
                break FinishReason::Stop;
            }
            out.push(tok);
            if let Some(delta) = detok.push(&self.tokenizer, tok)? {
                let (safe, hit) = stops.push(&delta);
                if emit(&safe, &mut text).is_break() {
                    break FinishReason::Aborted;
                }
                if hit {
                    break FinishReason::StopString;
                }
            }
            if usage.completion_tokens >= params.max_tokens {
                break FinishReason::Length;
            }
            logits = model.forward(&[tok], fed.len(), state)?;
            fed.push(tok);
            check(&logits)?;
        };

        if matches!(finish, FinishReason::Stop | FinishReason::Length) {
            // Release text held back by the detokenizer / stop matcher.
            if let Some(rest) = detok.flush(&self.tokenizer)? {
                let (safe, hit) = stops.push(&rest);
                let _ = emit(&safe, &mut text);
                if hit {
                    timings.decode = t1.elapsed();
                    timings.decode_tokens = usage.completion_tokens;
                    return Ok((text, out, FinishReason::StopString));
                }
            }
            let rest = stops.flush();
            let _ = emit(&rest, &mut text);
        }
        timings.decode = t1.elapsed();
        timings.decode_tokens = usage.completion_tokens;
        Ok((text, out, finish))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_matcher_holds_back_partial_matches() {
        let mut m = StopMatcher::new(&["\nUser:".to_string(), "STOP".to_string()]);
        assert_eq!(m.push("Hello"), ("Hello".to_string(), false));
        assert_eq!(m.push(" world\nUs"), (" world".to_string(), false));
        assert_eq!(m.push("eless"), ("\nUseless".to_string(), false));
        assert_eq!(m.push("ok ST"), ("ok ".to_string(), false));
        assert_eq!(m.push("OP and more"), (String::new(), true));
        let mut m = StopMatcher::new(&["ab".to_string()]);
        assert_eq!(m.push("xa"), ("x".to_string(), false));
        assert_eq!(m.flush(), "a");
        // Earliest match wins across stop strings.
        let mut m = StopMatcher::new(&["cd".to_string(), "b".to_string()]);
        assert_eq!(m.push("abcd"), ("a".to_string(), true));
        // Multi-byte stop strings split mid-character are safe.
        let mut m = StopMatcher::new(&["é!".to_string()]);
        assert_eq!(m.push("caf"), ("caf".to_string(), false));
        assert_eq!(m.push("é"), (String::new(), false));
        assert_eq!(m.push("!"), (String::new(), true));
    }

    #[test]
    fn diff_suffix_char_boundaries() {
        assert_eq!(diff_suffix("ab", "abc"), "c");
        assert_eq!(diff_suffix("aé", "aéz"), "z");
        assert_eq!(diff_suffix("a b", "a  bc"), " bc");
    }

    #[test]
    fn template_source_forms() {
        let s = serde_json::json!({"chat_template": "x"});
        assert_eq!(template_source(&s["chat_template"]).unwrap(), "x");
        let l = serde_json::json!([
            {"name": "tool_use", "template": "t"},
            {"name": "default", "template": "d"}
        ]);
        assert_eq!(template_source(&l).unwrap(), "d");
        assert_eq!(
            special_token_str(Some(&serde_json::json!({"content": "<s>"}))).unwrap(),
            "<s>"
        );
    }
}
