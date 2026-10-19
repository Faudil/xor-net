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
use crate::prompt_cache::{self, PromptCache};
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
    /// Extra variables for the chat template render context (HF
    /// `apply_chat_template(**kwargs)` / vLLM `chat_template_kwargs`), e.g.
    /// `enable_thinking`. They never override `messages`,
    /// `add_generation_prompt`, `bos_token` or `eos_token`.
    pub chat_template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// How `<think>…</think>` blocks in the output are reported.
    pub reasoning_format: ReasoningFormat,
}

/// How reasoning (`<think>…</think>`) in the model output is handled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    /// Leave the tags and their content in [`Generation::text`].
    #[default]
    None,
    /// Strip the tags; the content inside goes to [`Generation::reasoning`]
    /// and is streamed as [`StreamEvent::Reasoning`]. A prompt that ends
    /// inside an open `<think>` (e.g. the Qwen3.5 template with thinking on)
    /// starts the output in reasoning mode.
    Separate,
}

impl std::str::FromStr for ReasoningFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "raw" | "off" => Ok(Self::None),
            "separate" | "deepseek" | "on" => Ok(Self::Separate),
            other => anyhow::bail!("unknown reasoning format {other:?} (none | separate)"),
        }
    }
}

/// One streamed piece of output, see [`Generator::generate_stream_events`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEvent<'a> {
    /// Visible answer text.
    Text(&'a str),
    /// Text inside a `<think>` block (only with [`ReasoningFormat::Separate`]).
    Reasoning(&'a str),
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
            chat_template_kwargs: serde_json::Map::new(),
            reasoning_format: ReasoningFormat::None,
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
    /// Set one chat template variable (see [`Self::chat_template_kwargs`]).
    pub fn with_chat_template_kwarg(
        mut self,
        key: &str,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.chat_template_kwargs
            .insert(key.to_owned(), value.into());
        self
    }
    /// Shorthand for `with_chat_template_kwarg("enable_thinking", on)`.
    /// Only templates that read `enable_thinking` honour it (Bonsai 27B /
    /// Qwen3.5 and upstream Qwen3 do; the Bonsai 1.7B/4B/8B templates
    /// always emit an empty think block, i.e. thinking is always off).
    pub fn with_enable_thinking(self, on: bool) -> Self {
        self.with_chat_template_kwarg("enable_thinking", on)
    }
    pub fn with_reasoning_format(mut self, f: ReasoningFormat) -> Self {
        self.reasoning_format = f;
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
    /// Prompt tokens whose state was reused (from this generator's previous
    /// request, or restored from the shared [`PromptCache`]) and therefore
    /// not recomputed.
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
    /// concatenation of all streamed text deltas. With
    /// [`ReasoningFormat::Separate`] it excludes the reasoning.
    pub text: String,
    /// Reasoning content (concatenation of the [`StreamEvent::Reasoning`]
    /// deltas). `None` unless [`ReasoningFormat::Separate`] is used and the
    /// model produced non-empty reasoning.
    pub reasoning: Option<String>,
    /// The stop string that ended generation ([`FinishReason::StopString`]).
    pub stop_string: Option<String>,
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
        self.render_with_kwargs(messages, add_generation_prompt, &serde_json::Map::new())
    }

    /// Like [`render`](Self::render) with extra template variables (e.g.
    /// `enable_thinking`, `tools`). The fixed variables win on a clash.
    pub fn render_with_kwargs<M: Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> anyhow::Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let fixed = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.as_str(),
            eos_token => self.eos_token.as_str(),
        };
        let ctx = if kwargs.is_empty() {
            fixed
        } else {
            minijinja::context! { ..fixed, ..minijinja::Value::from_serialize(kwargs) }
        };
        tmpl.render(ctx)
            .map_err(|e| anyhow::anyhow!("chat template render failed: {e:#}"))
    }

    /// The template knows about `<think>` blocks (reasoning models).
    pub fn has_thinking(&self) -> bool {
        self.source.contains("<think>")
    }

    /// The template reads an `enable_thinking` variable.
    pub fn supports_enable_thinking(&self) -> bool {
        self.source.contains("enable_thinking")
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
    matched: Option<usize>,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
            matched: None,
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
            .enumerate()
            .filter_map(|(i, s)| self.pending.find(s.as_str()).map(|at| (at, i)))
            .min();
        if let Some((at, i)) = hit {
            self.matched = Some(i);
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

    /// The stop string that matched, if any.
    pub fn matched(&self) -> Option<&str> {
        self.matched.map(|i| self.stops[i].as_str())
    }
}

// ---------------------------------------------------------------------------
// Reasoning splitter
// ---------------------------------------------------------------------------

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Splits streamed text into visible text and the content of the first
/// `<think>…</think>` block, holding back tags split across deltas. Newlines
/// right after a tag and right before `</think>` are dropped (Qwen3 /
/// DeepSeek-R1 put `\n` / `\n\n` around the reasoning). A redundant `<think>`
/// at the very start of a block the prompt already opened is dropped. Tags
/// after the first block are left in the text.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    in_reasoning: bool,
    done: bool,
    strip_leading: bool,
    /// In reasoning, nothing emitted yet: a leading `<think>` is redundant.
    at_block_start: bool,
    pending: String,
}

impl ReasoningSplitter {
    /// `start_in_reasoning`: the prompt ended inside an open `<think>`.
    pub fn new(start_in_reasoning: bool) -> Self {
        Self {
            in_reasoning: start_in_reasoning,
            done: false,
            strip_leading: start_in_reasoning,
            at_block_start: start_in_reasoning,
            pending: String::new(),
        }
    }

    /// Whether `prompt` ends inside an unclosed `<think>` block.
    pub fn prompt_opens_reasoning(prompt: &str) -> bool {
        match (prompt.rfind(THINK_OPEN), prompt.rfind(THINK_CLOSE)) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        }
    }

    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// Feed text; appends `(is_reasoning, piece)` pairs (non-empty) to `out`.
    pub fn push(&mut self, text: &str, out: &mut Vec<(bool, String)>) {
        if self.done && self.pending.is_empty() && !self.strip_leading {
            if !text.is_empty() {
                out.push((false, text.to_owned()));
            }
            return;
        }
        self.pending.push_str(text);
        loop {
            if self.strip_leading {
                let n = self.pending.len() - self.pending.trim_start_matches('\n').len();
                self.pending.drain(..n);
                if self.pending.is_empty() {
                    return;
                }
                self.strip_leading = false;
            }
            if self.done {
                out.push((false, std::mem::take(&mut self.pending)));
                return;
            }
            if self.at_block_start {
                if self.pending.starts_with(THINK_OPEN) {
                    self.pending.drain(..THINK_OPEN.len());
                    self.at_block_start = false;
                    self.strip_leading = true;
                    continue;
                }
                if THINK_OPEN.starts_with(self.pending.as_str()) {
                    return; // could still become a redundant `<think>`
                }
                self.at_block_start = false;
            }
            let tag = if self.in_reasoning {
                THINK_CLOSE
            } else {
                THINK_OPEN
            };
            if let Some(i) = self.pending.find(tag) {
                let mut before = &self.pending[..i];
                if self.in_reasoning {
                    before = before.trim_end_matches('\n');
                }
                if !before.is_empty() {
                    out.push((self.in_reasoning, before.to_owned()));
                }
                self.pending.drain(..i + tag.len());
                if self.in_reasoning {
                    self.done = true;
                }
                self.in_reasoning = !self.in_reasoning;
                self.strip_leading = true;
                self.at_block_start = false;
                continue;
            }
            // Hold back a possible tag prefix (and, in reasoning, the
            // newlines before it).
            let mut hold = (1..tag.len().min(self.pending.len() + 1))
                .rev()
                .find(|&k| self.pending.ends_with(&tag[..k]))
                .unwrap_or(0);
            if self.in_reasoning {
                let rest = &self.pending[..self.pending.len() - hold];
                hold += rest.len() - rest.trim_end_matches('\n').len();
            }
            let cut = self.pending.len() - hold;
            if cut > 0 {
                let piece: String = self.pending.drain(..cut).collect();
                out.push((self.in_reasoning, piece));
            }
            return;
        }
    }

    /// End of output: release held-back text (a partial tag stays text of
    /// the current kind; trailing reasoning newlines are dropped).
    pub fn flush(&mut self, out: &mut Vec<(bool, String)>) {
        let mut rest = std::mem::take(&mut self.pending);
        if self.in_reasoning {
            rest.truncate(rest.trim_end_matches('\n').len());
        }
        if !rest.is_empty() {
            out.push((self.in_reasoning, rest));
        }
    }
}

/// Accumulates output and forwards it to the stream callback, splitting
/// reasoning off when requested.
struct OutputSink<'a> {
    splitter: Option<ReasoningSplitter>,
    text: String,
    reasoning: String,
    buf: Vec<(bool, String)>,
    on_event: &'a mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
}

impl OutputSink<'_> {
    fn dispatch(&mut self) -> ControlFlow<()> {
        let mut flow = ControlFlow::Continue(());
        for (reasoning, piece) in self.buf.drain(..) {
            if piece.is_empty() {
                continue;
            }
            let ev = if reasoning {
                self.reasoning.push_str(&piece);
                StreamEvent::Reasoning(&piece)
            } else {
                self.text.push_str(&piece);
                StreamEvent::Text(&piece)
            };
            // Keep accounting for the rest of the batch even after a Break.
            if flow.is_continue() {
                flow = (self.on_event)(ev);
            }
        }
        flow
    }

    fn push(&mut self, s: &str) -> ControlFlow<()> {
        if s.is_empty() {
            return ControlFlow::Continue(());
        }
        match &mut self.splitter {
            Some(sp) => sp.push(s, &mut self.buf),
            None => self.buf.push((false, s.to_owned())),
        }
        self.dispatch()
    }

    fn finish(&mut self) {
        if let Some(sp) = &mut self.splitter {
            sp.flush(&mut self.buf);
            let _ = self.dispatch();
        }
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
    /// Shared cross-request prefix cache (see [`Self::with_prompt_cache`]).
    prompt_cache: Option<Arc<PromptCache>>,
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
            prompt_cache: None,
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
        Self::from_dir_with_tokenizer(model, dir, tokenizer)
    }

    /// Like [`from_dir`](Self::from_dir) with an explicit tokenizer (the
    /// directory need not contain `tokenizer.json`); the chat template and
    /// EOS ids still come from the directory's config files when present.
    pub fn from_dir_with_tokenizer(
        model: Arc<dyn CausalLM>,
        dir: impl AsRef<Path>,
        tokenizer: impl Into<Arc<Tokenizer>>,
    ) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
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
            prompt_cache: self.prompt_cache.clone(),
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

    /// Reuse the model state across requests (multi-turn chat). The state
    /// is rolled back to the longest common prefix of the tokens already
    /// processed and the new prompt (chat templates often re-render the
    /// previous reply differently from what was sampled); states that cannot
    /// roll back arbitrarily (recurrent layers) fall back to the end of the
    /// previous prompt, which is checkpointed. On by default.
    pub fn set_prefix_caching(&mut self, on: bool) {
        self.prefix_caching = on;
    }

    /// Share a cross-request [`PromptCache`] (forks inherit it). Before each
    /// prefill the longest cached prefix of the prompt is restored when it
    /// beats what this generator's own state can reuse; after prefill a
    /// snapshot is inserted at each prefix boundary (for chat: the end of
    /// the system prompt and the end of the history before the generation
    /// prompt) that is at least [`PromptCache::min_tokens`] long and not
    /// cached yet. Use one cache per model.
    pub fn with_prompt_cache(mut self, cache: Arc<PromptCache>) -> Self {
        self.prompt_cache = Some(cache);
        self
    }

    /// Set or remove the shared prompt cache (see [`Self::with_prompt_cache`]).
    pub fn set_prompt_cache(&mut self, cache: Option<Arc<PromptCache>>) {
        self.prompt_cache = cache;
    }

    pub fn prompt_cache(&self) -> Option<&Arc<PromptCache>> {
        self.prompt_cache.as_ref()
    }

    /// Tokens held by this generator's state (what the next request can
    /// reuse without the prompt cache); empty if there is no usable state.
    pub fn processed_tokens(&self) -> &[u32] {
        if self.state.is_some() { &self.fed } else { &[] }
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
        self.apply_chat_template_with_kwargs(
            messages,
            add_generation_prompt,
            &serde_json::Map::new(),
        )
    }

    /// [`apply_chat_template`](Self::apply_chat_template) with extra
    /// template variables (e.g. `enable_thinking`).
    pub fn apply_chat_template_with_kwargs<M: Serialize>(
        &self,
        messages: &[M],
        add_generation_prompt: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> anyhow::Result<String> {
        match &self.chat_template {
            Some(t) => t.render_with_kwargs(messages, add_generation_prompt, kwargs),
            None => ChatTemplate::fallback(self.default_bos_str(), self.default_eos_str())
                .render_with_kwargs(messages, add_generation_prompt, kwargs),
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
    /// piece of visible text (reasoning deltas are not passed; see
    /// [`generate_stream_events`](Self::generate_stream_events)). Returning
    /// `ControlFlow::Break(())` stops generation (finish reason
    /// [`FinishReason::Aborted`]).
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        self.generate_stream_events(prompt, params, text_only(on_text))
    }

    /// Like [`generate_stream`](Self::generate_stream), reporting text and
    /// reasoning deltas separately.
    pub fn generate_stream_events(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_event: impl FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        let ids = self.encode(prompt, params.add_special_tokens)?;
        let starts_in_reasoning = ReasoningSplitter::prompt_opens_reasoning(prompt);
        self.generate_impl(&ids, params, starts_in_reasoning, &[], on_event)
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

    /// Streaming [`chat`](Self::chat); `on_text` sees visible text only.
    pub fn chat_stream<M: Serialize>(
        &mut self,
        messages: &[M],
        params: &GenerationParams,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        self.chat_stream_events(messages, params, text_only(on_text))
    }

    /// Streaming [`chat`](Self::chat) with text and reasoning events. The
    /// template is rendered with `params.chat_template_kwargs`.
    pub fn chat_stream_events<M: Serialize>(
        &mut self,
        messages: &[M],
        params: &GenerationParams,
        on_event: impl FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        let prompt =
            self.apply_chat_template_with_kwargs(messages, true, &params.chat_template_kwargs)?;
        let ids = self.encode(&prompt, false)?;
        let starts_in_reasoning = ReasoningSplitter::prompt_opens_reasoning(&prompt);
        let boundaries = if self.prefix_caching || self.prompt_cache.is_some() {
            self.chat_prefix_boundaries(messages, &params.chat_template_kwargs, &ids)
        } else {
            Vec::new()
        };
        self.generate_impl(&ids, params, starts_in_reasoning, &boundaries, on_event)
    }

    /// Core decode loop on an already tokenised prompt (`on_text` sees
    /// visible text only).
    pub fn generate_from_tokens(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        self.generate_from_tokens_events(prompt, params, text_only(on_text))
    }

    /// [`generate_from_tokens`](Self::generate_from_tokens) with text and
    /// reasoning events. Whether the output starts inside a `<think>`
    /// block is detected from the last 64 prompt tokens.
    pub fn generate_from_tokens_events(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        on_event: impl FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        self.generate_from_tokens_with_boundaries(prompt, params, &[], on_event)
    }

    /// [`generate_from_tokens_events`](Self::generate_from_tokens_events)
    /// with prefix `boundaries`: prompt positions where later requests are
    /// likely to diverge, e.g. from [`chat_prefix_boundaries`](Self::chat_prefix_boundaries)
    /// for a pre-tokenised chat prompt. The largest one is where a recurrent
    /// state checkpoints (default: the end of the prompt); each one is a
    /// [`PromptCache`] insertion point.
    pub fn generate_from_tokens_with_boundaries(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        boundaries: &[usize],
        on_event: impl FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<Generation> {
        let starts_in_reasoning = params.reasoning_format == ReasoningFormat::Separate && {
            let tail = &prompt[prompt.len().saturating_sub(64)..];
            ReasoningSplitter::prompt_opens_reasoning(&self.decode(tail, false)?)
        };
        self.generate_impl(prompt, params, starts_in_reasoning, boundaries, on_event)
    }

    /// Prefix boundaries of the chat prompt `ids` (`messages` rendered with
    /// the generation prompt and `kwargs`), ascending:
    /// - the end of the leading system message(s), which other
    ///   conversations with the same system prompt share: the system
    ///   messages rendered alone or, for templates that refuse that (e.g.
    ///   Qwen3.5: "No user query found"), the common prefix of two renders
    ///   whose next message differs only in content (`"a"` vs `"b"`), i.e.
    ///   the system part plus the next message's header;
    /// - the end of the conversation itself, i.e. `messages` rendered
    ///   *without* the generation prompt. That part renders identically in
    ///   the next turn (the reply and next user message are appended after
    ///   it), whereas the generation prompt (e.g. Qwen3's empty
    ///   `<think></think>` block) is usually not reproduced when the reply
    ///   is re-rendered.
    ///
    /// A boundary is only reported if its rendering tokenises to a prefix of
    /// `ids` (templates that cannot render a part alone just yield fewer).
    pub fn chat_prefix_boundaries<M: Serialize>(
        &self,
        messages: &[M],
        kwargs: &serde_json::Map<String, serde_json::Value>,
        ids: &[u32],
    ) -> Vec<usize> {
        let prefix_len = |msgs: &[M]| -> Option<usize> {
            let text = self
                .apply_chat_template_with_kwargs(msgs, false, kwargs)
                .ok()?;
            let toks = self.encode(&text, false).ok()?;
            (!toks.is_empty() && toks.len() <= ids.len() && ids.starts_with(&toks))
                .then_some(toks.len())
        };
        let n_system = messages
            .iter()
            .take_while(|m| {
                serde_json::to_value(m)
                    .ok()
                    .and_then(|v| v.get("role")?.as_str().map(|r| r == "system"))
                    .unwrap_or(false)
            })
            .count();
        let probe_len = || -> Option<usize> {
            let mut base = messages[..=n_system]
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            let role = base.pop()?.get("role")?.clone();
            let render = |content: &str| {
                let mut msgs = base.clone();
                msgs.push(serde_json::json!({"role": role, "content": content}));
                let text = self
                    .apply_chat_template_with_kwargs(&msgs, false, kwargs)
                    .ok()?;
                self.encode(&text, false).ok()
            };
            let (a, b) = (render("a")?, render("b")?);
            let n = common_prefix(&a, &b);
            (n > 0 && n < ids.len() && ids[..n] == a[..n]).then_some(n)
        };
        let mut out = Vec::new();
        if n_system > 0 && n_system < messages.len() {
            out.extend(prefix_len(&messages[..n_system]).or_else(probe_len));
        }
        out.extend(prefix_len(messages));
        out.sort_unstable();
        out.dedup();
        out
    }

    fn generate_impl(
        &mut self,
        prompt: &[u32],
        params: &GenerationParams,
        starts_in_reasoning: bool,
        boundaries: &[usize],
        mut on_event: impl FnMut(StreamEvent<'_>) -> ControlFlow<()>,
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
                reasoning: None,
                stop_string: None,
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
        let mut state = match self.state.take() {
            Some(s) => s,
            None => self.model.new_state()?,
        };
        // Shared prompt cache: the longest cached prefix, restored when it
        // beats what this state can keep (`own_max` bounds that from above,
        // so a clearly longer entry is restored without rolling back first).
        let cache_hit = self.prompt_cache.as_ref().and_then(|c| {
            c.lookup(
                prompt_cache::model_key(&self.model),
                prompt,
                prompt.len() - 1,
            )
        });
        let own_max = if self.prefix_caching {
            common_prefix(&fed, prompt).min(prompt.len() - 1)
        } else {
            0
        };
        let mut restored = None;
        if let Some(e) = &cache_hit
            && e.len() > own_max
            && state.restore(e.snapshot())
        {
            restored = Some(e);
        }
        let mut keep = 0;
        if restored.is_none() {
            keep = if self.prefix_caching {
                reusable_prefix(&mut *state, &fed, prompt)
            } else {
                0
            };
            if let Some(e) = &cache_hit
                && e.len() > keep
                && state.restore(e.snapshot())
            {
                restored = Some(e);
            }
        }
        if let Some(e) = restored {
            keep = e.len();
            fed.clear();
            fed.extend_from_slice(e.tokens());
            if let Some(c) = &self.prompt_cache {
                c.record_hit(keep);
            }
        } else {
            if keep == 0 {
                state.reset()?;
            }
            fed.truncate(keep);
        }
        usage.cached_prompt_tokens = keep;

        let mut sink = OutputSink {
            splitter: (params.reasoning_format == ReasoningFormat::Separate)
                .then(|| ReasoningSplitter::new(starts_in_reasoning)),
            text: String::new(),
            reasoning: String::new(),
            buf: Vec::new(),
            on_event: &mut on_event,
        };
        let result = self.run(
            prompt,
            params,
            boundaries,
            &mut *state,
            &mut fed,
            &mut usage,
            &mut timings,
            vocab,
            &mut sink,
        );
        // The state is valid for `fed` only if every forward succeeded.
        self.state = Some(state);
        match result {
            Ok((tokens, finish_reason, stop_string)) => {
                self.fed = fed;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
                Ok(Generation {
                    text: sink.text,
                    reasoning: (!sink.reasoning.is_empty()).then_some(sink.reasoning),
                    stop_string,
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
        boundaries: &[usize],
        state: &mut dyn ModelState,
        fed: &mut Vec<u32>,
        usage: &mut Usage,
        timings: &mut Timings,
        vocab: usize,
        sink: &mut OutputSink<'_>,
    ) -> anyhow::Result<(Vec<u32>, FinishReason, Option<String>)> {
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
        let start = fed.len();
        let recurrent = state.needs_checkpoint();
        // Restore point for the next turn (one per state): the end of the
        // prompt, or for chat the end of the conversation history before
        // the generation prompt (the largest boundary). Chat templates
        // re-render the assistant reply (and often the generation prompt)
        // differently from what was fed, so the next prompt diverges after
        // that point. States that can roll back to any position ignore it.
        let checkpoint_at = (self.prefix_caching && recurrent).then(|| {
            boundaries
                .iter()
                .copied()
                .filter(|&c| c >= start && c < prompt.len())
                .max()
                .unwrap_or(prompt.len())
        });
        // Prompt-cache insertion points not cached yet.
        let cache = self.prompt_cache.clone();
        let key = prompt_cache::model_key(&model);
        let mut snap_at: Vec<usize> = match &cache {
            Some(c) => boundaries
                .iter()
                .copied()
                .filter(|&b| b >= start && b <= prompt.len() && c.wants(b))
                .collect(),
            None => Vec::new(),
        };
        snap_at.sort_unstable();
        snap_at.dedup();
        if let Some(c) = &cache {
            snap_at.retain(|&b| !c.contains(key, &prompt[..b]));
        }
        let insert = |state: &dyn ModelState, b: usize| {
            if let (Some(c), Some(snap)) = (&cache, state.snapshot(b)) {
                c.insert(key, &prompt[..b], snap);
            }
        };
        // A recurrent state can only be snapshotted / checkpointed at its
        // current length, so prefill stops at each of those points; KV
        // states are snapshotted after the whole prefill.
        let mut stops = vec![prompt.len()];
        if recurrent {
            stops.extend(&snap_at);
            stops.extend(checkpoint_at);
            stops.sort_unstable();
            stops.dedup();
        }
        let mut logits = Vec::new();
        for end in stops {
            for piece in prompt[fed.len()..end].chunks(chunk) {
                logits = model.forward(piece, fed.len(), state)?;
                fed.extend_from_slice(piece);
            }
            if checkpoint_at == Some(end) {
                state.checkpoint();
            }
            if recurrent && snap_at.contains(&end) {
                insert(state, end);
            }
        }
        if !recurrent {
            for &b in &snap_at {
                insert(state, b);
            }
        }
        check(&logits)?;
        timings.prefill = t0.elapsed();
        timings.prefill_tokens = prompt.len() - usage.cached_prompt_tokens;

        // --- decode ----------------------------------------------------------
        let t1 = Instant::now();
        self.sampler.set_params(params.sampling.clone());
        let mut detok = StreamDecoder::new(prompt, true);
        let mut stops = StopMatcher::new(&params.stop);
        let mut out: Vec<u32> = Vec::with_capacity(params.max_tokens);
        let is_stop_id = |t: u32| {
            params.stop_token_ids.contains(&t) || (!params.ignore_eos && self.eos_ids.contains(&t))
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
                if sink.push(&safe).is_break() {
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
                let _ = sink.push(&safe);
                if hit {
                    sink.finish();
                    timings.decode = t1.elapsed();
                    timings.decode_tokens = usage.completion_tokens;
                    let stop = stops.matched().map(str::to_owned);
                    return Ok((out, FinishReason::StopString, stop));
                }
            }
            let rest = stops.flush();
            let _ = sink.push(&rest);
        }
        if finish != FinishReason::Aborted {
            sink.finish();
        }
        timings.decode = t1.elapsed();
        timings.decode_tokens = usage.completion_tokens;
        Ok((out, finish, stops.matched().map(str::to_owned)))
    }
}

/// How many leading positions of `state` (which holds `fed`) can be kept for
/// `prompt`, rolling the state back as needed. At least one prompt token is
/// always left to prefill (its logits start decoding). Tries the longest
/// common prefix, then the state's checkpoint; `0` means start over (the
/// caller resets). On a non-zero return the state holds exactly that prefix.
fn reusable_prefix(state: &mut dyn ModelState, fed: &[u32], prompt: &[u32]) -> usize {
    let lcp = common_prefix(fed, prompt);
    let target = lcp.min(prompt.len().saturating_sub(1));
    if target == 0 {
        return 0;
    }
    if target == fed.len() || state.truncate(target) {
        return target;
    }
    match state.checkpoint_len() {
        Some(cp) if cp > 0 && cp <= target && state.truncate(cp) => cp,
        _ => 0,
    }
}

/// Length of the longest common prefix of `a` and `b`.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn text_only(
    mut on_text: impl FnMut(&str) -> ControlFlow<()>,
) -> impl FnMut(StreamEvent<'_>) -> ControlFlow<()> {
    move |ev| match ev {
        StreamEvent::Text(s) => on_text(s),
        StreamEvent::Reasoning(_) => ControlFlow::Continue(()),
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

    fn split_all(start: bool, pieces: &[&str]) -> (String, String) {
        let mut sp = ReasoningSplitter::new(start);
        let mut out = Vec::new();
        for p in pieces {
            sp.push(p, &mut out);
        }
        sp.flush(&mut out);
        let mut t = String::new();
        let mut r = String::new();
        for (is_r, s) in out {
            assert!(!s.is_empty());
            if is_r { r.push_str(&s) } else { t.push_str(&s) }
        }
        (t, r)
    }

    #[test]
    fn reasoning_splitter_any_chunking() {
        let cases: &[(bool, &str, &str, &str)] = &[
            (
                false,
                "<think>\nabc\n\n</think>\n\nanswer\n",
                "answer\n",
                "abc",
            ),
            (
                true,
                "\nx<y\n</think>\n\nz </think> <think>",
                "z </think> <think>",
                "x<y",
            ),
            (false, "no tags <thin", "no tags <thin", ""),
            (true, "unterminated\n\n", "", "unterminated"),
            (false, "pre<think>a\nb</think>", "pre", "a\nb"),
            (false, "é<think>ü</think>ß", "éß", "ü"),
            // The prompt opened the block and the model repeats the tag.
            (true, "<think>\n</think>\n\nok", "ok", ""),
            (true, "\n<think>\nr</think>t", "t", "r"),
            (true, "<thin", "", "<thin"),
            (true, "a<think>b</think>", "", "a<think>b"),
        ];
        for &(start, input, text, reasoning) in cases {
            let whole = split_all(start, &[input]);
            assert_eq!(
                whole,
                (text.to_string(), reasoning.to_string()),
                "{input:?}"
            );
            let chars: Vec<String> = input.chars().map(String::from).collect();
            let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
            assert_eq!(split_all(start, &refs), whole, "char-by-char {input:?}");
        }
        assert!(ReasoningSplitter::prompt_opens_reasoning(
            "a</think>b<think>\n"
        ));
        assert!(!ReasoningSplitter::prompt_opens_reasoning(
            "<think>\n\n</think>\n\n"
        ));
        assert!(!ReasoningSplitter::prompt_opens_reasoning("plain"));
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
