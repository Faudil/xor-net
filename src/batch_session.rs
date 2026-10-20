//! Externally stepped generation sessions, for continuous batching.
//!
//! [`crate::Generator`] runs one request to completion, calling the model
//! itself. A [`DecodeSession`] holds the same per-request state (prompt,
//! model state, sampler, detokenizer, stop strings, reasoning split, usage,
//! prompt-cache bookkeeping) but lets a scheduler drive the model: it asks
//! the session for its next input ([`DecodeSession::input`]: a prefill
//! chunk or the last sampled token), runs one batched forward over many
//! sessions, and hands each its logits back ([`DecodeSession::feed`]).
//! Streamed text is buffered ([`DecodeSession::take_events`]).
//!
//! Semantics match `Generator::generate_from_tokens_with_boundaries`
//! (plain decoding, no speculative drafter): state reuse of the longest
//! common prefix, [`PromptCache`] restore / insertion at the prefix
//! boundaries, recurrent checkpoints, EOS / stop ids / stop strings /
//! `max_tokens`, and the same sampler calls, so a greedy (or seeded)
//! session produces the same tokens as the generator.

use std::sync::Arc;
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::generation::{
    FinishReason, GenerationError, GenerationParams, ReasoningFormat, ReasoningSplitter,
    StopMatcher, StreamDecoder, Timings, Usage, common_prefix,
};
use crate::models::batched::BatchItem;
use crate::models::traits::{CausalLM, Logits, ModelState};
use crate::prompt_cache::{self, PromptCache};
use crate::sampler::{Sampler, SamplingParams};

/// What every session of one model shares (taken from a
/// [`crate::Generator`]).
#[derive(Clone)]
pub struct SessionShared {
    pub model: Arc<dyn CausalLM>,
    pub tokenizer: Arc<Tokenizer>,
    pub eos_ids: Vec<u32>,
    pub prompt_cache: Option<Arc<PromptCache>>,
    /// Reuse the common prefix of the previous request on the same state.
    pub prefix_caching: bool,
}

impl SessionShared {
    /// Model, tokenizer, EOS ids and prompt cache of `g` (prefix caching on,
    /// the generator's default).
    pub fn from_generator(g: &crate::Generator) -> Self {
        Self {
            model: g.model().clone(),
            tokenizer: g.tokenizer().clone(),
            eos_ids: g.eos_token_ids().to_vec(),
            prompt_cache: g.prompt_cache().cloned(),
            prefix_caching: true,
        }
    }
}

/// How a finished session ended (see [`DecodeSession::outcome`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionOutcome {
    pub finish: FinishReason,
    pub usage: Usage,
    pub timings: Timings,
    pub stop_string: Option<String>,
    /// Completion tokens (without a terminating EOS / stop id).
    pub tokens: Vec<u32>,
}

/// One request being generated, stepped by an external scheduler.
pub struct DecodeSession {
    shared: SessionShared,
    prompt: Vec<u32>,
    params: GenerationParams,
    state: Box<dyn ModelState>,
    /// Tokens fed into `state`.
    fed: Vec<u32>,
    /// Prefill pause points (ascending, last = prompt length).
    pauses: Vec<usize>,
    checkpoint_at: Option<usize>,
    /// Prompt-cache insertion points still to snapshot.
    snap_at: Vec<usize>,
    recurrent: bool,
    sampler: Sampler,
    detok: StreamDecoder,
    stops: StopMatcher,
    splitter: Option<ReasoningSplitter>,
    out: Vec<u32>,
    /// The sampled token to feed next (decode phase).
    next: [u32; 1],
    decoding: bool,
    events: Vec<(bool, String)>,
    usage: Usage,
    timings: Timings,
    t_start: Instant,
    t_decode: Option<Instant>,
    finish: Option<FinishReason>,
    stop_string: Option<String>,
    /// A forward failed: the state no longer matches `fed`.
    poisoned: bool,
    error: Option<String>,
}

impl DecodeSession {
    /// Start a request. `state` / `fed` are a previous session's state and
    /// the tokens it holds (or `None` for a fresh state): the longest
    /// common prefix with `prompt` is reused, or a longer prefix restored
    /// from the prompt cache. `boundaries` as for
    /// `Generator::generate_from_tokens_with_boundaries`.
    pub fn start(
        shared: &SessionShared,
        prompt: &[u32],
        params: &GenerationParams,
        boundaries: &[usize],
        state: Option<(Box<dyn ModelState>, Vec<u32>)>,
    ) -> anyhow::Result<Self> {
        let model = &shared.model;
        let prompt: Vec<u32> = if prompt.is_empty() {
            match model.bos_token_id() {
                Some(bos) => vec![bos],
                None => return Err(GenerationError::EmptyPrompt.into()),
            }
        } else {
            prompt.to_vec()
        };
        let max_seq_len = model.max_seq_len();
        if prompt.len() + params.max_tokens > max_seq_len {
            return Err(GenerationError::ContextOverflow {
                prompt_tokens: prompt.len(),
                max_tokens: params.max_tokens,
                max_seq_len,
            }
            .into());
        }
        anyhow::ensure!(params.max_tokens > 0, "max_tokens must be at least 1");
        let starts_in_reasoning = params.reasoning_format == ReasoningFormat::Separate && {
            let tail = &prompt[prompt.len().saturating_sub(64)..];
            let text = shared
                .tokenizer
                .decode(tail, false)
                .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
            ReasoningSplitter::prompt_opens_reasoning(&text)
        };

        // --- state: reuse the processed prefix or the prompt cache --------
        let (mut state, mut fed) = match state {
            Some(s) => s,
            None => (model.new_state()?, Vec::new()),
        };
        let cache_hit = shared
            .prompt_cache
            .as_ref()
            .and_then(|c| c.lookup(prompt_cache::model_key(model), &prompt, prompt.len() - 1));
        let own_max = if shared.prefix_caching {
            common_prefix(&fed, &prompt).min(prompt.len() - 1)
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
            keep = if shared.prefix_caching {
                reusable_prefix(&mut *state, &fed, &prompt)
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
            if let Some(c) = &shared.prompt_cache {
                c.record_hit(keep);
            }
        } else {
            if keep == 0 {
                state.reset()?;
            }
            fed.truncate(keep);
        }
        let usage = Usage {
            prompt_tokens: prompt.len(),
            cached_prompt_tokens: keep,
            ..Usage::default()
        };

        // --- prefill plan (as `Generator::run`) ---------------------------
        let start = fed.len();
        let recurrent = state.needs_checkpoint();
        let checkpoint_at = (shared.prefix_caching && recurrent).then(|| {
            boundaries
                .iter()
                .copied()
                .filter(|&c| c >= start && c < prompt.len())
                .max()
                .unwrap_or(prompt.len())
        });
        let key = prompt_cache::model_key(model);
        let mut snap_at: Vec<usize> = match &shared.prompt_cache {
            Some(c) => boundaries
                .iter()
                .copied()
                .filter(|&b| b >= start && b <= prompt.len() && c.wants(b))
                .collect(),
            None => Vec::new(),
        };
        snap_at.sort_unstable();
        snap_at.dedup();
        if let Some(c) = &shared.prompt_cache {
            snap_at.retain(|&b| !c.contains(key, &prompt[..b]));
        }
        let mut pauses = vec![prompt.len()];
        if recurrent {
            pauses.extend(&snap_at);
            pauses.extend(checkpoint_at);
            pauses.sort_unstable();
            pauses.dedup();
        }
        let mut sampler = Sampler::from_params(SamplingParams::greedy().with_seed(Some(0)));
        sampler.set_params(params.sampling.clone());
        let detok = StreamDecoder::new(&prompt, true);
        let mut s = Self {
            shared: shared.clone(),
            detok,
            stops: StopMatcher::new(&params.stop),
            splitter: (params.reasoning_format == ReasoningFormat::Separate)
                .then(|| ReasoningSplitter::new(starts_in_reasoning)),
            prompt,
            params: params.clone(),
            state,
            fed,
            pauses,
            checkpoint_at,
            snap_at,
            recurrent,
            sampler,
            out: Vec::with_capacity(params.max_tokens),
            next: [0],
            decoding: false,
            events: Vec::new(),
            usage,
            timings: Timings::default(),
            t_start: Instant::now(),
            t_decode: None,
            finish: None,
            stop_string: None,
            poisoned: false,
            error: None,
        };
        // A pause at the current length (e.g. a checkpoint right at the
        // restored prefix) is handled before the first forward.
        s.handle_pause();
        Ok(s)
    }

    /// Whether the session still needs forwards.
    pub fn is_finished(&self) -> bool {
        self.finish.is_some()
    }

    /// In the prefill phase (the next input is a prompt chunk).
    pub fn is_prefilling(&self) -> bool {
        !self.decoding && self.finish.is_none()
    }

    /// Prompt tokens still to prefill.
    pub fn prefill_remaining(&self) -> usize {
        if self.decoding {
            0
        } else {
            self.prompt.len() - self.fed.len()
        }
    }

    /// The next forward of this session as a [`BatchItem`]: up to
    /// `max_prefill` prompt tokens (never past a checkpoint / snapshot
    /// point), or the last sampled token. `None` once finished.
    pub fn input(&mut self, max_prefill: usize) -> Option<BatchItem<'_>> {
        if self.finish.is_some() {
            return None;
        }
        let pos = self.fed.len();
        let tokens: &[u32] = if self.decoding {
            &self.next
        } else {
            let end = self.pauses.iter().copied().find(|&p| p > pos)?;
            let end = end.min(pos + max_prefill.max(1));
            &self.prompt[pos..end]
        };
        Some(BatchItem {
            tokens,
            pos,
            state: &mut *self.state,
        })
    }

    /// Account for the forward of the last [`input`](Self::input) (`n`
    /// tokens) that returned `logits` (its last position's): advance the
    /// prefill, or sample, detokenize and check the stop conditions.
    pub fn feed(&mut self, n: usize, mut logits: Logits) -> anyhow::Result<()> {
        anyhow::ensure!(self.finish.is_none(), "feed: session already finished");
        let vocab = self.shared.model.vocab_size();
        anyhow::ensure!(
            logits.len() == vocab,
            "model returned {} logits, expected vocab_size {vocab}",
            logits.len()
        );
        if self.decoding {
            anyhow::ensure!(n == 1, "feed: decode steps feed one token");
            self.fed.push(self.next[0]);
        } else {
            let pos = self.fed.len();
            anyhow::ensure!(
                n >= 1 && pos + n <= self.prompt.len(),
                "feed: prefill chunk out of range"
            );
            self.fed.extend_from_slice(&self.prompt[pos..pos + n]);
            self.handle_pause();
            if self.fed.len() < self.prompt.len() {
                return Ok(());
            }
            self.timings.prefill = self.t_start.elapsed();
            self.timings.prefill_tokens = self.prompt.len() - self.usage.cached_prompt_tokens;
            self.decoding = true;
            self.t_decode = Some(Instant::now());
        }
        self.sample(&mut logits)
    }

    /// Checkpoint / snapshot at the current length if it is a pause point;
    /// after the whole prompt, snapshot the KV-only boundaries.
    fn handle_pause(&mut self) {
        let len = self.fed.len();
        if self.checkpoint_at == Some(len) {
            self.state.checkpoint();
        }
        let key = prompt_cache::model_key(&self.shared.model);
        let insert = |state: &dyn ModelState, b: usize, prompt: &[u32]| {
            if let (Some(c), Some(snap)) = (&self.shared.prompt_cache, state.snapshot(b)) {
                c.insert(key, &prompt[..b], snap);
            }
        };
        if self.recurrent {
            if self.snap_at.contains(&len) {
                insert(&*self.state, len, &self.prompt);
            }
        } else if len == self.prompt.len() {
            for &b in &self.snap_at {
                insert(&*self.state, b, &self.prompt);
            }
        }
    }

    fn sample(&mut self, logits: &mut [f32]) -> anyhow::Result<()> {
        let tok = self.sampler.sample_split(logits, &self.prompt, &self.out)?;
        self.usage.completion_tokens += 1;
        let p = &self.params;
        let finish = if p.stop_token_ids.contains(&tok)
            || (!p.ignore_eos && self.shared.eos_ids.contains(&tok))
        {
            Some(FinishReason::Stop)
        } else {
            self.out.push(tok);
            let mut f = None;
            if let Some(delta) = self.detok.push(&self.shared.tokenizer, tok)? {
                let (safe, hit) = self.stops.push(&delta);
                self.emit(&safe);
                if hit {
                    f = Some(FinishReason::StopString);
                }
            }
            if f.is_none() && self.usage.completion_tokens >= self.params.max_tokens {
                f = Some(FinishReason::Length);
            }
            f
        };
        match finish {
            Some(f) => self.end(f)?,
            None => self.next = [tok],
        }
        Ok(())
    }

    fn emit(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        match &mut self.splitter {
            Some(sp) => sp.push(s, &mut self.events),
            None => self.events.push((false, s.to_owned())),
        }
    }

    /// Finish with `finish`, releasing held-back text as the generator does.
    fn end(&mut self, mut finish: FinishReason) -> anyhow::Result<()> {
        if matches!(finish, FinishReason::Stop | FinishReason::Length) {
            if let Some(rest) = self.detok.flush(&self.shared.tokenizer)? {
                let (safe, hit) = self.stops.push(&rest);
                self.emit(&safe);
                if hit {
                    finish = FinishReason::StopString;
                }
            }
            if finish != FinishReason::StopString {
                let rest = self.stops.flush();
                self.emit(&rest);
            }
        }
        if finish != FinishReason::Aborted
            && let Some(sp) = &mut self.splitter
        {
            sp.flush(&mut self.events);
        }
        self.stop_string = self.stops.matched().map(str::to_owned);
        self.finish_timings();
        self.finish = Some(finish);
        Ok(())
    }

    fn finish_timings(&mut self) {
        match self.t_decode {
            Some(t) => {
                self.timings.decode = t.elapsed();
                self.timings.decode_tokens = self.usage.completion_tokens;
            }
            None => self.timings.prefill = self.t_start.elapsed(),
        }
    }

    /// Stop now (client went away): finishes with
    /// [`FinishReason::Aborted`]; buffered events are kept.
    pub fn cancel(&mut self) {
        if self.finish.is_none() {
            self.stop_string = None;
            self.finish_timings();
            self.finish = Some(FinishReason::Aborted);
        }
    }

    /// Finish with an error (e.g. a failed forward): the state is treated
    /// as unusable.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.poisoned = true;
        if self.error.is_none() {
            self.error = Some(error.into());
        }
        self.cancel();
    }

    /// The error the session failed with, if any.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Streamed `(is_reasoning, text)` pieces since the last call.
    pub fn take_events(&mut self) -> Vec<(bool, String)> {
        std::mem::take(&mut self.events)
    }

    /// The result once finished.
    pub fn outcome(&self) -> Option<SessionOutcome> {
        let finish = self.finish?;
        let mut usage = self.usage;
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
        Some(SessionOutcome {
            finish,
            usage,
            timings: self.timings,
            stop_string: self.stop_string.clone(),
            tokens: self.out.clone(),
        })
    }

    /// The model state and the tokens it holds, for the next request on the
    /// same slot (`None` if a forward failed and the state is unusable).
    pub fn into_state(self) -> Option<(Box<dyn ModelState>, Vec<u32>)> {
        (!self.poisoned).then_some((self.state, self.fed))
    }
}

/// How many leading positions of `state` (which holds `fed`) can be kept for
/// `prompt` (see `generation::reusable_prefix`).
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

/// A batched forward: each item's last-position logits, in order
/// (`BatchedCausalLM::forward_batch`, or
/// [`crate::models::batched::forward_batch_looping`]).
pub type ForwardBatch<'f> = dyn Fn(&mut [BatchItem<'_>]) -> anyhow::Result<Vec<Logits>> + 'f;

/// One continuous-batching step over `sessions`: the next token of every
/// decoding session plus prefill chunks of the prefilling ones (in order,
/// each at most `prefill_chunk` tokens, `prefill_budget` tokens in total;
/// the first prefilling session always gets a chunk), through one
/// `forward`, the logits fed back. Finished sessions are skipped. Returns
/// the number of tokens forwarded. If the forward fails, every session in
/// it fails with the error; a session whose own bookkeeping fails (e.g.
/// detokenization) fails alone.
pub fn step(
    forward: &ForwardBatch<'_>,
    sessions: &mut [&mut DecodeSession],
    prefill_chunk: usize,
    prefill_budget: usize,
) -> usize {
    let mut budget = prefill_budget;
    let mut first_prefill = true;
    let mut chosen = Vec::new();
    for (i, s) in sessions.iter().enumerate() {
        if s.is_finished() {
            continue;
        }
        if s.is_prefilling() {
            let take = if first_prefill {
                prefill_chunk.min(budget.max(1))
            } else {
                prefill_chunk.min(budget)
            };
            if take == 0 {
                continue;
            }
            first_prefill = false;
            let take = take.min(s.prefill_remaining());
            budget = budget.saturating_sub(take);
            chosen.push((i, take));
        } else {
            chosen.push((i, 1));
        }
    }
    if chosen.is_empty() {
        return 0;
    }
    let mut items = Vec::with_capacity(chosen.len());
    let mut lens = Vec::with_capacity(chosen.len());
    let mut rest: &mut [&mut DecodeSession] = sessions;
    let mut base = 0;
    for &(i, take) in &chosen {
        let (_, tail) = rest.split_at_mut(i - base);
        let (one, tail) = tail.split_at_mut(1);
        rest = tail;
        base = i + 1;
        if let Some(it) = one[0].input(take) {
            lens.push(it.tokens.len());
            items.push(it);
        }
    }
    let total: usize = lens.iter().sum();
    let res = forward(&mut items);
    drop(items);
    match res {
        Ok(logits) if logits.len() == chosen.len() => {
            for ((&(i, _), &n), l) in chosen.iter().zip(&lens).zip(logits) {
                if let Err(e) = sessions[i].feed(n, l) {
                    sessions[i].fail(format!("{e:#}"));
                }
            }
        }
        Ok(logits) => {
            let msg = format!(
                "forward_batch returned {} results for {} items",
                logits.len(),
                chosen.len()
            );
            for &(i, _) in &chosen {
                sessions[i].fail(msg.clone());
            }
        }
        Err(e) => {
            let msg = format!("{e:#}");
            for &(i, _) in &chosen {
                sessions[i].fail(msg.clone());
            }
        }
    }
    total
}
