//! Model-family-agnostic interface used by [`crate::generation::Generator`].
//!
//! A model family implements [`CausalLM`]; its per-sequence state (a KV
//! cache, a recurrent state, or both) is opaque to callers and lives behind
//! [`ModelState`]. Both traits are object safe, so `Arc<dyn CausalLM>` and
//! `Box<dyn ModelState>` are the intended currency.

use std::any::Any;

/// Last-position logits, one `f32` per vocabulary entry. Owned so the
/// sampler can use it as scratch without an extra copy.
pub type Logits = Vec<f32>;

/// Opaque per-sequence inference state (KV cache, SSM/linear-attention
/// state, ...). Created by [`CausalLM::new_state`] and only ever passed back
/// to the model that created it.
pub trait ModelState: Any + Send {
    /// Make the state equivalent to a freshly created one so it can be reused
    /// for a new sequence starting at position 0 without reallocating.
    /// Position-indexed caches that are fully overwritten as positions are
    /// written may implement this as a no-op; recurrent states must zero.
    fn reset(&mut self) -> anyhow::Result<()>;

    /// Roll the state back so it holds exactly the first `len` positions of
    /// the sequence fed so far (`len <=` that count), as if only those had
    /// been fed. Returns `false`, leaving the state untouched, if this state
    /// cannot represent that prefix; the caller then falls back to
    /// [`checkpoint_len`](Self::checkpoint_len) or [`reset`](Self::reset).
    ///
    /// Position-indexed KV caches support any `len` (positions `>= len` are
    /// simply overwritten later). Recurrent states only support the current
    /// length, `0`, and the position saved by [`checkpoint`](Self::checkpoint).
    fn truncate(&mut self, len: usize) -> bool {
        let _ = len;
        false
    }

    /// Save a restore point at the current length so a later
    /// [`truncate`](Self::truncate) to exactly that length succeeds even if
    /// the state cannot otherwise roll back. At most one checkpoint is kept
    /// (a new one replaces it); [`reset`](Self::reset) discards it. A no-op
    /// for states that truncate to any length.
    fn checkpoint(&mut self) {}

    /// Length of the saved checkpoint, if any (see [`checkpoint`](Self::checkpoint)).
    fn checkpoint_len(&self) -> Option<usize> {
        None
    }

    /// `true` if rollback relies on [`checkpoint`](Self::checkpoint) (the
    /// state has recurrent parts), so callers should checkpoint where the
    /// next request is likely to diverge, even if that means splitting a
    /// prefill there. `false` (the default) for states that either truncate
    /// anywhere or not at all.
    fn needs_checkpoint(&self) -> bool {
        false
    }

    /// Deep copy of the first `len` positions (`len <=` the positions fed so
    /// far), for a prompt cache (see [`crate::prompt_cache::PromptCache`]).
    /// The copy holds only what those positions need (about `len` x the
    /// per-token KV size, plus any fixed-size recurrent state), never the
    /// state's spare capacity or scratch buffers. `None` if this state cannot
    /// represent that prefix: recurrent states only support the current
    /// length and the checkpoint (see [`checkpoint`](Self::checkpoint)), and
    /// states without snapshot support (the default) never do.
    fn snapshot(&self, len: usize) -> Option<StateSnapshot> {
        let _ = len;
        None
    }

    /// Overwrite this state with `snap` (taken from a state of the *same*
    /// model): afterwards it holds exactly [`snap.len()`](StateSnapshot::len)
    /// positions, as if those tokens had just been fed (any checkpoint is
    /// discarded). Returns `false`, leaving the state untouched, if the
    /// snapshot comes from another model type or shape.
    fn restore(&mut self, snap: &StateSnapshot) -> bool {
        let _ = snap;
        false
    }

    /// Finish a [`ForwardOpts::provisional`] forward (speculative
    /// verification): keep the first `len` positions of the sequence, where
    /// `len` lies between the length before that forward and the length
    /// after it. Afterwards the state is exactly as if only those positions
    /// had been fed. Also valid (and a no-op) without a provisional forward
    /// when `len` is the current length. The default truncates, which is
    /// right for position-indexed KV caches.
    fn commit(&mut self, len: usize) -> anyhow::Result<()> {
        anyhow::ensure!(self.truncate(len), "state cannot roll back to {len}");
        Ok(())
    }

    /// Downcasting hook for the owning model implementation.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// An immutable copy of a [`ModelState`] prefix, made by
/// [`ModelState::snapshot`] and consumed by [`ModelState::restore`]. The
/// payload is model specific; snapshots are `Send + Sync` so one can be
/// restored concurrently into several states (e.g. server slots).
pub struct StateSnapshot {
    len: usize,
    bytes: usize,
    data: Box<dyn Any + Send + Sync>,
}

impl StateSnapshot {
    /// Wrap a model-specific payload covering `len` positions and holding
    /// about `bytes` bytes of heap memory.
    pub fn new<T: Any + Send + Sync>(len: usize, bytes: usize, data: T) -> Self {
        Self {
            len,
            bytes,
            data: Box::new(data),
        }
    }

    /// Number of sequence positions the snapshot covers.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Approximate heap size (used for the prompt cache's byte budget).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The payload, if it has type `T`.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.data.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for StateSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateSnapshot")
            .field("len", &self.len)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

/// Hidden states copied out of a forward pass (for drafters such as DSpark
/// that condition on the target's intermediate layers).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HiddenTaps {
    /// Layer indices whose *output* (the residual stream after the layer's
    /// MLP) is captured, in output order.
    pub layers: Vec<usize>,
    /// Filled by the forward: `[n][layers.len() * hidden]`, row `t` being
    /// fed position `pos + t`, layers concatenated in `layers` order.
    pub data: Vec<f32>,
}

/// Extra outputs / modes of [`CausalLM::forward_with`]. `Default` is a plain
/// [`CausalLM::forward`].
#[derive(Debug, Default)]
pub struct ForwardOpts<'a> {
    /// Return the logits of *every* fed position (`[n][vocab]` row-major)
    /// instead of only the last one.
    pub all_logits: bool,
    /// Speculative verification: the fed positions are kept only
    /// provisionally; the caller must then call [`ModelState::commit`] with
    /// the number of positions to keep before anything else touches the
    /// state.
    pub provisional: bool,
    /// Capture hidden states (see [`HiddenTaps`]).
    pub taps: Option<&'a mut HiddenTaps>,
}

/// A decoder-only language model.
///
/// Contract for [`forward`](Self::forward):
/// - `tokens` are the next `tokens.len()` tokens of the sequence, placed at
///   positions `pos .. pos + tokens.len()`;
/// - the caller feeds every position exactly once, in order (prefill may be
///   split in chunks), so `pos` equals the number of tokens already fed into
///   `state` since it was created or [`reset`](ModelState::reset), minus any
///   positions dropped by a successful [`truncate`](ModelState::truncate);
/// - `pos + tokens.len() <= max_seq_len()`;
/// - the return value is the logits of the *last* fed position only.
pub trait CausalLM: Send + Sync {
    /// Allocate a fresh state for one sequence.
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>>;

    /// Run `tokens` through the model, updating `state`, and return the
    /// logits of the last position (`vocab_size()` entries).
    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits>;

    /// [`forward`](Self::forward) with the extras of [`ForwardOpts`] (all
    /// positions' logits, provisional verification, hidden taps), under the
    /// same feeding contract. Logits must be bit-identical to feeding the
    /// tokens one at a time, so speculative decoding stays exact.
    ///
    /// The default supports `all_logits` / `provisional` by feeding one
    /// token at a time (correct, but no faster than plain decoding; see
    /// [`batched_verify`](Self::batched_verify)) and rejects taps.
    fn forward_with(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
        opts: &mut ForwardOpts<'_>,
    ) -> anyhow::Result<Logits> {
        anyhow::ensure!(
            opts.taps.is_none(),
            "this model does not support hidden-state taps"
        );
        if !opts.all_logits {
            return self.forward(tokens, pos, state);
        }
        let mut out = Vec::with_capacity(tokens.len() * self.vocab_size());
        for (i, &t) in tokens.iter().enumerate() {
            out.extend(self.forward(&[t], pos + i, state)?);
        }
        Ok(out)
    }

    /// `true` if [`forward_with`](Self::forward_with) verifies several
    /// tokens in one pass over the weights (what makes speculative decoding
    /// pay off).
    fn batched_verify(&self) -> bool {
        false
    }

    /// Model width and depth, for validating [`HiddenTaps`] requests
    /// (`None` if taps are unsupported).
    fn hidden_shape(&self) -> Option<(usize, usize)> {
        None
    }

    /// The concrete model, for components that need more than this trait
    /// (e.g. a drafter sharing the target's embedding / output head).
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    fn vocab_size(&self) -> usize;

    /// Maximum number of positions a state can hold.
    fn max_seq_len(&self) -> usize;

    /// Token ids that end generation (from the model config).
    fn eos_token_ids(&self) -> &[u32];

    fn bos_token_id(&self) -> Option<u32>;
}

/// Helper for implementors: downcast an opaque state to the concrete type,
/// with a clear error if a state from another model was passed in.
pub fn downcast_state<T: ModelState>(state: &mut dyn ModelState) -> anyhow::Result<&mut T> {
    state.as_any_mut().downcast_mut::<T>().ok_or_else(|| {
        anyhow::anyhow!(
            "model state type mismatch: expected {}",
            std::any::type_name::<T>()
        )
    })
}
