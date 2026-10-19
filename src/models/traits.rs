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

    /// Downcasting hook for the owning model implementation.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A decoder-only language model.
///
/// Contract for [`forward`](Self::forward):
/// - `tokens` are the next `tokens.len()` tokens of the sequence, placed at
///   positions `pos .. pos + tokens.len()`;
/// - the caller feeds every position exactly once, in order (prefill may be
///   split in chunks), so `pos` equals the number of tokens already fed into
///   `state` since it was created or [`reset`](ModelState::reset);
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
