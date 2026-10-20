//! Batched forward over several independent sequences (continuous
//! batching).
//!
//! Decoding one sequence streams every weight once per token, so a
//! memory-bound model decodes at the DRAM-bandwidth ceiling no matter how
//! many cores it has. [`BatchedCausalLM::forward_batch`] runs the next
//! tokens of *several* sequences in one pass: the linear layers see the
//! concatenated rows of all items (each weight block is read once per step
//! for all of them), while attention / recurrent mixers run per item against
//! that item's own state.
//!
//! Contract: for every item, the returned logits are bit-identical to
//! `CausalLM::forward(item.tokens, item.pos, item.state)` on that item alone
//! (the Bonsai kernels use one fixed arithmetic order whatever the number of
//! rows), and each item's state ends up exactly as after that call.

use crate::models::traits::{CausalLM, Logits, ModelState};

/// One sequence of a batched forward: its next `tokens` at positions
/// `pos .. pos + tokens.len()`, fed into `state` (same contract as
/// [`CausalLM::forward`]). A prefill chunk has several tokens, a decode
/// step one.
pub struct BatchItem<'a> {
    pub tokens: &'a [u32],
    pub pos: usize,
    pub state: &'a mut dyn ModelState,
}

/// A [`CausalLM`] that can forward several sequences in one pass.
pub trait BatchedCausalLM: CausalLM {
    /// Forward every item (each under the [`CausalLM::forward`] contract,
    /// with distinct states) and return each item's last-position logits,
    /// in item order. On error, the states of all items are unspecified.
    fn forward_batch(&self, items: &mut [BatchItem<'_>]) -> anyhow::Result<Vec<Logits>>;
}

/// [`BatchedCausalLM::forward_batch`] for any model: one
/// [`CausalLM::forward`] per item (correct, no shared weight reads).
pub fn forward_batch_looping(
    model: &dyn CausalLM,
    items: &mut [BatchItem<'_>],
) -> anyhow::Result<Vec<Logits>> {
    items
        .iter_mut()
        .map(|it| model.forward(it.tokens, it.pos, &mut *it.state))
        .collect()
}
