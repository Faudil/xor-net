//! Prompt-lookup (n-gram) drafting: no model. The last `n` tokens of the
//! context are matched against earlier context (longest `n` first, most
//! recent occurrence first) and the tokens that followed the match are
//! proposed. Effective whenever the output copies from the input
//! (summarisation, code editing, quoting) or repeats itself.

use super::{DraftModel, DraftSampler, DraftToken, Drafter};

/// Prompt-lookup drafter configuration (also its [`DraftModel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookup {
    /// N-gram sizes tried, from `ngram_max` down to `ngram_min`.
    pub ngram_min: usize,
    pub ngram_max: usize,
    /// Longest continuation proposed per round (the round's `k` caps it
    /// further).
    pub max_draft: usize,
    /// Only the last `window` context tokens are searched (0 = all).
    pub window: usize,
}

/// Matching 3- to 5-grams: shorter n-grams propose many wrong drafts
/// (measured on Bonsai 1.7B / 27B), whose verification costs more than the
/// rare hits save.
impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            ngram_min: 3,
            ngram_max: 5,
            max_draft: 4,
            window: 0,
        }
    }
}

impl PromptLookup {
    pub fn new(ngram_min: usize, ngram_max: usize, max_draft: usize) -> Self {
        let ngram_min = ngram_min.max(1);
        Self {
            ngram_min,
            ngram_max: ngram_max.max(ngram_min),
            max_draft: max_draft.max(1),
            window: 0,
        }
    }

    /// The proposal for `context` (at most `max` tokens).
    pub fn lookup<'a>(&self, context: &'a [u32], max: usize) -> &'a [u32] {
        let len = context.len();
        let max = max.min(self.max_draft);
        if max == 0 {
            return &[];
        }
        let lo = if self.window > 0 {
            len.saturating_sub(self.window)
        } else {
            0
        };
        for n in (self.ngram_min..=self.ngram_max).rev() {
            if n >= len {
                continue;
            }
            let pat = &context[len - n..];
            // Most recent earlier occurrence (start i < len - n).
            for i in (lo..len - n).rev() {
                if context[i..i + n] == *pat {
                    let from = i + n;
                    let to = (from + max).min(len);
                    if to > from {
                        return &context[from..to];
                    }
                }
            }
        }
        &[]
    }
}

impl DraftModel for PromptLookup {
    fn name(&self) -> &str {
        "lookup"
    }
    fn new_drafter(&self) -> anyhow::Result<Box<dyn Drafter>> {
        Ok(Box::new(*self))
    }
    fn max_draft(&self) -> usize {
        self.max_draft
    }
}

impl Drafter for PromptLookup {
    fn draft(
        &mut self,
        context: &[u32],
        max: usize,
        _sampler: &mut dyn DraftSampler,
    ) -> anyhow::Result<Vec<DraftToken>> {
        Ok(self
            .lookup(context, max)
            .iter()
            .map(|&token| DraftToken { token, probs: None })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_longest_most_recent_match() {
        let pl = PromptLookup::new(1, 3, 4);
        // ... 1 2 3 [9 9] ... 1 2 3 [7 8] ... 1 2 3 -> most recent: 7 8
        let ctx = [1, 2, 3, 9, 9, 5, 1, 2, 3, 7, 8, 6, 1, 2, 3];
        assert_eq!(pl.lookup(&ctx, 4), &[7, 8, 6, 1]);
        assert_eq!(pl.lookup(&ctx, 2), &[7, 8]);
        // Longest n wins over a more recent shorter match.
        let ctx = [4, 5, 6, 10, 0, 6, 11, 4, 5, 6];
        assert_eq!(PromptLookup::new(1, 3, 1).lookup(&ctx, 8), &[10]);
        assert_eq!(PromptLookup::new(1, 1, 1).lookup(&ctx, 8), &[11]);
        // No match.
        assert!(pl.lookup(&[1, 2, 3, 4], 4).is_empty());
        assert!(pl.lookup(&[], 4).is_empty());
        // The continuation may run up to the end of the context.
        assert_eq!(PromptLookup::new(2, 2, 8).lookup(&[1, 2, 1, 2], 8), &[1, 2]);
    }
}
