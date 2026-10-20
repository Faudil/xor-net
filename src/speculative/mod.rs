//! Lossless speculative decoding.
//!
//! Each round a [`Drafter`] proposes up to `k` tokens after the last sampled
//! token `t`; the target model verifies `[t, d1 .. dk]` in **one** forward
//! ([`crate::models::traits::ForwardOpts::provisional`], logits of every
//! position) and the longest acceptable prefix is kept:
//!
//! - greedy: `d_i` is accepted iff it is the target's argmax (after
//!   penalties) at its position, so the output is token-identical to plain
//!   greedy decoding;
//! - sampling: standard speculative rejection sampling (Leviathan et al.
//!   2023, Chen et al. 2023): `d_i ~ q` is accepted with probability
//!   `min(1, p(d_i) / q(d_i))`, else a token is drawn from
//!   `norm(max(p - q, 0))` and the round ends. `p` and `q` both go through
//!   the request's [`crate::sampler::SamplingParams`] (penalties,
//!   temperature, top-k/top-p/min-p), so every emitted token is distributed
//!   exactly as plain sampling from the target would be. Deterministic
//!   drafters (prompt lookup) have a one-hot `q`.
//!
//! After all `k` drafts are accepted one bonus token is drawn from the last
//! position. The state is then committed to the accepted prefix
//! ([`crate::models::traits::ModelState::commit`]): KV caches truncate,
//! recurrent (Gated DeltaNet) layers replay the accepted tokens' saved
//! update inputs.
//!
//! Drafters: [`lookup::PromptLookup`] (n-gram match against the context, any
//! model) and DSpark (`crate::models::q1::dspark`, Bonsai 27B).
//!
//! Enable per request with
//! [`GenerationParams::with_speculative`](crate::generation::GenerationParams::with_speculative).

pub mod lookup;

use std::sync::Arc;

pub use lookup::PromptLookup;

/// One proposed token.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftToken {
    pub token: u32,
    /// Draft distribution over the vocabulary the token was sampled from
    /// (dense, after the request's sampling pipeline); `None` means the
    /// drafter is deterministic (`q` one-hot at `token`).
    pub probs: Option<Vec<f32>>,
}

/// How a drafter turns its own logits into a draft token, consistently with
/// the request's sampling parameters (see [`Drafter::draft`]).
pub trait DraftSampler {
    /// Pick the next draft token from `logits` (consumed as scratch).
    /// `drafted` are the tokens drafted so far this round (they extend the
    /// penalty context). Greedy requests return the argmax and `None`.
    fn pick(
        &mut self,
        logits: &mut [f32],
        drafted: &[u32],
    ) -> anyhow::Result<(u32, Option<Vec<f32>>)>;
}

/// Per-sequence drafting state, created by a [`DraftModel`]. The generator
/// keeps it in sync with the target state: [`rewind`](Self::rewind) at the
/// start of each request, [`observe`](Self::observe) for every committed
/// target forward.
pub trait Drafter: Send {
    /// Target layers whose outputs this drafter needs for every committed
    /// position (see [`crate::models::traits::HiddenTaps`]).
    fn taps(&self) -> Option<&[usize]> {
        None
    }

    /// The target state now holds exactly `fed` (start of a request, after
    /// prefix reuse / prompt-cache restore). Forget anything beyond the
    /// common prefix with what was observed.
    fn rewind(&mut self, fed: &[u32]) {
        let _ = fed;
    }

    /// Target positions `pos .. pos + tokens.len()` were committed.
    /// `taps`: their hidden-state rows (`[n][taps.len() * hidden]`) when
    /// [`taps`](Self::taps) asked for them.
    fn observe(&mut self, tokens: &[u32], pos: usize, taps: Option<&[f32]>) -> anyhow::Result<()> {
        let _ = (tokens, pos, taps);
        Ok(())
    }

    /// Propose up to `max` tokens following `context` (the whole sequence:
    /// prompt and output so far; its last token is the one about to be fed
    /// at position `context.len() - 1`). May return fewer, or none.
    fn draft(
        &mut self,
        context: &[u32],
        max: usize,
        sampler: &mut dyn DraftSampler,
    ) -> anyhow::Result<Vec<DraftToken>>;
}

/// Shared (weights / configuration) part of a drafter.
pub trait DraftModel: Send + Sync {
    /// Short name for logs and stats.
    fn name(&self) -> &str;
    /// Fresh per-sequence state.
    fn new_drafter(&self) -> anyhow::Result<Box<dyn Drafter>>;
    /// Largest useful draft length (e.g. DSpark's trained block size).
    fn max_draft(&self) -> usize {
        usize::MAX
    }
    /// Check that this drafter suits `target` (vocabulary, hidden taps).
    fn check_target(&self, target: &dyn crate::models::traits::CausalLM) -> anyhow::Result<()> {
        let _ = target;
        Ok(())
    }
}

/// Speculative decoding configuration (see
/// [`GenerationParams::with_speculative`](crate::generation::GenerationParams::with_speculative)).
#[derive(Clone)]
pub struct SpecConfig {
    pub drafter: Arc<dyn DraftModel>,
    /// Maximum draft tokens per round (capped by [`DraftModel::max_draft`]).
    pub k: usize,
    /// Pick the draft length per round (0 to `k`) from the measured
    /// acceptance per depth and the measured draft / verify costs (see
    /// [`AdaptiveK`]); drafting stops when it does not pay off.
    pub adaptive: bool,
}

impl SpecConfig {
    pub fn new(drafter: Arc<dyn DraftModel>, k: usize) -> Self {
        Self {
            drafter,
            k: k.max(1),
            adaptive: false,
        }
    }
    pub fn with_adaptive(mut self, on: bool) -> Self {
        self.adaptive = on;
        self
    }
    /// Draft length actually used per round.
    pub fn effective_k(&self) -> usize {
        self.k.clamp(1, self.drafter.max_draft().max(1))
    }
}

impl std::fmt::Debug for SpecConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecConfig")
            .field("drafter", &self.drafter.name())
            .field("k", &self.k)
            .field("adaptive", &self.adaptive)
            .finish()
    }
}

impl PartialEq for SpecConfig {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.drafter, &other.drafter)
            && self.k == other.k
            && self.adaptive == other.adaptive
    }
}

/// Per-request speculative decoding counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpecStats {
    /// Verification forwards (rounds with at least one draft token).
    pub rounds: usize,
    /// Draft tokens proposed / accepted.
    pub drafted: usize,
    pub accepted: usize,
    /// Plain single-token forwards (no draft available).
    pub plain_steps: usize,
    /// Time spent drafting (drafter forwards, lookup).
    pub draft_time: std::time::Duration,
    /// Time spent in target verification forwards + commits.
    pub verify_time: std::time::Duration,
}

impl SpecStats {
    /// Fraction of draft tokens accepted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
    /// Mean tokens committed per verification round (accepted + 1),
    /// a.k.a. mean accepted length τ.
    pub fn mean_accepted_len(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.accepted + self.rounds) as f64 / self.rounds as f64
        }
    }
}

/// Draft-length controller. Fixed (`k` every round) unless adaptive, in
/// which case each round picks the length `n` in `0..=k` maximising the
/// expected throughput `E[committed tokens | n] / E[time | n]`:
///
/// - `E[committed | n] = 1 + Σ_{i<=n} Π_{j<=i} a_j`, with `a_j` the
///   observed acceptance rate at draft depth `j` (given depth `j - 1` was
///   accepted), exponentially decayed so it follows the text;
/// - `E[time | n]` = measured drafting time + measured verification time
///   of `n + 1` tokens (per batch size, linear model for sizes not seen
///   yet); `n = 0` is a plain decode step without drafting.
///
/// Rounds that pick `n = 0` periodically probe with `k` so the acceptance
/// estimates stay current.
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    max: usize,
    on: bool,
    /// Decayed counts per depth `1..=max`: reached, accepted.
    reached: Vec<f64>,
    accepted: Vec<f64>,
    /// EMA of the drafting time (ms) per round.
    draft_ms: Option<f64>,
    /// EMA of the target forward time (ms) per batch size `1..=max + 1`.
    fwd_ms: Vec<Option<f64>>,
    idle: usize,
}

const DECAY: f64 = 0.9;
const EMA: f64 = 0.2;
/// After this many rounds without drafting, probe with the full length.
const PROBE_EVERY: usize = 16;

fn ema(old: &mut Option<f64>, x: f64) {
    *old = Some(match *old {
        Some(o) => o + EMA * (x - o),
        None => x,
    });
}

impl AdaptiveK {
    pub fn new(max: usize, adaptive: bool) -> Self {
        let max = max.max(1);
        Self {
            max,
            on: adaptive,
            reached: vec![0.0; max + 1],
            accepted: vec![0.0; max + 1],
            draft_ms: None,
            fwd_ms: vec![None; max + 2],
            idle: 0,
        }
    }

    /// Acceptance estimate at depth `i` (prior 1/2).
    fn rate(&self, i: usize) -> f64 {
        (self.accepted[i] + 1.0) / (self.reached[i] + 2.0)
    }

    /// Estimated forward time for a batch of `n` tokens.
    fn fwd(&self, n: usize) -> Option<f64> {
        if let Some(t) = self.fwd_ms[n] {
            return Some(t);
        }
        // Linear fit through the sizes seen (or one point and a prior
        // marginal cost of 15% of a 1-token step per extra token).
        let pts: Vec<(f64, f64)> = (1..self.fwd_ms.len())
            .filter_map(|m| self.fwd_ms[m].map(|t| (m as f64, t)))
            .collect();
        match pts.len() {
            0 => None,
            1 => {
                let (m, t) = pts[0];
                let base = t / (1.0 + 0.15 * (m - 1.0));
                Some(base * (1.0 + 0.15 * (n as f64 - 1.0)))
            }
            _ => {
                let k = pts.len() as f64;
                let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
                let (mx, my) = (sx / k, sy / k);
                let (sxy, sxx) = pts.iter().fold((0.0, 0.0), |(a, b), p| {
                    (a + (p.0 - mx) * (p.1 - my), b + (p.0 - mx) * (p.0 - mx))
                });
                let slope = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { 0.0 };
                Some((my + slope * (n as f64 - mx)).max(0.0))
            }
        }
    }

    /// Draft length for the next round (`0` = decode without drafting).
    pub fn get(&mut self) -> usize {
        if !self.on {
            return self.max;
        }
        let (Some(d), Some(_)) = (self.draft_ms, self.fwd(1)) else {
            return self.max; // no timings yet
        };
        let mut best = (0usize, 1.0 / self.fwd(1).unwrap_or(1.0));
        let mut expect = 1.0;
        let mut p = 1.0;
        for n in 1..=self.max {
            p *= self.rate(n);
            expect += p;
            if let Some(t) = self.fwd(n + 1) {
                let tput = expect / (d + t);
                if tput > best.1 {
                    best = (n, tput);
                }
            }
        }
        if best.0 == 0 {
            self.idle += 1;
            if self.idle >= PROBE_EVERY {
                self.idle = 0;
                return self.max;
            }
        } else {
            self.idle = 0;
        }
        best.0
    }

    /// Record a round: `drafted` tokens of which the first `accepted` were
    /// accepted, `draft_ms` spent drafting (if a draft was attempted) and
    /// `fwd_ms` in the target forward of `drafted + 1` tokens.
    pub fn update(&mut self, drafted: usize, accepted: usize, draft_ms: Option<f64>, fwd_ms: f64) {
        if let Some(d) = draft_ms {
            ema(&mut self.draft_ms, d);
        }
        if drafted + 1 < self.fwd_ms.len() {
            ema(&mut self.fwd_ms[drafted + 1], fwd_ms);
        }
        if drafted == 0 {
            return;
        }
        for i in 1..=self.max {
            self.reached[i] *= DECAY;
            self.accepted[i] *= DECAY;
        }
        for i in 1..=drafted.min(accepted + 1).min(self.max) {
            self.reached[i] += 1.0;
            if i <= accepted {
                self.accepted[i] += 1.0;
            }
        }
    }
}

/// Rejection-sampling acceptance of draft token `d` with draft distribution
/// `q` (`None` = one-hot) against target distribution `p`: returns `None`
/// if accepted, else the replacement token drawn from `norm(max(p - q, 0))`.
/// `u` is a uniform draw in `[0, 1)`; `resample` draws from weights.
pub fn accept_or_resample(
    p: &[f32],
    d: u32,
    q: Option<&[f32]>,
    u: f32,
    resample: &mut dyn FnMut(&[f32]) -> anyhow::Result<u32>,
) -> anyhow::Result<Option<u32>> {
    let pd = p[d as usize];
    let qd = q.map_or(1.0, |q| q[d as usize]);
    // Accept with probability min(1, p/q). A token q could not have drawn
    // (qd == 0) is accepted iff p > 0, which never happens for a proper q.
    if qd <= 0.0 {
        if pd > 0.0 {
            return Ok(None);
        }
    } else if u < pd / qd {
        return Ok(None);
    }
    let resid: Vec<f32> = match q {
        Some(q) => p.iter().zip(q).map(|(&a, &b)| (a - b).max(0.0)).collect(),
        None => {
            let mut r = p.to_vec();
            r[d as usize] = 0.0;
            r
        }
    };
    if resid.iter().all(|&x| x <= 0.0) {
        // p == q up to rounding: the rejection branch has probability ~0.
        return resample(p).map(Some);
    }
    resample(&resid).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_k_follows_costs_and_acceptance() {
        // Fixed mode always drafts k.
        let mut f = AdaptiveK::new(4, false);
        f.update(4, 0, Some(10.0), 100.0);
        assert_eq!(f.get(), 4);

        // Verify of n+1 tokens costs 80 + 10 n ms, drafting 20 ms.
        let cost = |n: usize| 80.0 + 10.0 * n as f64;
        // Drafts always accepted: the longest draft wins.
        let mut a = AdaptiveK::new(4, true);
        assert_eq!(a.get(), 4, "no timings yet");
        for _ in 0..20 {
            let n = a.get();
            a.update(n, n, Some(20.0), cost(n));
        }
        assert_eq!(a.get(), 4);
        // Drafts never accepted: stop drafting, but probe periodically.
        let mut r = AdaptiveK::new(4, true);
        let mut zeros = 0;
        for _ in 0..64 {
            let n = r.get();
            zeros += (n == 0) as usize;
            r.update(n, 0, (n > 0).then_some(20.0), cost(n));
        }
        assert!(zeros > 48, "{zeros}");
        assert!(zeros < 64, "probes");
    }

    #[test]
    fn rejection_sampling_one_hot_and_dense() {
        let p = [0.5f32, 0.3, 0.2];
        let mut draw = |w: &[f32]| -> anyhow::Result<u32> {
            Ok(w.iter().position(|&x| x > 0.0).unwrap() as u32)
        };
        // One-hot q: accept with probability p(d).
        assert_eq!(
            accept_or_resample(&p, 1, None, 0.29, &mut draw).unwrap(),
            None
        );
        // Rejected: residual is p without d.
        assert_eq!(
            accept_or_resample(&p, 0, None, 0.6, &mut draw).unwrap(),
            Some(1)
        );
        // Dense q: accept iff u < p/q.
        let q = [0.25f32, 0.6, 0.15];
        assert_eq!(
            accept_or_resample(&p, 0, Some(&q), 0.99, &mut draw).unwrap(),
            None
        );
        // p/q = 0.5 for token 1; rejection draws from max(p - q, 0) = [0.25, 0, 0.05].
        assert_eq!(
            accept_or_resample(&p, 1, Some(&q), 0.7, &mut draw).unwrap(),
            Some(0)
        );
    }
}
