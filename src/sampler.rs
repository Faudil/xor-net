//! Token sampling: penalties, temperature, top-k / top-p / min-p truncation.
//!
//! The order of operations follows HF `transformers` / vLLM:
//!
//! 1. penalties (repetition: HF semantics; frequency / presence: OpenAI
//!    semantics), applied to the raw logits over the last `penalty_last_n`
//!    tokens of the context;
//! 2. temperature (`temperature <= 0` selects greedy argmax);
//! 3. top-k, then top-p (the nucleus is computed over the top-k survivors),
//!    then min-p (relative to the most likely token);
//! 4. one draw from the renormalised survivors.
//!
//! Every path works on reusable scratch buffers: after the first call on a
//! given vocabulary size, sampling performs no heap allocation. The greedy
//! path is a single argmax pass over the logits.

use crate::util::with_best_simd;
use std::hash::{BuildHasher, Hasher};

/// Sampling configuration. `Default` matches the OpenAI defaults
/// (temperature 1, no truncation, no penalties, seed from entropy).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// `<= 0` (or NaN) selects greedy decoding.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. `0` disables.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of most likely tokens whose
    /// cumulative probability is `>= top_p`. `>= 1` disables.
    pub top_p: f32,
    /// Drop tokens whose probability is below `min_p * p_max`. `<= 0` disables.
    pub min_p: f32,
    /// HF repetition penalty: positive logits of seen tokens are divided by
    /// it, negative ones multiplied. `1.0` disables.
    pub repetition_penalty: f32,
    /// OpenAI frequency penalty: `logit -= count * frequency_penalty`.
    pub frequency_penalty: f32,
    /// OpenAI presence penalty: `logit -= presence_penalty` if seen.
    pub presence_penalty: f32,
    /// Penalties look at the last `n` tokens of the context only.
    /// `None` = the whole context.
    pub penalty_last_n: Option<usize>,
    /// `None` = seed from entropy (see [`Sampler::seed`] for the value used).
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: None,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }
    pub fn with_repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = p;
        self
    }
    pub fn with_frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = p;
        self
    }
    pub fn with_presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = p;
        self
    }
    pub fn with_penalty_last_n(mut self, n: Option<usize>) -> Self {
        self.penalty_last_n = n;
        self
    }
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// True when sampling reduces to argmax.
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature <= 0.0
    }

    fn has_penalties(&self) -> bool {
        (self.repetition_penalty != 1.0 && self.repetition_penalty > 0.0)
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }
}

// ---------------------------------------------------------------------------
// PRNG: xoshiro256** seeded through splitmix64.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        // splitmix64 never yields four zeros in a row, so the state is valid.
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` from the top 24 bits.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

fn entropy_seed() -> u64 {
    // `RandomState` is keyed from OS randomness once per process and then
    // incremented per instance; mix in the clock for good measure.
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    h.write_u128(nanos);
    h.finish()
}

// ---------------------------------------------------------------------------
// Vectorisable kernels (branch-free so LLVM emits SIMD with target-cpu=native)
// ---------------------------------------------------------------------------

const LANES: usize = 16;

/// `exp(x)` for `x <= 0`, ~1e-7 relative error, branch-free. Inputs below
/// -87 (including -inf and NaN) return exactly 0.
#[inline(always)]
fn fast_exp_neg(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: round-to-nearest trick
    let keep = x >= -87.0; // false for NaN too
    let xc = x.clamp(-87.0, 0.0);
    let t = xc * LOG2E + MAGIC;
    // Low mantissa bits of `t` hold round(xc * log2e) as an integer.
    let ni = (t.to_bits() as i32).wrapping_sub(0x4B40_0000);
    let n = t - MAGIC;
    let r = xc - n * LN2_HI - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let scale = f32::from_bits((ni.wrapping_add(127) << 23) as u32);
    let y = p * scale;
    f32::from_bits(y.to_bits() & (keep as u32).wrapping_neg())
}

/// In place: `v[i] = exp(v[i] - max)`; returns the sum.
fn exp_inplace(v: &mut [f32], max: f32) -> f32 {
    with_best_simd!(exp_inplace_lanes(v, max))
}

/// [`exp_inplace`] without the ISA dispatch (arguments, not captures, so LLVM
/// knows the slices do not alias).
#[inline(always)]
fn exp_inplace_lanes(v: &mut [f32], max: f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut chunks = v.chunks_exact_mut(LANES);
    for c in &mut chunks {
        for (x, a) in c.iter_mut().zip(acc.iter_mut()) {
            let e = fast_exp_neg(*x - max);
            *x = e;
            *a += e;
        }
    }
    let mut sum = 0.0f32;
    for x in chunks.into_remainder() {
        let e = fast_exp_neg(*x - max);
        *x = e;
        sum += e;
    }
    sum + acc.iter().sum::<f32>()
}

#[inline(always)]
fn sanitize(x: f32) -> f32 {
    let nan = x.is_nan();
    let y = x.min(f32::MAX);
    if nan { f32::NEG_INFINITY } else { y }
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// In place: scale by `inv_temp`, map NaN to -inf and +inf to `f32::MAX`;
/// returns the maximum.
fn scale_sanitize_max(v: &mut [f32], inv_temp: f32) -> f32 {
    with_best_simd!(scale_sanitize_max_lanes(v, inv_temp))
}

/// [`scale_sanitize_max`] without the ISA dispatch (arguments, not captures, so LLVM
/// knows the slices do not alias).
#[inline(always)]
fn scale_sanitize_max_lanes(v: &mut [f32], inv_temp: f32) -> f32 {
    let mut m = [f32::NEG_INFINITY; LANES];
    let mut chunks = v.chunks_exact_mut(LANES);
    for c in &mut chunks {
        for (x, mx) in c.iter_mut().zip(m.iter_mut()) {
            let y = sanitize(*x * inv_temp);
            *x = y;
            *mx = fmax(*mx, y);
        }
    }
    let mut mx = f32::NEG_INFINITY;
    for x in chunks.into_remainder() {
        let y = sanitize(*x * inv_temp);
        *x = y;
        mx = fmax(mx, y);
    }
    m.iter().fold(mx, |a, &b| fmax(a, b))
}

/// Maximum ignoring NaN (`-inf` if none).
fn max_of(v: &[f32]) -> f32 {
    with_best_simd!(max_of_lanes(v))
}

/// [`max_of`] without the ISA dispatch, for callers that already run under
/// `with_best_simd!` (or call it on tiny slices).
#[inline(always)]
fn max_of_lanes(v: &[f32]) -> f32 {
    let mut m = [f32::NEG_INFINITY; LANES];
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (x, mx) in c.iter().zip(m.iter_mut()) {
            *mx = fmax(*mx, *x);
        }
    }
    let tail = chunks
        .remainder()
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| fmax(a, b));
    m.iter().fold(tail, |a, &b| fmax(a, b))
}

/// First index of the maximum, ignoring NaN. `None` if every entry is NaN
/// or -inf. Two passes (SIMD max, then a scan for its first position) beat
/// a branchy single pass by ~4x on 128k-entry vocabularies.
#[inline]
fn argmax(v: &[f32]) -> Option<usize> {
    let m = max_of(v);
    if m == f32::NEG_INFINITY {
        return None;
    }
    v.iter().position(|&x| x == m)
}

/// Sum of the entries `>= thr` (NaN never counts).
#[inline]
fn masked_sum(w: &[f32], thr: f32) -> f32 {
    with_best_simd!(masked_sum_lanes(w, thr))
}

/// [`masked_sum`] without the ISA dispatch (arguments, not captures, so LLVM
/// knows the slices do not alias).
#[inline(always)]
fn masked_sum_lanes(w: &[f32], thr: f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut chunks = w.chunks_exact(LANES);
    for c in &mut chunks {
        for (x, a) in c.iter().zip(acc.iter_mut()) {
            *a += if *x >= thr { *x } else { 0.0 };
        }
    }
    let tail: f32 = chunks.remainder().iter().filter(|&&x| x >= thr).sum();
    tail + acc.iter().sum::<f32>()
}

/// Inverse-CDF draw over the entries `>= thr` of a dense weight vector.
/// Block sums are computed with SIMD; only the block containing the draw is
/// scanned sequentially.
fn scan_dense(w: &[f32], thr: f32, r: f32) -> Option<usize> {
    const BLOCK: usize = 512;
    let mut acc = 0.0f32;
    for (bi, block) in w.chunks(BLOCK).enumerate() {
        let s = masked_sum(block, thr);
        if acc + s > r {
            let mut a = acc;
            for (j, &x) in block.iter().enumerate() {
                if x >= thr {
                    a += x;
                    if r < a {
                        return Some(bi * BLOCK + j);
                    }
                }
            }
        }
        acc += s;
    }
    // Rounding pushed `r` past the total: take the last eligible entry.
    w.iter().rposition(|&x| x >= thr && x > 0.0)
}

/// Sum and count of the entries `>= thr` (NaN never counts).
#[inline]
fn masked_sum_count(w: &[f32], thr: f32) -> (f32, usize) {
    with_best_simd!(masked_sum_count_lanes(w, thr))
}

/// [`masked_sum_count`] without the ISA dispatch (arguments, not captures, so LLVM
/// knows the slices do not alias).
#[inline(always)]
fn masked_sum_count_lanes(w: &[f32], thr: f32) -> (f32, usize) {
    let mut acc = [0.0f32; LANES];
    let mut cnt = [0u32; LANES];
    let mut chunks = w.chunks_exact(LANES);
    for c in &mut chunks {
        for ((x, a), n) in c.iter().zip(acc.iter_mut()).zip(cnt.iter_mut()) {
            let keep = *x >= thr;
            *a += if keep { *x } else { 0.0 };
            *n += keep as u32;
        }
    }
    let mut s = acc.iter().sum::<f32>();
    let mut n = cnt.iter().map(|&c| c as usize).sum::<usize>();
    for &x in chunks.remainder() {
        if x >= thr {
            s += x;
            n += 1;
        }
    }
    (s, n)
}

/// Exact top-p threshold over a dense weight vector (all weights in
/// `[0, 1]`): the largest `T` such that `sum(w >= T) >= target`. The
/// nucleus is then `{ w >= T }`, i.e. the smallest prefix of the sorted
/// weights reaching `target` (exact ties at the boundary are all kept).
///
/// Bisection over the bit patterns of positive floats (monotonic in value)
/// with one SIMD masked pass per step, until at most `BAND` weights lie in
/// the undecided band `[lo, hi)`; those are then gathered (skipping
/// 16-wide blocks with no band member) and walked in sorted order.
fn top_p_threshold(w: &[f32], target: f32, lo_hint: f32, band: &mut Vec<f32>) -> f32 {
    const BAND: usize = 512;
    // Invariant: S(lo) >= target > S(hi), with S(t) = sum(w >= t).
    let mut lo = lo_hint.max(0.0).to_bits();
    let (s_lo, mut n_lo) = masked_sum_count(w, f32::from_bits(lo));
    if s_lo < target {
        lo = 0;
        n_lo = masked_sum_count(w, 0.0).1;
    }
    let mut hi = 1.0f32.to_bits() + 1;
    let (mut s_hi, mut n_hi) = (0.0f32, 0usize);
    while n_lo - n_hi > BAND {
        if hi - lo <= 1 {
            return f32::from_bits(lo);
        }
        let mid = lo + (hi - lo) / 2;
        let (s, n) = masked_sum_count(w, f32::from_bits(mid));
        if s >= target {
            (lo, n_lo) = (mid, n);
        } else {
            (hi, s_hi, n_hi) = (mid, s, n);
        }
    }
    let (flo, fhi) = (f32::from_bits(lo), f32::from_bits(hi));
    band.clear();
    for block in w.chunks(LANES) {
        let mut any = false;
        for &x in block {
            any |= x >= flo && x < fhi;
        }
        if any {
            band.extend(block.iter().copied().filter(|&x| x >= flo && x < fhi));
        }
    }
    band.sort_unstable_by(|a, b| b.total_cmp(a));
    let mut acc = s_hi;
    for &x in band.iter() {
        acc += x;
        if acc >= target {
            return x;
        }
    }
    // Rounding: the band's sum fell just short of target; keep all of it.
    flo
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Stateful sampler: owns the PRNG and reusable scratch buffers.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    seed: u64,
    rng: Xoshiro256,
    /// Candidate token indices (top-k / nucleus work list).
    idx: Vec<u32>,
    /// `(scaled logit, token)` min-sorted top-k buffer.
    topk: Vec<(f32, u32)>,
    /// Penalty bookkeeping: generation stamps + counts, sized to vocab.
    stamp: Vec<u32>,
    count: Vec<u32>,
    stamp_gen: u32,
    touched: Vec<u32>,
    /// Top-p threshold refinement scratch.
    band: Vec<f32>,
}

impl Sampler {
    /// Compatibility constructor. `seed == 0` seeds from entropy (legacy
    /// behaviour); `temperature` `None`/`<= 0` is greedy; `top_p` `None`
    /// disables nucleus sampling.
    pub fn new(
        seed: u64,
        temperature: Option<f32>,
        top_p: Option<f32>,
        repetition_penalty: f32,
    ) -> Self {
        Self::from_params(SamplingParams {
            temperature: temperature.unwrap_or(0.0),
            top_p: top_p.unwrap_or(1.0),
            repetition_penalty,
            seed: if seed == 0 { None } else { Some(seed) },
            ..SamplingParams::default()
        })
    }

    pub fn from_params(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or_else(entropy_seed);
        Self {
            params,
            seed,
            rng: Xoshiro256::new(seed),
            idx: Vec::new(),
            topk: Vec::new(),
            stamp: Vec::new(),
            count: Vec::new(),
            stamp_gen: 0,
            touched: Vec::new(),
            band: Vec::new(),
        }
    }

    /// The seed actually in use (useful to reproduce an entropy-seeded run).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Replace the parameters, keeping scratch buffers. Re-seeds the PRNG.
    pub fn set_params(&mut self, params: SamplingParams) {
        self.seed = params.seed.unwrap_or_else(entropy_seed);
        self.rng = Xoshiro256::new(self.seed);
        self.params = params;
    }

    /// Sample one token. `logits` is consumed as scratch (modified in place).
    /// All penalties use the last `penalty_last_n` tokens of `context`.
    pub fn sample(&mut self, logits: &mut [f32], context: &[u32]) -> anyhow::Result<u32> {
        self.sample_split(logits, &[], context)
    }

    /// Like [`sample`](Self::sample) but with vLLM penalty semantics: the
    /// repetition penalty sees `prompt ++ output`, frequency / presence
    /// penalties count `output` tokens only. The `penalty_last_n` window
    /// is taken from the tail of `prompt ++ output`.
    pub fn sample_split(
        &mut self,
        logits: &mut [f32],
        prompt: &[u32],
        output: &[u32],
    ) -> anyhow::Result<u32> {
        if logits.is_empty() {
            anyhow::bail!("sampler: empty logits");
        }
        if self.params.has_penalties() {
            self.apply_penalties(logits, prompt, output);
        }
        if self.params.is_greedy() {
            return argmax(logits)
                .map(|i| i as u32)
                .ok_or_else(|| anyhow::anyhow!("sampler: all logits are NaN or -inf"));
        }
        self.sample_stochastic(logits)
    }

    fn apply_penalties(&mut self, logits: &mut [f32], prompt: &[u32], output: &[u32]) {
        let vocab = logits.len();
        if self.stamp.len() < vocab {
            self.stamp.resize(vocab, 0);
            self.count.resize(vocab, 0);
        }
        self.stamp_gen = self.stamp_gen.wrapping_add(1);
        if self.stamp_gen == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.stamp_gen = 1;
        }
        let g = self.stamp_gen;
        let total = prompt.len() + output.len();
        let window = self.params.penalty_last_n.unwrap_or(total).min(total);
        let start = total - window;
        let p_part = &prompt[start.min(prompt.len())..];
        let o_part = &output[start.saturating_sub(prompt.len())..];
        self.touched.clear();
        for (part, counted) in [(p_part, 0u32), (o_part, 1u32)] {
            for &t in part {
                let t = t as usize;
                if t >= vocab {
                    continue;
                }
                if self.stamp[t] != g {
                    self.stamp[t] = g;
                    self.count[t] = 0;
                    self.touched.push(t as u32);
                }
                self.count[t] += counted;
            }
        }
        let rp = self.params.repetition_penalty;
        let use_rp = rp != 1.0 && rp > 0.0;
        let (fp, pp) = (self.params.frequency_penalty, self.params.presence_penalty);
        for &t in &self.touched {
            let t = t as usize;
            let mut v = logits[t];
            if use_rp {
                v = if v > 0.0 { v / rp } else { v * rp };
            }
            let c = self.count[t];
            if c > 0 {
                v -= c as f32 * fp + pp;
            }
            logits[t] = v;
        }
    }

    fn sample_stochastic(&mut self, logits: &mut [f32]) -> anyhow::Result<u32> {
        let vocab = logits.len();
        let p = &self.params;
        let inv_temp = 1.0 / p.temperature;
        let top_k = if p.top_k >= vocab { 0 } else { p.top_k };
        let top_p = if p.top_p.is_nan() { 1.0 } else { p.top_p };
        let min_p = if p.min_p.is_nan() {
            0.0
        } else {
            p.min_p.min(1.0)
        };

        let max = scale_sanitize_max(logits, inv_temp);
        if max == f32::NEG_INFINITY {
            anyhow::bail!("sampler: all logits are NaN or -inf");
        }

        if top_k > 0 {
            // Small explicit candidate list, sorted by probability.
            self.select_top_k(logits, top_k);
            self.topk.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let mut z = 0.0f32;
            for e in self.topk.iter_mut() {
                e.0 = fast_exp_neg(e.0 - max);
                z += e.0;
            }
            let mut n = self.topk.len();
            if top_p < 1.0 {
                let target = (top_p.max(0.0) * z).max(f32::MIN_POSITIVE);
                let mut acc = 0.0f32;
                for (i, e) in self.topk.iter().enumerate() {
                    acc += e.0;
                    if acc >= target {
                        n = i + 1;
                        break;
                    }
                }
            }
            if min_p > 0.0 {
                // The head has weight exactly 1 (= p_max), so it always stays.
                while n > 1 && self.topk[n - 1].0 < min_p {
                    n -= 1;
                }
            }
            let kept = &self.topk[..n];
            let mass: f32 = kept.iter().map(|e| e.0).sum();
            let r = self.rng.next_f32() * mass;
            let mut acc = 0.0f32;
            for e in kept {
                acc += e.0;
                if r < acc {
                    return Ok(e.1);
                }
            }
            return Ok(kept[kept.len() - 1].1);
        }

        // Dense path: weights in place, truncation expressed as a threshold.
        let z = exp_inplace(logits, max);
        let mut thr = 0.0f32;
        if top_p < 1.0 {
            let target = (top_p.max(0.0) * z).max(f32::MIN_POSITIVE);
            // Tokens below eps together carry < (1 - top_p) * z, so the
            // threshold is >= eps; start the search there.
            let eps = (1.0 - top_p) * z / vocab as f32;
            thr = top_p_threshold(logits, target, eps, &mut self.band);
        }
        if min_p > 0.0 {
            thr = thr.max(min_p);
        }
        let mass = if thr > 0.0 {
            masked_sum(logits, thr)
        } else {
            z
        };
        let r = self.rng.next_f32() * mass;
        scan_dense(logits, thr, r)
            .map(|i| i as u32)
            .ok_or_else(|| anyhow::anyhow!("sampler: no candidate survived truncation"))
    }

    /// Fill `self.topk` with the `k` largest `(logit, token)` pairs.
    fn select_top_k(&mut self, logits: &[f32], k: usize) {
        self.topk.clear();
        if k <= 256 {
            // Min-sorted bounded buffer. Blocks whose SIMD max cannot beat
            // the current k-th value are skipped wholesale.
            const BLOCK: usize = 64;
            let topk = &mut self.topk;
            with_best_simd!({
                for (bi, block) in logits.chunks(BLOCK).enumerate() {
                    if topk.len() == k && max_of_lanes(block) <= topk[0].0 {
                        continue;
                    }
                    for (j, &v) in block.iter().enumerate() {
                        let tok = (bi * BLOCK + j) as u32;
                        if topk.len() < k {
                            let pos = topk.partition_point(|&(x, _)| x < v);
                            topk.insert(pos, (v, tok));
                        } else if v > topk[0].0 {
                            let pos = topk.partition_point(|&(x, _)| x < v);
                            // Drop the minimum, shift the smaller entries down.
                            topk.copy_within(1..pos, 0);
                            topk[pos - 1] = (v, tok);
                        }
                    }
                }
            })
        } else {
            self.idx.clear();
            self.idx.extend(0..logits.len() as u32);
            self.idx.select_nth_unstable_by(k - 1, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            self.topk
                .extend(self.idx[..k].iter().map(|&i| (logits[i as usize], i)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_exp_accuracy() {
        let mut x = -90.0f32;
        while x <= 0.0 {
            let a = fast_exp_neg(x);
            let b = x.exp();
            if x >= -87.0 {
                assert!((a - b).abs() <= 2e-6 * b + 1e-37, "x={x} a={a} b={b}");
            } else {
                assert_eq!(a, 0.0);
            }
            x += 0.0137;
        }
        assert_eq!(fast_exp_neg(f32::NEG_INFINITY), 0.0);
        assert_eq!(fast_exp_neg(f32::NAN), 0.0);
        assert_eq!(fast_exp_neg(0.0), 1.0);
    }

    #[test]
    fn top_p_threshold_matches_sorted_prefix() {
        let mut rng = Xoshiro256::new(7);
        for trial in 0..300 {
            let n = 1 + (rng.next_u64() % 3000) as usize;
            let w: Vec<f32> = (0..n)
                .map(|_| {
                    let u = rng.next_f32();
                    // Integer-ish weights in (0, 1] give lots of ties.
                    if trial % 2 == 0 {
                        ((u * 8.0).floor() + 1.0) / 8.0
                    } else {
                        u * u
                    }
                })
                .collect();
            let total = masked_sum(&w, 0.0);
            let target = (rng.next_f32() * total).max(f32::MIN_POSITIVE);
            let thr = top_p_threshold(&w, target, 0.0, &mut Vec::new());
            let kept = masked_sum(&w, thr);
            assert!(kept >= target * (1.0 - 1e-5), "trial {trial}");
            // Minimality: dropping the boundary value class falls short.
            let above: f32 = w.iter().filter(|&&x| x > thr).sum();
            assert!(above < target * (1.0 + 1e-5), "trial {trial}: not minimal");
        }
    }

    #[test]
    fn argmax_ignores_nan_and_picks_first() {
        assert_eq!(argmax(&[1.0, f32::NAN, 3.0, 3.0]), Some(2));
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY]), None);
        let mut v = vec![0.0f32; 1000];
        v[777] = 1.0;
        assert_eq!(argmax(&v), Some(777));
    }

    #[test]
    fn scan_dense_respects_threshold() {
        let w = vec![0.1f32, 0.5, 0.2, 1.0, 0.05];
        for i in 0..100 {
            let r = i as f32 / 100.0 * 1.5;
            let t = scan_dense(&w, 0.5, r).unwrap();
            assert!(t == 1 || t == 3);
        }
    }
}
