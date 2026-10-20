//! Fast CPU implementations of the Bonsai (PrismML 1-bit `Q1_0`) models.
//!
//! - [`qwen3`]: dense Qwen3 (Bonsai 1.7B / 4B / 8B), GGUF arch `qwen3`.
//! - [`qwen35`]: hybrid Qwen3.5 (Bonsai 27B), GGUF arch `qwen35`: gated full
//!   attention every N-th layer, Gated DeltaNet linear attention otherwise.
//!
//! This module holds the shared building blocks: [`Linear`] (1-bit GEMV/GEMM
//! with a dense fallback), [`Embedding`] (dequantizes only looked-up rows),
//! RoPE tables (incl. YaRN), the growable [`KvCache`] and causal [`attention`].
//! All activations are plain row-major `f32` (`[tokens, features]`).

pub mod batch;
pub mod dspark;
pub mod flash;
pub mod format;
pub mod qwen3;
pub mod qwen35;

use std::path::Path;
use std::sync::Arc;

use half::f16;
use rayon::prelude::*;

use crate::bit1::q1_0::{Q1Weight, Q8Acts, gemm};
use crate::loader::gguf::{GgufFile, dequantize_data, is_prism_dtype};
use crate::models::traits::CausalLM;
use crate::nn::fast_attention_simd::{
    dot_product_f32, f16_to_f32_slice, f32_to_f16_slice, weighted_sum,
};

// ---------------------------------------------------------------------------
// Linear layers
// ---------------------------------------------------------------------------

/// A `[rows, cols]` projection `y = W x`.
#[derive(Clone, Debug)]
pub enum Linear {
    /// 1-bit weights (the Bonsai case).
    Q1(Q1Weight),
    /// Dense f32 fallback for tensors stored unquantized (row-major).
    F32 {
        rows: usize,
        cols: usize,
        w: Arc<Vec<f32>>,
    },
}

impl Linear {
    pub fn rows(&self) -> usize {
        match self {
            Linear::Q1(w) => w.rows(),
            Linear::F32 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Linear::Q1(w) => w.cols(),
            Linear::F32 { cols, .. } => *cols,
        }
    }

    /// Weight bytes streamed per token.
    pub fn nbytes(&self) -> usize {
        match self {
            Linear::Q1(w) => w.nbytes(),
            Linear::F32 { w, .. } => w.len() * 4,
        }
    }

    /// `out[t] = W x[t]` for `n` rows. `acts` is scratch for the int8
    /// quantized activations.
    pub fn forward(&self, x: &[f32], n: usize, acts: &mut Q8Acts, out: &mut [f32]) {
        assert_eq!(x.len(), n * self.cols(), "Linear: input length");
        assert_eq!(out.len(), n * self.rows(), "Linear: output length");
        match self {
            Linear::Q1(w) => {
                acts.quantize(x, n, w.cols(), act_block(w, n));
                gemm(w, acts, out);
            }
            Linear::F32 { rows, cols, w } => {
                let (rows, cols) = (*rows, *cols);
                out.par_chunks_mut(rows).enumerate().for_each(|(t, o)| {
                    let xr = &x[t * cols..(t + 1) * cols];
                    for (r, v) in o.iter_mut().enumerate() {
                        *v = dot_product_f32(&w[r * cols..(r + 1) * cols], xr);
                    }
                });
            }
        }
    }

    /// Load a 2-D GGUF tensor (`[rows, cols]` in our row-major convention).
    pub fn from_gguf(g: &GgufFile, name: &str) -> anyhow::Result<Self> {
        Self::from_gguf_concat(g, &[name])
    }

    /// Load several 2-D tensors with equal `cols` and stack their rows (one
    /// fused GEMV for Q/K/V or gate/up).
    pub fn from_gguf_concat(g: &GgufFile, names: &[&str]) -> anyhow::Result<Self> {
        let infos = names
            .iter()
            .map(|n| g.tensor_info(n))
            .collect::<anyhow::Result<Vec<_>>>()?;
        for i in &infos {
            anyhow::ensure!(i.shape.len() == 2, "{}: expected a 2-D tensor", i.name);
        }
        let cols = infos[0].shape[1];
        anyhow::ensure!(
            infos.iter().all(|i| i.shape[1] == cols),
            "concat {names:?}: cols differ"
        );
        if infos.iter().all(|i| is_prism_dtype(i.dtype_u32)) {
            let parts = infos
                .iter()
                .map(|i| {
                    Q1Weight::from_gguf(g.tensor_bytes(&i.name)?, i.dtype_u32, i.shape[0], cols)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            for n in names {
                g.release_tensor(n);
            }
            let w = if parts.len() == 1 {
                parts.into_iter().next().unwrap()
            } else {
                Q1Weight::concat_rows(&parts.iter().collect::<Vec<_>>())?
            };
            return Ok(Linear::Q1(w));
        }
        let mut w = Vec::new();
        let mut rows = 0;
        for i in &infos {
            w.extend(dequantize_data(
                g.tensor_bytes(&i.name)?,
                i.dtype_u32,
                i.elem_count,
            )?);
            rows += i.shape[0];
        }
        Ok(Linear::F32 {
            rows,
            cols,
            w: Arc::new(w),
        })
    }
}

/// Activation quantization block for `n` rows against `w`: 32 elements
/// (ggml `Q8_0` granularity) by default.
fn act_block(_w: &Q1Weight, _n: usize) -> usize {
    32
}

/// A 1-D f32 tensor from GGUF (norm weights, biases...), checked length.
pub fn gguf_vec(g: &GgufFile, name: &str, len: usize) -> anyhow::Result<Vec<f32>> {
    let info = g.tensor_info(name)?;
    anyhow::ensure!(
        info.elem_count == len,
        "{name}: {} elements, expected {len}",
        info.elem_count
    );
    dequantize_data(g.tensor_bytes(name)?, info.dtype_u32, info.elem_count)
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding table. Kept in its storage format; only the rows that are
/// looked up are dequantized.
#[derive(Clone, Debug)]
pub struct Embedding(pub Linear);

impl Embedding {
    pub fn dim(&self) -> usize {
        self.0.cols()
    }

    /// `out[t] = E[tokens[t]]`.
    pub fn lookup(&self, tokens: &[u32], out: &mut [f32]) -> anyhow::Result<()> {
        let dim = self.dim();
        anyhow::ensure!(out.len() == tokens.len() * dim, "embedding: output length");
        for (&t, o) in tokens.iter().zip(out.chunks_exact_mut(dim)) {
            let t = t as usize;
            anyhow::ensure!(t < self.0.rows(), "token id {t} out of range");
            match &self.0 {
                Linear::Q1(w) => w.dequantize_row_into(t, o),
                Linear::F32 { w, cols, .. } => o.copy_from_slice(&w[t * cols..(t + 1) * cols]),
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// YaRN context extension parameters (`rope.scaling.type == "yarn"`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Yarn {
    pub factor: f32,
    pub original_context: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// Rotary embedding (NEOX "rotate half" layout) on the first `rot_dim`
/// dimensions of each head.
#[derive(Clone, Debug)]
pub struct Rope {
    pub rot_dim: usize,
    inv_freq: Vec<f32>,
    /// Multiplies cos and sin (YaRN attention temperature), 1.0 otherwise.
    mscale: f32,
}

impl Rope {
    /// `inv_freq[i] = theta^(-2i/rot_dim)`, with the YaRN blend of
    /// interpolated / extrapolated frequencies when `yarn` is set (HF
    /// `_compute_yarn_parameters` / ggml `rope_yarn`).
    pub fn new(rot_dim: usize, theta: f32, yarn: Option<&Yarn>) -> Self {
        let half = rot_dim / 2;
        let base = theta as f64;
        let extrap: Vec<f64> = (0..half)
            .map(|i| 1.0 / base.powf((2 * i) as f64 / rot_dim as f64))
            .collect();
        let Some(y) = yarn.filter(|y| y.factor > 1.0) else {
            return Self {
                rot_dim,
                inv_freq: extrap.iter().map(|&f| f as f32).collect(),
                mscale: 1.0,
            };
        };
        let factor = y.factor as f64;
        let corr_dim = |rot: f64| {
            rot_dim as f64 * (y.original_context as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corr_dim(y.beta_fast as f64).floor().max(0.0);
        let high = corr_dim(y.beta_slow as f64)
            .ceil()
            .min(rot_dim as f64 - 1.0);
        let inv_freq = (0..half)
            .map(|i| {
                let ramp = ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                let extrap_w = 1.0 - ramp;
                let f = extrap[i] / factor * (1.0 - extrap_w) + extrap[i] * extrap_w;
                f as f32
            })
            .collect();
        Self {
            rot_dim,
            inv_freq,
            mscale: (0.1 * factor.ln() + 1.0) as f32,
        }
    }

    /// cos/sin tables for positions `pos .. pos + n`: `[n][rot_dim/2]` each.
    pub fn tables(&self, pos: usize, n: usize, cos: &mut Vec<f32>, sin: &mut Vec<f32>) {
        let half = self.rot_dim / 2;
        cos.resize(n * half, 0.0);
        sin.resize(n * half, 0.0);
        for t in 0..n {
            let p = (pos + t) as f64;
            for i in 0..half {
                let a = p * self.inv_freq[i] as f64;
                cos[t * half + i] = a.cos() as f32 * self.mscale;
                sin[t * half + i] = a.sin() as f32 * self.mscale;
            }
        }
    }

    /// Rotate one head vector in place with the tables of one position.
    #[inline]
    pub fn apply(&self, x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = self.rot_dim / 2;
        let (lo, hi) = x[..self.rot_dim].split_at_mut(half);
        for i in 0..half {
            let (a, b) = (lo[i], hi[i]);
            lo[i] = a * cos[i] - b * sin[i];
            hi[i] = a * sin[i] + b * cos[i];
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache + attention
// ---------------------------------------------------------------------------

/// Per-layer K/V storage `[kv_heads][capacity][head_dim]` in f16 (halves
/// the bytes attention streams per token; the same precision as llama.cpp's
/// default KV cache). Grown by doubling up to `max_len`, so a short
/// conversation does not pay for the full context window.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub kv_heads: usize,
    pub head_dim: usize,
    pub cap: usize,
    pub max_len: usize,
    k: Vec<f16>,
    v: Vec<f16>,
}

pub const KV_INITIAL_CAPACITY: usize = 256;

impl KvCache {
    pub fn new(kv_heads: usize, head_dim: usize, max_len: usize) -> Self {
        Self {
            kv_heads,
            head_dim,
            cap: 0,
            max_len,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Make room for positions `< len` (keeps positions `< keep`).
    pub fn reserve(&mut self, len: usize, keep: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            len <= self.max_len,
            "KV cache: {len} > max {}",
            self.max_len
        );
        if len <= self.cap {
            return Ok(());
        }
        let new_cap = len
            .max(self.cap * 2)
            .max(KV_INITIAL_CAPACITY)
            .min(self.max_len);
        let hd = self.head_dim;
        let grow = |old: &[f16], old_cap: usize| {
            let mut new = vec![f16::ZERO; self.kv_heads * new_cap * hd];
            let keep = keep.min(old_cap);
            for h in 0..self.kv_heads {
                new[h * new_cap * hd..(h * new_cap + keep) * hd]
                    .copy_from_slice(&old[h * old_cap * hd..(h * old_cap + keep) * hd]);
            }
            new
        };
        self.k = grow(&self.k, self.cap);
        self.v = grow(&self.v, self.cap);
        self.cap = new_cap;
        Ok(())
    }

    /// Store head `h` K/V vectors for position `p` (`p < cap`).
    #[inline]
    pub fn store(&mut self, h: usize, p: usize, k: &[f32], v: &[f32]) {
        let hd = self.head_dim;
        assert!(h < self.kv_heads && p < self.cap, "KV store out of range");
        let o = (h * self.cap + p) * hd;
        f32_to_f16_slice(k, &mut self.k[o..o + hd]);
        f32_to_f16_slice(v, &mut self.v[o..o + hd]);
    }

    /// K vectors of head `h` for positions `p0..p1`, contiguous.
    #[inline]
    fn k_rows(&self, h: usize, p0: usize, p1: usize) -> &[f16] {
        assert!(p0 <= p1 && p1 <= self.cap, "KV rows out of range");
        let base = h * self.cap;
        &self.k[(base + p0) * self.head_dim..(base + p1) * self.head_dim]
    }

    /// V vectors of head `h` for positions `p0..p1`, contiguous.
    #[inline]
    fn v_rows(&self, h: usize, p0: usize, p1: usize) -> &[f16] {
        assert!(p0 <= p1 && p1 <= self.cap, "KV rows out of range");
        let base = h * self.cap;
        &self.v[(base + p0) * self.head_dim..(base + p1) * self.head_dim]
    }

    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 2
    }

    /// Compact copy of positions `< len` (capacity exactly `len`), for state
    /// snapshots. `None` if `len` exceeds the capacity.
    pub fn prefix_copy(&self, len: usize) -> Option<KvCache> {
        if len > self.cap {
            return None;
        }
        let hd = self.head_dim;
        let take = |src: &[f16]| {
            let mut out = Vec::with_capacity(self.kv_heads * len * hd);
            for h in 0..self.kv_heads {
                let o = h * self.cap * hd;
                out.extend_from_slice(&src[o..o + len * hd]);
            }
            out
        };
        Some(KvCache {
            kv_heads: self.kv_heads,
            head_dim: hd,
            cap: len,
            max_len: self.max_len,
            k: take(&self.k),
            v: take(&self.v),
        })
    }

    /// Whether [`load_prefix`](Self::load_prefix) of `src` can succeed.
    pub fn can_load_prefix(&self, src: &KvCache) -> bool {
        src.kv_heads == self.kv_heads && src.head_dim == self.head_dim && src.cap <= self.max_len
    }

    /// Overwrite positions `< src.cap` with a [`prefix_copy`](Self::prefix_copy)
    /// (growing as needed); later positions are left as they are.
    pub fn load_prefix(&mut self, src: &KvCache) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.can_load_prefix(src),
            "KV cache: snapshot shape mismatch"
        );
        let len = src.cap;
        self.reserve(len, 0)?;
        let hd = self.head_dim;
        for h in 0..self.kv_heads {
            let (d, s) = (h * self.cap * hd, h * len * hd);
            self.k[d..d + len * hd].copy_from_slice(&src.k[s..s + len * hd]);
            self.v[d..d + len * hd].copy_from_slice(&src.v[s..s + len * hd]);
        }
        Ok(())
    }
}

/// Keys per split-K work item of [`attention`].
pub const ATTN_CHUNK: usize = 256;
/// Query rows per work item of [`attention`] (prefill): each K/V chunk is
/// converted from f16 once per block of queries.
pub const ATTN_QBLOCK: usize = 16;

/// Partial softmax state of one (query row, KV head, key chunk).
#[derive(Clone)]
struct AttnPartial {
    /// Per query head of the group: running max and sum of exp.
    m: Vec<f32>,
    l: Vec<f32>,
    /// `[group][hd]` unnormalized output.
    acc: Vec<f32>,
}

/// Causal GQA attention for `n` new query rows at positions `pos..pos+n`
/// whose K/V are already stored in `cache`.
///
/// `q`: `[n][heads * head_dim]`; `out`: `[n][heads * head_dim]`. Query head
/// `h` reads KV head `h / (heads / kv_heads)`. `scale` multiplies the
/// logits.
///
/// Head dims that [`flash::supported`] accepts (multiples of 64: all Bonsai
/// models) use the blocked, batch-invariant [`flash::attention`]: a row's
/// result is bit-identical whether it is computed alone (decode) or with
/// other rows (prefill). Other head dims use the split-K path below: items
/// of (block of up to [`ATTN_QBLOCK`] query rows, KV head, chunk of
/// [`ATTN_CHUNK`] keys) merged with log-sum-exp rescaling
/// (flash-decoding); it computes each row independently as well.
pub fn attention(
    q: &[f32],
    n: usize,
    pos: usize,
    heads: usize,
    cache: &KvCache,
    scale: f32,
    out: &mut [f32],
) {
    let hd = cache.head_dim;
    let nkv = cache.kv_heads;
    assert!(
        nkv > 0 && heads.is_multiple_of(nkv),
        "attention: heads not a multiple of kv heads"
    );
    let group = heads / nkv;
    assert_eq!(q.len(), n * heads * hd);
    assert_eq!(out.len(), n * heads * hd);
    assert!(pos + n <= cache.cap, "attention: positions beyond cache");
    if flash::supported(hd) {
        return flash::attention(q, n, pos, heads, cache, scale, out);
    }
    let max_chunks = (pos + n).div_ceil(ATTN_CHUNK);

    // Items: (query block, kv head, chunk) for every chunk visible to the
    // block's last row.
    let items: Vec<(usize, usize, usize)> = (0..n.div_ceil(ATTN_QBLOCK))
        .flat_map(|b| {
            let t_last = ((b + 1) * ATTN_QBLOCK).min(n) - 1;
            let chunks = (pos + t_last + 1).div_ceil(ATTN_CHUNK);
            (0..nkv).flat_map(move |h| (0..chunks).map(move |c| (b, h, c)))
        })
        .collect();
    // Partial for (t, kvh, c) lives at ((t * nkv) + kvh) * max_chunks + c.
    let results: Vec<Vec<(usize, AttnPartial)>> = items
        .par_iter()
        .map(|&(b, kvh, c)| {
            let (t0, t1) = (b * ATTN_QBLOCK, ((b + 1) * ATTN_QBLOCK).min(n));
            let p0 = c * ATTN_CHUNK;
            let p1 = ((c + 1) * ATTN_CHUNK).min(pos + t1);
            let cl = p1 - p0;
            let mut kf = vec![0.0f32; cl * hd];
            let mut vf = vec![0.0f32; cl * hd];
            // Positions p0..p1 of one head are contiguous in the cache.
            f16_to_f32_slice(cache.k_rows(kvh, p0, p1), &mut kf);
            f16_to_f32_slice(cache.v_rows(kvh, p0, p1), &mut vf);
            let mut res = Vec::with_capacity(t1 - t0);
            let mut sc = vec![0.0f32; cl];
            for t in t0..t1 {
                // Keys visible to row t: p <= pos + t.
                let vis = (pos + t + 1).min(p1).saturating_sub(p0);
                if vis == 0 {
                    continue;
                }
                let mut part = AttnPartial {
                    m: vec![0.0; group],
                    l: vec![0.0; group],
                    acc: vec![0.0; group * hd],
                };
                for g in 0..group {
                    let qv = &q[(t * heads + kvh * group + g) * hd..][..hd];
                    for (j, s) in sc[..vis].iter_mut().enumerate() {
                        *s = dot_product_f32(qv, &kf[j * hd..(j + 1) * hd]) * scale;
                    }
                    let m = sc[..vis].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut l = 0.0f32;
                    for s in sc[..vis].iter_mut() {
                        *s = (*s - m).exp();
                        l += *s;
                    }
                    // acc = sum_j e_j * V_j (SIMD-dispatched).
                    weighted_sum(
                        &mut part.acc[g * hd..(g + 1) * hd],
                        &sc,
                        &vf,
                        0,
                        cl,
                        hd,
                        vis,
                    );
                    part.m[g] = m;
                    part.l[g] = l;
                }
                res.push((((t * nkv) + kvh) * max_chunks + c, part));
            }
            res
        })
        .collect();
    let mut slots: Vec<Option<AttnPartial>> = vec![None; n * nkv * max_chunks];
    for (idx, p) in results.into_iter().flatten() {
        slots[idx] = Some(p);
    }

    // Merge the chunks of each (t, kvh).
    out.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(t, orow)| {
            for kvh in 0..nkv {
                let parts: Vec<&AttnPartial> = slots[(t * nkv + kvh) * max_chunks..][..max_chunks]
                    .iter()
                    .flatten()
                    .collect();
                for g in 0..group {
                    let m = parts
                        .iter()
                        .map(|p| p.m[g])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let o = &mut orow[(kvh * group + g) * hd..(kvh * group + g + 1) * hd];
                    o.fill(0.0);
                    let mut l = 0.0f32;
                    for p in &parts {
                        let w = (p.m[g] - m).exp();
                        l += w * p.l[g];
                        for (a, &b) in o.iter_mut().zip(&p.acc[g * hd..(g + 1) * hd]) {
                            *a += w * b;
                        }
                    }
                    let inv = 1.0 / l;
                    o.iter_mut().for_each(|a| *a *= inv);
                }
            }
        });
}

/// In-place RMS norm of each `dim`-sized row of `x`.
pub fn rms_norm_rows(x: &mut [f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    let f = |row: &mut [f32]| crate::nn::fast_attention_simd::rms_norm(row, weight, eps);
    if x.len() >= 8 * dim {
        x.par_chunks_exact_mut(dim).for_each(f);
    } else {
        x.chunks_exact_mut(dim).for_each(f);
    }
}

/// `act[t] = silu(gate[t]) * up[t]` for `gu` rows laid out `[gate; up]`
/// (`2 * inter` each). Overwrites the gate halves of `gu`.
pub fn swiglu_rows(gu: &mut [f32], act: &mut [f32], inter: usize) {
    let f = |(g2, a): (&mut [f32], &mut [f32])| {
        let (g, u) = g2.split_at_mut(inter);
        crate::bit1_58::quantization::silu_inplace(g, u, true);
        a.copy_from_slice(g);
    };
    if act.len() >= 8 * inter {
        gu.par_chunks_exact_mut(2 * inter)
            .zip(act.par_chunks_exact_mut(inter))
            .for_each(f);
    } else {
        gu.chunks_exact_mut(2 * inter)
            .zip(act.chunks_exact_mut(inter))
            .for_each(f);
    }
}

/// Copy the residual stream `x` (`[n][h]`) after layer `layer` into every
/// slot of `taps` that requests that layer (see
/// [`crate::models::traits::HiddenTaps`]; `taps.data` is pre-sized).
pub(crate) fn copy_taps(
    taps: &mut crate::models::traits::HiddenTaps,
    layer: usize,
    x: &[f32],
    n: usize,
    h: usize,
) {
    let nt = taps.layers.len();
    for c in (0..nt).filter(|&c| taps.layers[c] == layer) {
        for t in 0..n {
            taps.data[(t * nt + c) * h..(t * nt + c + 1) * h]
                .copy_from_slice(&x[t * h..(t + 1) * h]);
        }
    }
}

/// `[rows][head_rows]` lm-head output -> `[rows][vocab]` (drops padding rows
/// of the head beyond the vocabulary).
pub(crate) fn trim_logit_rows(mut logits: Vec<f32>, head_rows: usize, vocab: usize) -> Vec<f32> {
    if head_rows == vocab {
        return logits;
    }
    let rows = logits.len() / head_rows;
    for r in 0..rows {
        logits.copy_within(r * head_rows..r * head_rows + vocab, r * vocab);
    }
    logits.truncate(rows * vocab);
    logits
}

// ---------------------------------------------------------------------------
// GGUF metadata helpers
// ---------------------------------------------------------------------------

pub(crate) fn meta_u32(g: &GgufFile, key: &str) -> anyhow::Result<usize> {
    g.get_metadata(key)
        .and_then(|v| v.to_u32())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow::anyhow!("missing GGUF metadata {key}"))
}

pub(crate) fn meta_f32(g: &GgufFile, key: &str) -> Option<f32> {
    g.get_metadata(key).and_then(|v| v.to_f32())
}

pub(crate) fn meta_str<'a>(g: &'a GgufFile, key: &str) -> Option<&'a str> {
    g.get_metadata(key).and_then(|v| v.to_string())
}

/// YaRN parameters from `{arch}.rope.scaling.*`, if the GGUF asks for YaRN.
pub(crate) fn yarn_from_gguf(g: &GgufFile, arch: &str) -> Option<Yarn> {
    if meta_str(g, &format!("{arch}.rope.scaling.type")) != Some("yarn") {
        return None;
    }
    Some(Yarn {
        factor: meta_f32(g, &format!("{arch}.rope.scaling.factor"))?,
        original_context: meta_u32(g, &format!("{arch}.rope.scaling.original_context_length"))
            .ok()?,
        beta_fast: meta_f32(g, &format!("{arch}.rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
        beta_slow: meta_f32(g, &format!("{arch}.rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
    })
}

/// Special token ids: `(bos, eos list)`. EOS = the GGUF `eos_token_id` plus
/// the ChatML / Qwen end markers when they exist in the vocabulary.
pub(crate) fn special_tokens_from_gguf(g: &GgufFile) -> (Option<u32>, Vec<u32>) {
    let id = |k: &str| g.get_metadata(k).and_then(|v| v.to_u32());
    let add_bos = g
        .get_metadata("tokenizer.ggml.add_bos_token")
        .map(|v| matches!(v, crate::loader::gguf::Value::Bool(true)))
        .unwrap_or(false);
    let bos = if add_bos {
        id("tokenizer.ggml.bos_token_id")
    } else {
        None
    };
    let mut eos: Vec<u32> = id("tokenizer.ggml.eos_token_id").into_iter().collect();
    if let Some(tokens) = g
        .get_metadata("tokenizer.ggml.tokens")
        .and_then(|v| v.to_string_vec())
    {
        for marker in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(i) = tokens.iter().position(|t| *t == marker) {
                let i = i as u32;
                if !eos.contains(&i) {
                    eos.push(i);
                }
            }
        }
    }
    (bos, eos)
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a Bonsai model from a `.gguf` file or a converted `.xq1` file
/// (see [`mod@format`]), dispatching on the architecture (`qwen3` → dense,
/// `qwen35` → hybrid).
pub fn load_causal_lm(path: impl AsRef<Path>) -> anyhow::Result<Box<dyn CausalLM>> {
    let path = path.as_ref();
    if format::is_xq1(path)? {
        return format::load(path);
    }
    let g = GgufFile::open(path)?;
    match g.architecture()?.as_str() {
        "qwen3" => Ok(Box::new(qwen3::Qwen3Model::from_gguf(&g)?)),
        "qwen35" => qwen35::Qwen35Model::from_gguf(&g),
        other => {
            anyhow::bail!("unsupported GGUF architecture {other:?} (expected qwen3 or qwen35)")
        }
    }
}

/// Tokenizer and chat template shipped with a model file.
pub struct ModelAssets {
    pub tokenizer: tokenizers::Tokenizer,
    /// HF Jinja chat template (`tokenizer.chat_template`).
    pub chat_template: Option<String>,
}

impl ModelAssets {
    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        Ok(Self {
            tokenizer: crate::loader::gguf::build_tokenizer(g)?,
            chat_template: meta_str(g, "tokenizer.chat_template").map(str::to_owned),
        })
    }
}

/// Build a [`crate::Generator`] (tokenizer, chat template, EOS ids) for a
/// loaded model.
pub fn make_generator(
    model: Arc<dyn CausalLM>,
    assets: ModelAssets,
) -> anyhow::Result<crate::Generator> {
    let tok_str = |id: Option<u32>| {
        id.and_then(|i| assets.tokenizer.id_to_token(i))
            .unwrap_or_default()
    };
    let bos = tok_str(model.bos_token_id());
    let eos = tok_str(model.eos_token_ids().first().copied());
    let template = assets
        .chat_template
        .as_deref()
        .map(|src| crate::ChatTemplate::new(src, bos, eos))
        .transpose()?;
    let mut g = crate::Generator::new(model, assets.tokenizer);
    if let Some(t) = template {
        g = g.with_chat_template(t);
    }
    Ok(g)
}

/// Load a Bonsai model (`.gguf` or `.xq1`) and wrap it in a
/// [`crate::Generator`]. The tokenizer comes from `tokenizer_json` when
/// given, else from the model file's embedded tokenizer metadata.
pub fn load_generator(
    path: impl AsRef<Path>,
    tokenizer_json: Option<&Path>,
) -> anyhow::Result<crate::Generator> {
    let path = path.as_ref();
    let (model, mut assets) = if format::is_xq1(path)? {
        format::load_with_assets(path)?
    } else {
        let g = GgufFile::open(path)?;
        let model: Box<dyn CausalLM> = match g.architecture()?.as_str() {
            "qwen3" => Box::new(qwen3::Qwen3Model::from_gguf(&g)?),
            "qwen35" => qwen35::Qwen35Model::from_gguf(&g)?,
            other => anyhow::bail!("unsupported GGUF architecture {other:?}"),
        };
        (model, ModelAssets::from_gguf(&g)?)
    };
    if let Some(p) = tokenizer_json {
        assets.tokenizer = tokenizers::Tokenizer::from_file(p)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", p.display()))?;
    }
    make_generator(Arc::from(model), assets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive causal GQA attention over f16-rounded K/V.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n: usize,
        pos: usize,
        heads: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
    ) -> Vec<f32> {
        let r = |x: f32| f16::from_f32(x).to_f32();
        let total = pos + n;
        let mut out = vec![0.0; n * heads * hd];
        for t in 0..n {
            for h in 0..heads {
                let kvh = h / (heads / nkv);
                let qv = &q[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                let len = pos + t + 1;
                let s: Vec<f64> = (0..len)
                    .map(|p| {
                        (0..hd)
                            .map(|i| qv[i] as f64 * r(k[(kvh * total + p) * hd + i]) as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for i in 0..hd {
                    let acc: f64 = (0..len)
                        .map(|p| e[p] * r(v[(kvh * total + p) * hd + i]) as f64)
                        .sum();
                    out[(t * heads + h) * hd + i] = (acc / z) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn attention_matches_naive() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        for &(heads, nkv, hd, pos, n) in &[
            (4, 2, 16, 0, 5),
            (6, 2, 32, 300, 1),
            (3, 3, 8, 510, 7),
            (8, 1, 16, 0, 600),
        ] {
            let total = pos + n;
            let k: Vec<f32> = (0..nkv * total * hd).map(|_| rnd() * 2.0).collect();
            let v: Vec<f32> = (0..nkv * total * hd).map(|_| rnd()).collect();
            let q: Vec<f32> = (0..n * heads * hd).map(|_| rnd() * 2.0).collect();
            let mut cache = KvCache::new(nkv, hd, 4096);
            cache.reserve(total, 0).unwrap();
            for h in 0..nkv {
                for p in 0..total {
                    let o = (h * total + p) * hd;
                    cache.store(h, p, &k[o..o + hd], &v[o..o + hd]);
                }
            }
            let mut out = vec![0.0; n * heads * hd];
            let scale = 1.0 / (hd as f32).sqrt();
            attention(&q, n, pos, heads, &cache, scale, &mut out);
            let want = naive(&q, &k, &v, n, pos, heads, nkv, hd, scale);
            let err = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                err < 1e-4,
                "heads {heads} kv {nkv} pos {pos} n {n}: max err {err}"
            );
        }
    }

    #[test]
    fn kv_cache_growth_keeps_prefix() {
        let mut c = KvCache::new(2, 4, 1000);
        c.reserve(3, 0).unwrap();
        for p in 0..3 {
            c.store(1, p, &[p as f32; 4], &[-(p as f32); 4]);
        }
        c.reserve(700, 3).unwrap();
        assert!(c.cap >= 700 && c.cap <= 1000);
        for p in 0..3 {
            assert_eq!(c.k_rows(1, p, p + 1)[0].to_f32(), p as f32);
            assert_eq!(c.v_rows(1, p, p + 1)[3].to_f32(), -(p as f32));
        }
        assert!(c.reserve(1001, 700).is_err());
    }

    #[test]
    fn yarn_matches_hf_formula() {
        // Bonsai-1.7B: dim 128, theta 1e6, factor 4, original ctx 8192.
        let y = Yarn {
            factor: 4.0,
            original_context: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let r = Rope::new(128, 1e6, Some(&y));
        assert!((r.mscale - (0.1 * 4f32.ln() + 1.0)).abs() < 1e-6);
        // low = floor(17.17) = 17, high = ceil(33.2) = 34
        let base = |i: usize| 1.0 / 1e6f64.powf(2.0 * i as f64 / 128.0);
        assert!((r.inv_freq[5] as f64 - base(5)).abs() <= 1e-6 * base(5));
        assert!((r.inv_freq[60] as f64 - base(60) / 4.0).abs() <= 1e-6 * base(60));
        let ramp = (25.0 - 17.0) / (34.0 - 17.0);
        let want = base(25) / 4.0 * ramp + base(25) * (1.0 - ramp);
        assert!((r.inv_freq[25] as f64 - want).abs() <= 1e-6 * want);
    }
}
