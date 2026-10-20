//! PrismML's DSpark drafter for Bonsai 27B (GGUF arch `dspark`), a
//! [`DraftModel`] for [`crate::speculative`].
//!
//! Semantics follow PrismML's llama.cpp fork (`prism` branch,
//! `src/models/dspark.cpp` + `common/speculative.cpp`,
//! `common_speculative_impl_draft_dspark`), for checkpoints without hidden
//! correction (the published `Bonsai-27B-dspark` has none):
//!
//! - **Context.** For every committed target position the target's layer
//!   outputs at `dspark.target_layers` (five layers) are concatenated,
//!   projected `g = rms(fc · taps, hidden_norm)`, and each drafter layer
//!   stores `K = rope(rms_head(k_proj(g), k_norm))`, `V = v_proj(g)` at that
//!   position. Context rows never pass through the drafter's norms or MLPs.
//! - **Block.** A draft round runs `block_size` (4) rows at positions
//!   `p .. p+4`: the last sampled token (the anchor, at position `p`, not
//!   yet fed to the target) then `mask_token_id`s; their embeddings get a
//!   log-SNR conditioning vector added (anchor at `max_log_snr`, masks at
//!   `min_log_snr`, through `log_snr_fc1 -> SiLU -> log_snr_fc2`; constant,
//!   so precomputed at load). Six dense Qwen3-style layers follow (Q/K RMS
//!   norm, NEOX RoPE, GQA 40/4 heads) whose attention is **non-causal**:
//!   every block row sees all context rows and all block rows.
//! - **Head.** `base_k = output(rms(x_k, output_norm))` for the four rows
//!   in one GEMM, then a sequential Markov correction: `logits_k = base_k +
//!   markov_b · markov_a[prev]` with `prev` = the anchor for `k = 0` and the
//!   token drafted at `k - 1` after that; the mask token is excluded.
//!   llama.cpp drafts the argmax; here each step goes through the request's
//!   [`DraftSampler`] (the argmax for greedy requests), so sampled requests
//!   get a proper draft distribution for rejection sampling.
//!
//! The confidence head (used by llama.cpp only for optional truncation) is
//! not loaded.
//!
//! Precision: the drafter's Q4_1 weights are used exactly ([`q4_1`]),
//! activations are quantized to int8 per 32 as in llama.cpp's CPU backend
//! (including the Markov term, which llama.cpp's host loop evaluates in
//! f32), context K/V are stored in f16 as in llama.cpp's KV cache. By default the drafter's
//! own Q4_1 `output.weight` is the head, as in llama.cpp; with
//! [`DsparkHead::SharedTarget`] the target's 1-bit head is used instead (as
//! the model card describes: the drafter shares embeddings and head with
//! the target), which streams 3.3x fewer bytes per round. The token
//! embedding is the target's own (byte-identical in the published files).
//! None of this affects the generated text, only how often drafts are
//! accepted.

pub mod q4_1;

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use self::q4_1::{GGML_Q4_1, Q8Blocks, Q41Weight};
use super::qwen35::Qwen35Model;
use super::{Embedding, Linear, Rope, gguf_vec, meta_f32, meta_u32, rms_norm_rows, swiglu_rows};
use crate::bit1::q1_0::Q8Acts;
use crate::loader::gguf::{GgufFile, Value};
use crate::models::traits::CausalLM;
use crate::nn::fast_attention_simd::{
    dot_product_f32, f16_to_f32_slice, f32_to_f16_slice, rms_norm,
};
use crate::speculative::{DraftModel, DraftSampler, DraftToken, Drafter};
use half::f16;

#[derive(Clone, Debug, PartialEq)]
pub struct DsparkConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token: u32,
    /// Target layers whose outputs feed the drafter, in concatenation order.
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub vocab: usize,
    /// `(min_log_snr, max_log_snr)` if log-SNR conditioning is on.
    pub log_snr: Option<(f32, f32)>,
}

impl DsparkConfig {
    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        let arch = g.architecture()?;
        anyhow::ensure!(arch == "dspark", "not a DSpark drafter (arch {arch:?})");
        let k = |s: &str| format!("dspark.{s}");
        let flag = |s: &str| matches!(g.get_metadata(&k(s)), Some(Value::Bool(true)));
        anyhow::ensure!(
            !flag("dspark.hidden_correction"),
            "DSpark hidden-correction checkpoints are not supported"
        );
        let target_layers = match g.get_metadata(&k("dspark.target_layers")) {
            Some(Value::Array(v)) => v
                .iter()
                .map(|x| x.to_u32().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow::anyhow!("dspark.target_layers: not integers"))?,
            _ => anyhow::bail!("missing dspark.dspark.target_layers"),
        };
        let heads = meta_u32(g, &k("attention.head_count"))?;
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let log_snr = if flag("dspark.log_snr_conditioning") {
            let lo = meta_f32(g, &k("dspark.min_log_snr"))
                .ok_or_else(|| anyhow::anyhow!("missing min_log_snr"))?;
            let hi = meta_f32(g, &k("dspark.max_log_snr"))
                .ok_or_else(|| anyhow::anyhow!("missing max_log_snr"))?;
            anyhow::ensure!(
                lo.is_finite() && hi.is_finite() && hi > lo,
                "log-SNR bounds"
            );
            Some((lo, hi))
        } else {
            None
        };
        Ok(Self {
            hidden,
            intermediate: meta_u32(g, &k("feed_forward_length"))?,
            layers: meta_u32(g, &k("block_count"))?,
            heads,
            kv_heads: meta_u32(g, &k("attention.head_count_kv"))?,
            head_dim: meta_u32(g, &k("attention.key_length")).unwrap_or(hidden / heads),
            eps: meta_f32(g, &k("attention.layer_norm_rms_epsilon"))
                .ok_or_else(|| anyhow::anyhow!("missing rms epsilon"))?,
            rope_theta: meta_f32(g, &k("rope.freq_base")).unwrap_or(10_000.0),
            block_size: meta_u32(g, &k("dspark.block_size"))?,
            mask_token: meta_u32(g, &k("dspark.mask_token_id"))? as u32,
            target_layers,
            markov_rank: meta_u32(g, &k("dspark.markov_rank")).unwrap_or(0),
            vocab: g.tensor_info("token_embd.weight")?.shape[0],
            log_snr,
        })
    }
}

/// Which output head scores the drafts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DsparkHead {
    /// The drafter's own `output.weight` (Q4_1, 795 MB per round), as
    /// llama.cpp does by default.
    Own,
    /// The target's 1-bit output head (179 MB per round), as the model
    /// card describes and llama.cpp's `LLAMA_DSPARK_SHARED_HEAD=1`. Same
    /// acceptance as [`Own`](Self::Own) in our measurements, so the default.
    #[default]
    SharedTarget,
}

impl std::str::FromStr for DsparkHead {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "own" | "drafter" => Ok(Self::Own),
            "shared" | "target" => Ok(Self::SharedTarget),
            o => anyhow::bail!("unknown DSpark head {o:?} (own | shared)"),
        }
    }
}

enum Head {
    Q41(Q41Weight),
    Q1(Linear),
}

struct DLayer {
    attn_norm: Vec<f32>,
    q: Q41Weight,
    /// Rows `[k (kv*hd); v (kv*hd)]`.
    kv: Q41Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: Q41Weight,
    ffn_norm: Vec<f32>,
    /// Rows `[gate (inter); up (inter)]`.
    gate_up: Q41Weight,
    down: Q41Weight,
}

struct Weights {
    cfg: DsparkConfig,
    embed: Embedding,
    fc: Q41Weight,
    hidden_norm: Vec<f32>,
    layers: Vec<DLayer>,
    output_norm: Vec<f32>,
    head: Head,
    /// `[vocab][rank]` bf16 bits.
    markov_a: Vec<u16>,
    /// `[vocab][rank]`.
    markov_b: Option<Q41Weight>,
    /// Log-SNR conditioning added to the anchor / mask embeddings.
    snr_anchor: Vec<f32>,
    snr_mask: Vec<f32>,
    rope: Rope,
}

/// A loaded DSpark drafter (cheap to clone; weights are shared).
#[derive(Clone)]
pub struct DsparkModel {
    w: Arc<Weights>,
}

fn q41(g: &GgufFile, name: &str) -> anyhow::Result<Q41Weight> {
    let info = g.tensor_info(name)?;
    anyhow::ensure!(info.shape.len() == 2, "{name}: expected a 2-D tensor");
    anyhow::ensure!(
        info.dtype_u32 == GGML_Q4_1,
        "{name}: dtype {} (expected Q4_1)",
        info.dtype_u32
    );
    let w = Q41Weight::from_gguf(g.tensor_bytes(name)?, info.shape[0], info.shape[1])?;
    g.release_tensor(name);
    Ok(w)
}

/// f32 matrix (`[rows][cols]`) from any dense / standard GGUF dtype.
fn dense(g: &GgufFile, name: &str) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    let info = g.tensor_info(name)?;
    anyhow::ensure!(info.shape.len() == 2, "{name}: expected a 2-D tensor");
    let (r, c) = (info.shape[0], info.shape[1]);
    let v = g.dequantize_tensor(name)?;
    g.release_tensor(name);
    Ok((r, c, v))
}

/// `XORNET_DSPARK_PROFILE=1`: per-round phase timings on stderr.
fn profile() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var_os("XORNET_DSPARK_PROFILE").is_some())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `W x + b` for a dense row-major matrix.
fn dense_mv(w: &[f32], rows: usize, cols: usize, x: &[f32], b: &[f32]) -> Vec<f32> {
    (0..rows)
        .into_par_iter()
        .map(|r| dot_product_f32(&w[r * cols..(r + 1) * cols], x) + b[r])
        .collect()
}

impl DsparkModel {
    /// Load the drafter GGUF. `target` (the Bonsai 27B it was trained
    /// against) supplies the token embedding when it is byte-identical to
    /// the drafter's, and the head with [`DsparkHead::SharedTarget`].
    pub fn load(
        path: impl AsRef<Path>,
        target: Option<&Qwen35Model>,
        head: DsparkHead,
    ) -> anyhow::Result<Self> {
        let g = GgufFile::open(path.as_ref())?;
        let cfg = DsparkConfig::from_gguf(&g)?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let embed = match target {
            Some(t) if same_embedding(&g, t)? => t.embed.clone(),
            _ => Embedding(Linear::from_gguf(&g, "token_embd.weight")?),
        };
        g.release_tensor("token_embd.weight");
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = |n: &str| format!("blk.{i}.{n}");
                Ok(DLayer {
                    attn_norm: gguf_vec(&g, &p("attn_norm.weight"), h)?,
                    q: q41(&g, &p("attn_q.weight"))?,
                    kv: Q41Weight::concat_rows(&[
                        &q41(&g, &p("attn_k.weight"))?,
                        &q41(&g, &p("attn_v.weight"))?,
                    ])?,
                    q_norm: gguf_vec(&g, &p("attn_q_norm.weight"), hd)?,
                    k_norm: gguf_vec(&g, &p("attn_k_norm.weight"), hd)?,
                    o: q41(&g, &p("attn_output.weight"))?,
                    ffn_norm: gguf_vec(&g, &p("ffn_norm.weight"), h)?,
                    gate_up: Q41Weight::concat_rows(&[
                        &q41(&g, &p("ffn_gate.weight"))?,
                        &q41(&g, &p("ffn_up.weight"))?,
                    ])?,
                    down: q41(&g, &p("ffn_down.weight"))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let head = match head {
            DsparkHead::SharedTarget => {
                let t = target
                    .ok_or_else(|| anyhow::anyhow!("DSpark shared head needs the target model"))?;
                Head::Q1(t.lm_head().clone())
            }
            DsparkHead::Own => Head::Q41(q41(&g, "output.weight")?),
        };
        let (markov_a, markov_b) = if cfg.markov_rank > 0 {
            let info = g.tensor_info("dspark.markov_head_a.weight")?;
            anyhow::ensure!(
                info.shape == [cfg.vocab, cfg.markov_rank],
                "markov_head_a shape {:?}",
                info.shape
            );
            let a: Vec<u16> = match info.dtype_u32 {
                30 => g
                    .tensor_bytes("dspark.markov_head_a.weight")?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| u16::from_le_bytes(b))
                    .collect(),
                _ => g
                    .dequantize_tensor("dspark.markov_head_a.weight")?
                    .iter()
                    .map(|&x| half::bf16::from_f32(x).to_bits())
                    .collect(),
            };
            anyhow::ensure!(
                info.dtype_u32 == 30 || info.dtype_u32 == 0,
                "markov_head_a: dtype {} not supported",
                info.dtype_u32
            );
            g.release_tensor("dspark.markov_head_a.weight");
            let b = q41(&g, "dspark.markov_head_b.weight")?;
            anyhow::ensure!(
                (b.rows(), b.cols()) == (cfg.vocab, cfg.markov_rank),
                "markov_head_b shape"
            );
            (a, Some(b))
        } else {
            (Vec::new(), None)
        };
        let (snr_anchor, snr_mask) = match cfg.log_snr {
            Some((lo, hi)) => {
                let (r1, c1, w1) = dense(&g, "dspark.log_snr_fc1.weight")?;
                let b1 = gguf_vec(&g, "dspark.log_snr_fc1.bias", h)?;
                let (r2, c2, w2) = dense(&g, "dspark.log_snr_fc2.weight")?;
                let b2 = gguf_vec(&g, "dspark.log_snr_fc2.bias", h)?;
                anyhow::ensure!((r1, c1, r2, c2) == (h, 128, h, h), "log_snr fc shapes");
                let embed = |log_snr: f32| {
                    let feat = log_snr_features(log_snr, lo, hi);
                    let mut x = dense_mv(&w1, r1, c1, &feat, &b1);
                    x.iter_mut().for_each(|v| *v = silu(*v));
                    dense_mv(&w2, r2, c2, &x, &b2)
                };
                (embed(hi), embed(lo))
            }
            None => (vec![0.0; h], vec![0.0; h]),
        };
        let fc = q41(&g, "dspark.fc.weight")?;
        anyhow::ensure!(
            (fc.rows(), fc.cols()) == (h, cfg.target_layers.len() * h),
            "dspark.fc shape"
        );
        let w = Weights {
            embed,
            fc,
            hidden_norm: gguf_vec(&g, "dspark.hidden_norm.weight", h)?,
            layers,
            output_norm: gguf_vec(&g, "output_norm.weight", h)?,
            head,
            markov_a,
            markov_b,
            snr_anchor,
            snr_mask,
            rope: Rope::new(hd, cfg.rope_theta, None),
            cfg,
        };
        w.validate()?;
        Ok(Self { w: Arc::new(w) })
    }

    /// [`load`](Self::load) taking the target as a trait object: it must be
    /// a [`Qwen35Model`] (Bonsai 27B) for the embedding / head to be shared.
    pub fn load_for(
        path: impl AsRef<Path>,
        target: &dyn CausalLM,
        head: DsparkHead,
    ) -> anyhow::Result<Self> {
        let t = target
            .as_any()
            .and_then(|a| a.downcast_ref::<Qwen35Model>());
        anyhow::ensure!(
            t.is_some() || head == DsparkHead::Own,
            "DSpark shared head needs a qwen35 (Bonsai 27B) target"
        );
        Self::load(path, t, head)
    }

    pub fn config(&self) -> &DsparkConfig {
        &self.w.cfg
    }

    /// Weight bytes streamed by one draft round (excluding context rows).
    pub fn bytes_per_round(&self) -> usize {
        let w = &self.w;
        w.layers
            .iter()
            .map(|l| {
                l.q.nbytes() + l.kv.nbytes() + l.o.nbytes() + l.gate_up.nbytes() + l.down.nbytes()
            })
            .sum::<usize>()
            + match &w.head {
                Head::Q41(h) => h.nbytes(),
                Head::Q1(h) => h.nbytes(),
            }
            + w.markov_b
                .as_ref()
                .map_or(0, |b| b.nbytes() * w.cfg.block_size)
    }
}

fn same_embedding(g: &GgufFile, t: &Qwen35Model) -> anyhow::Result<bool> {
    let info = g.tensor_info("token_embd.weight")?;
    let Linear::Q1(te) = &t.embed.0 else {
        return Ok(false);
    };
    if info.shape != [te.rows(), te.cols()] {
        return Ok(false);
    }
    let mine = match Linear::from_gguf(g, "token_embd.weight")? {
        Linear::Q1(w) => w,
        Linear::F32 { .. } => return Ok(false),
    };
    Ok(mine.bits().as_slice() == te.bits().as_slice()
        && mine.scales().as_slice() == te.scales().as_slice()
        && mine.block() == te.block())
}

/// `LogSnrEmbed` featurisation (llama.cpp `dspark.cpp`): `t = (s - lo) /
/// (hi - lo) * 1000`, `[sin(t f_i), cos(t f_i)]`, `f_i = exp(-ln(1e4) i / 64)`.
fn log_snr_features(log_snr: f32, lo: f32, hi: f32) -> Vec<f32> {
    const N: usize = 128;
    const HALF: usize = N / 2;
    let t = (log_snr - lo) / (hi - lo) * 1000.0;
    let mut f = vec![0.0f32; N];
    for i in 0..HALF {
        let freq = (-(10000.0f32).ln() * i as f32 / HALF as f32).exp();
        let a = t * freq;
        f[i] = a.sin();
        f[HALF + i] = a.cos();
    }
    f
}

impl Weights {
    fn validate(&self) -> anyhow::Result<()> {
        let c = &self.cfg;
        let (h, hd) = (c.hidden, c.head_dim);
        anyhow::ensure!(c.heads.is_multiple_of(c.kv_heads), "head counts");
        anyhow::ensure!(self.embed.dim() == h, "embedding dim");
        anyhow::ensure!(
            (c.mask_token as usize) < c.vocab,
            "mask token outside vocab"
        );
        anyhow::ensure!(c.block_size >= 1, "block size");
        for (i, l) in self.layers.iter().enumerate() {
            let ok = (l.q.rows(), l.q.cols()) == (c.heads * hd, h)
                && (l.kv.rows(), l.kv.cols()) == (2 * c.kv_heads * hd, h)
                && (l.o.rows(), l.o.cols()) == (h, c.heads * hd)
                && (l.gate_up.rows(), l.gate_up.cols()) == (2 * c.intermediate, h)
                && (l.down.rows(), l.down.cols()) == (h, c.intermediate);
            anyhow::ensure!(ok, "drafter layer {i}: unexpected tensor shapes");
        }
        let (hr, hc) = match &self.head {
            Head::Q41(w) => (w.rows(), w.cols()),
            Head::Q1(w) => (w.rows(), w.cols()),
        };
        anyhow::ensure!(hc == h && hr >= c.vocab, "head shape");
        Ok(())
    }

    /// `[n][kv*hd]` K (normed, roped at `positions`) and V of rows `x`.
    fn kv_rows(
        &self,
        l: &DLayer,
        x: &[f32],
        n: usize,
        positions: &[usize],
        acts: &mut Q8Blocks,
    ) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let (hd, kvd) = (c.head_dim, c.kv_heads * c.head_dim);
        acts.quantize(x, n, c.hidden);
        let mut kv = vec![0.0f32; n * 2 * kvd];
        q4_1::gemm(&l.kv, acts, &mut kv);
        let mut k = vec![0.0f32; n * kvd];
        let mut v = vec![0.0f32; n * kvd];
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for t in 0..n {
            let row = &kv[t * 2 * kvd..(t + 1) * 2 * kvd];
            let kr = &mut k[t * kvd..(t + 1) * kvd];
            kr.copy_from_slice(&row[..kvd]);
            v[t * kvd..(t + 1) * kvd].copy_from_slice(&row[kvd..]);
            self.rope.tables(positions[t], 1, &mut cos, &mut sin);
            for hk in kr.chunks_exact_mut(hd) {
                rms_norm(hk, &l.k_norm, c.eps);
                self.rope.apply(hk, &cos, &sin);
            }
        }
        (k, v)
    }

    /// Project new context rows (`taps`: `[n][n_taps * hidden]`) into every
    /// layer's K/V.
    fn context_kv(&self, taps: &[f32], positions: &[usize]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let c = &self.cfg;
        let n = positions.len();
        let h = c.hidden;
        let mut acts = Q8Blocks::new();
        acts.quantize(taps, n, c.target_layers.len() * h);
        let mut g = vec![0.0f32; n * h];
        q4_1::gemm(&self.fc, &acts, &mut g);
        rms_norm_rows(&mut g, &self.hidden_norm, c.eps);
        self.layers
            .iter()
            .map(|l| self.kv_rows(l, &g, n, positions, &mut acts))
            .collect()
    }

    /// Run the draft block; returns `[block][vocab]` base logits (before
    /// the Markov term).
    fn block_logits(&self, ctx: &Context, anchor: u32, pos: usize) -> anyhow::Result<Vec<f32>> {
        let c = &self.cfg;
        let (h, hd, nh, nkv) = (c.hidden, c.head_dim, c.heads, c.kv_heads);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let bs = c.block_size;
        let toks: Vec<u32> = (0..bs)
            .map(|i| if i == 0 { anchor } else { c.mask_token })
            .collect();
        let positions: Vec<usize> = (pos..pos + bs).collect();
        let t0 = std::time::Instant::now();
        let mut x = vec![0.0f32; bs * h];
        self.embed.lookup(&toks, &mut x)?;
        for (r, row) in x.chunks_exact_mut(h).enumerate() {
            let s = if r % bs == 0 {
                &self.snr_anchor
            } else {
                &self.snr_mask
            };
            row.iter_mut().zip(s).for_each(|(a, b)| *a += b);
        }
        let mut acts = Q8Blocks::new();
        let mut hbuf = vec![0.0f32; bs * h];
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            hbuf.copy_from_slice(&x);
            rms_norm_rows(&mut hbuf, &l.attn_norm, c.eps);
            acts.quantize(&hbuf, bs, h);
            let mut q = vec![0.0f32; bs * qd];
            q4_1::gemm(&l.q, &acts, &mut q);
            for (t, qr) in q.chunks_exact_mut(qd).enumerate() {
                self.rope.tables(positions[t], 1, &mut cos, &mut sin);
                for hq in qr.chunks_exact_mut(hd) {
                    rms_norm(hq, &l.q_norm, c.eps);
                    self.rope.apply(hq, &cos, &sin);
                }
            }
            let (kb, vb) = self.kv_rows(l, &hbuf, bs, &positions, &mut acts);
            let (kc, vc) = &ctx.kv[li];
            let nctx = ctx.positions.len();
            debug_assert_eq!(kc.len(), nctx * kvd);
            // Non-causal attention: each block row over all context rows
            // and all block rows. Keys / values per KV head gathered to
            // contiguous f32 first.
            let total = nctx + bs;
            let heads: Vec<(Vec<f32>, Vec<f32>)> = (0..nkv)
                .into_par_iter()
                .map(|kvh| {
                    let mut k = vec![0.0f32; total * hd];
                    let mut v = vec![0.0f32; total * hd];
                    for j in 0..nctx {
                        let o = j * kvd + kvh * hd;
                        f16_to_f32_slice(&kc[o..o + hd], &mut k[j * hd..(j + 1) * hd]);
                        f16_to_f32_slice(&vc[o..o + hd], &mut v[j * hd..(j + 1) * hd]);
                    }
                    for j in 0..bs {
                        let o = j * kvd + kvh * hd;
                        k[(nctx + j) * hd..(nctx + j + 1) * hd].copy_from_slice(&kb[o..o + hd]);
                        v[(nctx + j) * hd..(nctx + j + 1) * hd].copy_from_slice(&vb[o..o + hd]);
                    }
                    (k, v)
                })
                .collect();
            let group = nh / nkv;
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att = vec![0.0f32; bs * qd];
            att.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
                let (t, hh) = (i / nh, i % nh);
                let (kh, vh) = &heads[hh / group];
                let qv = &q[t * qd + hh * hd..t * qd + (hh + 1) * hd];
                let mut sc: Vec<f32> = kh
                    .chunks_exact(hd)
                    .map(|kj| dot_product_f32(qv, kj) * scale)
                    .collect();
                let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0f32;
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    z += *s;
                }
                o.fill(0.0);
                for (&e, vj) in sc.iter().zip(vh.chunks_exact(hd)) {
                    for (a, &b) in o.iter_mut().zip(vj) {
                        *a += e * b;
                    }
                }
                let inv = 1.0 / z;
                o.iter_mut().for_each(|a| *a *= inv);
            });
            acts.quantize(&att, bs, qd);
            let mut proj = vec![0.0f32; bs * h];
            q4_1::gemm(&l.o, &acts, &mut proj);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);

            hbuf.copy_from_slice(&x);
            rms_norm_rows(&mut hbuf, &l.ffn_norm, c.eps);
            acts.quantize(&hbuf, bs, h);
            let inter = c.intermediate;
            let mut gu = vec![0.0f32; bs * 2 * inter];
            q4_1::gemm(&l.gate_up, &acts, &mut gu);
            let mut act = vec![0.0f32; bs * inter];
            swiglu_rows(&mut gu, &mut act, inter);
            acts.quantize(&act, bs, inter);
            q4_1::gemm(&l.down, &acts, &mut proj);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
        }
        rms_norm_rows(&mut x, &self.output_norm, c.eps);
        let t_layers = t0.elapsed();
        let vocab = c.vocab;
        let logits = match &self.head {
            Head::Q41(w) => {
                acts.quantize(&x, bs, h);
                let mut out = vec![0.0f32; bs * w.rows()];
                q4_1::gemm(w, &acts, &mut out);
                super::trim_logit_rows(out, w.rows(), vocab)
            }
            Head::Q1(w) => {
                let mut out = vec![0.0f32; bs * w.rows()];
                w.forward(&x, bs, &mut Q8Acts::new(), &mut out);
                super::trim_logit_rows(out, w.rows(), vocab)
            }
        };
        if profile() {
            eprintln!(
                "dspark block: layers {:.1} ms, head {:.1} ms (ctx {})",
                t_layers.as_secs_f64() * 1e3,
                (t0.elapsed() - t_layers).as_secs_f64() * 1e3,
                ctx.positions.len()
            );
        }
        Ok(logits)
    }

    /// `logits += markov_b · markov_a[prev]`.
    fn add_markov(&self, prev: u32, logits: &mut [f32]) {
        let Some(b) = &self.markov_b else {
            return;
        };
        let r = self.cfg.markov_rank;
        let a: Vec<f32> = self.markov_a[prev as usize * r..(prev as usize + 1) * r]
            .iter()
            .map(|&x| half::bf16::from_bits(x).to_f32())
            .collect();
        // Int8 activations like every other drafter matmul (llama.cpp's
        // host path uses f32 here; drafts only, 4.5x faster).
        let mut acts = Q8Blocks::new();
        acts.quantize(&a, 1, r);
        let mut bias = vec![0.0f32; b.rows()];
        q4_1::gemm(b, &acts, &mut bias);
        logits.iter_mut().zip(&bias).for_each(|(l, b)| *l += b);
    }
}

/// Drafter K/V of the committed target positions it has seen.
#[derive(Default)]
struct Context {
    /// Position of each stored row (ascending).
    positions: Vec<usize>,
    /// Per layer `[rows][kv*hd]` K and V (f16, as llama.cpp's KV cache).
    kv: Vec<(Vec<f16>, Vec<f16>)>,
}

impl Context {
    fn truncate_positions(&mut self, len: usize, kvd: usize) {
        let keep = self.positions.partition_point(|&p| p < len);
        self.positions.truncate(keep);
        for (k, v) in self.kv.iter_mut() {
            k.truncate(keep * kvd);
            v.truncate(keep * kvd);
        }
    }
}

/// Per-sequence DSpark state.
pub struct DsparkDrafter {
    w: Arc<Weights>,
    /// Tokens of target positions `0..seq.len()` this drafter knows about.
    seq: Vec<u32>,
    ctx: Context,
    /// Observed tap rows not projected into `ctx` yet (positions, rows):
    /// projecting is batched until the next draft (as llama.cpp does), so
    /// plain decode steps cost no drafter weight reads.
    pending_pos: Vec<usize>,
    pending_taps: Vec<f32>,
}

/// Pending tap rows are projected once this many accumulate (prefill).
const FLUSH_ROWS: usize = 128;

impl DraftModel for DsparkModel {
    fn name(&self) -> &str {
        "dspark"
    }
    fn new_drafter(&self) -> anyhow::Result<Box<dyn Drafter>> {
        let n = self.w.cfg.layers;
        Ok(Box::new(DsparkDrafter {
            w: self.w.clone(),
            seq: Vec::new(),
            ctx: Context {
                positions: Vec::new(),
                kv: (0..n).map(|_| (Vec::new(), Vec::new())).collect(),
            },
            pending_pos: Vec::new(),
            pending_taps: Vec::new(),
        }))
    }
    fn max_draft(&self) -> usize {
        self.w.cfg.block_size
    }
    fn check_target(&self, target: &dyn CausalLM) -> anyhow::Result<()> {
        let c = &self.w.cfg;
        anyhow::ensure!(
            target.vocab_size() == c.vocab,
            "DSpark drafter vocab {} != target vocab {}",
            c.vocab,
            target.vocab_size()
        );
        let (h, layers) = target
            .hidden_shape()
            .ok_or_else(|| anyhow::anyhow!("target model has no hidden-state taps"))?;
        anyhow::ensure!(h == c.hidden, "DSpark hidden {} != target {h}", c.hidden);
        anyhow::ensure!(
            c.target_layers.iter().all(|&l| l < layers),
            "DSpark taps layers {:?} beyond the target's {layers}",
            c.target_layers
        );
        Ok(())
    }
}

impl DsparkDrafter {
    /// Project the pending tap rows into the context K/V.
    fn flush(&mut self) {
        if self.pending_pos.is_empty() {
            return;
        }
        let c = &self.w.cfg;
        let width = c.target_layers.len() * c.hidden;
        // Bounded scratch for long prefills.
        for (i, p) in self.pending_pos.chunks(FLUSH_ROWS * 2).enumerate() {
            let at = i * FLUSH_ROWS * 2 * width;
            let t = &self.pending_taps[at..at + p.len() * width];
            let kv = self.w.context_kv(t, p);
            for ((k, v), (nk, nv)) in self.ctx.kv.iter_mut().zip(kv) {
                let at = k.len();
                k.resize(at + nk.len(), f16::ZERO);
                v.resize(at + nv.len(), f16::ZERO);
                f32_to_f16_slice(&nk, &mut k[at..]);
                f32_to_f16_slice(&nv, &mut v[at..]);
            }
            self.ctx.positions.extend_from_slice(p);
        }
        self.pending_pos.clear();
        self.pending_taps.clear();
    }

    fn truncate_pending(&mut self, len: usize) {
        let keep = self.pending_pos.partition_point(|&p| p < len);
        let width = self.w.cfg.target_layers.len() * self.w.cfg.hidden;
        self.pending_pos.truncate(keep);
        self.pending_taps.truncate(keep * width);
    }
}

impl Drafter for DsparkDrafter {
    fn taps(&self) -> Option<&[usize]> {
        Some(&self.w.cfg.target_layers)
    }

    fn rewind(&mut self, fed: &[u32]) {
        let l = self.seq.iter().zip(fed).take_while(|(a, b)| a == b).count();
        let kvd = self.w.cfg.kv_heads * self.w.cfg.head_dim;
        self.ctx.truncate_positions(l, kvd);
        self.truncate_pending(l);
        // Positions l.. of `fed` have no context rows (e.g. restored from
        // the prompt cache): drafting continues without them.
        self.seq.clear();
        self.seq.extend_from_slice(fed);
    }

    fn observe(&mut self, tokens: &[u32], pos: usize, taps: Option<&[f32]>) -> anyhow::Result<()> {
        let c = &self.w.cfg;
        let kvd = c.kv_heads * c.head_dim;
        let taps = taps.ok_or_else(|| anyhow::anyhow!("DSpark needs hidden taps"))?;
        let n = tokens.len();
        let width = c.target_layers.len() * c.hidden;
        anyhow::ensure!(
            taps.len() == n * width,
            "DSpark: tap rows {} != {n} x {width}",
            taps.len()
        );
        if pos < self.seq.len() {
            self.seq.truncate(pos);
            self.ctx.truncate_positions(pos, kvd);
            self.truncate_pending(pos);
        }
        self.seq.resize(pos, u32::MAX);
        self.seq.extend_from_slice(tokens);
        self.pending_pos.extend(pos..pos + n);
        self.pending_taps.extend_from_slice(taps);
        if self.pending_pos.len() >= FLUSH_ROWS {
            self.flush();
        }
        Ok(())
    }

    fn draft(
        &mut self,
        context: &[u32],
        max: usize,
        sampler: &mut dyn DraftSampler,
    ) -> anyhow::Result<Vec<DraftToken>> {
        let Some(&anchor) = context.last() else {
            return Ok(Vec::new());
        };
        let pos = context.len() - 1;
        let n = max.min(self.w.cfg.block_size);
        if n == 0 {
            return Ok(Vec::new());
        }
        self.flush();
        let c = &self.w.cfg;
        if self.ctx.positions.is_empty() {
            return Ok(Vec::new());
        }
        let base = self.w.block_logits(&self.ctx, anchor, pos)?;
        let vocab = c.vocab;
        let mut out = Vec::with_capacity(n);
        let mut drafted = Vec::with_capacity(n);
        let mut prev = anchor;
        let t0 = std::time::Instant::now();
        for k in 0..n {
            let mut l = base[k * vocab..(k + 1) * vocab].to_vec();
            self.w.add_markov(prev, &mut l);
            l[c.mask_token as usize] = f32::NEG_INFINITY;
            let (tok, probs) = sampler.pick(&mut l, &drafted)?;
            drafted.push(tok);
            out.push(DraftToken { token: tok, probs });
            prev = tok;
        }
        if profile() {
            eprintln!(
                "dspark markov+pick x{n}: {:.1} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(out)
    }
}
