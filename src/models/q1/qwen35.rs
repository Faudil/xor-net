//! Hybrid Qwen3.5 (Bonsai 27B, GGUF arch `qwen35`) on the 1-bit fast path.
//!
//! Layers alternate between Gated DeltaNet linear attention (recurrent) and
//! gated full attention (every `full_attention_interval`-th layer). Both are
//! followed by a SwiGLU MLP:
//!
//! ```text
//! x += mixer(rms(x, attn_norm));  x += down(silu(gate(h)) * up(h)), h = rms(x, post_attention_norm)
//! ```
//!
//! Semantics follow llama.cpp `src/models/qwen35.cpp` + `delta-net-base.cpp`
//! and the GGUF conventions of `convert_hf_to_gguf.py`:
//! - `ssm_a` already holds `-exp(A_log)`; decay `= exp(softplus(alpha + dt_bias) * ssm_a)`;
//! - V heads are stored in *tiled* order: value head `vh` uses key head
//!   `vh % k_heads`;
//! - RMS norm weights other than `ssm_norm` already include the `+1` offset;
//! - full attention: `attn_q` rows interleave `[q (hd) | gate (hd)]` per head,
//!   Q/K RMS norm, partial NEOX RoPE on the first `rope.dimension_count`
//!   dims, output multiplied by `sigmoid(gate)`.

use rayon::prelude::*;

use crate::bit1::q1_0::Q8Acts;
use crate::loader::gguf::GgufFile;
use crate::models::traits::{CausalLM, Logits, ModelState, downcast_state};
use crate::nn::fast_attention_simd::{dot_product_f32, rms_norm};
use crate::util::with_best_simd;

use super::{
    Embedding, KvCache, Linear, Rope, Yarn, attention, gguf_vec, meta_f32, meta_u32, rms_norm_rows,
    special_tokens_from_gguf, swiglu_rows, yarn_from_gguf,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Qwen35Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// `true` = Gated DeltaNet layer, `false` = full attention.
    pub recurrent: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub yarn: Option<Yarn>,
    pub context_length: usize,
    pub vocab: usize,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub conv_kernel: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

impl Qwen35Config {
    pub fn key_dim(&self) -> usize {
        self.k_heads * self.head_k_dim
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.head_v_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        let arch = g.architecture()?;
        let k = |s: &str| format!("{arch}.{s}");
        let nextn = meta_u32(g, &k("nextn_predict_layers")).unwrap_or(0);
        let layers = meta_u32(g, &k("block_count"))? - nextn;
        let interval = meta_u32(g, &k("full_attention_interval")).unwrap_or(4);
        let head_dim = meta_u32(g, &k("attention.key_length"))?;
        let v_heads = meta_u32(g, &k("ssm.time_step_rank"))?;
        let (bos, eos) = special_tokens_from_gguf(g);
        Ok(Self {
            hidden: meta_u32(g, &k("embedding_length"))?,
            intermediate: meta_u32(g, &k("feed_forward_length"))?,
            layers,
            recurrent: (0..layers).map(|i| (i + 1) % interval != 0).collect(),
            heads: meta_u32(g, &k("attention.head_count"))?,
            kv_heads: meta_u32(g, &k("attention.head_count_kv"))?,
            head_dim,
            rot_dim: meta_u32(g, &k("rope.dimension_count")).unwrap_or(head_dim),
            eps: meta_f32(g, &k("attention.layer_norm_rms_epsilon"))
                .ok_or_else(|| anyhow::anyhow!("missing rms epsilon"))?,
            rope_theta: meta_f32(g, &k("rope.freq_base")).unwrap_or(10_000.0),
            yarn: yarn_from_gguf(g, &arch),
            context_length: meta_u32(g, &k("context_length"))?,
            vocab: g.tensor_info("token_embd.weight")?.shape[0],
            bos,
            eos,
            conv_kernel: meta_u32(g, &k("ssm.conv_kernel"))?,
            k_heads: meta_u32(g, &k("ssm.group_count"))?,
            v_heads,
            head_k_dim: meta_u32(g, &k("ssm.state_size"))?,
            head_v_dim: meta_u32(g, &k("ssm.inner_size"))? / v_heads,
        })
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.recurrent.len() == self.layers, "recurrent flags");
        anyhow::ensure!(
            self.heads.is_multiple_of(self.kv_heads) && self.v_heads.is_multiple_of(self.k_heads),
            "head counts"
        );
        anyhow::ensure!(
            self.head_k_dim == self.head_v_dim,
            "Gated DeltaNet needs head_k_dim == head_v_dim"
        );
        anyhow::ensure!(
            self.rot_dim.is_multiple_of(2) && self.rot_dim <= self.head_dim,
            "rotary dim"
        );
        anyhow::ensure!(self.conv_kernel >= 1, "conv kernel");
        Ok(())
    }
}

/// Gated full attention.
#[derive(Clone, Debug)]
pub struct AttnMixer {
    /// Rows: `[q|gate interleaved per head (2*heads*hd); k; v]`.
    pub qkv: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub o: Linear,
}

/// Gated DeltaNet linear attention.
#[derive(Clone, Debug)]
pub struct GdnMixer {
    /// Rows: `[qkv (conv_dim); z (value_dim)]`.
    pub qkvz: Linear,
    /// Rows: `[beta (v_heads); alpha (v_heads)]`.
    pub ba: Linear,
    /// `[conv_dim][conv_kernel]`.
    pub conv: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` per value head.
    pub ssm_a: Vec<f32>,
    /// `[head_v_dim]` gated RMS norm weight.
    pub norm: Vec<f32>,
    pub out: Linear,
}

#[derive(Clone, Debug)]
pub enum Mixer {
    Attn(AttnMixer),
    Gdn(GdnMixer),
}

#[derive(Clone, Debug)]
pub struct Qwen35Layer {
    pub attn_norm: Vec<f32>,
    pub mixer: Mixer,
    pub post_norm: Vec<f32>,
    /// Rows: `[gate (inter); up (inter)]`.
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Clone, Debug)]
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    pub embed: Embedding,
    pub layers: Vec<Qwen35Layer>,
    pub output_norm: Vec<f32>,
    pub lm_head: Option<Linear>,
    rope: Rope,
}

/// Per-sequence state: KV caches (attention layers), conv history and
/// DeltaNet state matrices (recurrent layers), plus scratch.
pub struct Qwen35State {
    kv: Vec<Option<KvCache>>,
    /// Per layer: last `conv_kernel - 1` conv inputs, `[K-1][conv_dim]`.
    conv: Vec<Vec<f32>>,
    /// Per layer: `[v_heads][head_v_dim][head_k_dim]` (row = value dim).
    ssm: Vec<Vec<f32>>,
    /// Positions fed so far.
    len: usize,
    /// Saved recurrent state (see [`ModelState::checkpoint`]).
    ckpt: RecurrentCheckpoint,
    s: Scratch,
}

/// Copy of every recurrent layer's conv history and DeltaNet state at
/// position `len` (valid when `len` is `Some`). The buffers are kept across
/// checkpoints so re-checkpointing does not allocate.
#[derive(Default)]
struct RecurrentCheckpoint {
    len: Option<usize>,
    conv: Vec<Vec<f32>>,
    ssm: Vec<Vec<f32>>,
}

fn copy_layers(dst: &mut Vec<Vec<f32>>, src: &[Vec<f32>]) {
    dst.resize_with(src.len(), Vec::new);
    for (d, s) in dst.iter_mut().zip(src) {
        d.clear();
        d.extend_from_slice(s);
    }
}

impl Qwen35State {
    /// Bytes held by one checkpoint (recurrent conv + DeltaNet state).
    pub fn checkpoint_bytes(&self) -> usize {
        self.conv.iter().chain(&self.ssm).map(|v| v.len() * 4).sum()
    }
}

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    acts: Q8Acts,
}

impl ModelState for Qwen35State {
    fn reset(&mut self) -> anyhow::Result<()> {
        // Recurrent state must restart from zero; KV positions are
        // overwritten before they are read.
        for v in self.conv.iter_mut().chain(self.ssm.iter_mut()) {
            v.fill(0.0);
        }
        self.len = 0;
        self.ckpt.len = None;
        Ok(())
    }

    /// KV caches roll back to any length; the recurrent state only to the
    /// current length, 0 or the checkpoint.
    fn truncate(&mut self, len: usize) -> bool {
        if len == self.len {
            true
        } else if len == 0 {
            self.reset().is_ok()
        } else if len < self.len && self.ckpt.len == Some(len) {
            for (d, s) in self.conv.iter_mut().zip(&self.ckpt.conv) {
                d.copy_from_slice(s);
            }
            for (d, s) in self.ssm.iter_mut().zip(&self.ckpt.ssm) {
                d.copy_from_slice(s);
            }
            self.len = len;
            true
        } else {
            false
        }
    }

    fn checkpoint(&mut self) {
        copy_layers(&mut self.ckpt.conv, &self.conv);
        copy_layers(&mut self.ckpt.ssm, &self.ssm);
        self.ckpt.len = Some(self.len);
    }

    fn checkpoint_len(&self) -> Option<usize> {
        self.ckpt.len
    }

    fn needs_checkpoint(&self) -> bool {
        true
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn resize(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    v.resize(n, 0.0);
    &mut v[..n]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// One state row of the gated delta rule: `s *= decay`, then
/// `s += (v_j - s·k) * beta * k`; returns `s·q`. A function so `s` and `k`
/// are known not to alias and the loops vectorize (under `with_best_simd!`).
#[inline(always)]
fn delta_row(sj: &mut [f32], k: &[f32], q: &[f32], vj: f32, beta: f32, decay: f32) -> f32 {
    sj.iter_mut().for_each(|e| *e *= decay);
    let sk = dot_product_f32(sj, k);
    let d = (vj - sk) * beta;
    for (e, &ki) in sj.iter_mut().zip(k) {
        *e += d * ki;
    }
    dot_product_f32(sj, q)
}

/// `x / sqrt(Σx² + eps)` (llama.cpp `build_gdn_l2_norm`).
fn l2_norm(x: &mut [f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

impl GdnMixer {
    /// `hbuf`: normed input `[n][hidden]` -> `out`: `[n][hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        c: &Qwen35Config,
        hbuf: &[f32],
        n: usize,
        conv_state: &mut [f32],
        ssm: &mut [f32],
        s: &mut Scratch,
        out: &mut [f32],
    ) {
        let (cd, vd) = (c.conv_dim(), c.value_dim());
        let (kd, kk) = (c.key_dim(), c.conv_kernel);
        let (nv, nk, hk, hv) = (c.v_heads, c.k_heads, c.head_k_dim, c.head_v_dim);
        let w = cd + vd;
        let qkvz = resize(&mut s.a, n * w);
        self.qkvz.forward(hbuf, n, &mut s.acts, qkvz);
        let ba = resize(&mut s.b, n * 2 * nv);
        self.ba.forward(hbuf, n, &mut s.acts, ba);

        // Causal depthwise conv over [history (K-1 rows); new rows] + SiLU.
        let conv_out = resize(&mut s.c, n * cd);
        let hist = kk - 1;
        for t in 0..n {
            let o = &mut conv_out[t * cd..(t + 1) * cd];
            o.fill(0.0);
            for j in 0..kk {
                // window row t + j of [history; new]
                let r = t + j;
                let src: &[f32] = if r < hist {
                    &conv_state[r * cd..(r + 1) * cd]
                } else {
                    &qkvz[(r - hist) * w..(r - hist) * w + cd]
                };
                for ch in 0..cd {
                    o[ch] += self.conv[ch * kk + j] * src[ch];
                }
            }
            for v in o.iter_mut() {
                *v *= sigmoid(*v);
            }
        }
        // New history = last K-1 rows of [history; new].
        if hist > 0 {
            let mut new_hist = vec![0.0f32; hist * cd];
            for r in 0..hist {
                let src_row = n + r; // index in [history; new] (len hist + n)
                let src: &[f32] = if src_row < hist {
                    &conv_state[src_row * cd..(src_row + 1) * cd]
                } else {
                    &qkvz[(src_row - hist) * w..(src_row - hist) * w + cd]
                };
                new_hist[r * cd..(r + 1) * cd].copy_from_slice(src);
            }
            conv_state.copy_from_slice(&new_hist);
        }

        // Per token: L2-normalize q, k per key head; q *= 1/sqrt(hk).
        let qscale = 1.0 / (hk as f32).sqrt();
        for t in 0..n {
            let row = &mut conv_out[t * cd..(t + 1) * cd];
            for h in 0..2 * nk {
                l2_norm(&mut row[h * hk..(h + 1) * hk], c.eps);
            }
            row[..kd].iter_mut().for_each(|v| *v *= qscale);
        }

        // Delta rule, parallel over value heads, sequential over tokens.
        let o = resize(&mut s.d, n * vd);
        let mut tmp = vec![0.0f32; nv * n * hv]; // [head][t][hv]
        let conv_out = &*conv_out;
        let ba = &*ba;
        ssm.par_chunks_mut(hv * hk)
            .zip(tmp.par_chunks_mut(n * hv))
            .enumerate()
            .for_each(|(vh, (st, oh))| {
                let kh = vh % nk; // tiled V-head order (see module docs)
                with_best_simd!({
                    for t in 0..n {
                        let row = &conv_out[t * cd..(t + 1) * cd];
                        let q = &row[kh * hk..(kh + 1) * hk];
                        let k = &row[kd + kh * hk..kd + (kh + 1) * hk];
                        let v = &row[2 * kd + vh * hv..2 * kd + (vh + 1) * hv];
                        let beta = sigmoid(ba[t * 2 * nv + vh]);
                        let alpha = ba[t * 2 * nv + nv + vh];
                        let decay = (softplus(alpha + self.dt_bias[vh]) * self.ssm_a[vh]).exp();
                        let ot = &mut oh[t * hv..(t + 1) * hv];
                        for (j, sj) in st.chunks_exact_mut(hk).enumerate() {
                            ot[j] = delta_row(sj, k, q, v[j], beta, decay);
                        }
                    }
                });
            });
        // Gated RMS norm per head: rms(o_h) * norm * silu(z_h).
        for t in 0..n {
            let z = &qkvz[t * w + cd..(t + 1) * w];
            for vh in 0..nv {
                let oo = &mut o[t * vd + vh * hv..t * vd + (vh + 1) * hv];
                oo.copy_from_slice(&tmp[(vh * n + t) * hv..(vh * n + t + 1) * hv]);
                rms_norm(oo, &self.norm, c.eps);
                for (a, &zz) in oo.iter_mut().zip(&z[vh * hv..(vh + 1) * hv]) {
                    *a *= zz * sigmoid(zz);
                }
            }
        }
        self.out.forward(o, n, &mut s.acts, out);
    }
}

impl AttnMixer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        c: &Qwen35Config,
        rope: &Rope,
        hbuf: &[f32],
        n: usize,
        pos: usize,
        kv: &mut KvCache,
        s: &mut Scratch,
        out: &mut [f32],
    ) {
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let w = 2 * qd + 2 * kvd;
        let qkv = resize(&mut s.a, n * w);
        self.qkv.forward(hbuf, n, &mut s.acts, qkv);
        let q = resize(&mut s.b, n * qd);
        let gate = resize(&mut s.c, n * qd);
        let half = c.rot_dim / 2;
        for t in 0..n {
            let row = &mut qkv[t * w..(t + 1) * w];
            let (cs, sn) = (
                &s.cos[t * half..(t + 1) * half],
                &s.sin[t * half..(t + 1) * half],
            );
            let (qg, kvr) = row.split_at_mut(2 * qd);
            for h in 0..nh {
                let qh = &mut qg[2 * h * hd..(2 * h + 1) * hd];
                rms_norm(qh, &self.q_norm, c.eps);
                rope.apply(qh, cs, sn);
                q[(t * nh + h) * hd..(t * nh + h + 1) * hd].copy_from_slice(qh);
                gate[(t * nh + h) * hd..(t * nh + h + 1) * hd]
                    .copy_from_slice(&qg[(2 * h + 1) * hd..(2 * h + 2) * hd]);
            }
            let (kr, vr) = kvr.split_at_mut(kvd);
            for (j, (hk, hv)) in kr.chunks_exact_mut(hd).zip(vr.chunks_exact(hd)).enumerate() {
                rms_norm(hk, &self.k_norm, c.eps);
                rope.apply(hk, cs, sn);
                kv.store(j, pos + t, hk, hv);
            }
        }
        let att = resize(&mut s.d, n * qd);
        attention(q, n, pos, nh, kv, 1.0 / (hd as f32).sqrt(), att);
        for (a, &g) in att.iter_mut().zip(gate.iter()) {
            *a *= sigmoid(g);
        }
        self.o.forward(att, n, &mut s.acts, out);
    }
}

impl Qwen35Model {
    pub fn new(
        cfg: Qwen35Config,
        embed: Embedding,
        layers: Vec<Qwen35Layer>,
        output_norm: Vec<f32>,
        lm_head: Option<Linear>,
    ) -> anyhow::Result<Self> {
        cfg.validate()?;
        let c = &cfg;
        let h = c.hidden;
        anyhow::ensure!(embed.dim() == h, "embedding dim");
        anyhow::ensure!(layers.len() == c.layers, "layer count");
        anyhow::ensure!(output_norm.len() == h, "output norm");
        for (i, l) in layers.iter().enumerate() {
            let common = l.attn_norm.len() == h
                && l.post_norm.len() == h
                && (l.gate_up.rows(), l.gate_up.cols()) == (2 * c.intermediate, h)
                && (l.down.rows(), l.down.cols()) == (h, c.intermediate);
            let mixer = match (&l.mixer, c.recurrent[i]) {
                (Mixer::Attn(a), false) => {
                    (a.qkv.rows(), a.qkv.cols())
                        == (2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim, h)
                        && a.q_norm.len() == c.head_dim
                        && a.k_norm.len() == c.head_dim
                        && (a.o.rows(), a.o.cols()) == (h, c.heads * c.head_dim)
                }
                (Mixer::Gdn(g), true) => {
                    (g.qkvz.rows(), g.qkvz.cols()) == (c.conv_dim() + c.value_dim(), h)
                        && (g.ba.rows(), g.ba.cols()) == (2 * c.v_heads, h)
                        && g.conv.len() == c.conv_dim() * c.conv_kernel
                        && g.dt_bias.len() == c.v_heads
                        && g.ssm_a.len() == c.v_heads
                        && g.norm.len() == c.head_v_dim
                        && (g.out.rows(), g.out.cols()) == (h, c.value_dim())
                }
                _ => false,
            };
            anyhow::ensure!(common && mixer, "layer {i}: unexpected tensor shapes");
        }
        let head = lm_head.as_ref().unwrap_or(&embed.0);
        anyhow::ensure!(head.cols() == h && head.rows() >= c.vocab, "lm_head shape");
        let rope = Rope::new(c.rot_dim, c.rope_theta, c.yarn.as_ref());
        Ok(Self {
            cfg,
            embed,
            layers,
            output_norm,
            lm_head,
            rope,
        })
    }

    /// Load from GGUF, as a boxed [`CausalLM`].
    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Box<dyn CausalLM>> {
        Ok(Box::new(Self::load_gguf(g)?))
    }

    pub fn load_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        let cfg = Qwen35Config::from_gguf(g)?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = |n: &str| format!("blk.{i}.{n}");
                let mixer = if cfg.recurrent[i] {
                    let cd = cfg.conv_dim();
                    Mixer::Gdn(GdnMixer {
                        qkvz: Linear::from_gguf_concat(
                            g,
                            &[&p("attn_qkv.weight"), &p("attn_gate.weight")],
                        )?,
                        ba: Linear::from_gguf_concat(
                            g,
                            &[&p("ssm_beta.weight"), &p("ssm_alpha.weight")],
                        )?,
                        conv: gguf_vec(g, &p("ssm_conv1d.weight"), cd * cfg.conv_kernel)?,
                        dt_bias: gguf_vec(g, &p("ssm_dt.bias"), cfg.v_heads)?,
                        ssm_a: gguf_vec(g, &p("ssm_a"), cfg.v_heads)?,
                        norm: gguf_vec(g, &p("ssm_norm.weight"), cfg.head_v_dim)?,
                        out: Linear::from_gguf(g, &p("ssm_out.weight"))?,
                    })
                } else {
                    Mixer::Attn(AttnMixer {
                        qkv: Linear::from_gguf_concat(
                            g,
                            &[
                                &p("attn_q.weight"),
                                &p("attn_k.weight"),
                                &p("attn_v.weight"),
                            ],
                        )?,
                        q_norm: gguf_vec(g, &p("attn_q_norm.weight"), hd)?,
                        k_norm: gguf_vec(g, &p("attn_k_norm.weight"), hd)?,
                        o: Linear::from_gguf(g, &p("attn_output.weight"))?,
                    })
                };
                Ok(Qwen35Layer {
                    attn_norm: gguf_vec(g, &p("attn_norm.weight"), h)?,
                    mixer,
                    post_norm: gguf_vec(g, &p("post_attention_norm.weight"), h)?,
                    gate_up: Linear::from_gguf_concat(
                        g,
                        &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                    )?,
                    down: Linear::from_gguf(g, &p("ffn_down.weight"))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let embed = Embedding(Linear::from_gguf(g, "token_embd.weight")?);
        let output_norm = gguf_vec(g, "output_norm.weight", h)?;
        let lm_head = if g.has_tensor("output.weight") {
            Some(Linear::from_gguf(g, "output.weight")?)
        } else {
            None
        };
        Self::new(cfg, embed, layers, output_norm, lm_head)
    }

    pub fn write_xq1(&self, w: &mut impl super::format::TensorSink) {
        w.put_linear("embed", &self.embed.0);
        w.put_vec("output_norm", &self.output_norm);
        if let Some(h) = &self.lm_head {
            w.put_linear("lm_head", h);
        }
        for (i, l) in self.layers.iter().enumerate() {
            let p = |n: &str| format!("blk.{i}.{n}");
            w.put_vec(&p("attn_norm"), &l.attn_norm);
            w.put_vec(&p("post_norm"), &l.post_norm);
            w.put_linear(&p("gate_up"), &l.gate_up);
            w.put_linear(&p("down"), &l.down);
            match &l.mixer {
                Mixer::Attn(a) => {
                    w.put_linear(&p("qkv"), &a.qkv);
                    w.put_vec(&p("q_norm"), &a.q_norm);
                    w.put_vec(&p("k_norm"), &a.k_norm);
                    w.put_linear(&p("o"), &a.o);
                }
                Mixer::Gdn(g) => {
                    w.put_linear(&p("qkvz"), &g.qkvz);
                    w.put_linear(&p("ba"), &g.ba);
                    w.put_vec(&p("conv"), &g.conv);
                    w.put_vec(&p("dt_bias"), &g.dt_bias);
                    w.put_vec(&p("ssm_a"), &g.ssm_a);
                    w.put_vec(&p("ssm_norm"), &g.norm);
                    w.put_linear(&p("ssm_out"), &g.out);
                }
            }
        }
    }

    pub fn read_xq1(r: &super::format::XqReader) -> anyhow::Result<Self> {
        let cfg: Qwen35Config = r.config()?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = |n: &str| format!("blk.{i}.{n}");
                let mixer = if cfg.recurrent[i] {
                    Mixer::Gdn(GdnMixer {
                        qkvz: r.linear(&p("qkvz"))?,
                        ba: r.linear(&p("ba"))?,
                        conv: r.vec(&p("conv"), cfg.conv_dim() * cfg.conv_kernel)?,
                        dt_bias: r.vec(&p("dt_bias"), cfg.v_heads)?,
                        ssm_a: r.vec(&p("ssm_a"), cfg.v_heads)?,
                        norm: r.vec(&p("ssm_norm"), cfg.head_v_dim)?,
                        out: r.linear(&p("ssm_out"))?,
                    })
                } else {
                    Mixer::Attn(AttnMixer {
                        qkv: r.linear(&p("qkv"))?,
                        q_norm: r.vec(&p("q_norm"), hd)?,
                        k_norm: r.vec(&p("k_norm"), hd)?,
                        o: r.linear(&p("o"))?,
                    })
                };
                Ok(Qwen35Layer {
                    attn_norm: r.vec(&p("attn_norm"), h)?,
                    mixer,
                    post_norm: r.vec(&p("post_norm"), h)?,
                    gate_up: r.linear(&p("gate_up"))?,
                    down: r.linear(&p("down"))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let lm_head = if r.has("lm_head") {
            Some(r.linear("lm_head")?)
        } else {
            None
        };
        Self::new(
            cfg.clone(),
            r.embedding("embed")?,
            layers,
            r.vec("output_norm", h)?,
            lm_head,
        )
    }

    pub fn lm_head(&self) -> &Linear {
        self.lm_head.as_ref().unwrap_or(&self.embed.0)
    }

    /// Weight bytes streamed per decoded token.
    pub fn weight_bytes_per_token(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.gate_up.nbytes()
                    + l.down.nbytes()
                    + match &l.mixer {
                        Mixer::Attn(a) => a.qkv.nbytes() + a.o.nbytes(),
                        Mixer::Gdn(g) => g.qkvz.nbytes() + g.ba.nbytes() + g.out.nbytes(),
                    }
            })
            .sum::<usize>()
            + self.lm_head().nbytes()
    }

    pub fn new_qwen35_state(&self) -> Qwen35State {
        let c = &self.cfg;
        let mut kv = Vec::new();
        let mut conv = Vec::new();
        let mut ssm = Vec::new();
        for &rec in &c.recurrent {
            if rec {
                kv.push(None);
                conv.push(vec![0.0; (c.conv_kernel - 1) * c.conv_dim()]);
                ssm.push(vec![0.0; c.v_heads * c.head_v_dim * c.head_k_dim]);
            } else {
                kv.push(Some(KvCache::new(c.kv_heads, c.head_dim, c.context_length)));
                conv.push(Vec::new());
                ssm.push(Vec::new());
            }
        }
        Qwen35State {
            kv,
            conv,
            ssm,
            len: 0,
            ckpt: RecurrentCheckpoint::default(),
            s: Scratch::default(),
        }
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut Qwen35State,
    ) -> anyhow::Result<Vec<f32>> {
        let c = &self.cfg;
        let n = tokens.len();
        anyhow::ensure!(n > 0, "forward: no tokens");
        anyhow::ensure!(
            pos + n <= c.context_length,
            "forward: positions {pos}..{} exceed context {}",
            pos + n,
            c.context_length
        );
        // The recurrent state has no positions: it must be fed in order.
        anyhow::ensure!(
            pos == state.len,
            "forward: pos {pos} but the state holds {} positions",
            state.len
        );
        for kv in state.kv.iter_mut().flatten() {
            kv.reserve(pos + n, pos)?;
        }
        let (h, inter) = (c.hidden, c.intermediate);
        let mut x = std::mem::take(&mut state.s.x);
        x.resize(n * h, 0.0);
        self.embed.lookup(tokens, &mut x)?;
        self.rope.tables(pos, n, &mut state.s.cos, &mut state.s.sin);
        let mut hbuf = std::mem::take(&mut state.s.h);
        let mut proj = vec![0.0f32; n * h];

        for (i, l) in self.layers.iter().enumerate() {
            hbuf.resize(n * h, 0.0);
            hbuf.copy_from_slice(&x);
            rms_norm_rows(&mut hbuf, &l.attn_norm, c.eps);
            match &l.mixer {
                Mixer::Attn(a) => {
                    let kv = state.kv[i]
                        .as_mut()
                        .expect("attention layer has a KV cache");
                    a.forward(c, &self.rope, &hbuf, n, pos, kv, &mut state.s, &mut proj);
                }
                Mixer::Gdn(g) => {
                    g.forward(
                        c,
                        &hbuf,
                        n,
                        &mut state.conv[i],
                        &mut state.ssm[i],
                        &mut state.s,
                        &mut proj,
                    );
                }
            }
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);

            hbuf.copy_from_slice(&x);
            rms_norm_rows(&mut hbuf, &l.post_norm, c.eps);
            let s = &mut state.s;
            let gu = resize(&mut s.a, n * 2 * inter);
            l.gate_up.forward(&hbuf, n, &mut s.acts, gu);
            let act = resize(&mut s.b, n * inter);
            swiglu_rows(gu, act, inter);
            l.down.forward(act, n, &mut s.acts, &mut proj);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
        }

        hbuf.resize(h, 0.0);
        hbuf.copy_from_slice(&x[(n - 1) * h..]);
        rms_norm_rows(&mut hbuf, &self.output_norm, c.eps);
        let head = self.lm_head();
        let mut logits = vec![0.0f32; head.rows()];
        head.forward(&hbuf, 1, &mut state.s.acts, &mut logits);
        logits.truncate(c.vocab);
        state.s.x = x;
        state.s.h = hbuf;
        state.len = pos + n;
        Ok(logits)
    }
}

impl CausalLM for Qwen35Model {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(self.new_qwen35_state()))
    }

    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        let st = downcast_state::<Qwen35State>(state)?;
        self.forward_tokens(tokens, pos, st)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn max_seq_len(&self) -> usize {
        self.cfg.context_length
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.cfg.eos
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.cfg.bos
    }
}
