//! Dense Qwen3 (Bonsai 1.7B / 4B / 8B) on the 1-bit fast path.
//!
//! Per layer: `x += o(attn(rope(k_norm/q_norm(qkv(rms(x))))))`,
//! `x += down(silu(gate(h)) * up(h))` with `h = rms(x)`. Q/K/V and gate/up
//! are fused into one GEMV each at load time. Q/K RMS norms are applied per
//! head *before* RoPE (as in HF `Qwen3Attention`).

use crate::bit1::q1_0::Q8Acts;
use crate::loader::gguf::GgufFile;
use crate::models::traits::{CausalLM, Logits, ModelState, downcast_state};
use rayon::prelude::*;

use super::{
    Embedding, KvCache, Linear, Rope, Yarn, attention, gguf_vec, meta_f32, meta_u32, rms_norm_rows,
    special_tokens_from_gguf, swiglu_rows, yarn_from_gguf,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Qwen3Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub yarn: Option<Yarn>,
    pub context_length: usize,
    pub vocab: usize,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
}

impl Qwen3Config {
    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        let arch = g.architecture()?;
        let k = |s: &str| format!("{arch}.{s}");
        let hidden = meta_u32(g, &k("embedding_length"))?;
        let heads = meta_u32(g, &k("attention.head_count"))?;
        let head_dim = meta_u32(g, &k("attention.key_length")).unwrap_or(hidden / heads);
        let (bos, eos) = special_tokens_from_gguf(g);
        let vocab = g.tensor_info("token_embd.weight")?.shape[0];
        Ok(Self {
            hidden,
            intermediate: meta_u32(g, &k("feed_forward_length"))?,
            layers: meta_u32(g, &k("block_count"))?,
            heads,
            kv_heads: meta_u32(g, &k("attention.head_count_kv"))?,
            head_dim,
            eps: meta_f32(g, &k("attention.layer_norm_rms_epsilon"))
                .ok_or_else(|| anyhow::anyhow!("missing rms epsilon"))?,
            rope_theta: meta_f32(g, &k("rope.freq_base")).unwrap_or(10_000.0),
            yarn: yarn_from_gguf(g, &arch),
            context_length: meta_u32(g, &k("context_length"))?,
            vocab,
            bos,
            eos,
        })
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.heads > 0 && self.kv_heads > 0 && self.heads.is_multiple_of(self.kv_heads),
            "heads {} not a multiple of kv heads {}",
            self.heads,
            self.kv_heads
        );
        anyhow::ensure!(self.head_dim.is_multiple_of(2), "odd head_dim");
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3Layer {
    pub attn_norm: Vec<f32>,
    /// Rows: `[q (heads*hd); k (kv*hd); v (kv*hd)]`.
    pub qkv: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub o: Linear,
    pub ffn_norm: Vec<f32>,
    /// Rows: `[gate (inter); up (inter)]`.
    pub gate_up: Linear,
    pub down: Linear,
}

#[derive(Clone, Debug)]
pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    pub embed: Embedding,
    pub layers: Vec<Qwen3Layer>,
    pub output_norm: Vec<f32>,
    /// `None` = tied to the embedding table.
    pub lm_head: Option<Linear>,
    rope: Rope,
}

/// Per-sequence state: KV caches plus reusable scratch buffers.
pub struct Qwen3State {
    kv: Vec<KvCache>,
    s: Scratch,
}

#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    gu: Vec<f32>,
    act: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    acts: Q8Acts,
}

impl ModelState for Qwen3State {
    fn reset(&mut self) -> anyhow::Result<()> {
        // Positions are overwritten before they are read.
        Ok(())
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Qwen3State {
    /// Bytes currently held by the KV caches.
    pub fn kv_bytes(&self) -> usize {
        self.kv.iter().map(|c| c.bytes()).sum()
    }
}

fn resize(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    v.resize(n, 0.0);
    &mut v[..n]
}

impl Qwen3Model {
    pub fn new(
        cfg: Qwen3Config,
        embed: Embedding,
        layers: Vec<Qwen3Layer>,
        output_norm: Vec<f32>,
        lm_head: Option<Linear>,
    ) -> anyhow::Result<Self> {
        cfg.validate()?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let qkv_rows = (cfg.heads + 2 * cfg.kv_heads) * hd;
        anyhow::ensure!(embed.dim() == h, "embedding dim");
        anyhow::ensure!(layers.len() == cfg.layers, "layer count");
        anyhow::ensure!(output_norm.len() == h, "output norm");
        for (i, l) in layers.iter().enumerate() {
            let ok = l.attn_norm.len() == h
                && l.ffn_norm.len() == h
                && l.q_norm.len() == hd
                && l.k_norm.len() == hd
                && (l.qkv.rows(), l.qkv.cols()) == (qkv_rows, h)
                && (l.o.rows(), l.o.cols()) == (h, cfg.heads * hd)
                && (l.gate_up.rows(), l.gate_up.cols()) == (2 * cfg.intermediate, h)
                && (l.down.rows(), l.down.cols()) == (h, cfg.intermediate);
            anyhow::ensure!(ok, "layer {i}: unexpected tensor shapes");
        }
        let head = lm_head.as_ref().unwrap_or(&embed.0);
        anyhow::ensure!(
            head.cols() == h && head.rows() >= cfg.vocab,
            "lm_head shape"
        );
        let rope = Rope::new(hd, cfg.rope_theta, cfg.yarn.as_ref());
        Ok(Self {
            cfg,
            embed,
            layers,
            output_norm,
            lm_head,
            rope,
        })
    }

    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Self> {
        let cfg = Qwen3Config::from_gguf(g)?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let embed = Embedding(Linear::from_gguf(g, "token_embd.weight")?);
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = |n: &str| format!("blk.{i}.{n}");
                Ok(Qwen3Layer {
                    attn_norm: gguf_vec(g, &p("attn_norm.weight"), h)?,
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
                    ffn_norm: gguf_vec(g, &p("ffn_norm.weight"), h)?,
                    gate_up: Linear::from_gguf_concat(
                        g,
                        &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                    )?,
                    down: Linear::from_gguf(g, &p("ffn_down.weight"))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let output_norm = gguf_vec(g, "output_norm.weight", h)?;
        let lm_head = if g.has_tensor("output.weight") {
            Some(Linear::from_gguf(g, "output.weight")?)
        } else {
            None
        };
        Self::new(cfg, embed, layers, output_norm, lm_head)
    }

    /// Serialize all tensors into an `.xq1` writer (see [`super::format`]).
    pub fn write_xq1(&self, w: &mut impl super::format::TensorSink) {
        w.put_linear("embed", &self.embed.0);
        w.put_vec("output_norm", &self.output_norm);
        if let Some(h) = &self.lm_head {
            w.put_linear("lm_head", h);
        }
        for (i, l) in self.layers.iter().enumerate() {
            let p = |n: &str| format!("blk.{i}.{n}");
            w.put_vec(&p("attn_norm"), &l.attn_norm);
            w.put_linear(&p("qkv"), &l.qkv);
            w.put_vec(&p("q_norm"), &l.q_norm);
            w.put_vec(&p("k_norm"), &l.k_norm);
            w.put_linear(&p("o"), &l.o);
            w.put_vec(&p("ffn_norm"), &l.ffn_norm);
            w.put_linear(&p("gate_up"), &l.gate_up);
            w.put_linear(&p("down"), &l.down);
        }
    }

    /// Load from a mapped `.xq1` file (1-bit weights are zero-copy).
    pub fn read_xq1(r: &super::format::XqReader) -> anyhow::Result<Self> {
        let cfg: Qwen3Config = r.config()?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = |n: &str| format!("blk.{i}.{n}");
                Ok(Qwen3Layer {
                    attn_norm: r.vec(&p("attn_norm"), h)?,
                    qkv: r.linear(&p("qkv"))?,
                    q_norm: r.vec(&p("q_norm"), hd)?,
                    k_norm: r.vec(&p("k_norm"), hd)?,
                    o: r.linear(&p("o"))?,
                    ffn_norm: r.vec(&p("ffn_norm"), h)?,
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
            .map(|l| l.qkv.nbytes() + l.o.nbytes() + l.gate_up.nbytes() + l.down.nbytes())
            .sum::<usize>()
            + self.lm_head().nbytes()
    }

    pub fn new_qwen3_state(&self) -> Qwen3State {
        Qwen3State {
            kv: (0..self.cfg.layers)
                .map(|_| {
                    KvCache::new(
                        self.cfg.kv_heads,
                        self.cfg.head_dim,
                        self.cfg.context_length,
                    )
                })
                .collect(),
            s: Scratch::default(),
        }
    }

    /// Run `tokens` at positions `pos..`; returns the last position's logits.
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut Qwen3State,
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
        for kv in state.kv.iter_mut() {
            kv.reserve(pos + n, pos)?;
        }
        let (h, hd, nh, nkv) = (c.hidden, c.head_dim, c.heads, c.kv_heads);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let inter = c.intermediate;
        let s = &mut state.s;

        let x = resize(&mut s.x, n * h);
        self.embed.lookup(tokens, x)?;
        self.rope.tables(pos, n, &mut s.cos, &mut s.sin);
        let half = hd / 2;
        let scale = 1.0 / (hd as f32).sqrt();

        for (l, kv) in self.layers.iter().zip(state.kv.iter_mut()) {
            // --- attention ---
            let hbuf = resize(&mut s.h, n * h);
            hbuf.copy_from_slice(x);
            rms_norm_rows(hbuf, &l.attn_norm, c.eps);
            let qkv = resize(&mut s.qkv, n * (qd + 2 * kvd));
            l.qkv.forward(hbuf, n, &mut s.acts, qkv);
            let q = resize(&mut s.q, n * qd);
            // Per token (parallel for prefill): Q/K RMS norm per head, RoPE.
            let w = qd + 2 * kvd;
            let (cos, sin) = (&s.cos, &s.sin);
            let prep = |(t, (row, qo)): (usize, (&mut [f32], &mut [f32]))| {
                let (cs, sn) = (
                    &cos[t * half..(t + 1) * half],
                    &sin[t * half..(t + 1) * half],
                );
                let (qr, kvr) = row.split_at_mut(qd);
                for hq in qr.chunks_exact_mut(hd) {
                    crate::nn::fast_attention_simd::rms_norm(hq, &l.q_norm, c.eps);
                    self.rope.apply(hq, cs, sn);
                }
                for hk in kvr[..kvd].chunks_exact_mut(hd) {
                    crate::nn::fast_attention_simd::rms_norm(hk, &l.k_norm, c.eps);
                    self.rope.apply(hk, cs, sn);
                }
                qo.copy_from_slice(qr);
            };
            if n >= 8 {
                qkv.par_chunks_exact_mut(w)
                    .zip(q.par_chunks_exact_mut(qd))
                    .enumerate()
                    .for_each(prep);
            } else {
                qkv.chunks_exact_mut(w)
                    .zip(q.chunks_exact_mut(qd))
                    .enumerate()
                    .for_each(prep);
            }
            for (t, row) in qkv.chunks_exact(w).enumerate() {
                let (kr, vr) = row[qd..].split_at(kvd);
                for (j, (hk, hv)) in kr.chunks_exact(hd).zip(vr.chunks_exact(hd)).enumerate() {
                    kv.store(j, pos + t, hk, hv);
                }
            }
            let att = resize(&mut s.att, n * qd);
            attention(q, n, pos, nh, kv, scale, att);
            let proj = resize(&mut s.proj, n * h);
            l.o.forward(att, n, &mut s.acts, proj);
            for (a, b) in x.iter_mut().zip(proj.iter()) {
                *a += b;
            }

            // --- MLP ---
            let hbuf = resize(&mut s.h, n * h);
            hbuf.copy_from_slice(x);
            rms_norm_rows(hbuf, &l.ffn_norm, c.eps);
            let gu = resize(&mut s.gu, n * 2 * inter);
            l.gate_up.forward(hbuf, n, &mut s.acts, gu);
            let act = resize(&mut s.act, n * inter);
            swiglu_rows(gu, act, inter);
            let proj = resize(&mut s.proj, n * h);
            l.down.forward(act, n, &mut s.acts, proj);
            for (a, b) in x.iter_mut().zip(proj.iter()) {
                *a += b;
            }
        }

        let last = resize(&mut s.h, h);
        last.copy_from_slice(&x[(n - 1) * h..]);
        rms_norm_rows(last, &self.output_norm, c.eps);
        let head = self.lm_head();
        let mut logits = vec![0.0f32; head.rows()];
        head.forward(last, 1, &mut s.acts, &mut logits);
        logits.truncate(c.vocab);
        Ok(logits)
    }
}

impl CausalLM for Qwen3Model {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(self.new_qwen3_state()))
    }

    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        let st = downcast_state::<Qwen3State>(state)?;
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
