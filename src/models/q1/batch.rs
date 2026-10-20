//! Batched forward ([`BatchedCausalLM`]) for the Bonsai models: the next
//! tokens of several sequences in one pass over the weights.
//!
//! The rows of all items are concatenated (`[N][hidden]`, `N = Σ n_i`) for
//! everything that is row-wise: embedding, RMS norms, every linear layer
//! (one small-batch GEMM, `q1_0::gemm`), SwiGLU, residuals and
//! the LM head (last row of each item). Attention (each item against its own
//! KV cache, all decode items in one [`flash::attention_batch`] pass) and
//! the Gated DeltaNet recurrence (each item's own conv / delta state, items
//! in parallel) run per item.
//!
//! Every per-row operation is the one the single-sequence forward performs
//! (same functions, same arithmetic order), and the Q1 kernels and the
//! blocked attention are batch invariant, so each item's logits are
//! bit-identical to forwarding it alone.

use std::cell::RefCell;

use rayon::prelude::*;

use super::qwen3::{Qwen3Model, Qwen3State};
use super::qwen35::{self, Qwen35Model, Qwen35State};
use super::{KvCache, Linear, Rope, attention, flash, rms_norm_rows, swiglu_rows};
use crate::bit1::q1_0::Q8Acts;
use crate::models::batched::{BatchItem, BatchedCausalLM};
use crate::models::traits::{CausalLM, Logits, downcast_state};
use crate::nn::fast_attention_simd::rms_norm;

/// `out = W x` for `n` rows: the small-batch Q1 GEMM for decode batches
/// (`q1_0::gemm` reads the weights once for up to 16 rows), bit-identical
/// to the single-sequence forward.
fn linear(l: &Linear, x: &[f32], n: usize, acts: &mut Q8Acts, out: &mut [f32]) {
    l.forward(x, n, acts, out)
}

/// Reusable buffers of the batched forward (one set per calling thread).
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
    tc: Vec<f32>,
    ts: Vec<f32>,
    tg: Vec<f32>,
    acts: Q8Acts,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

fn resize(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    v.resize(n, 0.0);
    &mut v[..n]
}

/// Row layout of a batch: item `i` owns rows `off[i] .. off[i] + len[i]`.
struct Layout {
    off: Vec<usize>,
    len: Vec<usize>,
    pos: Vec<usize>,
    total: usize,
}

impl Layout {
    fn new(items: &[BatchItem<'_>], ctx: usize) -> anyhow::Result<Self> {
        let mut off = Vec::with_capacity(items.len());
        let mut total = 0;
        for it in items {
            let n = it.tokens.len();
            anyhow::ensure!(n > 0, "forward_batch: an item has no tokens");
            anyhow::ensure!(
                it.pos + n <= ctx,
                "forward_batch: positions {}..{} exceed context {ctx}",
                it.pos,
                it.pos + n
            );
            off.push(total);
            total += n;
        }
        Ok(Self {
            off,
            len: items.iter().map(|it| it.tokens.len()).collect(),
            pos: items.iter().map(|it| it.pos).collect(),
            total,
        })
    }

    /// Concatenated tokens.
    fn tokens(items: &[BatchItem<'_>]) -> Vec<u32> {
        items
            .iter()
            .flat_map(|it| it.tokens.iter().copied())
            .collect()
    }

    /// cos / sin tables of every row (each item at its own positions).
    fn rope_tables(&self, rope: &Rope, s: &mut Scratch) {
        let half = rope.rot_dim / 2;
        s.cos.clear();
        s.sin.clear();
        s.cos.reserve(self.total * half);
        s.sin.reserve(self.total * half);
        for (&p, &n) in self.pos.iter().zip(&self.len) {
            rope.tables(p, n, &mut s.tc, &mut s.ts);
            s.cos.extend_from_slice(&s.tc[..n * half]);
            s.sin.extend_from_slice(&s.ts[..n * half]);
        }
    }

    /// Split `buf` (`[total][width]`) into the items' row ranges.
    fn split_mut<'b>(&self, mut buf: &'b mut [f32], width: usize) -> Vec<&'b mut [f32]> {
        let mut out = Vec::with_capacity(self.len.len());
        for &n in &self.len {
            let (a, b) = buf.split_at_mut(n * width);
            out.push(a);
            buf = b;
        }
        out
    }

    /// Copy the last row of every item (`[B][width]`) into `dst`.
    fn last_rows(&self, src: &[f32], width: usize, dst: &mut Vec<f32>) {
        dst.clear();
        for (&o, &n) in self.off.iter().zip(&self.len) {
            dst.extend_from_slice(&src[(o + n - 1) * width..(o + n) * width]);
        }
    }
}

/// LM head over the last row of each item: `[B][vocab]` split per item.
fn lm_head(
    head: &Linear,
    norm: &[f32],
    eps: f32,
    vocab: usize,
    lay: &Layout,
    s: &mut Scratch,
    hidden: usize,
) -> Vec<Logits> {
    let b = lay.len.len();
    lay.last_rows(&s.x, hidden, &mut s.h);
    rms_norm_rows(&mut s.h, norm, eps);
    let rows = head.rows();
    let mut all = vec![0.0f32; b * rows];
    linear(head, &s.h, b, &mut s.acts, &mut all);
    all.chunks_exact(rows)
        .map(|r| r[..vocab].to_vec())
        .collect()
}

// ---------------------------------------------------------------------------
// Qwen3 (dense)
// ---------------------------------------------------------------------------

impl BatchedCausalLM for Qwen3Model {
    fn forward_batch(&self, items: &mut [BatchItem<'_>]) -> anyhow::Result<Vec<Logits>> {
        if items.len() == 1 {
            let it = &mut items[0];
            return Ok(vec![self.forward(it.tokens, it.pos, &mut *it.state)?]);
        }
        let c = &self.cfg;
        let lay = Layout::new(items, c.context_length)?;
        let tokens = Layout::tokens(items);
        let mut states: Vec<&mut Qwen3State> = items
            .iter_mut()
            .map(|it| downcast_state::<Qwen3State>(&mut *it.state))
            .collect::<anyhow::Result<_>>()?;
        for (st, (&p, &n)) in states.iter_mut().zip(lay.pos.iter().zip(&lay.len)) {
            for kv in st.kv_caches_mut() {
                kv.reserve(p + n, p)?;
            }
        }
        let rope = Rope::new(c.head_dim, c.rope_theta, c.yarn.as_ref());
        SCRATCH.with_borrow_mut(|s| -> anyhow::Result<Vec<Logits>> {
            let n = lay.total;
            let (h, hd, nh, nkv) = (c.hidden, c.head_dim, c.heads, c.kv_heads);
            let (qd, kvd) = (nh * hd, nkv * hd);
            let inter = c.intermediate;
            let half = hd / 2;
            let scale = 1.0 / (hd as f32).sqrt();
            let w = qd + 2 * kvd;
            resize(&mut s.x, n * h);
            self.embed.lookup(&tokens, &mut s.x)?;
            lay.rope_tables(&rope, s);

            for (li, l) in self.layers.iter().enumerate() {
                // --- attention ---
                let hbuf = resize(&mut s.h, n * h);
                hbuf.copy_from_slice(&s.x);
                rms_norm_rows(hbuf, &l.attn_norm, c.eps);
                let qkv = resize(&mut s.a, n * w);
                linear(&l.qkv, hbuf, n, &mut s.acts, qkv);
                let q = resize(&mut s.b, n * qd);
                // Per row: Q/K RMS norm per head, RoPE (as `forward_tokens`).
                let (cos, sin) = (&s.cos, &s.sin);
                let prep = |(t, (row, qo)): (usize, (&mut [f32], &mut [f32]))| {
                    let (cs, sn) = (
                        &cos[t * half..(t + 1) * half],
                        &sin[t * half..(t + 1) * half],
                    );
                    let (qr, kvr) = row.split_at_mut(qd);
                    for hq in qr.chunks_exact_mut(hd) {
                        crate::nn::fast_attention_simd::rms_norm(hq, &l.q_norm, c.eps);
                        rope.apply(hq, cs, sn);
                    }
                    for hk in kvr[..kvd].chunks_exact_mut(hd) {
                        crate::nn::fast_attention_simd::rms_norm(hk, &l.k_norm, c.eps);
                        rope.apply(hk, cs, sn);
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
                let att = resize(&mut s.d, n * qd);
                let qkv = &*qkv;
                let q = &*q;
                for (i, st) in states.iter_mut().enumerate() {
                    let kv = &mut st.kv_caches_mut()[li];
                    let (o, len, pos) = (lay.off[i], lay.len[i], lay.pos[i]);
                    for t in 0..len {
                        let row = &qkv[(o + t) * w..(o + t + 1) * w];
                        let (kr, vr) = row[qd..].split_at(kvd);
                        for (j, (hk, hv)) in
                            kr.chunks_exact(hd).zip(vr.chunks_exact(hd)).enumerate()
                        {
                            kv.store(j, pos + t, hk, hv);
                        }
                    }
                }
                let caches: Vec<&KvCache> = states
                    .iter_mut()
                    .map(|st| &st.kv_caches_mut()[li])
                    .collect();
                attend(&caches, &lay, q, att, nh, hd, scale);
                let proj = resize(&mut s.c, n * h);
                linear(&l.o, att, n, &mut s.acts, proj);
                for (a, b) in s.x.iter_mut().zip(proj.iter()) {
                    *a += b;
                }

                // --- MLP ---
                let hbuf = resize(&mut s.h, n * h);
                hbuf.copy_from_slice(&s.x);
                rms_norm_rows(hbuf, &l.ffn_norm, c.eps);
                let gu = resize(&mut s.a, n * 2 * inter);
                linear(&l.gate_up, hbuf, n, &mut s.acts, gu);
                let act = resize(&mut s.b, n * inter);
                swiglu_rows(gu, act, inter);
                let proj = resize(&mut s.c, n * h);
                linear(&l.down, act, n, &mut s.acts, proj);
                for (a, b) in s.x.iter_mut().zip(proj.iter()) {
                    *a += b;
                }
            }
            Ok(lm_head(
                self.lm_head(),
                &self.output_norm,
                c.eps,
                c.vocab,
                &lay,
                s,
                h,
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Qwen3.5 (hybrid: gated attention + Gated DeltaNet)
// ---------------------------------------------------------------------------

/// Causal attention of every item against its own cache: one
/// [`flash::attention_batch`] pass (bit-identical to [`attention`] per
/// item) when the head dim allows, else [`attention`] item by item.
fn attend(
    caches: &[&KvCache],
    lay: &Layout,
    q: &[f32],
    att: &mut [f32],
    heads: usize,
    hd: usize,
    scale: f32,
) {
    let qd = heads * hd;
    let outs = lay.split_mut(att, qd);
    if flash::supported(hd) {
        let mut calls: Vec<flash::BatchCall<'_>> = outs
            .into_iter()
            .enumerate()
            .map(|(i, out)| flash::BatchCall {
                q: &q[lay.off[i] * qd..(lay.off[i] + lay.len[i]) * qd],
                n: lay.len[i],
                pos: lay.pos[i],
                cache: caches[i],
                out,
            })
            .collect();
        flash::attention_batch(&mut calls, heads, scale);
    } else {
        // One after another: `attention` is parallel inside and keeps
        // per-thread scratch that must not be re-entered.
        for (i, out) in outs.into_iter().enumerate() {
            let (o, len) = (lay.off[i], lay.len[i]);
            attention(
                &q[o * qd..(o + len) * qd],
                len,
                lay.pos[i],
                heads,
                caches[i],
                scale,
                out,
            );
        }
    }
}

/// Per-item part of [`qwen35::AttnMixer`] before attention: Q/K norm +
/// RoPE of the item's `qkv` rows (`q` and the gate split out), K/V into its
/// cache. Same operations as `AttnMixer::forward`.
#[allow(clippy::too_many_arguments)]
fn attn_prep(
    a: &qwen35::AttnMixer,
    c: &qwen35::Qwen35Config,
    rope: &Rope,
    qkv: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n: usize,
    pos: usize,
    kv: &mut KvCache,
    q: &mut [f32],
    gate: &mut [f32],
) {
    let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let w = 2 * qd + 2 * kvd;
    let half = c.rot_dim / 2;
    for t in 0..n {
        let row = &mut qkv[t * w..(t + 1) * w];
        let (cs, sn) = (
            &cos[t * half..(t + 1) * half],
            &sin[t * half..(t + 1) * half],
        );
        let (qg, kvr) = row.split_at_mut(2 * qd);
        for h in 0..nh {
            let qh = &mut qg[2 * h * hd..(2 * h + 1) * hd];
            rms_norm(qh, &a.q_norm, c.eps);
            rope.apply(qh, cs, sn);
            q[(t * nh + h) * hd..(t * nh + h + 1) * hd].copy_from_slice(qh);
            gate[(t * nh + h) * hd..(t * nh + h + 1) * hd]
                .copy_from_slice(&qg[(2 * h + 1) * hd..(2 * h + 2) * hd]);
        }
        let (kr, vr) = kvr.split_at_mut(kvd);
        for (j, (hk, hv)) in kr.chunks_exact_mut(hd).zip(vr.chunks_exact(hd)).enumerate() {
            rms_norm(hk, &a.k_norm, c.eps);
            rope.apply(hk, cs, sn);
            kv.store(j, pos + t, hk, hv);
        }
    }
}

/// Per-item part of [`qwen35::GdnMixer`]: causal conv + SiLU + q/k L2 norm
/// over the item's `qkvz` rows and conv history, the gated delta rule on its
/// DeltaNet state, gated RMS norm into `o` (`[n][value_dim]`). Same
/// operations as `GdnMixer::forward` between its projections.
#[allow(clippy::too_many_arguments)]
fn gdn_item(
    g: &qwen35::GdnMixer,
    c: &qwen35::Qwen35Config,
    qkvz: &[f32],
    ba: &[f32],
    n: usize,
    conv_state: &mut [f32],
    ssm: &mut [f32],
    o: &mut [f32],
) {
    let (cd, vd) = (c.conv_dim(), c.value_dim());
    let (kd, kk) = (c.key_dim(), c.conv_kernel);
    let (nv, nk, hk, hv) = (c.v_heads, c.k_heads, c.head_k_dim, c.head_v_dim);
    let w = cd + vd;
    let mut conv_out = vec![0.0f32; n * cd];
    let hist = kk - 1;
    let qscale = 1.0 / (hk as f32).sqrt();
    let window = |r: usize| -> &[f32] {
        if r < hist {
            &conv_state[r * cd..(r + 1) * cd]
        } else {
            &qkvz[(r - hist) * w..(r - hist) * w + cd]
        }
    };
    let norm_qk = |row: &mut [f32]| {
        for h in 0..2 * nk {
            qwen35::l2_norm(&mut row[h * hk..(h + 1) * hk], c.eps);
        }
        row[..kd].iter_mut().for_each(|v| *v *= qscale);
    };
    if n >= qwen35::PAR_TOKENS {
        let taps: Vec<f32> = (0..kk)
            .flat_map(|j| (0..cd).map(move |ch| g.conv[ch * kk + j]))
            .collect();
        conv_out
            .par_chunks_exact_mut(cd)
            .enumerate()
            .for_each(|(t, o)| {
                o.fill(0.0);
                for j in 0..kk {
                    qwen35::conv_tap(o, &taps[j * cd..(j + 1) * cd], window(t + j));
                }
                for v in o.iter_mut() {
                    *v *= qwen35::sigmoid(*v);
                }
                norm_qk(o);
            });
    } else {
        for t in 0..n {
            let o = &mut conv_out[t * cd..(t + 1) * cd];
            o.fill(0.0);
            for j in 0..kk {
                let src = window(t + j);
                for ch in 0..cd {
                    o[ch] += g.conv[ch * kk + j] * src[ch];
                }
            }
            for v in o.iter_mut() {
                *v *= qwen35::sigmoid(*v);
            }
            norm_qk(o);
        }
    }
    let mut beta = vec![0.0f32; n * nv];
    let mut decay = vec![0.0f32; n * nv];
    for t in 0..n {
        for vh in 0..nv {
            beta[t * nv + vh] = qwen35::sigmoid(ba[t * 2 * nv + vh]);
            let alpha = ba[t * 2 * nv + nv + vh];
            decay[t * nv + vh] = (qwen35::softplus(alpha + g.dt_bias[vh]) * g.ssm_a[vh]).exp();
        }
    }
    let dims = qwen35::GdnDims::new(c);
    let mut tmp = vec![0.0f32; nv * n * hv]; // [head][t][hv]
    let rows: Vec<&[f32]> = (0..n).map(|t| &qkvz[t * w..t * w + cd]).collect();
    qwen35::update_conv_history(conv_state, &rows, cd);
    qwen35::gdn_heads(&dims, ssm, &conv_out, &beta, &decay, n, Some(&mut tmp));
    for (t, ot) in o.chunks_exact_mut(vd).enumerate() {
        let z = &qkvz[t * w + cd..(t + 1) * w];
        for vh in 0..nv {
            let oo = &mut ot[vh * hv..(vh + 1) * hv];
            oo.copy_from_slice(&tmp[(vh * n + t) * hv..(vh * n + t + 1) * hv]);
            rms_norm(oo, &g.norm, c.eps);
            for (a, &zz) in oo.iter_mut().zip(&z[vh * hv..(vh + 1) * hv]) {
                *a *= zz * qwen35::sigmoid(zz);
            }
        }
    }
}

/// Mutable views of one item's [`Qwen35State`].
type Parts<'a> = (
    &'a mut [Option<KvCache>],
    &'a mut [Vec<f32>],
    &'a mut [Vec<f32>],
    &'a mut usize,
);

impl BatchedCausalLM for Qwen35Model {
    fn forward_batch(&self, items: &mut [BatchItem<'_>]) -> anyhow::Result<Vec<Logits>> {
        if items.len() == 1 {
            let it = &mut items[0];
            return Ok(vec![self.forward(it.tokens, it.pos, &mut *it.state)?]);
        }
        let c = &self.cfg;
        let lay = Layout::new(items, c.context_length)?;
        let tokens = Layout::tokens(items);
        let mut parts: Vec<Parts<'_>> = items
            .iter_mut()
            .map(|it| downcast_state::<Qwen35State>(&mut *it.state)?.batch_parts())
            .collect::<anyhow::Result<_>>()?;
        for (p, (&pos, &n)) in parts.iter_mut().zip(lay.pos.iter().zip(&lay.len)) {
            // The recurrent state has no positions: it must be fed in order.
            anyhow::ensure!(
                pos == *p.3,
                "forward: pos {pos} but the state holds {} positions",
                *p.3
            );
            for kv in p.0.iter_mut().flatten() {
                kv.reserve(pos + n, pos)?;
            }
        }
        let rope = Rope::new(c.rot_dim, c.rope_theta, c.yarn.as_ref());
        let logits = SCRATCH.with_borrow_mut(|s| -> anyhow::Result<Vec<Logits>> {
            let n = lay.total;
            let (h, inter) = (c.hidden, c.intermediate);
            let half = c.rot_dim / 2;
            resize(&mut s.x, n * h);
            self.embed.lookup(&tokens, &mut s.x)?;
            lay.rope_tables(&rope, s);
            for (li, l) in self.layers.iter().enumerate() {
                let hbuf = resize(&mut s.h, n * h);
                hbuf.copy_from_slice(&s.x);
                rms_norm_rows(hbuf, &l.attn_norm, c.eps);
                let proj = match &l.mixer {
                    qwen35::Mixer::Attn(a) => {
                        let (qd, kvd) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
                        let w = 2 * qd + 2 * kvd;
                        let qkv = resize(&mut s.a, n * w);
                        linear(&a.qkv, hbuf, n, &mut s.acts, qkv);
                        let q = resize(&mut s.b, n * qd);
                        let gate = resize(&mut s.tg, n * qd);
                        let (cos, sin) = (&s.cos, &s.sin);
                        parts
                            .par_iter_mut()
                            .zip(lay.split_mut(qkv, w))
                            .zip(lay.split_mut(q, qd).into_par_iter())
                            .zip(lay.split_mut(gate, qd).into_par_iter())
                            .enumerate()
                            .for_each(|(i, (((p, qkv), q), gate))| {
                                let (o, len) = (lay.off[i], lay.len[i]);
                                let kv = p.0[li].as_mut().expect("attention layer has a KV cache");
                                attn_prep(
                                    a,
                                    c,
                                    &rope,
                                    qkv,
                                    &cos[o * half..(o + len) * half],
                                    &sin[o * half..(o + len) * half],
                                    len,
                                    lay.pos[i],
                                    kv,
                                    q,
                                    gate,
                                );
                            });
                        let att = resize(&mut s.d, n * qd);
                        let caches: Vec<&KvCache> = parts
                            .iter()
                            .map(|p| p.0[li].as_ref().expect("attention layer has a KV cache"))
                            .collect();
                        attend(
                            &caches,
                            &lay,
                            q,
                            att,
                            c.heads,
                            c.head_dim,
                            1.0 / (c.head_dim as f32).sqrt(),
                        );
                        let apply_gate = |(a, g): (&mut f32, &f32)| *a *= qwen35::sigmoid(*g);
                        if n >= qwen35::PAR_TOKENS {
                            att.par_iter_mut().zip(gate.par_iter()).for_each(apply_gate);
                        } else {
                            att.iter_mut().zip(gate.iter()).for_each(apply_gate);
                        }
                        let proj = resize(&mut s.c, n * h);
                        linear(&a.o, att, n, &mut s.acts, proj);
                        proj
                    }
                    qwen35::Mixer::Gdn(g) => {
                        let (cd, vd, nv) = (c.conv_dim(), c.value_dim(), c.v_heads);
                        let w = cd + vd;
                        let qkvz = resize(&mut s.a, n * w);
                        linear(&g.qkvz, hbuf, n, &mut s.acts, qkvz);
                        let ba = resize(&mut s.b, n * 2 * nv);
                        linear(&g.ba, hbuf, n, &mut s.acts, ba);
                        let o = resize(&mut s.d, n * vd);
                        let (qkvz, ba) = (&*qkvz, &*ba);
                        parts
                            .par_iter_mut()
                            .zip(lay.split_mut(o, vd))
                            .enumerate()
                            .for_each(|(i, (p, o))| {
                                let (off, len) = (lay.off[i], lay.len[i]);
                                gdn_item(
                                    g,
                                    c,
                                    &qkvz[off * w..(off + len) * w],
                                    &ba[off * 2 * nv..(off + len) * 2 * nv],
                                    len,
                                    &mut p.1[li],
                                    &mut p.2[li],
                                    o,
                                );
                            });
                        let proj = resize(&mut s.c, n * h);
                        linear(&g.out, o, n, &mut s.acts, proj);
                        proj
                    }
                };
                s.x.iter_mut().zip(proj.iter()).for_each(|(a, b)| *a += b);

                let hbuf = resize(&mut s.h, n * h);
                hbuf.copy_from_slice(&s.x);
                rms_norm_rows(hbuf, &l.post_norm, c.eps);
                let gu = resize(&mut s.a, n * 2 * inter);
                linear(&l.gate_up, hbuf, n, &mut s.acts, gu);
                let act = resize(&mut s.b, n * inter);
                swiglu_rows(gu, act, inter);
                let proj = resize(&mut s.c, n * h);
                linear(&l.down, act, n, &mut s.acts, proj);
                s.x.iter_mut().zip(proj.iter()).for_each(|(a, b)| *a += b);
            }
            Ok(lm_head(
                self.lm_head(),
                &self.output_norm,
                c.eps,
                c.vocab,
                &lay,
                s,
                h,
            ))
        })?;
        for (p, (&pos, &n)) in parts.iter_mut().zip(lay.pos.iter().zip(&lay.len)) {
            *p.3 = pos + n;
        }
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a Bonsai model (`.gguf` or `.xq1`) as a [`BatchedCausalLM`] and a
/// [`crate::Generator`] over the same `Arc` (as
/// [`super::load_generator`] does, tokenizer from `tokenizer_json` or the
/// model file).
pub fn load_batched(
    path: impl AsRef<std::path::Path>,
    tokenizer_json: Option<&std::path::Path>,
) -> anyhow::Result<(std::sync::Arc<dyn BatchedCausalLM>, crate::Generator)> {
    use std::sync::Arc;
    let path = path.as_ref();
    let (model, mut assets): (Arc<dyn BatchedCausalLM>, _) = if super::format::is_xq1(path)? {
        let r = super::format::XqReader::open(path)?;
        let m: Arc<dyn BatchedCausalLM> = match r.arch() {
            "qwen3" => Arc::new(Qwen3Model::read_xq1(&r)?),
            "qwen35" => Arc::new(Qwen35Model::read_xq1(&r)?),
            other => anyhow::bail!(".xq1: unsupported arch {other}"),
        };
        (m, r.assets()?)
    } else {
        let g = crate::loader::gguf::GgufFile::open(path)?;
        let m: Arc<dyn BatchedCausalLM> = match g.architecture()?.as_str() {
            "qwen3" => Arc::new(Qwen3Model::from_gguf(&g)?),
            "qwen35" => Arc::new(Qwen35Model::load_gguf(&g)?),
            other => anyhow::bail!("unsupported GGUF architecture {other:?}"),
        };
        (m, super::ModelAssets::from_gguf(&g)?)
    };
    if let Some(p) = tokenizer_json {
        assets.tokenizer = tokenizers::Tokenizer::from_file(p)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", p.display()))?;
    }
    let base: Arc<dyn CausalLM> = model.clone();
    Ok((model, super::make_generator(base, assets)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::q1::qwen3::{Qwen3Config, Qwen3Layer};
    use crate::models::traits::ModelState;

    pub(crate) fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Random Q1 linear `[rows, cols]` (block 32).
    pub(crate) fn q1(rows: usize, cols: usize, rnd: &mut impl FnMut() -> u64) -> Linear {
        use crate::bit1::q1_0::{ByteBuf, Q1Weight};
        let bits: Vec<u8> = (0..rows * cols / 8).map(|_| rnd() as u8).collect();
        let scales: Vec<u8> = (0..rows * cols / 32)
            .flat_map(|_| {
                half::f16::from_f32(0.02 + (rnd() % 64) as f32 * 1e-3)
                    .to_le_bytes()
                    .into_iter()
            })
            .collect();
        Linear::Q1(
            Q1Weight::new(
                rows,
                cols,
                32,
                ByteBuf::from_vec(bits),
                ByteBuf::from_vec(scales),
            )
            .unwrap(),
        )
    }

    pub(crate) fn vecf(n: usize, base: f32, rnd: &mut impl FnMut() -> u64) -> Vec<f32> {
        (0..n)
            .map(|_| base + ((rnd() % 1000) as f32 / 1000.0 - 0.5) * 0.4)
            .collect()
    }

    fn tiny_qwen3() -> Qwen3Model {
        let mut r = rng(77);
        let (h, hd, nh, nkv, inter, vocab, layers) = (128, 64, 4, 2, 256, 300, 3);
        let cfg = Qwen3Config {
            hidden: h,
            intermediate: inter,
            layers,
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            eps: 1e-6,
            rope_theta: 10_000.0,
            yarn: None,
            context_length: 512,
            vocab,
            bos: None,
            eos: vec![0],
        };
        let layers = (0..layers)
            .map(|_| Qwen3Layer {
                attn_norm: vecf(h, 1.0, &mut r),
                qkv: q1((nh + 2 * nkv) * hd, h, &mut r),
                q_norm: vecf(hd, 1.0, &mut r),
                k_norm: vecf(hd, 1.0, &mut r),
                o: q1(h, nh * hd, &mut r),
                ffn_norm: vecf(h, 1.0, &mut r),
                gate_up: q1(2 * inter, h, &mut r),
                down: q1(h, inter, &mut r),
            })
            .collect();
        let embed = super::super::Embedding(q1(vocab + 20, h, &mut r));
        Qwen3Model::new(cfg, embed, layers, vecf(h, 1.0, &mut r), None).unwrap()
    }

    /// Several sequences at different positions, mixing prefill chunks and
    /// single-token steps, batched vs alone: bit-identical logits and
    /// states (checked by continuing to decode).
    pub(crate) fn check_batch_exact<M: BatchedCausalLM>(m: &M) {
        let vocab = m.vocab_size() as u32;
        let mut r = rng(5);
        // (prompt length, steps): each step feeds `chunk` tokens.
        let plans: [(usize, &[usize]); 5] = [
            (5, &[1, 1, 3, 1, 1, 1]),
            (17, &[1, 1, 1, 1, 9, 1]),
            (1, &[2, 1, 1, 1, 1, 1]),
            (40, &[1, 1, 1, 1, 1, 1]),
            (9, &[1, 16, 1, 1, 1, 1]),
        ];
        let seqs: Vec<Vec<u32>> = plans
            .iter()
            .map(|(p, steps)| {
                let total = p + steps.iter().sum::<usize>();
                (0..total).map(|_| (r() % vocab as u64) as u32).collect()
            })
            .collect();
        let mut solo: Vec<Box<dyn ModelState>> =
            plans.iter().map(|_| m.new_state().unwrap()).collect();
        let mut bat: Vec<Box<dyn ModelState>> =
            plans.iter().map(|_| m.new_state().unwrap()).collect();
        let mut fed = vec![0usize; plans.len()];
        // Step 0 feeds the prompts; step k the k-th chunk. Items join at
        // different steps (sequence i starts at step i % 3).
        for step in 0..(plans[0].1.len() + 3) {
            let mut want = Vec::new();
            let mut idx = Vec::new();
            let mut spans = Vec::new();
            for (i, (p, steps)) in plans.iter().enumerate() {
                let k = match step.checked_sub(i % 3) {
                    Some(k) if k <= steps.len() => k,
                    _ => continue,
                };
                let len = if k == 0 { *p } else { steps[k - 1] };
                let toks = &seqs[i][fed[i]..fed[i] + len];
                want.push(m.forward(toks, fed[i], solo[i].as_mut()).unwrap());
                idx.push(i);
                spans.push((fed[i], len));
                fed[i] += len;
            }
            if idx.is_empty() {
                continue;
            }
            let mut items: Vec<BatchItem<'_>> = Vec::new();
            let mut rest: &mut [Box<dyn ModelState>] = &mut bat;
            let mut base = 0;
            for (&i, &(p, len)) in idx.iter().zip(&spans) {
                let (_, tail) = rest.split_at_mut(i - base);
                let (one, tail) = tail.split_at_mut(1);
                rest = tail;
                base = i + 1;
                items.push(BatchItem {
                    tokens: &seqs[i][p..p + len],
                    pos: p,
                    state: one[0].as_mut(),
                });
            }
            let got = m.forward_batch(&mut items).unwrap();
            assert_eq!(got.len(), want.len());
            for ((g, w), &i) in got.iter().zip(&want).zip(&idx) {
                assert_eq!(g.len(), w.len());
                let same = g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "step {step} seq {i}: batched logits differ");
            }
        }
    }

    #[test]
    fn qwen3_batch_matches_single() {
        check_batch_exact(&tiny_qwen3());
    }

    fn tiny_qwen35() -> Qwen35Model {
        use crate::models::q1::qwen35::{AttnMixer, GdnMixer, Mixer, Qwen35Config, Qwen35Layer};
        let mut r = rng(91);
        let (h, inter, vocab) = (128, 256, 300);
        let (nh, nkv, hd) = (4, 2, 64);
        let (nk, nv, hk) = (2, 4, 32);
        let recurrent = vec![true, true, false, true, false];
        let cfg = Qwen35Config {
            hidden: h,
            intermediate: inter,
            layers: recurrent.len(),
            recurrent: recurrent.clone(),
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            rot_dim: 32,
            eps: 1e-6,
            rope_theta: 10_000.0,
            yarn: None,
            context_length: 512,
            vocab,
            bos: None,
            eos: vec![0],
            conv_kernel: 4,
            k_heads: nk,
            v_heads: nv,
            head_k_dim: hk,
            head_v_dim: hk,
        };
        let (cd, vd) = (cfg.conv_dim(), cfg.value_dim());
        let layers = recurrent
            .iter()
            .map(|&rec| {
                let mixer = if rec {
                    Mixer::Gdn(GdnMixer {
                        qkvz: q1(cd + vd, h, &mut r),
                        ba: q1(2 * nv, h, &mut r),
                        conv: vecf(cd * 4, 0.0, &mut r),
                        dt_bias: vecf(nv, 0.0, &mut r),
                        ssm_a: vecf(nv, -1.0, &mut r),
                        norm: vecf(hk, 1.0, &mut r),
                        out: q1(h, vd, &mut r),
                    })
                } else {
                    Mixer::Attn(AttnMixer {
                        qkv: q1(2 * nh * hd + 2 * nkv * hd, h, &mut r),
                        q_norm: vecf(hd, 1.0, &mut r),
                        k_norm: vecf(hd, 1.0, &mut r),
                        o: q1(h, nh * hd, &mut r),
                    })
                };
                Qwen35Layer {
                    attn_norm: vecf(h, 1.0, &mut r),
                    mixer,
                    post_norm: vecf(h, 1.0, &mut r),
                    gate_up: q1(2 * inter, h, &mut r),
                    down: q1(h, inter, &mut r),
                }
            })
            .collect();
        let embed = super::super::Embedding(q1(vocab + 4, h, &mut r));
        Qwen35Model::new(cfg, embed, layers, vecf(h, 1.0, &mut r), None).unwrap()
    }

    #[test]
    fn qwen35_batch_matches_single() {
        check_batch_exact(&tiny_qwen35());
    }
}
