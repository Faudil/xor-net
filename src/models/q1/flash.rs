//! Blocked causal attention, batch-invariant: every query row is computed
//! with the same operations whether it comes alone (decode), in a chunk or
//! in a whole prompt.
//!
//! A row's keys are split at multiples of [`SC`] into super-chunks; each is
//! reduced to an online-softmax partial over [`KC`]-key chunks and the
//! partials are merged in order. Within a chunk, `S = Q Kᵀ` and `O += P V`
//! are register-blocked matrix products (4 query rows × 64 keys / 64 dims),
//! each element a fused multiply-add chain in a fixed order.
//!
//! Prefill items are (block of [`QB`] query rows, KV head): the `QB · group`
//! query vectors of a KV head share every converted K/V chunk (converted to
//! f32 once per call). Decode items are (KV head, super-chunk), so long
//! contexts still use all threads.

use rayon::prelude::*;

use super::KvCache;
use crate::nn::fast_attention_simd::f16_to_f32_slice;
use crate::util::{cpu_features, with_best_simd};

/// Query rows per work item.
pub const QB: usize = 16;
/// Keys per chunk (also the width of the score block).
pub const KC: usize = 64;
/// Head-dimension block of the `P V` product.
const DC: usize = 64;

/// Whether [`attention`] handles this shape (`head_dim` a multiple of
/// [`DC`], at most [`MAX_HD`]).
pub fn supported(head_dim: usize) -> bool {
    head_dim > 0 && head_dim.is_multiple_of(DC) && head_dim <= MAX_HD
}

/// Largest supported head dimension.
pub const MAX_HD: usize = 512;

#[derive(Default)]
struct Scratch {
    q: Vec<f32>,
    s: Vec<f32>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: items write disjoint (row, head) ranges of the output.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// `e^x` to ~2 ulp for `-87 ≤ x ≤ 88` (clamped to that range), written so loops
/// over it vectorize: `x = n ln2 + r`, a degree-6 polynomial for `e^r` and
/// the exponent bits for `2^n`.
#[inline(always)]
pub fn exp_approx(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668 + r * (0.008_333_452 + r * 0.001_388_888_9)))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `s[r][0..KC] = q[r] · kt[.][0..KC]` for `rp` (multiple of 4) rows.
#[inline(always)]
fn qk<const FMA: bool>(q: &[f32], kt: &[f32], s: &mut [f32], rp: usize, hd: usize) {
    let kt = &kt[..hd * KC];
    for (qr, sr) in q
        .chunks_exact(4 * hd)
        .zip(s.as_chunks_mut::<{ 4 * KC }>().0.iter_mut())
        .take(rp / 4)
    {
        let qrow: [&[f32]; 4] = std::array::from_fn(|i| &qr[i * hd..(i + 1) * hd]);
        let mut acc = [[0.0f32; KC]; 4];
        for (d, k) in kt.as_chunks::<KC>().0.iter().enumerate() {
            for (a, qi) in acc.iter_mut().zip(&qrow) {
                let qv = qi[d];
                for (x, &kv) in a.iter_mut().zip(k) {
                    *x = madd::<FMA>(qv, kv, *x);
                }
            }
        }
        for (a, o) in acc.iter().zip(sr.as_chunks_mut::<KC>().0.iter_mut()) {
            *o = *a;
        }
    }
}

/// `o[r] += Σ_{j < kc} p[r][j] · v[j]` for `rp` (multiple of 4) rows.
#[inline(always)]
fn pv<const FMA: bool>(p: &[f32], v: &[f32], o: &mut [f32], rp: usize, hd: usize, kc: usize) {
    assert!(kc <= KC);
    let v = &v[..kc * hd];
    for (pr, or) in p
        .as_chunks::<{ 4 * KC }>()
        .0
        .iter()
        .zip(o.chunks_exact_mut(4 * hd))
        .take(rp / 4)
    {
        let prow: [&[f32; KC]; 4] =
            std::array::from_fn(|i| pr[i * KC..(i + 1) * KC].try_into().unwrap());
        for dc in (0..hd).step_by(DC) {
            let mut acc = [[0.0f32; DC]; 4];
            for (i, a) in acc.iter_mut().enumerate() {
                a.copy_from_slice(&or[i * hd + dc..i * hd + dc + DC]);
            }
            for (j, vrow) in v.chunks_exact(hd).enumerate() {
                let vv: &[f32; DC] = vrow[dc..dc + DC].try_into().unwrap();
                for (a, pi) in acc.iter_mut().zip(&prow) {
                    let pj = pi[j & (KC - 1)];
                    for (x, &e) in a.iter_mut().zip(vv) {
                        *x = madd::<FMA>(pj, e, *x);
                    }
                }
            }
            for (i, a) in acc.iter().enumerate() {
                or[i * hd + dc..i * hd + dc + DC].copy_from_slice(a);
            }
        }
    }
}

/// Lane-wise reductions (a plain fold over `f32` does not vectorize).
#[inline(always)]
fn reduce16(x: &[f32], init: f32, f: impl Fn(f32, f32) -> f32) -> f32 {
    let mut acc = [init; 16];
    let (chunks, tail) = x.as_chunks::<16>();
    for c in chunks {
        for (a, &v) in acc.iter_mut().zip(c) {
            *a = f(*a, v);
        }
    }
    let mut r = tail.iter().fold(init, |a, &v| f(a, v));
    for a in acc {
        r = f(r, a);
    }
    r
}

/// Online softmax for one score row: keys `0..vis` of the chunk are
/// visible. Turns `s` into probabilities relative to the new running max,
/// updates `m`/`l` and returns the factor to rescale the row's output by.
#[inline(always)]
fn softmax_row(s: &mut [f32], vis: usize, m: &mut f32, l: &mut f32) -> f32 {
    if vis == 0 {
        s.fill(0.0);
        return 1.0;
    }
    let cmax = reduce16(&s[..vis], f32::NEG_INFINITY, f32::max);
    let mnew = m.max(cmax);
    let corr = if *m == f32::NEG_INFINITY {
        0.0
    } else {
        exp_approx(*m - mnew)
    };
    for x in s[..vis].iter_mut() {
        *x = exp_approx(*x - mnew);
    }
    s[vis..].fill(0.0);
    let sum = reduce16(&s[..vis], 0.0, |a, b| a + b);
    *l = *l * corr + sum;
    *m = mnew;
    corr
}

/// K (transposed, zero-padded to [`KC`] keys) and V of keys `p0..p1`
/// (`p1 - p0 <= KC`) of head `h`, in f32: `kt` is `[hd][KC]`, `v` `[KC][hd]`.
#[inline(always)]
fn fill_chunk(cache: &KvCache, h: usize, p0: usize, p1: usize, kt: &mut [f32], v: &mut [f32]) {
    let hd = cache.head_dim;
    let kc = p1 - p0;
    assert!(kc <= KC && hd <= MAX_HD && kt.len() == hd * KC && v.len() == KC * hd);
    f16_to_f32_slice(cache.v_rows(h, p0, p1), &mut v[..kc * hd]);
    let mut row = [0.0f32; MAX_HD];
    kt.fill(0.0);
    for (j, kj) in cache.k_rows(h, p0, p1).chunks_exact(hd).enumerate() {
        let r = &mut row[..hd];
        f16_to_f32_slice(kj, r);
        for (d, &x) in r.iter().enumerate() {
            kt[d * KC + j] = x;
        }
    }
}

/// The keys `0..len` of every KV head in f32, chunked by [`KC`] (see
/// [`fill_chunk`]); built once per prefill call and shared by all items.
struct KvF32<'a> {
    hd: usize,
    nchunks: usize,
    kt: &'a [f32],
    v: &'a [f32],
}

impl<'a> KvF32<'a> {
    fn new(cache: &KvCache, len: usize, kt: &'a mut Vec<f32>, v: &'a mut Vec<f32>) -> Self {
        let (nkv, hd) = (cache.kv_heads, cache.head_dim);
        let nchunks = len.div_ceil(KC);
        kt.resize(nkv * nchunks * KC * hd, 0.0);
        v.resize(nkv * nchunks * KC * hd, 0.0);
        kt.par_chunks_mut(KC * hd)
            .zip(v.par_chunks_mut(KC * hd))
            .enumerate()
            .for_each(|(i, (ktc, vc))| {
                let (h, c) = (i / nchunks, i % nchunks);
                with_best_simd!(fill_chunk(
                    cache,
                    h,
                    c * KC,
                    ((c + 1) * KC).min(len),
                    ktc,
                    vc
                ));
            });
        Self { hd, nchunks, kt, v }
    }

    /// `(Kᵀ, V)` of chunk `c` of head `h`.
    #[inline(always)]
    fn chunk(&self, h: usize, c: usize) -> (&[f32], &[f32]) {
        let sz = KC * self.hd;
        let o = (h * self.nchunks + c) * sz;
        (&self.kt[o..o + sz], &self.v[o..o + sz])
    }
}

/// Keys per super-chunk: every query row's keys are split at multiples of
/// `SC`, each part reduced to an independent softmax partial and the parts
/// merged in order ([`merge_row`]). The split is fixed, so a row's result does
/// not depend on how many rows are computed together (prefill, chunked
/// prefill or decode all give bit-identical outputs), while decode can still
/// spread long contexts over threads.
pub const SC: usize = 256;

/// Online-softmax partial `(m, l, o)` of `rp` rows (multiple of 4; rows
/// `>= rows` are padding) over the keys of super-chunk `c` that are below
/// `keys`. Row `r` sees keys `< end(r)`. `chunk(i)` returns `(Kᵀ, V)` of
/// key chunk `i` (keys `i·KC ..`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn superchunk<'k, const FMA: bool>(
    s: &mut [f32],
    q: &[f32],
    rp: usize,
    rows: usize,
    hd: usize,
    c: usize,
    keys: usize,
    end: impl Fn(usize) -> usize,
    chunk: impl Fn(usize) -> (&'k [f32], &'k [f32]),
    m: &mut [f32],
    l: &mut [f32],
    o: &mut [f32],
) {
    m[..rp].fill(f32::NEG_INFINITY);
    l[..rp].fill(0.0);
    o[..rp * hd].fill(0.0);
    for c0 in (c * SC..((c + 1) * SC).min(keys)).step_by(KC) {
        let kc = KC.min(keys - c0);
        let (kt, vf) = chunk(c0 / KC);
        qk::<FMA>(q, kt, s, rp, hd);
        for r in 0..rp {
            let srow = &mut s[r * KC..(r + 1) * KC];
            let vis = if r < rows {
                end(r).saturating_sub(c0).min(kc)
            } else {
                0
            };
            let corr = softmax_row(srow, vis, &mut m[r], &mut l[r]);
            if corr != 1.0 {
                o[r * hd..(r + 1) * hd].iter_mut().for_each(|x| *x *= corr);
            }
        }
        pv::<FMA>(s, vf, o, rp, hd, kc);
    }
}

/// Merge the super-chunk partials `part(c) = (m, l, o)`, `c < nsc`, of one
/// row in order (skipping parts the row cannot see) into `out = o / l`.
#[inline(always)]
fn merge_row<'p, const FMA: bool>(
    nsc: usize,
    part: impl Fn(usize) -> (f32, f32, &'p [f32]),
    out: &mut [f32],
) {
    let mx = (0..nsc)
        .map(|c| part(c).0)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut lt = 0.0f32;
    out.fill(0.0);
    for c in 0..nsc {
        let (mc, lc, oc) = part(c);
        if mc == f32::NEG_INFINITY {
            continue;
        }
        let wgt = exp_approx(mc - mx);
        lt = madd::<FMA>(wgt, lc, lt);
        for (x, &v) in out.iter_mut().zip(oc) {
            *x = madd::<FMA>(wgt, v, *x);
        }
    }
    let inv = 1.0 / lt;
    out.iter_mut().for_each(|x| *x *= inv);
}

#[derive(Default)]
struct Parts {
    m: Vec<f32>,
    l: Vec<f32>,
    o: Vec<f32>,
    kt: Vec<f32>,
    v: Vec<f32>,
}

thread_local! {
    static PARTS: std::cell::RefCell<Parts> = std::cell::RefCell::new(Parts::default());
    /// Per-call buffers owned by the calling thread (no page faults after
    /// the first call).
    static CALL: std::cell::RefCell<Parts> = std::cell::RefCell::new(Parts::default());
}

#[allow(clippy::too_many_arguments)]
/// Gather the `rows` query vectors of KV head `kvh` for query rows
/// `t0..t1`, pre-scaled, into `dst` (`rp × hd`, zero padding).
fn gather_q(
    q: &[f32],
    t0: usize,
    t1: usize,
    kvh: usize,
    heads: usize,
    group: usize,
    hd: usize,
    scale: f32,
    rp: usize,
    dst: &mut Vec<f32>,
) {
    dst.clear();
    dst.resize(rp * hd, 0.0);
    for t in t0..t1 {
        for g in 0..group {
            let r = (t - t0) * group + g;
            let src = &q[(t * heads + kvh * group + g) * hd..][..hd];
            for (d, &v) in dst[r * hd..(r + 1) * hd].iter_mut().zip(src) {
                *d = v * scale;
            }
        }
    }
}

/// Causal GQA attention of `n` query rows at positions `pos..pos + n` over
/// the keys `0..pos + n` of `cache` (the contract of [`super::attention`]).
/// Every row is computed with the same operations whatever `n` and `pos`.
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
    assert!(supported(hd), "flash attention: head_dim {hd}");
    assert!(
        nkv > 0 && heads.is_multiple_of(nkv),
        "attention: heads not a multiple of kv heads"
    );
    assert_eq!(q.len(), n * heads * hd);
    assert_eq!(out.len(), n * heads * hd);
    assert!(pos + n <= cache.cap, "attention: positions beyond cache");
    if n == 0 {
        return;
    }
    let group = heads / nkv;
    let fma = cpu_features().avx2_tier();
    if n > QB {
        let optr = OutPtr(out.as_mut_ptr());
        let out_len = out.len();
        CALL.with_borrow_mut(|call| {
            let Parts { kt, v, .. } = call;
            let kv = KvF32::new(cache, pos + n, kt, v);
            let kv = &kv;
            let nqb = n.div_ceil(QB);
            // Later query blocks see more keys: schedule them first.
            (0..nqb * nkv).into_par_iter().for_each(|i| {
                let (b, kvh) = (nqb - 1 - i / nkv, i % nkv);
                let (t0, t1) = (b * QB, ((b + 1) * QB).min(n));
                SCRATCH.with_borrow_mut(|sc| {
                    PARTS.with_borrow_mut(|pt| {
                        with_best_simd!(if fma {
                            block_item::<true>(
                                sc, pt, q, t0, t1, kvh, pos, heads, group, kv, scale, optr, out_len,
                            )
                        } else {
                            block_item::<false>(
                                sc, pt, q, t0, t1, kvh, pos, heads, group, kv, scale, optr, out_len,
                            )
                        })
                    })
                });
            });
        });
    } else {
        // Few rows (decode): one item per (KV head, super-chunk), partials
        // merged afterwards.
        let rows = n * group;
        let rp = rows.next_multiple_of(4);
        let keys = pos + n;
        let nsc = keys.div_ceil(SC);
        CALL.with_borrow_mut(|call| {
            call.m.resize(nkv * nsc * rp, 0.0);
            call.l.resize(nkv * nsc * rp, 0.0);
            call.o.resize(nkv * nsc * rp * hd, 0.0);
            call.m
                .par_chunks_mut(rp)
                .zip(call.l.par_chunks_mut(rp))
                .zip(call.o.par_chunks_mut(rp * hd))
                .enumerate()
                .for_each(|(i, ((m, l), o))| {
                    let (kvh, c) = (i / nsc, i % nsc);
                    SCRATCH.with_borrow_mut(|sc| {
                        PARTS.with_borrow_mut(|pt| {
                            with_best_simd!(if fma {
                                chunk_item::<true>(
                                    sc, pt, q, n, kvh, c, pos, heads, group, cache, scale, m, l, o,
                                )
                            } else {
                                chunk_item::<false>(
                                    sc, pt, q, n, kvh, c, pos, heads, group, cache, scale, m, l, o,
                                )
                            })
                        })
                    });
                });
            let (m, l, o) = (&call.m, &call.l, &call.o);
            out.par_chunks_mut(heads * hd)
                .enumerate()
                .for_each(|(t, orow)| {
                    for kvh in 0..nkv {
                        for g in 0..group {
                            let r = t * group + g;
                            let part = |c: usize| {
                                let i = (kvh * nsc + c) * rp + r;
                                (m[i], l[i], &o[i * hd..(i + 1) * hd])
                            };
                            let dst = &mut orow[(kvh * group + g) * hd..(kvh * group + g + 1) * hd];
                            with_best_simd!(if fma {
                                merge_row::<true>(nsc, part, dst)
                            } else {
                                merge_row::<false>(nsc, part, dst)
                            });
                        }
                    }
                });
        });
    }
}

/// Prefill item: query rows `t0..t1` × KV head `kvh`, all super-chunks.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn block_item<const FMA: bool>(
    sc: &mut Scratch,
    pt: &mut Parts,
    q: &[f32],
    t0: usize,
    t1: usize,
    kvh: usize,
    pos: usize,
    heads: usize,
    group: usize,
    kv: &KvF32<'_>,
    scale: f32,
    out: OutPtr,
    out_len: usize,
) {
    let hd = kv.hd;
    let rows = (t1 - t0) * group;
    let rp = rows.next_multiple_of(4);
    let keys = pos + t1;
    let nsc = keys.div_ceil(SC);
    gather_q(q, t0, t1, kvh, heads, group, hd, scale, rp, &mut sc.q);
    sc.s.resize(rp * KC, 0.0);
    pt.m.resize(nsc * rp, 0.0);
    pt.l.resize(nsc * rp, 0.0);
    pt.o.resize(nsc * rp * hd, 0.0);
    let end = |r: usize| pos + t0 + r / group + 1;
    for c in 0..nsc {
        superchunk::<FMA>(
            &mut sc.s,
            &sc.q,
            rp,
            rows,
            hd,
            c,
            keys,
            end,
            |i| kv.chunk(kvh, i),
            &mut pt.m[c * rp..(c + 1) * rp],
            &mut pt.l[c * rp..(c + 1) * rp],
            &mut pt.o[c * rp * hd..(c + 1) * rp * hd],
        );
    }
    for t in t0..t1 {
        let off = (t * heads + kvh * group) * hd;
        assert!(off + group * hd <= out_len);
        for g in 0..group {
            let r = (t - t0) * group + g;
            let part = |c: usize| {
                let i = c * rp + r;
                (pt.m[i], pt.l[i], &pt.o[i * hd..(i + 1) * hd])
            };
            // SAFETY: in bounds (checked above); items write disjoint
            // (row, head) ranges.
            let dst = unsafe { std::slice::from_raw_parts_mut(out.0.add(off + g * hd), hd) };
            merge_row::<FMA>(nsc, part, dst);
        }
    }
}

/// Decode item: all `n` query rows × KV head `kvh` × super-chunk `c`,
/// converting that super-chunk's K/V itself.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn chunk_item<const FMA: bool>(
    sc: &mut Scratch,
    pt: &mut Parts,
    q: &[f32],
    n: usize,
    kvh: usize,
    c: usize,
    pos: usize,
    heads: usize,
    group: usize,
    cache: &KvCache,
    scale: f32,
    m: &mut [f32],
    l: &mut [f32],
    o: &mut [f32],
) {
    let hd = cache.head_dim;
    let rows = n * group;
    let rp = rows.next_multiple_of(4);
    let keys = pos + n;
    gather_q(q, 0, n, kvh, heads, group, hd, scale, rp, &mut sc.q);
    sc.s.resize(rp * KC, 0.0);
    // Key chunks of this super-chunk, converted once.
    let first = c * SC / KC;
    let last = ((c + 1) * SC).min(keys).div_ceil(KC);
    pt.kt.resize((last - first) * KC * hd, 0.0);
    pt.v.resize((last - first) * KC * hd, 0.0);
    for (i, (ktc, vc)) in pt
        .kt
        .chunks_exact_mut(KC * hd)
        .zip(pt.v.chunks_exact_mut(KC * hd))
        .enumerate()
    {
        let p0 = (first + i) * KC;
        fill_chunk(cache, kvh, p0, (p0 + KC).min(keys), ktc, vc);
    }
    let (kt, vv) = (&pt.kt, &pt.v);
    let sz = KC * hd;
    superchunk::<FMA>(
        &mut sc.s,
        &sc.q,
        rp,
        rows,
        hd,
        c,
        keys,
        |r| pos + r / group + 1,
        |i| {
            (
                &kt[(i - first) * sz..(i - first + 1) * sz],
                &vv[(i - first) * sz..(i - first + 1) * sz],
            )
        },
        m,
        l,
        o,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn exp_approx_is_accurate() {
        let mut x = -90.0f32;
        while x <= 20.0 {
            let (got, want) = (exp_approx(x), (x as f64).exp());
            if x >= -87.0 {
                let rel = ((got as f64 - want) / want).abs();
                assert!(rel < 4e-7, "exp({x}) = {got}, want {want}");
            } else {
                assert!(got as f64 <= 2e-38, "exp({x}) = {got}");
            }
            x += 0.0137;
        }
        assert_eq!(exp_approx(0.0), 1.0);
    }

    /// Causal GQA attention in f64 over f16-rounded K/V.
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
        let r = |x: f32| f16::from_f32(x).to_f32() as f64;
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
                            .map(|i| qv[i] as f64 * r(k[(kvh * total + p) * hd + i]))
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for i in 0..hd {
                    let acc: f64 = (0..len)
                        .map(|p| e[p] * r(v[(kvh * total + p) * hd + i]))
                        .sum();
                    out[(t * heads + h) * hd + i] = (acc / z) as f32;
                }
            }
        }
        out
    }

    /// Fill a cache with `total` random positions; returns (K, V) in f32.
    fn random_cache(nkv: usize, hd: usize, total: usize, rnd: &mut impl FnMut() -> f32) -> KvCache {
        let mut cache = KvCache::new(nkv, hd, 4096);
        cache.reserve(total, 0).unwrap();
        for h in 0..nkv {
            for p in 0..total {
                let k: Vec<f32> = (0..hd).map(|_| rnd() * 2.0).collect();
                let v: Vec<f32> = (0..hd).map(|_| rnd()).collect();
                cache.store(h, p, &k, &v);
            }
        }
        cache
    }

    /// The result of a row does not depend on how many rows are computed
    /// together: whole prompt, chunks and one row at a time agree bit for
    /// bit (prefill, chunked prefill and decode share KV caches).
    #[test]
    fn batch_invariant() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        for &(heads, nkv, hd, n) in &[(4, 2, 64, 70), (6, 2, 128, 300), (2, 1, 256, 40)] {
            let cache = random_cache(nkv, hd, n, &mut rnd);
            let q: Vec<f32> = (0..n * heads * hd).map(|_| rnd() * 2.0).collect();
            let scale = 1.0 / (hd as f32).sqrt();
            let w = heads * hd;
            let mut all = vec![0.0; n * w];
            attention(&q, n, 0, heads, &cache, scale, &mut all);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            for t in 0..n {
                let mut one = vec![0.0; w];
                attention(&q[t * w..(t + 1) * w], 1, t, heads, &cache, scale, &mut one);
                assert_eq!(
                    bits(&one),
                    bits(&all[t * w..(t + 1) * w]),
                    "row {t} alone, n {n}"
                );
            }
            for split in [1, 7, 17, n / 2] {
                let mut got = vec![0.0; n * w];
                let (a, b) = got.split_at_mut(split * w);
                attention(&q[..split * w], split, 0, heads, &cache, scale, a);
                attention(&q[split * w..], n - split, split, heads, &cache, scale, b);
                assert_eq!(bits(&got), bits(&all), "split at {split}, n {n}");
            }
        }
    }

    #[test]
    fn prefill_matches_naive() {
        let mut seed = 777u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        // (heads, kv heads, head_dim, pos, n): partial query blocks, odd
        // group sizes (rows not a multiple of 4), chunks crossing KC, a
        // cached prefix and head_dim 256.
        for &(heads, nkv, hd, pos, n) in &[
            (2, 2, 64, 0, 1),
            (4, 2, 64, 0, 37),
            (6, 2, 128, 5, 70),
            (3, 1, 64, 130, 17),
            (4, 1, 256, 0, 65),
            (16, 8, 128, 0, 150),
            (16, 8, 128, 700, 1),
            (8, 2, 64, 513, 3),
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
            let scale = 1.0 / (hd as f32).sqrt();
            let mut out = vec![f32::NAN; n * heads * hd];
            attention(&q, n, pos, heads, &cache, scale, &mut out);
            let want = naive(&q, &k, &v, n, pos, heads, nkv, hd, scale);
            let err = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                err < 1e-4,
                "heads {heads} kv {nkv} hd {hd} pos {pos} n {n}: max err {err}"
            );
        }
    }
}
