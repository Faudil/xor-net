//! 1-bit "Q1_0" weights (PrismML Bonsai / ggml `Q1_0`) and their GEMV/GEMM
//! kernels.
//!
//! # Weight format
//!
//! A row of `cols` weights is split into blocks of `block` (32 or 128)
//! elements. Each block has one f16 scale `d` and one sign bit per element:
//! `w = bit ? +d : -d`. This is the ggml / PrismML convention for all three
//! GGUF block types (see [`Q1Weight::from_gguf`]):
//!
//! | GGUF type | block | bytes | layout                       |
//! |-----------|-------|-------|------------------------------|
//! | 41 `Q1_0` (upstream) | 128 | 18 | `[d: f16][bits: 16 B]` |
//! | 42 `Q1_0` (PrismML)  | 32  | 6  | `[d: f16][bits: 4 B]`  |
//! | 43 `Q1_0_G128`       | 128 | 18 | `[bits: 16 B][d: f16]` |
//!
//! [`Q1Weight`] stores the same information *losslessly* in a split layout:
//! all sign bits of a row contiguous (`cols / 8` bytes, bit `i` of byte `k`
//! is element `8k + i`), and the f16 scales in a separate array. Kernels read
//! whole sign words of a row (and 8 or 16 rows' words side by side) and
//! convert 8 scales with one `vcvtph2ps`; the layout is identical for all
//! three source types.
//!
//! # Activations
//!
//! Activations are quantized to int8 per row in blocks of 32 elements (the
//! ggml `Q8_0` granularity that llama.cpp uses against Q1_0) or of 128
//! elements, with one f32 scale per block. The activation block must divide
//! the weight block. Block scales keep the error of large-outlier
//! activations local to their block, unlike a single per-row scale.
//! [`Q8Acts`] also keeps, per 32-element sub-block, a [`perm32`]-ordered copy
//! of the values and their negated sum `-Σq`.
//!
//! # Numerics: one canonical order
//!
//! For row `r` and activation block `j` inside weight block `b` the integer
//! dot `I_j = Σ_i (bit_i ? q_i : -q_i)` is exact (|q| ≤ 127). Every kernel
//! then combines the blocks in the same fused-multiply-add order (see
//! [`dot_rows_scalar`]): `p_b = fma(I_j, d_x[j], p_b)` over the activation
//! blocks of `b`, `y = fma(p_b, d_w[b], y)` over the weight blocks. Results
//! are therefore bit-identical across ISAs, row groupings and token counts:
//! a prompt prefilled in one call gives the same numbers as feeding it
//! token by token, so prefill, chunked prefill and decode can share KV
//! caches exactly.
//!
//! # Kernels
//!
//! - GEMV / row kernels ([`dot_rows_isa`]): AVX-512 processes 16 rows at a
//!   time: their sign words are transposed (16×16 × 32 bit) so one vector
//!   holds the 16 rows' bits of a sub-block; per 4-element group a shift and
//!   an AND expand them to bytes `{0, 1}` and `vpdpbusd` against the
//!   broadcast permuted activations sums `S = Σ bit·q` for all 16 rows, so
//!   `I = 2S - Σq` lands one row per lane, ready for the float part. AVX2
//!   expands one row's bits with `vpsrlvd` + AND, sums with `vpmaddubsw` and
//!   a `vphaddw` tree, and transposes 8 rows × 8 blocks before the float
//!   part.
//! - Token-tiled GEMM ([`gemm_with`], [`TileIsa`]): a slab of 8/16 rows is
//!   expanded once to bytes and reused by every token; each `vpdpbusd` (or
//!   `vpmaddubsw`) does `4 · lanes` MACs of a tile of tokens held in
//!   registers.

use std::sync::Arc;

use rayon::prelude::*;

use crate::loader::gguf::{PRISM_Q1_0, PRISM_Q1_0_G128, PRISM_Q1_0_UPSTREAM, fp16_to_f32};
use crate::util::cpu_features;

/// Elements per AVX-512 step / activation lane-scale group (32-element
/// blocks only).
const SUB: usize = 32;

// ---------------------------------------------------------------------------
// Byte storage: owned, or borrowed from a shared buffer such as an mmap
// ---------------------------------------------------------------------------

/// An immutable byte range kept alive by a shared owner (a `Vec<u8>`, an
/// `Mmap`, ...). Cloning is cheap.
#[derive(Clone)]
pub struct ByteBuf {
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    off: usize,
    len: usize,
}

impl std::fmt::Debug for ByteBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ByteBuf({} bytes)", self.len)
    }
}

impl ByteBuf {
    pub fn from_vec(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            owner: Arc::new(v),
            off: 0,
            len,
        }
    }

    /// `owner[off..off + len]`. Fails if the range is out of bounds.
    pub fn from_shared(
        owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
        off: usize,
        len: usize,
    ) -> anyhow::Result<Self> {
        let total = (*owner).as_ref().len();
        anyhow::ensure!(
            off.checked_add(len).is_some_and(|end| end <= total),
            "ByteBuf range {off}+{len} exceeds buffer of {total} bytes"
        );
        Ok(Self { owner, off, len })
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &(*self.owner).as_ref()[self.off..self.off + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// A `[rows, cols]` 1-bit weight matrix in the split layout described in the
/// module docs. Invariants (checked by [`Q1Weight::new`]): `block` is 32 or
/// 128, `cols.is_multiple_of(block)`, `bits.len() == rows * cols / 8` and
/// `scales.len() == rows * (cols / block) * 2`.
#[derive(Clone, Debug)]
pub struct Q1Weight {
    rows: usize,
    cols: usize,
    block: usize,
    bits: ByteBuf,
    scales: ByteBuf,
}

/// Block geometry of a GGUF Q1 dtype: `(block elems, block bytes, scale
/// offset, bits offset)`.
fn gguf_q1_geometry(dtype: u32) -> anyhow::Result<(usize, usize, usize, usize)> {
    Ok(match dtype {
        PRISM_Q1_0_UPSTREAM => (128, 18, 0, 2),
        PRISM_Q1_0 => (32, 6, 0, 2),
        PRISM_Q1_0_G128 => (128, 18, 16, 0),
        _ => anyhow::bail!("dtype {dtype} is not a Q1_0 type"),
    })
}

impl Q1Weight {
    pub fn new(
        rows: usize,
        cols: usize,
        block: usize,
        bits: ByteBuf,
        scales: ByteBuf,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(block == 32 || block == 128, "Q1 block must be 32 or 128");
        anyhow::ensure!(
            cols > 0 && cols.is_multiple_of(block),
            "Q1 cols {cols} not a multiple of block {block}"
        );
        let nbits = rows
            .checked_mul(cols / 8)
            .ok_or_else(|| anyhow::anyhow!("Q1 size overflow"))?;
        let nsc = rows
            .checked_mul(cols / block * 2)
            .ok_or_else(|| anyhow::anyhow!("Q1 size overflow"))?;
        anyhow::ensure!(
            bits.len() == nbits,
            "Q1 bits: {} bytes, expected {nbits}",
            bits.len()
        );
        anyhow::ensure!(
            scales.len() == nsc,
            "Q1 scales: {} bytes, expected {nsc}",
            scales.len()
        );
        Ok(Self {
            rows,
            cols,
            block,
            bits,
            scales,
        })
    }

    /// Repack GGUF Q1 blocks (`dtype` 41, 42 or 43) of a `[rows, cols]`
    /// tensor. Lossless: [`Self::to_gguf`] restores the exact input bytes.
    pub fn from_gguf(data: &[u8], dtype: u32, rows: usize, cols: usize) -> anyhow::Result<Self> {
        let (block, bbytes, soff, boff) = gguf_q1_geometry(dtype)?;
        anyhow::ensure!(
            cols.is_multiple_of(block),
            "cols {cols} not a multiple of {block}"
        );
        let nb = cols / block;
        anyhow::ensure!(
            data.len() == rows * nb * bbytes,
            "GGUF Q1 tensor has {} bytes, expected {}",
            data.len(),
            rows * nb * bbytes
        );
        let nbb = block / 8;
        let mut bits = vec![0u8; rows * cols / 8];
        let mut scales = vec![0u8; rows * nb * 2];
        for (i, blk) in data.chunks_exact(bbytes).enumerate() {
            bits[i * nbb..(i + 1) * nbb].copy_from_slice(&blk[boff..boff + nbb]);
            scales[i * 2..i * 2 + 2].copy_from_slice(&blk[soff..soff + 2]);
        }
        Self::new(
            rows,
            cols,
            block,
            ByteBuf::from_vec(bits),
            ByteBuf::from_vec(scales),
        )
    }

    /// Re-encode as GGUF blocks of `dtype` (must have the same block size).
    pub fn to_gguf(&self, dtype: u32) -> anyhow::Result<Vec<u8>> {
        let (block, bbytes, soff, boff) = gguf_q1_geometry(dtype)?;
        anyhow::ensure!(block == self.block, "block size mismatch");
        let nbb = block / 8;
        let total = self.rows * self.cols / block;
        let bits = self.bits.as_slice();
        let sc = self.scales.as_slice();
        let mut out = vec![0u8; total * bbytes];
        for (i, blk) in out.chunks_exact_mut(bbytes).enumerate() {
            blk[boff..boff + nbb].copy_from_slice(&bits[i * nbb..(i + 1) * nbb]);
            blk[soff..soff + 2].copy_from_slice(&sc[i * 2..i * 2 + 2]);
        }
        Ok(out)
    }

    /// Stack matrices with equal `cols` and `block` vertically (e.g. fuse
    /// Q/K/V or gate/up projections into one GEMV).
    pub fn concat_rows(parts: &[&Q1Weight]) -> anyhow::Result<Self> {
        anyhow::ensure!(!parts.is_empty(), "concat_rows: no parts");
        let (cols, block) = (parts[0].cols, parts[0].block);
        let mut bits = Vec::new();
        let mut scales = Vec::new();
        let mut rows = 0;
        for p in parts {
            anyhow::ensure!(
                p.cols == cols && p.block == block,
                "concat_rows: shape mismatch"
            );
            bits.extend_from_slice(p.bits.as_slice());
            scales.extend_from_slice(p.scales.as_slice());
            rows += p.rows;
        }
        Self::new(
            rows,
            cols,
            block,
            ByteBuf::from_vec(bits),
            ByteBuf::from_vec(scales),
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn bits(&self) -> &ByteBuf {
        &self.bits
    }
    pub fn scales(&self) -> &ByteBuf {
        &self.scales
    }

    /// Bytes streamed by one full GEMV.
    pub fn nbytes(&self) -> usize {
        self.bits.len() + self.scales.len()
    }

    /// Scale of block `b` of row `r`.
    // `always`: otherwise it stays out of line in the `#[target_feature]`
    // kernels (one call per weight block) unless built with target-cpu=native.
    #[inline(always)]
    pub fn scale(&self, r: usize, b: usize) -> f32 {
        let i = (r * (self.cols / self.block) + b) * 2;
        let s = self.scales.as_slice();
        fp16_to_f32(u16::from_le_bytes([s[i], s[i + 1]]))
    }

    /// Dequantize row `r` into `out[..cols]` (exact: `±d` in f32).
    pub fn dequantize_row_into(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows, "row {r} out of range");
        assert!(out.len() >= self.cols, "output too short");
        let rb = self.cols / 8;
        let bits = &self.bits.as_slice()[r * rb..(r + 1) * rb];
        for b in 0..self.cols / self.block {
            let d = self.scale(r, b);
            for e in b * self.block..(b + 1) * self.block {
                out[e] = if (bits[e >> 3] >> (e & 7)) & 1 == 1 {
                    d
                } else {
                    -d
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// `n` activation rows of `cols` elements quantized to int8 in blocks of
/// `block` (32 or 128) elements. Reusable: [`Q8Acts::quantize`] only grows
/// the buffers.
///
/// Besides the plain int8 values it keeps, per 32-element sub-block, the
/// values in the [`perm32`] order (AVX2 and token-tiled kernels) and the
/// negated integer sum `-Σq`.
#[derive(Clone, Debug, Default)]
pub struct Q8Acts {
    n: usize,
    cols: usize,
    block: usize,
    q: Vec<i8>,
    /// One scale per block: `x ≈ q * d`.
    d: Vec<f32>,
    /// `q` with every 32-element sub-block permuted by [`perm32`].
    qp: Vec<i8>,
    /// `-Σ q` per 32-element sub-block.
    nsum: Vec<i32>,
    /// 128-element blocks only: `-Σ q` per block (`nsum` otherwise).
    nsumb: Vec<i32>,
}

/// One quantized activation row (borrowed from a [`Q8Acts`]).
#[derive(Clone, Copy)]
pub struct ActRow<'a> {
    pub block: usize,
    pub q: &'a [i8],
    pub d: &'a [f32],
    /// `q` with each 32-element sub-block permuted by [`perm32`].
    pub qp: &'a [i8],
    /// `-Σ q` per 32-element sub-block.
    pub nsum: &'a [i32],
}

/// Position of element `e` (0..32) of a sub-block in the permuted layout
/// used by the AVX2 and token-tiled kernels: byte `4j + p` holds element
/// `8p + j`. With it, the four bytes of 32-bit lane `j` pair with sign bits
/// `j, 8 + j, 16 + j, 24 + j` of the block's `u32`, which one shift and one
/// AND by `0x0101_0101` extract (no byte shuffle).
#[inline(always)]
pub const fn perm32(e: usize) -> usize {
    4 * (e % 8) + e / 8
}

/// Quantize one block: returns `d` with `x ≈ q * d`, `|q| ≤ 127`
/// (round half to even). NaN elements map to 0; an infinite element makes
/// `d` NaN so the non-finite input stays visible in the output.
// `always`, and callers run it under `with_best_simd!`: on the x86_64
// baseline (no SSE4.1) `round_ties_even` is a libm call per element.
#[inline(always)]
fn quantize_block(x: &[f32], q: &mut [i8]) -> f32 {
    // Lane-wise max (a plain fold over f32 does not vectorize); the max of
    // a set does not depend on the order, NaNs are skipped either way.
    let (chunks, tail) = x.as_chunks::<8>();
    let mut lanes = [0.0f32; 8];
    for c in chunks {
        for (m, v) in lanes.iter_mut().zip(c) {
            *m = m.max(v.abs());
        }
    }
    let amax = tail
        .iter()
        .chain(&lanes)
        .fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        q.fill(0);
        return if amax == 0.0 { 0.0 } else { f32::NAN };
    }
    let id = 127.0 / amax;
    for (o, &v) in q.iter_mut().zip(x) {
        let r = (v * id).round_ties_even();
        *o = if r.is_nan() {
            0
        } else {
            r.clamp(-127.0, 127.0) as i8
        };
    }
    amax / 127.0
}

impl Q8Acts {
    pub fn new() -> Self {
        Self::default()
    }

    /// Quantize `x` = `n` rows of `cols` in blocks of `block` (32 or 128,
    /// `cols % block == 0`). Use the block size of the weights it will meet.
    pub fn quantize(&mut self, x: &[f32], n: usize, cols: usize, block: usize) {
        assert!(
            block == 32 || block == 128,
            "activation block must be 32 or 128"
        );
        assert!(
            cols.is_multiple_of(block),
            "cols {cols} not a multiple of {block}"
        );
        assert_eq!(x.len(), n * cols, "Q8Acts::quantize: length mismatch");
        self.n = n;
        self.cols = cols;
        self.block = block;
        let nblk = n * cols / block;
        self.q.resize(n * cols, 0);
        self.d.resize(nblk, 0.0);
        let nsub = n * cols / SUB;
        self.qp.resize(n * cols, 0);
        self.nsum.resize(nsub, 0);
        let avx2 = cpu_features().avx2_tier();
        let quantize_rows =
            |x: &[f32], q: &mut [i8], d: &mut [f32], qp: &mut [i8], ns: &mut [i32]| {
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 tier detected; lengths are consistent.
                    return unsafe { quantize_blocks_avx2(x, q, d, qp, ns, block) };
                }
                let _ = avx2;
                crate::util::with_best_simd!(quantize_blocks(x, q, d, qp, ns, block))
            };
        if nblk * block >= 1 << 17 {
            // Parallel over groups of whole blocks (one dispatch per group).
            let per = 64;
            x.par_chunks(per * block)
                .zip(self.q.par_chunks_mut(per * block))
                .zip(self.d.par_chunks_mut(per))
                .zip(self.qp.par_chunks_mut(per * block))
                .zip(self.nsum.par_chunks_mut(per * block / SUB))
                .for_each(|((((xb, qb), d), qp), ns)| quantize_rows(xb, qb, d, qp, ns));
        } else {
            quantize_rows(x, &mut self.q, &mut self.d, &mut self.qp, &mut self.nsum);
        }
        if block == SUB {
            self.nsumb.clear();
        } else {
            self.nsumb.clear();
            self.nsumb.extend(
                self.nsum
                    .as_chunks::<{ 128 / SUB }>()
                    .0
                    .iter()
                    .map(|s| s.iter().sum::<i32>()),
            );
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn block(&self) -> usize {
        self.block
    }

    pub fn row(&self, t: usize) -> ActRow<'_> {
        assert!(t < self.n, "activation row {t} out of range");
        let c = self.cols;
        let nb = c / self.block;
        let ns = c / SUB;
        let d = &self.d[t * nb..(t + 1) * nb];
        ActRow {
            block: self.block,
            q: &self.q[t * c..(t + 1) * c],
            d,
            qp: &self.qp[t * c..(t + 1) * c],
            nsum: &self.nsum[t * ns..(t + 1) * ns],
        }
    }
}

/// Quantize whole blocks of `x` (see [`quantize_block`]) and fill the
/// [`perm32`]-ordered copy and the negated sub-block sums.
#[inline(always)]
fn quantize_blocks(
    x: &[f32],
    q: &mut [i8],
    d: &mut [f32],
    qp: &mut [i8],
    nsum: &mut [i32],
    block: usize,
) {
    for ((xb, qb), d) in x
        .chunks_exact(block)
        .zip(q.chunks_exact_mut(block))
        .zip(d.iter_mut())
    {
        *d = quantize_block(xb, qb);
    }
    for ((qs, ps), ns) in q
        .as_chunks::<SUB>()
        .0
        .iter()
        .zip(qp.as_chunks_mut::<SUB>().0.iter_mut())
        .zip(nsum.iter_mut())
    {
        // Byte 4j + p <- element 8p + j (a fixed shuffle).
        *ps = std::array::from_fn(|o| qs[8 * (o % 4) + o / 4]);
        *ns = -qs.iter().map(|&v| v as i32).sum::<i32>();
    }
}

/// AVX2 [`quantize_blocks`], bit-identical to it: the same block max, the
/// same `127 / amax` and product, and `vcvtps2dq` rounds half to even like
/// `round_ties_even`. Blocks whose max is 0 or not finite go through
/// [`quantize_block`].
///
/// # Safety
/// The CPU must support `avx2`, `fma` and `f16c` (the AVX2 tier). Lengths
/// are checked (panics).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::needless_range_loop)]
unsafe fn quantize_blocks_avx2(
    x: &[f32],
    q: &mut [i8],
    d: &mut [f32],
    qp: &mut [i8],
    nsum: &mut [i32],
    block: usize,
) {
    use std::arch::x86_64::*;
    assert!(block == 32 || block == 128);
    let nblk = x.len() / block;
    assert!(x.len() == nblk * block && q.len() == x.len() && qp.len() == x.len());
    assert!(d.len() == nblk && nsum.len() == x.len() / SUB);
    let absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let lim = _mm256_set1_epi32(127);
    let nlim = _mm256_set1_epi32(-127);
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let pair = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let tr = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10,
        14, 3, 7, 11, 15,
    );
    let bias = _mm256_set1_epi8(-128);
    let xp = x.as_ptr();
    for bi in 0..nblk {
        let xb = xp.add(bi * block);
        // max |x|; NaN lanes keep the running max (second operand).
        let mut m = _mm256_setzero_ps();
        for c in 0..block / 8 {
            m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(xb.add(8 * c)), absm), m);
        }
        let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
        if amax == 0.0 || !amax.is_finite() {
            let r = bi * block..(bi + 1) * block;
            d[bi] = quantize_block(&x[r.clone()], &mut q[r.clone()]);
            for k in r.start / SUB..r.end / SUB {
                let qs: &[i8; SUB] = q[k * SUB..(k + 1) * SUB].try_into().unwrap();
                let ps: &mut [i8; SUB] = (&mut qp[k * SUB..(k + 1) * SUB]).try_into().unwrap();
                *ps = std::array::from_fn(|o| qs[8 * (o % 4) + o / 4]);
                nsum[k] = -qs.iter().map(|&v| v as i32).sum::<i32>();
            }
            continue;
        }
        let id = _mm256_set1_ps(127.0 / amax);
        d[bi] = amax / 127.0;
        for c in 0..block / SUB {
            let k = bi * block / SUB + c;
            let xs = xb.add(SUB * c);
            let qi = |o: usize| {
                let v = _mm256_loadu_ps(xs.add(8 * o));
                let i = _mm256_cvtps_epi32(_mm256_mul_ps(v, id));
                // NaN -> 0, then clamp to ±127.
                let i = _mm256_and_si256(i, _mm256_castps_si256(_mm256_cmp_ps::<_CMP_ORD_Q>(v, v)));
                _mm256_max_epi32(_mm256_min_epi32(i, lim), nlim)
            };
            let a = _mm256_packs_epi32(qi(0), qi(1));
            let b = _mm256_packs_epi32(qi(2), qi(3));
            let qv = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, b), order);
            _mm256_storeu_si256(q.as_mut_ptr().add(SUB * k) as *mut __m256i, qv);
            let pv = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(qv, pair), tr);
            _mm256_storeu_si256(qp.as_mut_ptr().add(SUB * k) as *mut __m256i, pv);
            // Σq = Σ(q + 128) - 32·128.
            let sad = _mm256_sad_epu8(_mm256_xor_si256(qv, bias), _mm256_setzero_si256());
            let s2 = _mm_add_epi64(
                _mm256_castsi256_si128(sad),
                _mm256_extracti128_si256(sad, 1),
            );
            let s = _mm_cvtsi128_si64(_mm_add_epi64(s2, _mm_unpackhi_epi64(s2, s2))) as i32;
            nsum[k] = 32 * 128 - s;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Instruction-set variant of the row kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    /// `avx2` + `fma` + `f16c`.
    Avx2,
    /// `avx512f` + `avx512bw` (+ the AVX2 tier for F16C / FMA).
    Avx512Bw,
    /// `avx512f` + `avx512bw` + `avx512vnni` (+ the AVX2 tier).
    Avx512Vnni,
}

impl Isa {
    /// Fastest variant supported by this CPU.
    pub fn best() -> Self {
        [Isa::Avx512Vnni, Isa::Avx512Bw, Isa::Avx2]
            .into_iter()
            .find(|i| i.supported())
            .unwrap_or(Isa::Scalar)
    }

    /// Whether this CPU can run the variant.
    pub fn supported(self) -> bool {
        let f = cpu_features();
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => f.avx2_fma() && f.f16c,
            Isa::Avx512Bw => f.avx2_fma() && f.f16c && f.avx512bw(),
            Isa::Avx512Vnni => f.avx2_fma() && f.f16c && f.avx512_vnni(),
        }
    }

    /// All variants this CPU supports.
    pub fn all_supported() -> Vec<Isa> {
        [Isa::Scalar, Isa::Avx2, Isa::Avx512Bw, Isa::Avx512Vnni]
            .into_iter()
            .filter(|i| i.supported())
            .collect()
    }
}

fn check_rows(w: &Q1Weight, r0: usize, a: &ActRow<'_>, out: &[f32]) {
    assert!(
        r0.checked_add(out.len()).is_some_and(|e| e <= w.rows),
        "Q1 kernel: rows {r0}+{} exceed {}",
        out.len(),
        w.rows
    );
    assert!(
        (a.block == 32 || a.block == 128) && w.block.is_multiple_of(a.block),
        "Q1 kernel: activation block {} incompatible with weight block {}",
        a.block,
        w.block
    );
    assert_eq!(a.q.len(), w.cols, "Q1 kernel: activation length mismatch");
    assert_eq!(a.d.len(), w.cols / a.block, "Q1 kernel: act scales length");
    assert_eq!(a.qp.len(), w.cols, "Q1 kernel: permuted activation length");
    assert_eq!(a.nsum.len(), w.cols / SUB, "Q1 kernel: act sums length");
}

/// Reference kernel; defines the result every other kernel reproduces bit
/// for bit. For row `r`, with `I_j = Σ_{e in block j} (bit_e ? q_e : -q_e)`
/// (exact) over the activation blocks `j` of weight block `b`:
///
/// ```text
/// y = 0
/// for b:  p = 0
///         for j in b:  p = fma(I_j, d_x[j], p)
///         y = fma(p, d_w[r, b], y)
/// ```
///
/// All float operations are fused multiply-adds in this fixed order, so the
/// result does not depend on the ISA, on how rows are grouped, or on how
/// many tokens a GEMM processes at once.
pub fn dot_rows_scalar(w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
    check_rows(w, r0, &a, out);
    let rb = w.cols / 8;
    let nb = w.cols / w.block;
    let apw = w.block / a.block;
    let bits_all = w.bits.as_slice();
    for (i, o) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let bits = &bits_all[r * rb..(r + 1) * rb];
        let mut y = 0.0f32;
        for b in 0..nb {
            let mut p = 0.0f32;
            for j in b * apw..(b + 1) * apw {
                let mut sumi = 0i32;
                for e in j * a.block..(j + 1) * a.block {
                    let v = a.q[e] as i32;
                    sumi += if (bits[e >> 3] >> (e & 7)) & 1 == 1 {
                        v
                    } else {
                        -v
                    };
                }
                p = (sumi as f32).mul_add(a.d[j], p);
            }
            y = p.mul_add(w.scale(r, b), y);
        }
        *o = y;
    }
}

/// Weight scales of `lanes` rows starting at the row bit / scale pointers
/// `rows`, converted to f32 as `[lane][block]` into `dst` (F16C).
///
/// # Safety
/// The CPU must support `f16c`; every row has `nb` scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn row_scales_f32(w: &Q1Weight, rows: &[usize], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = w.cols / w.block;
    assert!(dst.len() >= rows.len() * nb);
    let sc = w.scales.as_slice();
    for (l, &r) in rows.iter().enumerate() {
        assert!(r < w.rows);
        let src = &sc[r * nb * 2..(r + 1) * nb * 2];
        let d = &mut dst[l * nb..(l + 1) * nb];
        let mut b = 0;
        while b + 8 <= nb {
            let h = _mm_loadu_si128(src.as_ptr().add(2 * b) as *const __m128i);
            _mm256_storeu_ps(d.as_mut_ptr().add(b), _mm256_cvtph_ps(h));
            b += 8;
        }
        for (b, o) in d.iter_mut().enumerate().skip(b) {
            *o = w.scale(r, b);
        }
    }
}

/// AVX2 row kernel, bit-identical to [`dot_rows_scalar`].
///
/// Per 32-element sub-block `k` of a row with sign bits `m` (a `u32`):
/// `vpsrlvd` shifts 32-bit lane `j` of `broadcast(m)` right by `j` and an AND
/// with `0x0101_0101` leaves byte `p` of lane `j` = bit `8p + j` (0 or 1),
/// which is exactly the element the [`perm32`]-ordered activations hold
/// there. `vpmaddubsw` then gives `S = Σ bit·q` in 16 i16 pair sums (no
/// saturation: ≤ 2·127); a `vphaddw` tree reduces eight consecutive
/// sub-blocks of a row to `[S_0 .. S_7]` and `2S - Σq` is the exact block
/// dot. Eight rows are processed together and their block sums transposed
/// (8×8) so the float part runs with one row per lane in the canonical
/// order.
///
/// # Safety
/// The CPU must support `avx2`, `fma` and `f16c`. Shapes are checked
/// (panics); all loads stay inside the weight rows and activation row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_rows_avx2(w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
    use std::arch::x86_64::*;
    check_rows(w, r0, &a, out);
    let cols = w.cols;
    let rb = cols / 8;
    let nb = cols / w.block;
    let nsub = cols / SUB;
    let apw = w.block / a.block; // activation blocks per weight block
    let spa = a.block / SUB; // sub-blocks per activation block
    let full = nsub / 8;
    let bits_all = w.bits.as_slice().as_ptr();
    let qp = a.qp.as_ptr();
    let shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let lo1 = _mm256_set1_epi8(1);
    let one16 = _mm256_set1_epi16(1);
    let gidx = _mm256_mullo_epi32(shifts, _mm256_set1_epi32(nb as i32));
    // i16 pair sums of `Σ bit·q` for sub-block k of the row bits at `rb_ptr`.
    let sub_dot = |rb_ptr: *const u8, k: usize| -> __m256i {
        let m = (rb_ptr.add(4 * k) as *const u32).read_unaligned();
        let wb = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(m as i32), shifts), lo1);
        _mm256_maddubs_epi16(wb, _mm256_loadu_si256(qp.add(SUB * k) as *const __m256i))
    };
    let mut dwr = vec![0.0f32; 8 * nb];
    let n = out.len();
    let mut i = 0;
    while i < n {
        let rr = (n - i).min(8);
        // Rows beyond `rr` repeat the last row (results discarded).
        let rows: [usize; 8] = std::array::from_fn(|k| r0 + i + k.min(rr - 1));
        let rbits: [*const u8; 8] = std::array::from_fn(|k| bits_all.add(rows[k] * rb));
        row_scales_f32(w, &rows, &mut dwr);
        let dw = |b: usize| _mm256_i32gather_ps::<4>(dwr.as_ptr().add(b), gidx);
        if i == 0 {
            // The first group of a call has nothing ahead of it.
            for off in (0..rb * rr).step_by(64) {
                _mm_prefetch::<_MM_HINT_T0>(bits_all.add(r0 * rb + off) as *const i8);
            }
        }
        // Prefetch the next group's rows (8 interleaved streams), one line
        // per row every 64 bytes of progress.
        let pf_rows = (n - i).saturating_sub(8).min(8);
        let pf = bits_all.add((r0 + i + 8).min(r0 + n - 1) * rb);
        let mut y = _mm256_setzero_ps();
        let mut part = _mm256_setzero_ps();
        let mut iacc = _mm256_setzero_si256();
        for g in 0..full {
            let k0 = 8 * g;
            if g % 2 == 0 {
                for r in 0..pf_rows {
                    _mm_prefetch::<_MM_HINT_T0>(pf.add(r * rb + 4 * k0) as *const i8);
                }
            }
            let ns = _mm256_loadu_si256(a.nsum.as_ptr().add(k0) as *const __m256i);
            // Per row: [I_k0 .. I_k0+7] = 2·S + (-Σq).
            let sv: [__m256i; 8] = std::array::from_fn(|r| {
                let p = |s: usize| sub_dot(rbits[r], k0 + s);
                let h01 = _mm256_hadd_epi16(p(0), p(1));
                let h23 = _mm256_hadd_epi16(p(2), p(3));
                let h45 = _mm256_hadd_epi16(p(4), p(5));
                let h67 = _mm256_hadd_epi16(p(6), p(7));
                let h = _mm256_hadd_epi16(_mm256_hadd_epi16(h01, h23), _mm256_hadd_epi16(h45, h67));
                // Lane halves hold the two 16-element halves of each block
                // (|S| ≤ 32·127 fits i16).
                let s16 = _mm_add_epi16(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepi16_epi32(s16), 1), ns)
            });
            // Transpose: tv[s] = sub-block k0 + s of the eight rows.
            let u0 = _mm256_unpacklo_epi32(sv[0], sv[1]);
            let u1 = _mm256_unpackhi_epi32(sv[0], sv[1]);
            let u2 = _mm256_unpacklo_epi32(sv[2], sv[3]);
            let u3 = _mm256_unpackhi_epi32(sv[2], sv[3]);
            let u4 = _mm256_unpacklo_epi32(sv[4], sv[5]);
            let u5 = _mm256_unpackhi_epi32(sv[4], sv[5]);
            let u6 = _mm256_unpacklo_epi32(sv[6], sv[7]);
            let u7 = _mm256_unpackhi_epi32(sv[6], sv[7]);
            let v0 = _mm256_unpacklo_epi64(u0, u2);
            let v1 = _mm256_unpackhi_epi64(u0, u2);
            let v2 = _mm256_unpacklo_epi64(u1, u3);
            let v3 = _mm256_unpackhi_epi64(u1, u3);
            let v4 = _mm256_unpacklo_epi64(u4, u6);
            let v5 = _mm256_unpackhi_epi64(u4, u6);
            let v6 = _mm256_unpacklo_epi64(u5, u7);
            let v7 = _mm256_unpackhi_epi64(u5, u7);
            let tv = [
                _mm256_permute2x128_si256::<0x20>(v0, v4),
                _mm256_permute2x128_si256::<0x20>(v1, v5),
                _mm256_permute2x128_si256::<0x20>(v2, v6),
                _mm256_permute2x128_si256::<0x20>(v3, v7),
                _mm256_permute2x128_si256::<0x31>(v0, v4),
                _mm256_permute2x128_si256::<0x31>(v1, v5),
                _mm256_permute2x128_si256::<0x31>(v2, v6),
                _mm256_permute2x128_si256::<0x31>(v3, v7),
            ];
            for (s, &iv) in tv.iter().enumerate() {
                let k = k0 + s;
                if spa == 1 {
                    part = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(iv),
                        _mm256_set1_ps(*a.d.get_unchecked(k)),
                        part,
                    );
                    if (k + 1).is_multiple_of(apw) {
                        y = _mm256_fmadd_ps(part, dw(k / apw), y);
                        part = _mm256_setzero_ps();
                    }
                } else {
                    iacc = _mm256_add_epi32(iacc, iv);
                    if (k + 1).is_multiple_of(spa) {
                        let j = k / spa;
                        let p = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(iacc),
                            _mm256_set1_ps(*a.d.get_unchecked(j)),
                            _mm256_setzero_ps(),
                        );
                        y = _mm256_fmadd_ps(p, dw(j / apw), y);
                        iacc = _mm256_setzero_si256();
                    }
                }
            }
        }
        // Tail sub-blocks (nsub % 8; starts on a weight-block boundary):
        // continue each row's chain in scalar code.
        let mut ys = [0.0f32; 8];
        _mm256_storeu_ps(ys.as_mut_ptr(), y);
        for (r, yr) in ys.iter_mut().enumerate().take(rr) {
            let (mut p, mut isum) = (0.0f32, 0i32);
            for k in 8 * full..nsub {
                let m = _mm256_madd_epi16(sub_dot(rbits[r], k), one16);
                let s = _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256(m, 1));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
                isum += 2 * _mm_cvtsi128_si32(s) + *a.nsum.get_unchecked(k);
                if (k + 1).is_multiple_of(spa) {
                    let j = k / spa;
                    p = (isum as f32).mul_add(*a.d.get_unchecked(j), p);
                    isum = 0;
                    if (j + 1).is_multiple_of(apw) {
                        *yr = p.mul_add(dwr[r * nb + j / apw], *yr);
                        p = 0.0;
                    }
                }
            }
        }
        out[i..i + rr].copy_from_slice(&ys[..rr]);
        i += 8;
    }
}

/// Row groups the 16-row AVX-512 GEMV prefetches ahead.
#[cfg(target_arch = "x86_64")]
const PF_GROUPS: usize = 1;

/// 16×16 transpose of 32-bit lanes: `out[s]` lane `r` = `v[r]` lane `s`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn transpose16_epi32(
    v: &[std::arch::x86_64::__m512i; 16],
) -> [std::arch::x86_64::__m512i; 16] {
    use std::arch::x86_64::*;
    // Within each 128-bit lane: 4×4 transposes of rows (4q .. 4q+4).
    let a: [__m512i; 16] = std::array::from_fn(|i| {
        let (q, h) = (i / 2, i % 2);
        if h == 0 {
            _mm512_unpacklo_epi32(v[2 * q], v[2 * q + 1])
        } else {
            _mm512_unpackhi_epi32(v[2 * q], v[2 * q + 1])
        }
    });
    let b: [__m512i; 16] = std::array::from_fn(|i| {
        let (q, h) = (i / 4, i % 4);
        let (x, y) = (a[4 * q + (h / 2)], a[4 * q + 2 + (h / 2)]);
        if h % 2 == 0 {
            _mm512_unpacklo_epi64(x, y)
        } else {
            _mm512_unpackhi_epi64(x, y)
        }
    });
    // b[4q + c] lane L holds rows 4q..4q+4 of column 4L + c. Combine the
    // 128-bit lanes across row quads.
    let c: [__m512i; 16] = std::array::from_fn(|i| {
        let (q2, h, col) = (i / 8, (i / 4) % 2, i % 4);
        let (x, y) = (b[8 * q2 + col], b[8 * q2 + 4 + col]);
        if h == 0 {
            _mm512_shuffle_i32x4::<0x88>(x, y)
        } else {
            _mm512_shuffle_i32x4::<0xDD>(x, y)
        }
    });
    // c[8·q2 + 4h + col] lanes: [rq 2q2 (L=h), rq 2q2+1 (L=h), rq 2q2 (L=h+2), rq 2q2+1 (L=h+2)]
    std::array::from_fn(|s| {
        let (l, col) = (s / 4, s % 4);
        let (h, hi) = (l % 2, l / 2);
        let (x, y) = (c[4 * h + col], c[8 + 4 * h + col]);
        if hi == 0 {
            _mm512_shuffle_i32x4::<0x88>(x, y)
        } else {
            _mm512_shuffle_i32x4::<0xDD>(x, y)
        }
    })
}

#[cfg(target_arch = "x86_64")]
macro_rules! avx512_kernel {
    ($(#[$doc:meta])* $name:ident, $feat:literal, $vnni:expr) => {
        $(#[$doc])*
        #[target_feature(enable = $feat)]
        pub unsafe fn $name(w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
            use std::arch::x86_64::*;
            check_rows(w, r0, &a, out);
            let cols = w.cols;
            let rb = cols / 8;
            let nb = cols / w.block;
            let nsub = cols / SUB;
            let apw = w.block / a.block; // activation blocks per weight block
            let spa = a.block / SUB; // sub-blocks per activation block
            let bits_all = w.bits.as_slice().as_ptr();
            let qp = a.qp.as_ptr();
            let m01 = _mm512_set1_epi32(0x0101_0101);
            let one16 = _mm512_set1_epi16(1);
            let gidx = _mm512_mullo_epi32(
                _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                _mm512_set1_epi32(nb as i32),
            );
            // S = Σ bit·q of one sub-block for 16 rows: `t` holds the
            // rows' sign words, `xp` the sub-block's permuted activations.
            let sub_dot = |t: __m512i, xp: *const u32| -> __m512i {
                macro_rules! grp {
                    ($g:literal) => {
                        (
                            _mm512_and_si512(_mm512_srli_epi32::<$g>(t), m01),
                            _mm512_set1_epi32(xp.add($g).read_unaligned() as i32),
                        )
                    };
                }
                let gs = [grp!(0), grp!(1), grp!(2), grp!(3), grp!(4), grp!(5), grp!(6), grp!(7)];
                if $vnni {
                    let (mut s0, mut s1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                    for g in (0..8).step_by(2) {
                        s0 = _mm512_dpbusd_epi32(s0, gs[g].0, gs[g].1);
                        s1 = _mm512_dpbusd_epi32(s1, gs[g + 1].0, gs[g + 1].1);
                    }
                    _mm512_add_epi32(s0, s1)
                } else {
                    // i16 pair sums ≤ 2·127, eight groups ≤ 2032.
                    let mut s16 = _mm512_setzero_si512();
                    for (wv, xv) in gs {
                        s16 = _mm512_add_epi16(s16, _mm512_maddubs_epi16(wv, xv));
                    }
                    _mm512_madd_epi16(s16, one16)
                }
            };
            let mut dwr = vec![0.0f32; 16 * nb];
            let n = out.len();
            let mut i = 0;
            while i < n {
                let rr = (n - i).min(16);
                // Rows beyond `rr` repeat the last row (results discarded).
                let rows: [usize; 16] = std::array::from_fn(|k| r0 + i + k.min(rr - 1));
                let rbits: [*const u8; 16] = std::array::from_fn(|k| bits_all.add(rows[k] * rb));
                // Software prefetch of the same chunk of the row group
                // PF_GROUPS ahead (16 interleaved row streams are more than
                // the hardware prefetchers follow), spread over the loop.
                if i == 0 {
                    // The first group of a call has nothing ahead of it.
                    for off in (0..rb * rr).step_by(64) {
                        _mm_prefetch::<_MM_HINT_T0>(bits_all.add(r0 * rb + off) as *const i8);
                    }
                }
                let pf_rows = (n - i).saturating_sub(16 * PF_GROUPS).min(16);
                let pf = bits_all.add((r0 + i + 16 * PF_GROUPS).min(r0 + n - 1) * rb);
                row_scales_f32(w, &rows, &mut dwr);
                let dw = |b: usize| _mm512_i32gather_ps::<4>(gidx, dwr.as_ptr().add(b));
                let mut y = _mm512_setzero_ps();
                let mut part = _mm512_setzero_ps();
                let mut iacc = _mm512_setzero_si512();
                for kc in (0..nsub).step_by(16) {
                    // Sign words of sub-blocks kc .. kc+16 of the 16 rows,
                    // transposed to one vector per sub-block.
                    let cnt = (nsub - kc).min(16);
                    for r in 0..pf_rows {
                        _mm_prefetch::<_MM_HINT_T0>(pf.add(r * rb + 4 * kc) as *const i8);
                    }
                    let mask = ((1u32 << cnt) - 1) as u16;
                    let v: [__m512i; 16] = std::array::from_fn(|r| {
                        _mm512_maskz_loadu_epi32(mask, rbits[r].add(4 * kc) as *const i32)
                    });
                    let t = transpose16_epi32(&v);
                    for (s, &ts) in t.iter().enumerate().take(cnt) {
                        let k = kc + s;
                        let sv = sub_dot(ts, qp.add(SUB * k) as *const u32);
                        // Exact block dot: 2·S - Σq.
                        let iv = _mm512_add_epi32(
                            _mm512_slli_epi32::<1>(sv),
                            _mm512_set1_epi32(*a.nsum.get_unchecked(k)),
                        );
                        if spa == 1 {
                            part = _mm512_fmadd_ps(
                                _mm512_cvtepi32_ps(iv),
                                _mm512_set1_ps(*a.d.get_unchecked(k)),
                                part,
                            );
                            if (k + 1).is_multiple_of(apw) {
                                y = _mm512_fmadd_ps(part, dw(k / apw), y);
                                part = _mm512_setzero_ps();
                            }
                        } else {
                            iacc = _mm512_add_epi32(iacc, iv);
                            if (k + 1).is_multiple_of(spa) {
                                let j = k / spa;
                                let pj = _mm512_fmadd_ps(
                                    _mm512_cvtepi32_ps(iacc),
                                    _mm512_set1_ps(*a.d.get_unchecked(j)),
                                    _mm512_setzero_ps(),
                                );
                                y = _mm512_fmadd_ps(pj, dw(j / apw), y);
                                iacc = _mm512_setzero_si512();
                            }
                        }
                    }
                }
                let mut ys = [0.0f32; 16];
                _mm512_storeu_ps(ys.as_mut_ptr(), y);
                out[i..i + rr].copy_from_slice(&ys[..rr]);
                i += 16;
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
avx512_kernel!(
    /// AVX-512BW row kernel (`vpmaddubsw` + `vpmaddwd` byte sums),
    /// bit-identical to [`dot_rows_scalar`]: 16 rows at a time, per row the
    /// exact block sums are transposed to one row per lane before the float
    /// part.
    ///
    /// # Safety
    /// The CPU must support `avx512f`, `avx512bw`, `avx2`, `fma` and `f16c`.
    /// Shapes are checked (panics); all loads stay inside the weight rows
    /// and activation row.
    dot_rows_avx512bw,
    "avx512f,avx512bw,avx2,fma,f16c",
    false
);

#[cfg(target_arch = "x86_64")]
avx512_kernel!(
    /// AVX-512 VNNI row kernel (`vpdpbusd` byte sums), bit-identical to
    /// [`dot_rows_scalar`] (see [`dot_rows_avx512bw`]).
    ///
    /// # Safety
    /// The CPU must support `avx512f`, `avx512bw`, `avx512vnni`, `avx2`, `fma`
    /// and `f16c`. Shapes are checked (panics); all loads stay inside the
    /// weight rows and activation row.
    dot_rows_avx512vnni,
    "avx512f,avx512bw,avx512vnni,avx2,fma,f16c",
    true
);

/// Rows `r0 .. r0 + out.len()` of `W · x` with an explicit kernel variant.
/// Panics if the CPU does not support `isa` or on a shape mismatch.
pub fn dot_rows_isa(isa: Isa, w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
    assert!(
        isa.supported(),
        "Q1 kernel {isa:?} not supported on this CPU"
    );
    match isa {
        Isa::Scalar => dot_rows_scalar(w, r0, a, out),
        // SAFETY: support checked above; the kernels check shapes.
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { dot_rows_avx2(w, r0, a, out) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bw => unsafe { dot_rows_avx512bw(w, r0, a, out) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Vnni => unsafe { dot_rows_avx512vnni(w, r0, a, out) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!(),
    }
}

/// Output rows per parallel task: enough tasks for load balance, multiple of 4.
fn task_rows(rows: usize, per_thread: usize, min: usize) -> usize {
    let t = rayon::current_num_threads().max(1);
    rows.div_ceil(t * per_thread).max(min).next_multiple_of(16)
}

/// `out = W · x` for a single quantized activation row (`acts.n() == 1`),
/// parallel over output rows.
pub fn gemv(w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    assert_eq!(acts.n(), 1, "gemv: expected one activation row");
    assert_eq!(out.len(), w.rows, "gemv: output length mismatch");
    let a = acts.row(0);
    let isa = Isa::best();
    let chunk = task_rows(w.rows, 4, 32);
    if chunk >= w.rows {
        dot_rows_isa(isa, w, 0, a, out);
        return;
    }
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(c, o)| dot_rows_isa(isa, w, c * chunk, a, o));
}

/// From this many tokens on, [`gemm`] uses the token-tiled kernel; below,
/// expanding each slab's bits costs more than it saves.
pub const GEMM_TILED_MIN_TOKENS: usize = 8;

/// `out[t] = W · x[t]` for every activation row `t` (`out` is
/// `[acts.n(), rows]` row-major). One row uses [`gemv`]; a few use the row
/// kernels; [`GEMM_TILED_MIN_TOKENS`] or more the token-tiled kernel
/// ([`GemmKernel::best`]).
pub fn gemm(w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    let n = acts.n();
    assert_eq!(out.len(), n * w.rows, "gemm: output length mismatch");
    if n == 1 {
        return gemv(w, acts, out);
    }
    let k = if n < GEMM_TILED_MIN_TOKENS {
        GemmKernel::Rows(Isa::best())
    } else {
        GemmKernel::best()
    };
    gemm_with(k, w, acts, out);
}

/// Multi-token kernel variant of [`gemm_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKernel {
    /// The row kernel of the given ISA, once per token, over slabs of rows
    /// kept hot in cache.
    Rows(Isa),
    /// Token-tiled kernel (see [`gemm_with`]) with the given inner product.
    Tiled(TileIsa),
}

/// Inner product of the token-tiled GEMM kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileIsa {
    /// ymm, `vpmaddubsw` + `vpmaddwd` (`avx2`, `fma`, `f16c`).
    Avx2,
    /// ymm, VEX `vpdpbusd` (AVX2 tier + `avxvnni`).
    AvxVnni,
    /// zmm, `vpmaddubsw` + `vpmaddwd` (`avx512f`, `avx512bw`).
    Avx512Bw,
    /// zmm, `vpdpbusd` (`avx512f`, `avx512bw`, `avx512vnni`).
    Avx512Vnni,
}

impl TileIsa {
    pub fn supported(self) -> bool {
        let f = cpu_features();
        let avx2 = f.avx2_fma() && f.f16c;
        match self {
            TileIsa::Avx2 => avx2,
            TileIsa::AvxVnni => avx2 && f.avxvnni,
            TileIsa::Avx512Bw => avx2 && f.avx512bw(),
            TileIsa::Avx512Vnni => avx2 && f.avx512_vnni(),
        }
    }

    /// Tokens per register tile.
    pub fn tile(self) -> usize {
        match self {
            TileIsa::Avx2 | TileIsa::AvxVnni => 4,
            TileIsa::Avx512Bw | TileIsa::Avx512Vnni => 8,
        }
    }

    /// Output rows per kernel call (SIMD lanes).
    pub fn lanes(self) -> usize {
        match self {
            TileIsa::Avx2 | TileIsa::AvxVnni => 8,
            TileIsa::Avx512Bw | TileIsa::Avx512Vnni => 16,
        }
    }
}

impl GemmKernel {
    /// Fastest variant supported by this CPU.
    pub fn best() -> Self {
        [
            TileIsa::Avx512Vnni,
            TileIsa::Avx512Bw,
            TileIsa::AvxVnni,
            TileIsa::Avx2,
        ]
        .into_iter()
        .find(|t| t.supported())
        .map_or(GemmKernel::Rows(Isa::best()), GemmKernel::Tiled)
    }

    pub fn supported(self) -> bool {
        match self {
            GemmKernel::Rows(i) => i.supported(),
            GemmKernel::Tiled(t) => t.supported(),
        }
    }

    /// All variants this CPU supports.
    pub fn all_supported() -> Vec<GemmKernel> {
        Isa::all_supported()
            .into_iter()
            .map(GemmKernel::Rows)
            .chain(
                [
                    TileIsa::Avx2,
                    TileIsa::AvxVnni,
                    TileIsa::Avx512Bw,
                    TileIsa::Avx512Vnni,
                ]
                .into_iter()
                .filter(|t| t.supported())
                .map(GemmKernel::Tiled),
            )
            .collect()
    }
}

/// [`gemm`] with an explicit kernel (any `acts.n() >= 1`). Panics if the CPU
/// does not support it or on a shape mismatch.
///
/// # Token-tiled kernel
///
/// The rows are cut into slabs of `L` = 8 (ymm) or 16 (zmm) rows, one row
/// per i32/f32 SIMD lane. A slab's sign bits are expanded once into bytes
/// `{0, 2}` laid out `[sub-block k][group j][lane][4 bytes]`, where group `j`
/// holds bits `j, 8+j, 16+j, 24+j` of the sub-block (the [`perm32`] order of
/// the activations), together with its weight scales per lane. Each
/// activation `u32` (four permuted int8 values of one token) is broadcast
/// and multiplied with one expanded group by `vpdpbusd` (or `vpmaddubsw` +
/// `vpmaddwd`), so one instruction does `4·L` MACs and the bit decoding is
/// paid once per slab instead of once per token. Starting each activation
/// block's i32 accumulator at `-Σq` gives the exact `Σ ±q`; it is then scaled
/// by the per-lane weight scales and the token's activation scale (one
/// multiply + one FMA per block, token and `L` rows). A tile of `T` tokens
/// keeps its `T` accumulators in registers and reuses each expanded group
/// `T` times.
///
/// Work is split over (token block, slab range) items so every thread works
/// on the same activations at a time (shared in L3) and its slab stays in L2.
pub fn gemm_with(kernel: GemmKernel, w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    assert!(
        kernel.supported(),
        "Q1 GEMM kernel {kernel:?} not supported on this CPU"
    );
    let n = acts.n();
    assert_eq!(out.len(), n * w.rows, "gemm: output length mismatch");
    if n == 0 || w.rows == 0 {
        return;
    }
    // Shape checks shared by every kernel (row 0 stands for all rows).
    check_rows(w, 0, &acts.row(0), &out[..0]);
    match kernel {
        GemmKernel::Rows(isa) => gemm_rows(isa, w, acts, out),
        GemmKernel::Tiled(t) => gemm_tiled(t, w, acts, out),
    }
}

/// Per-token row kernels over slabs of rows kept hot in cache: each task
/// sweeps all activation rows for its slab, so the weights are streamed from
/// memory once per call instead of once per token.
fn gemm_rows(isa: Isa, w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    let n = acts.n();
    let rows = w.rows;
    let chunk = task_rows(rows, 2, 16);
    // tmp: per chunk c, a [n][chunk rows] slab.
    let mut tmp = vec![0.0f32; n * rows];
    tmp.par_chunks_mut(chunk * n)
        .enumerate()
        .for_each(|(c, slab)| {
            let cr = slab.len() / n;
            for (t, o) in slab.chunks_exact_mut(cr).enumerate() {
                dot_rows_isa(isa, w, c * chunk, acts.row(t), o);
            }
        });
    for (c, slab) in tmp.chunks(chunk * n).enumerate() {
        let cr = slab.len() / n;
        for (t, s) in slab.chunks_exact(cr).enumerate() {
            out[t * rows + c * chunk..t * rows + c * chunk + cr].copy_from_slice(s);
        }
    }
}

/// Expanded weights of one slab of `lanes` rows (see [`gemm_with`]).
#[derive(Default)]
struct Slab {
    /// `[sub-block][group][lane]`, four bytes `{0, 2}` each.
    wexp: Vec<u32>,
    /// Weight scales `[weight block][lane]` (0 for lanes past the last row).
    dw: Vec<f32>,
    /// Scratch: the slab's scales `[lane][weight block]`.
    rs: Vec<f32>,
}

impl Slab {
    /// Expand rows `r0 .. r0 + lanes` of `w` for the kernels of `isa`.
    fn fill(&mut self, isa: TileIsa, w: &Q1Weight, r0: usize) {
        let lanes = isa.lanes();
        let nsub = w.cols / SUB;
        let nb = w.cols / w.block;
        assert!(r0 < w.rows);
        assert!(
            lanes * (w.cols / 8) < i32::MAX as usize,
            "Q1 GEMM: rows too long"
        );
        self.wexp.resize(nsub * 8 * lanes, 0);
        self.dw.clear();
        self.dw.resize(nb * lanes, 0.0);
        let valid = lanes.min(w.rows - r0);
        let rows: Vec<usize> = (r0..r0 + valid).collect();
        self.rs.resize(valid * nb, 0.0);
        // SAFETY: every tiled ISA includes the AVX2 tier (F16C).
        #[cfg(target_arch = "x86_64")]
        unsafe {
            row_scales_f32(w, &rows, &mut self.rs)
        };
        for (lane, rsl) in self.rs.chunks_exact(nb).enumerate() {
            for (b, &d) in rsl.iter().enumerate() {
                self.dw[b * lanes + lane] = d;
            }
        }
        // SAFETY: `isa` is supported (the caller dispatches on it); the
        // gathers read sub-block words of rows r0 .. r0 + valid only.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            match lanes {
                16 => expand_slab_avx512(w, r0, valid, &mut self.wexp),
                _ => expand_slab_avx2(w, r0, valid, &mut self.wexp),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unreachable!("tiled GEMM is x86_64 only");
    }
}

/// Transpose and expand the sign bits of `valid` (≤ 16) rows from `r0`:
/// entry `[k][j][lane]` holds byte `p` = 2 · bit `8p + j` of sub-block `k`
/// of that lane's row (0 for lanes ≥ `valid`).
///
/// # Safety
/// The CPU must support `avx512f`. `r0 + valid <= rows`, `wexp` holds
/// `cols / 32 · 8 · 16` entries and `16 · cols / 8 < 2^31`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn expand_slab_avx512(w: &Q1Weight, r0: usize, valid: usize, wexp: &mut [u32]) {
    use std::arch::x86_64::*;
    let rb = w.cols / 8;
    let nsub = w.cols / SUB;
    assert!(r0 + valid <= w.rows && valid <= 16 && wexp.len() == nsub * 8 * 16);
    let base = w.bits.as_slice().as_ptr().add(r0 * rb);
    let idx = _mm512_mullo_epi32(
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        _mm512_set1_epi32(rb as i32),
    );
    let mask: __mmask16 = ((1u32 << valid) - 1) as u16;
    let m01 = _mm512_set1_epi32(0x0101_0101);
    let dst = wexp.as_mut_ptr() as *mut __m512i;
    for k in 0..nsub {
        let v = _mm512_mask_i32gather_epi32::<1>(
            _mm512_setzero_si512(),
            mask,
            idx,
            base.add(4 * k) as *const i32,
        );
        macro_rules! group {
            ($j:literal) => {
                let e = _mm512_and_si512(_mm512_srli_epi32::<$j>(v), m01);
                _mm512_storeu_si512(dst.add(8 * k + $j), _mm512_add_epi32(e, e));
            };
        }
        group!(0);
        group!(1);
        group!(2);
        group!(3);
        group!(4);
        group!(5);
        group!(6);
        group!(7);
    }
}

/// [`expand_slab_avx512`] for 8 lanes.
///
/// # Safety
/// The CPU must support `avx2`. `r0 + valid <= rows`, `wexp` holds
/// `cols / 32 · 8 · 8` entries and `8 · cols / 8 < 2^31`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn expand_slab_avx2(w: &Q1Weight, r0: usize, valid: usize, wexp: &mut [u32]) {
    use std::arch::x86_64::*;
    let rb = w.cols / 8;
    let nsub = w.cols / SUB;
    assert!(r0 + valid <= w.rows && valid <= 8 && wexp.len() == nsub * 8 * 8);
    let base = w.bits.as_slice().as_ptr().add(r0 * rb);
    let idx = _mm256_mullo_epi32(
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        _mm256_set1_epi32(rb as i32),
    );
    let mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(valid as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    );
    let m01 = _mm256_set1_epi32(0x0101_0101);
    let dst = wexp.as_mut_ptr() as *mut __m256i;
    for k in 0..nsub {
        let v = _mm256_mask_i32gather_epi32::<1>(
            _mm256_setzero_si256(),
            base.add(4 * k) as *const i32,
            idx,
            mask,
        );
        macro_rules! group {
            ($j:literal) => {
                let e = _mm256_and_si256(_mm256_srli_epi32::<$j>(v), m01);
                _mm256_storeu_si256(dst.add(8 * k + $j), _mm256_add_epi32(e, e));
            };
        }
        group!(0);
        group!(1);
        group!(2);
        group!(3);
        group!(4);
        group!(5);
        group!(6);
        group!(7);
    }
}

thread_local! {
    static SLAB: std::cell::RefCell<Slab> = std::cell::RefCell::new(Slab::default());
}

/// Raw output pointer shared by the GEMM tasks, which write disjoint
/// `(token, row)` ranges.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: only used to write disjoint elements from different tasks.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Tokens per activation block of a task (L3-resident).
const GEMM_TOKEN_BLOCK: usize = 256;

/// The activations of all tokens regrouped by token tile of `tile` tokens
/// (tokens past `n` are zero), so the kernel addresses a whole tile from one
/// base pointer with constant offsets.
#[derive(Default)]
struct PackedActs {
    tile: usize,
    nsub: usize,
    nab: usize,
    /// `[tile][sub-block][token][32]` ([`perm32`] order).
    x: Vec<i8>,
    /// `-Σq` per activation block: `[tile][block][token]`.
    ns: Vec<i32>,
    /// Activation scales: `[tile][block][token]`.
    dx: Vec<f32>,
}

impl PackedActs {
    fn fill(&mut self, a: &Q8Acts, tile: usize) {
        let (n, cols) = (a.n, a.cols);
        let nsub = cols / SUB;
        let nab = cols / a.block;
        let ntiles = n.div_ceil(tile);
        self.tile = tile;
        self.nsub = nsub;
        self.nab = nab;
        self.x.resize(ntiles * tile * cols, 0);
        self.ns.resize(ntiles * tile * nab, 0);
        self.dx.resize(ntiles * tile * nab, 0.0);
        let nsb: &[i32] = if a.block == SUB { &a.nsum } else { &a.nsumb };
        let fill_tile = |c: usize, x: &mut [i8], ns: &mut [i32], dx: &mut [f32]| {
            for i in 0..tile {
                let t = c * tile + i;
                if t < n {
                    let src = &a.qp[t * cols..(t + 1) * cols];
                    for (k, blk) in src.as_chunks::<SUB>().0.iter().enumerate() {
                        x[(k * tile + i) * SUB..(k * tile + i + 1) * SUB].copy_from_slice(blk);
                    }
                    for j in 0..nab {
                        ns[j * tile + i] = nsb[t * nab + j];
                        dx[j * tile + i] = a.d[t * nab + j];
                    }
                } else {
                    for k in 0..nsub {
                        x[(k * tile + i) * SUB..(k * tile + i + 1) * SUB].fill(0);
                    }
                    for j in 0..nab {
                        ns[j * tile + i] = 0;
                        dx[j * tile + i] = 0.0;
                    }
                }
            }
        };
        let it = self
            .x
            .par_chunks_mut(tile * cols)
            .zip(self.ns.par_chunks_mut(tile * nab))
            .zip(self.dx.par_chunks_mut(tile * nab))
            .enumerate();
        if n * cols >= 1 << 16 {
            it.for_each(|(c, ((x, ns), dx))| fill_tile(c, x, ns, dx));
        } else {
            for (c, ((x, ns), dx)) in self
                .x
                .chunks_mut(tile * cols)
                .zip(self.ns.chunks_mut(tile * nab))
                .zip(self.dx.chunks_mut(tile * nab))
                .enumerate()
            {
                fill_tile(c, x, ns, dx);
            }
        }
    }
}

thread_local! {
    static PACKED: std::cell::RefCell<PackedActs> = std::cell::RefCell::new(PackedActs::default());
}

fn gemm_tiled(isa: TileIsa, w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    PACKED.with_borrow_mut(|pa| {
        pa.fill(acts, isa.tile());
        gemm_tiled_packed(isa, w, acts, pa, out)
    })
}

fn gemm_tiled_packed(isa: TileIsa, w: &Q1Weight, acts: &Q8Acts, pa: &PackedActs, out: &mut [f32]) {
    let n = acts.n();
    let lanes = isa.lanes();
    let tile = pa.tile;
    let nslab = w.rows.div_ceil(lanes);
    let ntiles = n.div_ceil(tile);
    let ntb = n.div_ceil(GEMM_TOKEN_BLOCK);
    let tpb = ntiles.div_ceil(ntb); // tiles per token block
    let ntb = ntiles.div_ceil(tpb);
    let threads = rayon::current_num_threads().max(1);
    // Items per token block: enough for load balance, at most one per slab.
    let per_tb = (16 * threads).div_ceil(ntb).clamp(1, nslab);
    let spi = nslab.div_ceil(per_tb);
    let per_tb = nslab.div_ceil(spi);
    let optr = OutPtr(out.as_mut_ptr());
    let total = out.len();
    (0..ntb * per_tb).into_par_iter().for_each(|item| {
        let (tbi, si) = (item / per_tb, item % per_tb);
        let (c0, c1) = (tbi * tpb, ((tbi + 1) * tpb).min(ntiles));
        SLAB.with_borrow_mut(|slab| {
            for sl in si * spi..((si + 1) * spi).min(nslab) {
                let r0 = sl * lanes;
                slab.fill(isa, w, r0);
                let job = TileJob {
                    w,
                    acts,
                    pa,
                    slab,
                    r0,
                    c0,
                    c1,
                    out: optr,
                    out_len: total,
                };
                // SAFETY: `isa` is supported (checked by `gemm_with`); the
                // shapes were checked there and `job.check` re-checks what
                // the kernel indexes; tasks write disjoint (token, row)
                // ranges of `out`.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    match isa {
                        TileIsa::Avx2 => tile_avx2(&job),
                        TileIsa::AvxVnni => tile_avxvnni(&job),
                        TileIsa::Avx512Bw => tile_avx512bw(&job),
                        TileIsa::Avx512Vnni => tile_avx512vnni(&job),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let _ = (isa, job);
                    unreachable!()
                }
            }
        });
    });
}

/// One slab × range of token tiles of the tiled GEMM.
struct TileJob<'a> {
    w: &'a Q1Weight,
    acts: &'a Q8Acts,
    pa: &'a PackedActs,
    slab: &'a Slab,
    r0: usize,
    /// Token tiles `c0..c1`.
    c0: usize,
    c1: usize,
    out: OutPtr,
    out_len: usize,
}

impl TileJob<'_> {
    /// Checks the invariants the kernels rely on (cheap; once per slab).
    fn check(&self, lanes: usize, tile: usize) {
        let (w, a, s, pa) = (self.w, self.acts, self.slab, self.pa);
        let nsub = w.cols / SUB;
        let nab = w.cols / a.block;
        assert!(self.r0 < w.rows && self.c0 < self.c1);
        assert!(self.c1 <= a.n.div_ceil(tile) && pa.tile == tile);
        assert_eq!(a.cols, w.cols);
        assert!(w.block.is_multiple_of(a.block) && a.block.is_multiple_of(SUB));
        assert!(pa.nsub == nsub && pa.nab == nab);
        assert!(pa.x.len() >= self.c1 * tile * w.cols);
        assert!(pa.ns.len() >= self.c1 * tile * nab && pa.dx.len() >= self.c1 * tile * nab);
        assert_eq!(s.wexp.len(), nsub * 8 * lanes);
        assert_eq!(s.dw.len(), w.cols / w.block * lanes);
        assert_eq!(self.out_len, a.n * w.rows);
    }

    /// Store `vals` (lane `i` = row `r0 + i`) for token `t`, clipped to the
    /// matrix rows; tokens past the last are dropped.
    #[inline(always)]
    fn store(&self, t: usize, vals: &[f32]) {
        if t >= self.acts.n {
            return;
        }
        let rows = self.w.rows;
        let cnt = vals.len().min(rows - self.r0);
        let off = t * rows + self.r0;
        assert!(off + cnt <= self.out_len);
        // SAFETY: in bounds (checked); tasks write disjoint ranges.
        unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), self.out.0.add(off), cnt) }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! tile_kernel {
    (
        $(#[$doc:meta])* $name:ident, $feat:literal, lanes = $lanes:literal, tile = $tile:literal,
        vec = $vi:ty, fvec = $vf:ty,
        set1 = $set1:path, loadi = $loadi:path, loadf = $loadf:path,
        storef = $storef:path, setzf = $setzf:path, set1f = $set1f:path, cvt = $cvt:path,
        fma = $fma:path,
        step = |$iacc:ident, $wv:ident, $xp:ident| $step:expr
    ) => {
        $(#[$doc])*
        #[target_feature(enable = $feat)]
        unsafe fn $name(job: &TileJob<'_>) {
            use std::arch::x86_64::*;
            const L: usize = $lanes;
            const T: usize = $tile;
            job.check(L, T);
            let (w, a, slab, pa) = (job.w, job.acts, job.slab, job.pa);
            let cols = w.cols;
            let nab = cols / a.block;
            let nsub = cols / SUB;
            let spa = a.block / SUB; // sub-blocks per activation block
            let apw = w.block / a.block; // activation blocks per weight block
            let nb = cols / w.block;
            let wexp = slab.wexp.as_ptr() as *const u8;
            let dw = slab.dw.as_ptr();
            for c in job.c0..job.c1 {
                // Tile c: token i of sub-block k at xb + (k·T + i)·32.
                let xb = pa.x.as_ptr().add(c * nsub * T * SUB) as *const u8;
                let nsb = pa.ns.as_ptr().add(c * nab * T);
                let dxb = pa.dx.as_ptr().add(c * nab * T);
                let mut facc: [$vf; T] = [$setzf(); T];
                for b in 0..nb {
                    let dwv = $loadf(dw.add(b * L));
                    // Σ_j cvt(I_j) · d_x[j] over the activation blocks of
                    // weight block b, then one FMA with the lane scales.
                    let mut part: [$vf; T] = [$setzf(); T];
                    for j in b * apw..(b + 1) * apw {
                        let mut iacc: [$vi; T] = std::array::from_fn(|i| $set1(*nsb.add(j * T + i)));
                        for s in 0..spa {
                            let k = j * spa + s;
                            let wk = wexp.add(k * 8 * 4 * L);
                            let wv: [$vi; 8] =
                                std::array::from_fn(|g| $loadi(wk.add(g * 4 * L) as *const _));
                            let xk = xb.add(k * T * SUB);
                            for i in 0..T {
                                let $xp = xk.add(i * SUB) as *const u32;
                                let ($iacc, $wv) = (iacc[i], &wv);
                                iacc[i] = $step;
                            }
                        }
                        for i in 0..T {
                            part[i] = $fma($cvt(iacc[i]), $set1f(*dxb.add(j * T + i)), part[i]);
                        }
                    }
                    for i in 0..T {
                        facc[i] = $fma(part[i], dwv, facc[i]);
                    }
                }
                for i in 0..T {
                    let mut v = [0.0f32; L];
                    $storef(v.as_mut_ptr(), facc[i]);
                    job.store(c * T + i, &v);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
tile_kernel!(
    /// Token-tiled GEMM, ymm, `vpmaddubsw` (i16 sums per sub-block) +
    /// `vpmaddwd` to i32.
    ///
    /// # Safety
    /// The CPU must support `avx2`, `fma` and `f16c`. `job` is checked.
    tile_avx2, "avx2,fma,f16c", lanes = 8, tile = 4,
    vec = __m256i, fvec = __m256,
    set1 = _mm256_set1_epi32, loadi = _mm256_loadu_si256,
    loadf = _mm256_loadu_ps, storef = _mm256_storeu_ps, setzf = _mm256_setzero_ps,
    set1f = _mm256_set1_ps, cvt = _mm256_cvtepi32_ps,
    fma = _mm256_fmadd_ps,
    // i16 pair sums ≤ 2·2·127, eight groups ≤ 4064: no overflow.
    step = |acc, wv, xp| {
        let mut s16 = _mm256_setzero_si256();
        for (g, &wg) in wv.iter().enumerate() {
            let x = _mm256_set1_epi32(xp.add(g).read_unaligned() as i32);
            s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(wg, x));
        }
        _mm256_add_epi32(acc, _mm256_madd_epi16(s16, _mm256_set1_epi16(1)))
    }
);

#[cfg(target_arch = "x86_64")]
tile_kernel!(
    /// Token-tiled GEMM, ymm, VEX `vpdpbusd`.
    ///
    /// # Safety
    /// The CPU must support `avx2`, `fma`, `f16c` and `avxvnni`. `job` is
    /// checked.
    tile_avxvnni, "avx2,fma,f16c,avxvnni", lanes = 8, tile = 4,
    vec = __m256i, fvec = __m256,
    set1 = _mm256_set1_epi32, loadi = _mm256_loadu_si256,
    loadf = _mm256_loadu_ps, storef = _mm256_storeu_ps, setzf = _mm256_setzero_ps,
    set1f = _mm256_set1_ps, cvt = _mm256_cvtepi32_ps,
    fma = _mm256_fmadd_ps,
    step = |acc, wv, xp| {
        let x = |g: usize| _mm256_set1_epi32(xp.add(g).read_unaligned() as i32);
        let (mut a0, mut a1) = (acc, _mm256_setzero_si256());
        for g in (0..8).step_by(2) {
            a0 = _mm256_dpbusd_avx_epi32(a0, wv[g], x(g));
            a1 = _mm256_dpbusd_avx_epi32(a1, wv[g + 1], x(g + 1));
        }
        _mm256_add_epi32(a0, a1)
    }
);

#[cfg(target_arch = "x86_64")]
tile_kernel!(
    /// Token-tiled GEMM, zmm, `vpmaddubsw` + `vpmaddwd`.
    ///
    /// # Safety
    /// The CPU must support `avx512f`, `avx512bw`, `avx2`, `fma` and `f16c`.
    /// `job` is checked.
    tile_avx512bw, "avx512f,avx512bw,avx2,fma,f16c", lanes = 16, tile = 8,
    vec = __m512i, fvec = __m512,
    set1 = _mm512_set1_epi32, loadi = _mm512_loadu_si512,
    loadf = _mm512_loadu_ps, storef = _mm512_storeu_ps, setzf = _mm512_setzero_ps,
    set1f = _mm512_set1_ps, cvt = _mm512_cvtepi32_ps,
    fma = _mm512_fmadd_ps,
    step = |acc, wv, xp| {
        let mut s16 = _mm512_setzero_si512();
        for (g, &wg) in wv.iter().enumerate() {
            let x = _mm512_set1_epi32(xp.add(g).read_unaligned() as i32);
            s16 = _mm512_add_epi16(s16, _mm512_maddubs_epi16(wg, x));
        }
        _mm512_add_epi32(acc, _mm512_madd_epi16(s16, _mm512_set1_epi16(1)))
    }
);

#[cfg(target_arch = "x86_64")]
tile_kernel!(
    /// Token-tiled GEMM, zmm, `vpdpbusd`.
    ///
    /// # Safety
    /// The CPU must support `avx512f`, `avx512bw`, `avx512vnni`, `avx2`,
    /// `fma` and `f16c`. `job` is checked.
    tile_avx512vnni, "avx512f,avx512bw,avx512vnni,avx2,fma,f16c", lanes = 16, tile = 8,
    vec = __m512i, fvec = __m512,
    set1 = _mm512_set1_epi32, loadi = _mm512_loadu_si512,
    loadf = _mm512_loadu_ps, storef = _mm512_storeu_ps, setzf = _mm512_setzero_ps,
    set1f = _mm512_set1_ps, cvt = _mm512_cvtepi32_ps,
    fma = _mm512_fmadd_ps,
    // One chain per token: the next sub-block's chains are independent, so
    // out-of-order execution overlaps them (splitting measured no faster).
    step = |acc, wv, xp| {
        let x = |g: usize| _mm512_set1_epi32(xp.add(g).read_unaligned() as i32);
        let mut a0 = acc;
        for (g, &wg) in wv.iter().enumerate() {
            a0 = _mm512_dpbusd_epi32(a0, wg, x(g));
        }
        a0
    }
);

/// Convenience: quantize `x` (`n` rows) and compute `W · x` into `out`.
pub fn matmul_f32(w: &Q1Weight, x: &[f32], n: usize, acts: &mut Q8Acts, out: &mut [f32]) {
    acts.quantize(x, n, w.cols, w.block);
    gemm(w, acts, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gguf::dequantize_data;

    fn rand_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                *seed as u8
            })
            .collect()
    }

    /// Random GGUF blocks with finite, positive f16 scales.
    fn rand_gguf(dtype: u32, rows: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
        let (block, bbytes, soff, _) = gguf_q1_geometry(dtype).unwrap();
        let mut data = rand_bytes(rows * cols / block * bbytes, seed);
        for blk in data.chunks_exact_mut(bbytes) {
            // exponent 10..20 (0.001 .. 32), random mantissa
            let h = (((blk[soff] as u16) % 10 + 10) << 10) | (blk[soff + 1] as u16 & 0x3ff);
            blk[soff..soff + 2].copy_from_slice(&h.to_le_bytes());
        }
        data
    }

    #[test]
    fn gguf_roundtrip_and_dequant_are_lossless() {
        let mut seed = 0x1234_5678_9abc_def1;
        for dtype in [PRISM_Q1_0_UPSTREAM, PRISM_Q1_0, PRISM_Q1_0_G128] {
            for (rows, cols) in [(1, 128), (7, 256), (3, 384)] {
                let data = rand_gguf(dtype, rows, cols, &mut seed);
                let w = Q1Weight::from_gguf(&data, dtype, rows, cols).unwrap();
                assert_eq!(w.to_gguf(dtype).unwrap(), data, "round trip dtype {dtype}");
                let reference = dequantize_data(&data, dtype, rows * cols).unwrap();
                let mut row = vec![0.0f32; cols];
                for r in 0..rows {
                    w.dequantize_row_into(r, &mut row);
                    assert_eq!(
                        row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        reference[r * cols..(r + 1) * cols]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "dequant dtype {dtype} row {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn concat_rows_matches_parts() {
        let mut seed = 99;
        let a = Q1Weight::from_gguf(&rand_gguf(41, 3, 256, &mut seed), 41, 3, 256).unwrap();
        let b = Q1Weight::from_gguf(&rand_gguf(41, 5, 256, &mut seed), 41, 5, 256).unwrap();
        let c = Q1Weight::concat_rows(&[&a, &b]).unwrap();
        assert_eq!(c.rows(), 8);
        let (mut x, mut y) = (vec![0.0; 256], vec![0.0; 256]);
        for r in 0..8 {
            c.dequantize_row_into(r, &mut x);
            if r < 3 {
                a.dequantize_row_into(r, &mut y)
            } else {
                b.dequantize_row_into(r - 3, &mut y)
            }
            assert_eq!(x, y);
        }
    }

    #[test]
    fn quantized_dot_close_to_f32_dot() {
        let mut seed = 7;
        let (rows, cols) = (9, 512);
        let w = Q1Weight::from_gguf(&rand_gguf(41, rows, cols, &mut seed), 41, rows, cols).unwrap();
        let x: Vec<f32> = rand_bytes(cols, &mut seed)
            .iter()
            .map(|&b| (b as f32 - 128.0) / 37.0)
            .collect();
        let mut acts = Q8Acts::new();
        let mut out = vec![0.0; rows];
        matmul_f32(&w, &x, 1, &mut acts, &mut out);
        let mut row = vec![0.0; cols];
        for (r, &got) in out.iter().enumerate() {
            w.dequantize_row_into(r, &mut row);
            let exact: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            let norm: f32 = row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            assert!(
                (got - exact).abs() <= 0.01 * norm,
                "row {r}: {got} vs {exact}"
            );
        }
    }
}
