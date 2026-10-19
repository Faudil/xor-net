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
//! is element `8k + i`), and the f16 scales in a separate array. This lets the
//! AVX-512 kernel load 64 sign bits straight into a mask register and 16
//! scales with one `vcvtph2ps`, and it is identical for all three source
//! types.
//!
//! # Activations
//!
//! Activations are quantized to int8 per row in blocks of 32 elements (the
//! ggml `Q8_0` granularity that llama.cpp uses against Q1_0) or of 128
//! elements, with one f32 scale per block. The activation block must divide
//! the weight block. Block scales keep the error of large-outlier
//! activations local to their block, unlike a single per-row scale.
//!
//! # Kernel
//!
//! For one activation block `j` inside weight block `b`:
//! `Σ_i w_i x_i ≈ d_w[b] · d_x[j] · Σ_i (bit_i ? q_i : -q_i)`. The integer
//! sum is exact (|q| ≤ 127). The AVX-512 kernel negates the activation bytes
//! whose bit is clear with one masked subtract, sums groups of four bytes
//! with `vpdpbusd` (or `vpmaddubsw` + `vpmaddwd` without VNNI), converts the
//! i32 lanes to f32 and applies the scales with one multiply and one FMA per
//! 64 elements. With 128-element activation blocks the integer sum spans the
//! whole weight block and the float work drops to one convert + one FMA per
//! 128 elements. Four weight rows share every activation load.

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
#[derive(Clone, Debug, Default)]
pub struct Q8Acts {
    n: usize,
    cols: usize,
    block: usize,
    q: Vec<i8>,
    /// One scale per block: `x ≈ q * d`.
    d: Vec<f32>,
    /// 32-element blocks only: `d` expanded to one entry per 4 elements
    /// (one per i32 lane of the AVX-512 kernel).
    lane_d: Vec<f32>,
}

/// One quantized activation row (borrowed from a [`Q8Acts`]).
#[derive(Clone, Copy)]
pub struct ActRow<'a> {
    pub block: usize,
    pub q: &'a [i8],
    pub d: &'a [f32],
    /// Empty unless `block == 32`.
    pub lane_d: &'a [f32],
}

/// Quantize one block: returns `d` with `x ≈ q * d`, `|q| ≤ 127`
/// (round half to even). NaN elements map to 0; an infinite element makes
/// `d` NaN so the non-finite input stays visible in the output.
// `always`, and callers run it under `with_best_simd!`: on the x86_64
// baseline (no SSE4.1) `round_ties_even` is a libm call per element.
#[inline(always)]
fn quantize_block(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
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
        let quantize_rows = |x: &[f32], q: &mut [i8], d: &mut [f32]| {
            crate::util::with_best_simd!({
                for ((xb, qb), d) in x
                    .chunks_exact(block)
                    .zip(q.chunks_exact_mut(block))
                    .zip(d.iter_mut())
                {
                    *d = quantize_block(xb, qb);
                }
            })
        };
        if nblk * block >= 1 << 17 {
            // Parallel over groups of whole blocks (one dispatch per group).
            let per = 64;
            x.par_chunks(per * block)
                .zip(self.q.par_chunks_mut(per * block))
                .zip(self.d.par_chunks_mut(per))
                .for_each(|((xb, qb), d)| quantize_rows(xb, qb, d));
        } else {
            quantize_rows(x, &mut self.q, &mut self.d);
        }
        if block == SUB {
            self.lane_d.resize(n * cols / 4, 0.0);
            for (ld, &d) in self
                .lane_d
                .as_chunks_mut::<{ SUB / 4 }>()
                .0
                .iter_mut()
                .zip(&self.d)
            {
                ld.fill(d);
            }
        } else {
            self.lane_d.clear();
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
        ActRow {
            block: self.block,
            q: &self.q[t * c..(t + 1) * c],
            d: &self.d[t * nb..(t + 1) * nb],
            lane_d: if self.lane_d.is_empty() {
                &[]
            } else {
                &self.lane_d[t * c / 4..(t + 1) * c / 4]
            },
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
    /// `avx2` + `fma`.
    Avx2,
    /// `avx512f` + `avx512bw`.
    Avx512Bw,
    /// `avx512f` + `avx512bw` + `avx512vnni`.
    Avx512Vnni,
}

impl Isa {
    /// Fastest variant supported by this CPU.
    pub fn best() -> Self {
        let f = cpu_features();
        if f.avx512_vnni() {
            Isa::Avx512Vnni
        } else if f.avx512bw() {
            Isa::Avx512Bw
        } else if f.avx2_fma() {
            Isa::Avx2
        } else {
            Isa::Scalar
        }
    }

    /// Whether this CPU can run the variant.
    pub fn supported(self) -> bool {
        let f = cpu_features();
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => f.avx2_fma(),
            Isa::Avx512Bw => f.avx512bw(),
            Isa::Avx512Vnni => f.avx512_vnni(),
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
    if a.block == SUB {
        assert_eq!(
            a.lane_d.len(),
            w.cols / 4,
            "Q1 kernel: act lane scales length"
        );
    }
}

/// Reference: for each weight block `b` and each activation block `j` in
/// it, the exact integer sum `Σ_{c in j} ±q[c]` times `d_x[j]`, summed over
/// `j`, times `d_w[r0 + i, b]`, accumulated in f32.
pub fn dot_rows_scalar(w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
    check_rows(w, r0, &a, out);
    let rb = w.cols / 8;
    let nb = w.cols / w.block;
    let sub = w.block / a.block;
    let bits_all = w.bits.as_slice();
    for (i, o) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let bits = &bits_all[r * rb..(r + 1) * rb];
        let mut acc = 0.0f32;
        for b in 0..nb {
            let mut bacc = 0.0f32;
            for j in b * sub..(b + 1) * sub {
                let mut sumi = 0i32;
                for e in j * a.block..(j + 1) * a.block {
                    let v = a.q[e] as i32;
                    sumi += if (bits[e >> 3] >> (e & 7)) & 1 == 1 {
                        v
                    } else {
                        -v
                    };
                }
                bacc += sumi as f32 * a.d[j];
            }
            acc += bacc * w.scale(r, b);
        }
        *o = acc;
    }
}

/// AVX2 row kernel.
///
/// # Safety
/// The CPU must support `avx2` and `fma`. Shapes are checked (panics).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_rows_avx2(w: &Q1Weight, r0: usize, a: ActRow<'_>, out: &mut [f32]) {
    use std::arch::x86_64::*;
    check_rows(w, r0, &a, out);
    let rb = w.cols / 8;
    let nab = w.cols / a.block;
    let per_w = w.block / a.block;
    let sub = a.block / SUB;
    let bits_all = w.bits.as_slice();
    // byte i of each 128-bit lane gets source byte (lane*2 + i/8).
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let bitsel = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let one8 = _mm256_set1_epi8(1);
    let one16 = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    for (i, o) in out.iter_mut().enumerate() {
        let r = r0 + i;
        let bits = &bits_all[r * rb..(r + 1) * rb];
        let mut acc = _mm256_setzero_ps();
        let mut ws = 0.0f32;
        for b in 0..nab {
            if b % per_w == 0 {
                ws = w.scale(r, b / per_w);
            }
            let mut iacc = _mm256_setzero_si256();
            for j in b * sub..(b + 1) * sub {
                // SAFETY: j*32+32 <= cols = a.q.len(); 4*j+4 <= rb.
                let x = _mm256_loadu_si256(a.q.as_ptr().add(j * SUB) as *const __m256i);
                let m = (bits.as_ptr().add(4 * j) as *const u32).read_unaligned();
                let v = _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32(m as i32), shuf),
                    bitsel,
                );
                // bit clear -> -1 (negate), bit set -> +1.
                let sgn = _mm256_or_si256(_mm256_cmpeq_epi8(v, zero), one8);
                let xs = _mm256_sign_epi8(x, sgn);
                iacc = _mm256_add_epi32(
                    iacc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(one8, xs), one16),
                );
            }
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_set1_ps(a.d[b] * ws), acc);
        }
        let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        *o = _mm_cvtss_f32(s);
    }
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
            let bits_all = w.bits.as_slice().as_ptr();
            let sc_all = w.scales.as_slice().as_ptr();
            let zero = _mm512_setzero_si512();
            let one8 = _mm512_set1_epi8(1);
            let one16 = _mm512_set1_epi16(1);
            // Sum of the 64 sign-applied activation bytes, in 16 i32 lanes.
            let dot64 = |acc: __m512i, x: __m512i, m: u64| -> __m512i {
                // Negate the bytes whose sign bit is clear.
                let xs = _mm512_mask_sub_epi8(x, !m, zero, x);
                if $vnni {
                    _mm512_dpbusd_epi32(acc, one8, xs)
                } else {
                    _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(one8, xs), one16))
                }
            };
            // 16 f16 scales starting at block c0 of a row (zero-padded tail).
            let load_scales = |p: *const u8, cnt: usize| -> __m512 {
                let h = if cnt == 16 {
                    _mm256_loadu_si256(p as *const __m256i)
                } else {
                    let mut tmp = [0u8; 32];
                    std::ptr::copy_nonoverlapping(p, tmp.as_mut_ptr(), cnt * 2);
                    _mm256_loadu_si256(tmp.as_ptr() as *const __m256i)
                };
                _mm512_cvtph_ps(h)
            };
            let n = out.len();
            let mut i = 0;
            while i < n {
                let rr = (n - i).min(4);
                // Rows beyond `rr` repeat the last row (results discarded).
                let rows: [usize; 4] = std::array::from_fn(|k| r0 + i + k.min(rr - 1));
                let rbits: [*const u8; 4] = std::array::from_fn(|k| bits_all.add(rows[k] * rb));
                let rsc: [*const u8; 4] = std::array::from_fn(|k| sc_all.add(rows[k] * nb * 2));
                let mut acc = [_mm512_setzero_ps(); 4];
                if a.block == 128 {
                    // One i32 accumulation per 128-element block, then one
                    // FMA with the combined scale d_w * d_x.
                    let mut ws = [[0.0f32; 16]; 4];
                    for b in 0..nb {
                        if b % 16 == 0 {
                            let cnt = (nb - b).min(16);
                            for r in 0..4 {
                                _mm512_storeu_ps(ws[r].as_mut_ptr(), load_scales(rsc[r].add(b * 2), cnt));
                            }
                        }
                        let x0 = _mm512_loadu_si512(a.q.as_ptr().add(128 * b) as *const _);
                        let x1 = _mm512_loadu_si512(a.q.as_ptr().add(128 * b + 64) as *const _);
                        let da = *a.d.get_unchecked(b);
                        for r in 0..4 {
                            let p = rbits[r].add(16 * b) as *const u64;
                            let s = dot64(zero, x0, p.read_unaligned());
                            let s = dot64(s, x1, p.add(1).read_unaligned());
                            acc[r] = _mm512_fmadd_ps(
                                _mm512_cvtepi32_ps(s),
                                _mm512_set1_ps(da * ws[r][b % 16]),
                                acc[r],
                            );
                        }
                    }
                } else {
                    // 32-element activation blocks: a 64-element step covers
                    // two of them (lanes 0-7 / 8-15), each scaled per lane;
                    // the weight scale vector is gathered per step. A final
                    // 32-element half step when cols % 64 == 32 (32-element
                    // weight blocks only).
                    let big = w.block == 128;
                    let full = cols / 64;
                    let steps = cols.div_ceil(64);
                    // Steps covered by one chunk of 16 weight scales.
                    let spc = if big { 32 } else { 8 };
                    let hi_one = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
                    let mut wz = [_mm512_setzero_ps(); 4];
                    for k in 0..steps {
                        if k % spc == 0 {
                            let c0 = (k / spc) * 16;
                            let cnt = (nb - c0).min(16);
                            for r in 0..4 {
                                wz[r] = load_scales(rsc[r].add(c0 * 2), cnt);
                            }
                        }
                        let idx = if big {
                            _mm512_set1_epi32(((k / 2) % 16) as i32)
                        } else {
                            _mm512_add_epi32(_mm512_set1_epi32(((2 * k) % 16) as i32), hi_one)
                        };
                        let (x, s) = if k < full {
                            (
                                _mm512_loadu_si512(a.q.as_ptr().add(64 * k) as *const _),
                                _mm512_loadu_ps(a.lane_d.as_ptr().add(16 * k)),
                            )
                        } else {
                            (
                                _mm512_maskz_loadu_epi8(0xFFFF_FFFF, a.q.as_ptr().add(64 * k)),
                                _mm512_maskz_loadu_ps(0x00FF, a.lane_d.as_ptr().add(16 * k)),
                            )
                        };
                        for r in 0..4 {
                            let m: u64 = if k < full {
                                (rbits[r].add(8 * k) as *const u64).read_unaligned()
                            } else {
                                (rbits[r].add(8 * k) as *const u32).read_unaligned() as u64
                            };
                            let p = dot64(zero, x, m);
                            let f = _mm512_mul_ps(_mm512_cvtepi32_ps(p), s);
                            let wv = _mm512_permutexvar_ps(idx, wz[r]);
                            acc[r] = _mm512_fmadd_ps(f, wv, acc[r]);
                        }
                    }
                }
                for r in 0..rr {
                    out[i + r] = _mm512_reduce_add_ps(acc[r]);
                }
                i += 4;
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
avx512_kernel!(
    /// AVX-512BW row kernel (`vpmaddubsw` + `vpmaddwd` byte sums).
    ///
    /// # Safety
    /// The CPU must support `avx512f` and `avx512bw`. Shapes are checked
    /// (panics); all loads stay inside the weight rows and activation row.
    dot_rows_avx512bw,
    "avx512f,avx512bw",
    false
);

#[cfg(target_arch = "x86_64")]
avx512_kernel!(
    /// AVX-512 VNNI row kernel (`vpdpbusd` byte sums).
    ///
    /// # Safety
    /// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`. Shapes are
    /// checked (panics); all loads stay inside the weight rows and activation
    /// row.
    dot_rows_avx512vnni,
    "avx512f,avx512bw,avx512vnni",
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
    rows.div_ceil(t * per_thread).max(min).next_multiple_of(4)
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

/// `out[t] = W · x[t]` for every activation row `t` (`out` is
/// `[acts.n(), rows]` row-major). Each task keeps a slab of weight rows hot
/// in cache while it sweeps all activation rows, so the weights are streamed
/// from memory once per call instead of once per token.
pub fn gemm(w: &Q1Weight, acts: &Q8Acts, out: &mut [f32]) {
    let n = acts.n();
    assert_eq!(out.len(), n * w.rows, "gemm: output length mismatch");
    if n == 1 {
        return gemv(w, acts, out);
    }
    let isa = Isa::best();
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
