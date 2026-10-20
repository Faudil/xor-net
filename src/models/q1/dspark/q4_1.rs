//! ggml `Q4_1` weights (the DSpark drafter's format) and their GEMV/GEMM.
//!
//! # Format
//!
//! A row is cut into blocks of 32 elements; each GGUF block is 20 bytes:
//! `[d: f16][m: f16][qs: 16 B]`, element `e < 16` is `qs[e] & 15`, element
//! `16 + e` is `qs[e] >> 4`, and `w = d * q + m` (`q` in `0..16`).
//! [`Q41Weight`] keeps exactly this information, split into a nibble array
//! and two f16 scale arrays (same 20 bytes per block, so the same memory
//! traffic), which lets the kernels load 16 nibble bytes per block directly.
//!
//! # Kernels
//!
//! As in llama.cpp (`Q4_1 x Q8_1`), activations are quantized to int8 in
//! blocks of 32 ([`Q8Blocks`]: `x ≈ dx * q`, plus `sx = dx * Σq`), so a
//! block contributes `dw·dx·Σ(qw·qx) + mw·sx`. The integer part is exact;
//! every kernel then combines blocks in one canonical order (see
//! [`row_dot_scalar`]): eight `f32` lanes, lane `l` taking elements
//! `4l..4l+4` of each block (`lane = fma(I_l · dx, dw, lane)`), the `mw·sx`
//! terms in eight lanes by block index mod 8, and a fixed horizontal sum.
//! AVX-VNNI (`vpdpbusd`), AVX2 (`vpmaddubsw` + `vpmaddwd`) and scalar
//! results are therefore bit-identical, for any token count.
//!
//! Weights are streamed once per call (rows are split over threads, each
//! row reused for every token), so a few-token GEMM costs about one GEMV.

use half::f16;
use rayon::prelude::*;

use crate::nn::fast_attention_simd::f16_to_f32_slice;
use crate::util::{cpu_features, with_best_simd};

/// GGML type id of `Q4_1`.
pub const GGML_Q4_1: u32 = 3;
const QK: usize = 32;
const BLOCK_BYTES: usize = 20;

/// A `[rows, cols]` Q4_1 matrix (see the module docs).
#[derive(Clone, Debug)]
pub struct Q41Weight {
    rows: usize,
    cols: usize,
    /// `[rows][cols / 32][16]` nibble pairs.
    qs: Vec<u8>,
    /// `[rows][cols / 32]` block scales / minimums.
    d: Vec<f16>,
    m: Vec<f16>,
}

impl Q41Weight {
    /// From GGUF `Q4_1` data (`rows * cols / 32` blocks of 20 bytes).
    pub fn from_gguf(data: &[u8], rows: usize, cols: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cols.is_multiple_of(QK),
            "Q4_1: cols {cols} not a multiple of 32"
        );
        let nb = cols / QK;
        anyhow::ensure!(
            data.len() == rows * nb * BLOCK_BYTES,
            "Q4_1: {} bytes for a {rows}x{cols} matrix",
            data.len()
        );
        let mut qs = vec![0u8; rows * nb * 16];
        let mut d = vec![f16::ZERO; rows * nb];
        let mut m = vec![f16::ZERO; rows * nb];
        qs.par_chunks_mut(nb * 16)
            .zip(d.par_chunks_mut(nb).zip(m.par_chunks_mut(nb)))
            .zip(data.par_chunks(nb * BLOCK_BYTES))
            .for_each(|((q, (d, m)), src)| {
                for b in 0..nb {
                    let blk = &src[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
                    d[b] = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]]));
                    m[b] = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]]));
                    q[b * 16..(b + 1) * 16].copy_from_slice(&blk[4..20]);
                }
            });
        Ok(Self {
            rows,
            cols,
            qs,
            d,
            m,
        })
    }

    /// Stack the rows of matrices with equal `cols` (one fused GEMM).
    pub fn concat_rows(parts: &[&Q41Weight]) -> anyhow::Result<Self> {
        anyhow::ensure!(!parts.is_empty(), "concat_rows: no parts");
        let cols = parts[0].cols;
        anyhow::ensure!(
            parts.iter().all(|p| p.cols == cols),
            "concat_rows: cols differ"
        );
        Ok(Self {
            rows: parts.iter().map(|p| p.rows).sum(),
            cols,
            qs: parts.iter().flat_map(|p| p.qs.iter().copied()).collect(),
            d: parts.iter().flat_map(|p| p.d.iter().copied()).collect(),
            m: parts.iter().flat_map(|p| p.m.iter().copied()).collect(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    fn nb(&self) -> usize {
        self.cols / QK
    }
    /// Bytes streamed per pass (same as the GGUF blocks).
    pub fn nbytes(&self) -> usize {
        self.qs.len() + 2 * (self.d.len() + self.m.len())
    }

    /// `out = row r` dequantized (`d * q + m`).
    pub fn dequantize_row_into(&self, r: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols);
        let nb = self.nb();
        for b in 0..nb {
            let (d, m) = (self.d[r * nb + b].to_f32(), self.m[r * nb + b].to_f32());
            let q = &self.qs[(r * nb + b) * 16..(r * nb + b + 1) * 16];
            for e in 0..16 {
                out[b * QK + e] = (q[e] & 15) as f32 * d + m;
                out[b * QK + 16 + e] = (q[e] >> 4) as f32 * d + m;
            }
        }
    }

    /// Row scales as f32: `(dw, mw)`, `nb` each.
    fn row_scales(&self, r: usize, dw: &mut [f32], mw: &mut [f32]) {
        let nb = self.nb();
        f16_to_f32_slice(&self.d[r * nb..(r + 1) * nb], dw);
        f16_to_f32_slice(&self.m[r * nb..(r + 1) * nb], mw);
    }
}

/// `n` activation rows quantized to int8 in blocks of 32.
#[derive(Clone, Debug, Default)]
pub struct Q8Blocks {
    n: usize,
    cols: usize,
    /// `[n][cols]` int8 values.
    q: Vec<i8>,
    /// `[n][cols / 32]` block scales and `dx * Σq`.
    dx: Vec<f32>,
    sx: Vec<f32>,
}

impl Q8Blocks {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn n(&self) -> usize {
        self.n
    }

    /// Quantize `x` (`[n][cols]`): `dx = amax / 127`, `q = round(x / dx)`
    /// (ties to even). Reuses the buffers.
    pub fn quantize(&mut self, x: &[f32], n: usize, cols: usize) {
        assert_eq!(x.len(), n * cols, "Q8Blocks: input length");
        assert!(
            cols.is_multiple_of(QK),
            "Q8Blocks: cols not a multiple of 32"
        );
        let nb = cols / QK;
        self.n = n;
        self.cols = cols;
        self.q.resize(n * cols, 0);
        self.dx.resize(n * nb, 0.0);
        self.sx.resize(n * nb, 0.0);
        type Block<'a> = ((&'a mut [i8], &'a mut f32), (&'a mut f32, &'a [f32]));
        let f = |((q, dx), (sx, x)): Block<'_>| {
            let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut s = 0i32;
            for (o, &v) in q.iter_mut().zip(x) {
                let r = (v * id).round_ties_even().clamp(-127.0, 127.0) as i32;
                *o = r as i8;
                s += r;
            }
            *dx = d;
            *sx = d * s as f32;
        };
        if n * nb >= 4096 {
            self.q
                .par_chunks_mut(QK)
                .zip(self.dx.par_iter_mut())
                .zip(self.sx.par_iter_mut().zip(x.par_chunks(QK)))
                .for_each(|((q, dx), (sx, x))| f(((q, dx), (sx, x))));
        } else {
            let (q, dx, sx) = (&mut self.q, &mut self.dx, &mut self.sx);
            with_best_simd!(
                q.chunks_mut(QK)
                    .zip(dx.iter_mut())
                    .zip(sx.iter_mut().zip(x.chunks(QK)))
                    .for_each(f)
            );
        }
    }
}

/// Kernel variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    /// `vpmaddubsw` + `vpmaddwd` (AVX2 + FMA).
    Avx2,
    /// VEX `vpdpbusd` (AVX-VNNI).
    AvxVnni,
    /// EVEX `vpdpbusd` on ymm (AVX-512 VNNI + VL).
    Avx512Vnni,
}

impl Isa {
    pub fn supported(self) -> bool {
        let f = cpu_features();
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => f.avx2_tier(),
            Isa::AvxVnni => f.avx_vnni(),
            Isa::Avx512Vnni => f.avx2_tier() && f.avx512_vnni() && f.avx512vl,
        }
    }
    pub fn best() -> Self {
        [Isa::AvxVnni, Isa::Avx512Vnni, Isa::Avx2]
            .into_iter()
            .find(|i| i.supported())
            .unwrap_or(Isa::Scalar)
    }
    pub fn all_supported() -> Vec<Isa> {
        [Isa::Scalar, Isa::Avx2, Isa::AvxVnni, Isa::Avx512Vnni]
            .into_iter()
            .filter(|i| i.supported())
            .collect()
    }
}

/// `((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7))` (what the SIMD
/// reduction computes).
#[inline(always)]
fn hsum8(v: &[f32; 8]) -> f32 {
    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))
}

/// Reference dot of row `r` with activation row `t`; defines the canonical
/// arithmetic every kernel reproduces bit for bit.
pub fn row_dot_scalar(w: &Q41Weight, r: usize, a: &Q8Blocks, t: usize) -> f32 {
    let nb = w.nb();
    let mut dw = vec![0.0; nb];
    let mut mw = vec![0.0; nb];
    w.row_scales(r, &mut dw, &mut mw);
    row_dot_scalar_with(w, r, a, t, &dw, &mw)
}

fn row_dot_scalar_with(
    w: &Q41Weight,
    r: usize,
    a: &Q8Blocks,
    t: usize,
    dw: &[f32],
    mw: &[f32],
) -> f32 {
    let nb = w.nb();
    let mut acc = [0.0f32; 8];
    for (b, &dwb) in dw.iter().enumerate().take(nb) {
        let q = &w.qs[(r * nb + b) * 16..(r * nb + b + 1) * 16];
        let x = &a.q[(t * nb + b) * QK..(t * nb + b + 1) * QK];
        let dx = a.dx[t * nb + b];
        for (l, lane) in acc.iter_mut().enumerate() {
            let mut s = 0i32;
            for e in 4 * l..4 * l + 4 {
                let qw = if e < 16 { q[e] & 15 } else { q[e - 16] >> 4 };
                s += qw as i32 * x[e] as i32;
            }
            *lane = (s as f32 * dx).mul_add(dwb, *lane);
        }
    }
    let mut macc = [0.0f32; 8];
    for b in 0..nb {
        macc[b % 8] = mw[b].mul_add(a.sx[t * nb + b], macc[b % 8]);
    }
    hsum8(&acc) + hsum8(&macc)
}

/// Max tokens per register tile.
const TILE: usize = 4;

#[cfg(target_arch = "x86_64")]
macro_rules! tile_kernel {
    ($name:ident, $feat:literal, |$nv:ident, $av:ident| $isum:expr) => {
        /// Tile of `NT` tokens `t0..` against row `r` (see [`row_dot_scalar`]).
        #[target_feature(enable = $feat)]
        unsafe fn $name<const NT: usize>(
            w: &Q41Weight,
            r: usize,
            a: &Q8Blocks,
            t0: usize,
            dw: &[f32],
            mw: &[f32],
            out: &mut [f32; TILE],
        ) {
            use std::arch::x86_64::*;
            let nb = w.nb();
            let qs = w.qs.as_ptr().add(r * nb * 16);
            let mask = _mm_set1_epi8(0x0F);
            let mut acc = [_mm256_setzero_ps(); NT];
            let xq = a.q.as_ptr().add(t0 * nb * QK);
            let dx = a.dx.as_ptr().add(t0 * nb);
            for b in 0..nb {
                let q = _mm_loadu_si128(qs.add(b * 16) as *const __m128i);
                let lo = _mm_and_si128(q, mask);
                let hi = _mm_and_si128(_mm_srli_epi16(q, 4), mask);
                let $nv = _mm256_set_m128i(hi, lo);
                let dwb = _mm256_set1_ps(*dw.get_unchecked(b));
                for t in 0..NT {
                    let $av = _mm256_loadu_si256(xq.add((t * nb + b) * QK) as *const __m256i);
                    let is: __m256i = $isum;
                    let p =
                        _mm256_mul_ps(_mm256_cvtepi32_ps(is), _mm256_set1_ps(*dx.add(t * nb + b)));
                    acc[t] = _mm256_fmadd_ps(p, dwb, acc[t]);
                }
            }
            for t in 0..NT {
                let sx = a.sx.as_ptr().add((t0 + t) * nb);
                let mut macc = _mm256_setzero_ps();
                let groups = nb / 8;
                for g in 0..groups {
                    macc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(mw.as_ptr().add(g * 8)),
                        _mm256_loadu_ps(sx.add(g * 8)),
                        macc,
                    );
                }
                let mut m8 = [0.0f32; 8];
                _mm256_storeu_ps(m8.as_mut_ptr(), macc);
                for b in groups * 8..nb {
                    m8[b % 8] = mw[b].mul_add(*sx.add(b), m8[b % 8]);
                }
                let mut a8 = [0.0f32; 8];
                _mm256_storeu_ps(a8.as_mut_ptr(), acc[t]);
                out[t] = hsum8(&a8) + hsum8(&m8);
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
tile_kernel!(tile_avx2, "avx2,fma,f16c", |n, a| {
    _mm256_madd_epi16(_mm256_maddubs_epi16(n, a), _mm256_set1_epi16(1))
});
#[cfg(target_arch = "x86_64")]
tile_kernel!(tile_avxvnni, "avx2,fma,f16c,avxvnni", |n, a| {
    _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), n, a)
});
#[cfg(target_arch = "x86_64")]
tile_kernel!(
    tile_avx512vnni,
    "avx2,fma,f16c,avx512f,avx512vl,avx512vnni",
    |n, a| { _mm256_dpbusd_epi32(_mm256_setzero_si256(), n, a) }
);

/// Dots of row `r` with tokens `t0 .. t0 + nt` (`nt <= TILE`).
#[allow(clippy::too_many_arguments)]
fn tile(
    isa: Isa,
    w: &Q41Weight,
    r: usize,
    a: &Q8Blocks,
    t0: usize,
    nt: usize,
    dw: &[f32],
    mw: &[f32],
    out: &mut [f32; TILE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! go {
            ($k:ident) => {
                // SAFETY: `isa.supported()` was checked by the caller;
                // shapes are checked in `gemm_with`.
                unsafe {
                    match nt {
                        1 => $k::<1>(w, r, a, t0, dw, mw, out),
                        2 => $k::<2>(w, r, a, t0, dw, mw, out),
                        3 => $k::<3>(w, r, a, t0, dw, mw, out),
                        _ => $k::<4>(w, r, a, t0, dw, mw, out),
                    }
                }
            };
        }
        match isa {
            Isa::Avx2 => return go!(tile_avx2),
            Isa::AvxVnni => return go!(tile_avxvnni),
            Isa::Avx512Vnni => return go!(tile_avx512vnni),
            Isa::Scalar => {}
        }
    }
    for (t, o) in out.iter_mut().enumerate().take(nt) {
        *o = row_dot_scalar_with(w, r, a, t0 + t, dw, mw);
    }
}

/// `out[t][r] = W[r] · x[t]` for every activation row (`out`: `[n][rows]`).
pub fn gemm(w: &Q41Weight, a: &Q8Blocks, out: &mut [f32]) {
    gemm_with(Isa::best(), w, a, out)
}

/// [`gemm`] with an explicit kernel. Panics on a shape mismatch or an
/// unsupported ISA.
pub fn gemm_with(isa: Isa, w: &Q41Weight, a: &Q8Blocks, out: &mut [f32]) {
    assert!(
        isa.supported(),
        "Q4_1 kernel {isa:?} not supported on this CPU"
    );
    let (n, rows) = (a.n, w.rows);
    assert_eq!(a.cols, w.cols, "Q4_1 gemm: cols mismatch");
    assert_eq!(out.len(), n * rows, "Q4_1 gemm: output length");
    if n == 0 || rows == 0 {
        return;
    }
    let nb = w.nb();
    let threads = rayon::current_num_threads().max(1);
    let chunk = rows.div_ceil(threads * 8).max(16);
    let work = |r0: usize, r1: usize, sink: &mut dyn FnMut(usize, usize, f32)| {
        let mut dw = vec![0.0f32; nb];
        let mut mw = vec![0.0f32; nb];
        let mut res = [0.0f32; TILE];
        for r in r0..r1 {
            w.row_scales(r, &mut dw, &mut mw);
            let mut t0 = 0;
            while t0 < n {
                let nt = (n - t0).min(TILE);
                tile(isa, w, r, a, t0, nt, &dw, &mw, &mut res);
                for (k, &v) in res[..nt].iter().enumerate() {
                    sink(t0 + k, r, v);
                }
                t0 += nt;
            }
        }
    };
    if n == 1 {
        out.par_chunks_mut(chunk).enumerate().for_each(|(c, o)| {
            let r0 = c * chunk;
            work(r0, r0 + o.len(), &mut |_, r, v| o[r - r0] = v);
        });
        return;
    }
    // Per chunk a [n][chunk] slab, scattered into `out` afterwards.
    let nchunks = rows.div_ceil(chunk);
    let slabs: Vec<Vec<f32>> = (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let (r0, r1) = (c * chunk, ((c + 1) * chunk).min(rows));
            let cr = r1 - r0;
            let mut slab = vec![0.0f32; n * cr];
            work(r0, r1, &mut |t, r, v| slab[t * cr + r - r0] = v);
            slab
        })
        .collect();
    out.par_chunks_mut(rows).enumerate().for_each(|(t, o)| {
        for (c, slab) in slabs.iter().enumerate() {
            let r0 = c * chunk;
            let cr = slab.len() / n;
            o[r0..r0 + cr].copy_from_slice(&slab[t * cr..(t + 1) * cr]);
        }
    });
}

/// `out[r] = W[r] · x` with an `f32` activation (no activation
/// quantization; used for the small DSpark Markov head, which llama.cpp's
/// host path also evaluates in f32): per block `dw · Σ q·x + mw · Σx`.
pub fn gemv_f32(w: &Q41Weight, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "Q4_1 gemv_f32: input length");
    assert_eq!(out.len(), w.rows, "Q4_1 gemv_f32: output length");
    let nb = w.nb();
    let xs: Vec<f32> = x.chunks(QK).map(|c| c.iter().sum()).collect();
    let chunk = w
        .rows
        .div_ceil(rayon::current_num_threads().max(1) * 8)
        .max(64);
    out.par_chunks_mut(chunk).enumerate().for_each(|(c, o)| {
        let r0 = c * chunk;
        let mut dw = vec![0.0f32; o.len() * nb];
        let mut mw = vec![0.0f32; o.len() * nb];
        f16_to_f32_slice(&w.d[r0 * nb..(r0 + o.len()) * nb], &mut dw);
        f16_to_f32_slice(&w.m[r0 * nb..(r0 + o.len()) * nb], &mut mw);
        let qs = &w.qs[r0 * nb * 16..(r0 + o.len()) * nb * 16];
        with_best_simd!(gemv_f32_rows(qs, &dw, &mw, x, &xs, nb, o))
    });
}

#[inline(always)]
fn gemv_f32_rows(
    qs: &[u8],
    dw: &[f32],
    mw: &[f32],
    x: &[f32],
    xs: &[f32],
    nb: usize,
    o: &mut [f32],
) {
    for (i, y) in o.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for b in 0..nb {
            let q = &qs[(i * nb + b) * 16..(i * nb + b + 1) * 16];
            let (lo, hi) = x[b * QK..(b + 1) * QK].split_at(16);
            let mut s = [0.0f32; 16];
            for e in 0..16 {
                s[e] = (q[e] & 15) as f32 * lo[e] + (q[e] >> 4) as f32 * hi[e];
            }
            let s: f32 = s.iter().sum();
            acc += dw[i * nb + b] * s + mw[i * nb + b] * xs[b];
        }
        *y = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn random_q41(rows: usize, cols: usize, seed: &mut u64) -> (Q41Weight, Vec<u8>) {
        let nb = cols / 32;
        let mut data = Vec::with_capacity(rows * nb * 20);
        for _ in 0..rows * nb {
            let d = f16::from_f32(0.01 + 0.02 * rng(seed).abs());
            let m = f16::from_f32(-0.1 * rng(seed).abs());
            data.extend_from_slice(&d.to_bits().to_le_bytes());
            data.extend_from_slice(&m.to_bits().to_le_bytes());
            for _ in 0..16 {
                data.push((rng(seed) * 1e6) as i64 as u8);
            }
        }
        (Q41Weight::from_gguf(&data, rows, cols).unwrap(), data)
    }

    /// ggml `dequantize_row_q4_1`.
    fn ggml_dequant(data: &[u8], cols: usize, r: usize) -> Vec<f32> {
        let nb = cols / 32;
        let mut out = vec![0.0; cols];
        for b in 0..nb {
            let blk = &data[(r * nb + b) * 20..(r * nb + b + 1) * 20];
            let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
            let m = f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
            for j in 0..16 {
                out[b * 32 + j] = (blk[4 + j] & 15) as f32 * d + m;
                out[b * 32 + 16 + j] = (blk[4 + j] >> 4) as f32 * d + m;
            }
        }
        out
    }

    #[test]
    fn all_kernels_bit_identical_and_close_to_f64() {
        let mut seed = 0x1234_5678u64;
        for &(rows, cols, n) in &[(37, 256, 1), (130, 5120 / 4, 5), (19, 96, 9), (64, 32, 4)] {
            let (w, data) = random_q41(rows, cols, &mut seed);
            let x: Vec<f32> = (0..n * cols).map(|_| rng(&mut seed) * 3.0).collect();
            let mut a = Q8Blocks::new();
            a.quantize(&x, n, cols);
            let mut want = vec![0.0; n * rows];
            for t in 0..n {
                for r in 0..rows {
                    want[t * rows + r] = row_dot_scalar(&w, r, &a, t);
                }
            }
            for isa in Isa::all_supported() {
                let mut got = vec![0.0; n * rows];
                gemm_with(isa, &w, &a, &mut got);
                assert!(
                    got.iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{isa:?} {rows}x{cols} n={n}"
                );
            }
            // Against an f64 dot of the exact dequantized weights.
            for t in 0..n {
                for r in 0..rows {
                    let wr = ggml_dequant(&data, cols, r);
                    let mut mine = vec![0.0; cols];
                    w.dequantize_row_into(r, &mut mine);
                    assert_eq!(wr, mine);
                    let exact: f64 = wr
                        .iter()
                        .zip(&x[t * cols..(t + 1) * cols])
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    let scale: f64 = wr
                        .iter()
                        .zip(&x[t * cols..(t + 1) * cols])
                        .map(|(&a, &b)| (a as f64 * b as f64).abs())
                        .sum();
                    let got = want[t * rows + r] as f64;
                    assert!(
                        (got - exact).abs() <= 0.02 * scale + 1e-6,
                        "{got} vs {exact}"
                    );
                }
            }
            // f32-activation GEMV.
            let mut y = vec![0.0; rows];
            gemv_f32(&w, &x[..cols], &mut y);
            for (r, &yr) in y.iter().enumerate() {
                let wr = ggml_dequant(&data, cols, r);
                let exact: f64 = wr
                    .iter()
                    .zip(&x[..cols])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let scale: f64 = wr
                    .iter()
                    .zip(&x[..cols])
                    .map(|(&a, &b)| (a as f64 * b as f64).abs())
                    .sum();
                assert!((yr as f64 - exact).abs() <= 1e-5 * scale + 1e-6);
            }
        }
    }
}
