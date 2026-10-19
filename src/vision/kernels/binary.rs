//! Binary (XNOR / popcount) 2-D convolution on bit-packed NHWC activations.
//!
//! # Math
//!
//! With activations and weights in `{-1, +1}` encoded as bits (`1 => +1`),
//! `a·w = matches - mismatches = n - 2·popcount(a XOR w)` where `n` is the
//! number of products. For a `kh x kw` window over `cin` channels,
//! `n = kh·kw·cin`. Unused high bits of each pixel's last word are zero in
//! both operands, so they never produce a mismatch and `n` stays exact.
//!
//! # Padding
//!
//! Larq binary convs quantise **then** pad, with a constant `pad_values`
//! (QuickNet uses `1.0`). A padded tap therefore contributes `pad·sign(w)`,
//! not zero. [`PadValue::PlusOne`] uses an all-ones (valid bits) word as the
//! padded pixel, [`PadValue::MinusOne`] an all-zeros word. Zero padding is
//! not representable in a pure XNOR kernel and is rejected at load time.
//!
//! # Kernels
//!
//! Weights are re-laid out at load into blocks of [`BLOCK`] = 32 output
//! channels: `[block][kpos][word][lane]`. The packed input is copied into a
//! buffer bordered with padding pixels, so tap `k` of any output pixel is a
//! fixed offset from the pixel's base and the inner loop has no bounds
//! branches. The kernel broadcasts one input word, XORs it against 32
//! output channels' weight words at once and accumulates per-lane popcounts:
//!
//! * `Avx512`: `vpxorq` + `vpopcntq` (AVX-512 VPOPCNTDQ), 4 zmm per pixel,
//!   4 output pixels per tile (16 accumulators) so each weight load is
//!   reused 4 times.
//! * `Avx2`: nibble-LUT popcount (`vpshufb` + `vpsadbw`), 8 ymm per pixel.
//! * `Scalar`: `u64::count_ones`.
//!
//! All three are checked bit-exactly against a float reference
//! ([`binary_conv2d_reference`]) in the tests.

use super::epilogue::Epilogue;
use crate::vision::tensor::{FeatureMap, Geometry, PackedMap, Padding};
use rayon::prelude::*;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Output channels per weight block (one AVX-512 tile column).
pub const BLOCK: usize = 32;
/// Output pixels per AVX-512 tile.
const PIX_TILE: usize = 4;
/// Largest supported `kh * kw`.
pub const MAX_TAPS: usize = 49;

/// Value used for padded positions of a binary conv's (binarised) input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadValue {
    PlusOne,
    MinusOne,
}

impl PadValue {
    pub fn from_f32(v: f32) -> Option<Self> {
        if v == 1.0 {
            Some(Self::PlusOne)
        } else if v == -1.0 {
            Some(Self::MinusOne)
        } else {
            None
        }
    }
}

/// Which popcount kernel to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKernel {
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Clone, Copy)]
struct Support {
    avx2: bool,
    avx512: bool,
}

fn support() -> Support {
    static S: OnceLock<Support> = OnceLock::new();
    *S.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let f = crate::util::cpu_features();
            Support {
                avx2: f.avx2,
                avx512: f.avx512f && f.avx512bw && std::is_x86_feature_detected!("avx512vpopcntdq"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Support {
                avx2: false,
                avx512: false,
            }
        }
    })
}

impl BinaryKernel {
    /// The fastest kernel this CPU supports.
    pub fn detect() -> Self {
        let s = support();
        if s.avx512 {
            Self::Avx512
        } else if s.avx2 {
            Self::Avx2
        } else {
            Self::Scalar
        }
    }

    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Avx2 => support().avx2,
            Self::Avx512 => support().avx512,
        }
    }

    /// All kernels runnable on this CPU (for tests/benchmarks).
    pub fn available() -> Vec<Self> {
        [Self::Scalar, Self::Avx2, Self::Avx512]
            .into_iter()
            .filter(|k| k.is_supported())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Activation packing
// ---------------------------------------------------------------------------

/// Binarise a feature map: bit = `x >= 0` (larq's `sign`, which maps 0 to +1;
/// `-0.0` also maps to +1 and NaN to -1).
pub fn pack_signs(input: &FeatureMap) -> PackedMap {
    let mut out = PackedMap::zeros(input.h, input.w, input.c);
    let (c, words) = (input.c, out.words);
    if c == 0 {
        return out;
    }
    let use_avx512 = support().avx512;
    out.data
        .par_chunks_mut(words * input.w)
        .zip(input.data.par_chunks(c * input.w))
        .for_each(|(orow, irow)| {
            for (dst, src) in orow.chunks_exact_mut(words).zip(irow.chunks_exact(c)) {
                #[cfg(target_arch = "x86_64")]
                if use_avx512 {
                    // SAFETY: avx512f detected; `dst.len() == ceil(src.len()/64)`.
                    unsafe { pack_pixel_avx512(src, dst) };
                    continue;
                }
                let _ = use_avx512;
                pack_pixel_scalar(src, dst);
            }
        });
    out
}

/// Scalar reference for one pixel. `dst.len()` must be `ceil(src.len()/64)`.
pub fn pack_pixel_scalar(src: &[f32], dst: &mut [u64]) {
    assert_eq!(
        dst.len(),
        src.len().div_ceil(64),
        "pack_pixel: word count mismatch"
    );
    for (word, chunk) in dst.iter_mut().zip(src.chunks(64)) {
        let mut bits = 0u64;
        for (b, &x) in chunk.iter().enumerate() {
            bits |= ((x >= 0.0) as u64) << b;
        }
        *word = bits;
    }
}

/// # Safety
/// The CPU must support `avx512f`. `dst.len()` must equal
/// `src.len().div_ceil(64)` (asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn pack_pixel_avx512(src: &[f32], dst: &mut [u64]) {
    assert_eq!(
        dst.len(),
        src.len().div_ceil(64),
        "pack_pixel: word count mismatch"
    );
    let zero = _mm512_setzero_ps();
    let full = src.len() / 64;
    for j in 0..full {
        let p = src.as_ptr().add(j * 64);
        let m0 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_loadu_ps(p), zero) as u64;
        let m1 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_loadu_ps(p.add(16)), zero) as u64;
        let m2 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_loadu_ps(p.add(32)), zero) as u64;
        let m3 = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_loadu_ps(p.add(48)), zero) as u64;
        *dst.get_unchecked_mut(j) = m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);
    }
    if full < dst.len() {
        pack_pixel_scalar(&src[full * 64..], &mut dst[full..]);
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Binary conv weights re-laid out for the popcount kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryConvWeights {
    pub cout: usize,
    pub cin: usize,
    pub kh: usize,
    pub kw: usize,
    /// `ceil(cin / 64)`.
    pub words: usize,
    /// `[ceil(cout/BLOCK)][kh*kw][words][BLOCK]`; lanes past `cout` are zero.
    blocks: Vec<u64>,
}

impl BinaryConvWeights {
    /// From the on-disk layout `[cout][kh][kw][words]` (see the module docs
    /// of `crate::vision`). Errors if the length is wrong or any bit beyond
    /// `cin` is set.
    pub fn from_packed_ohwi(
        packed: &[u64],
        cout: usize,
        kh: usize,
        kw: usize,
        cin: usize,
    ) -> Result<Self, String> {
        let words = cin.div_ceil(64);
        if cout == 0 || cin == 0 || kh == 0 || kw == 0 {
            return Err("binary conv with an empty dimension".into());
        }
        if kh.checked_mul(kw).is_none_or(|t| t > MAX_TAPS) {
            return Err(format!(
                "binary conv kernel {kh}x{kw} exceeds {MAX_TAPS} taps"
            ));
        }
        let expect = cout
            .checked_mul(kh * kw)
            .and_then(|v| v.checked_mul(words))
            .ok_or("binary conv weights: size overflows usize")?;
        if packed.len() != expect {
            return Err(format!(
                "binary conv weights: {} words, expected {expect}",
                packed.len()
            ));
        }
        let tail_mask = if cin.is_multiple_of(64) {
            0
        } else {
            !0u64 << (cin % 64)
        };
        let nk = kh * kw;
        let nb = cout.div_ceil(BLOCK);
        let mut blocks = vec![0u64; nb * nk * words * BLOCK];
        for o in 0..cout {
            let (b, lane) = (o / BLOCK, o % BLOCK);
            for k in 0..nk {
                for j in 0..words {
                    let v = packed[(o * nk + k) * words + j];
                    if j == words - 1 && v & tail_mask != 0 {
                        return Err(format!("binary conv weights: padding bits set (out {o})"));
                    }
                    blocks[((b * nk + k) * words + j) * BLOCK + lane] = v;
                }
            }
        }
        Ok(Self {
            cout,
            cin,
            kh,
            kw,
            words,
            blocks,
        })
    }

    /// From real-valued HWIO weights `[kh][kw][cin][cout]`, binarised with
    /// `sign` (`w >= 0 => +1`).
    pub fn from_f32_hwio(
        w: &[f32],
        kh: usize,
        kw: usize,
        cin: usize,
        cout: usize,
    ) -> Result<Self, String> {
        let n = [kh, kw, cin, cout]
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b));
        if n != Some(w.len()) {
            return Err("binary conv: weight length does not match [kh, kw, cin, cout]".into());
        }
        let words = cin.div_ceil(64);
        let mut packed = vec![0u64; cout * kh * kw * words];
        for k in 0..kh * kw {
            for ci in 0..cin {
                for o in 0..cout {
                    if w[(k * cin + ci) * cout + o] >= 0.0 {
                        packed[(o * kh * kw + k) * words + ci / 64] |= 1u64 << (ci % 64);
                    }
                }
            }
        }
        Self::from_packed_ohwi(&packed, cout, kh, kw, cin)
    }

    fn block(&self, b: usize) -> &[u64] {
        let n = self.kh * self.kw * self.words * BLOCK;
        &self.blocks[b * n..(b + 1) * n]
    }
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

/// Binary convolution with the fastest available kernel.
#[allow(clippy::too_many_arguments)]
pub fn binary_conv2d(
    input: &PackedMap,
    w: &BinaryConvWeights,
    stride: [usize; 2],
    padding: Padding,
    pad_value: PadValue,
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
) -> FeatureMap {
    binary_conv2d_with(
        BinaryKernel::detect(),
        input,
        w,
        stride,
        padding,
        pad_value,
        epi,
        residual,
    )
}

/// Binary convolution with an explicit kernel. Panics if `kernel` is not
/// supported by this CPU or the shapes are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn binary_conv2d_with(
    kernel: BinaryKernel,
    input: &PackedMap,
    w: &BinaryConvWeights,
    stride: [usize; 2],
    padding: Padding,
    pad_value: PadValue,
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
) -> FeatureMap {
    assert!(
        kernel.is_supported(),
        "binary_conv2d: {kernel:?} kernel not supported on this CPU"
    );
    assert_eq!(
        input.c, w.cin,
        "binary_conv2d: input channels != weight cin"
    );
    assert_eq!(input.words, w.words, "binary_conv2d: word count mismatch");
    assert_eq!(
        input.data.len(),
        input.h * input.w * input.words,
        "binary_conv2d: bad PackedMap"
    );
    let g = Geometry::new(input.h, input.w, [w.kh, w.kw], stride, padding)
        .expect("binary_conv2d: kernel does not fit the input");
    let cout = w.cout;
    epi.check(cout);
    if let Some(r) = residual {
        assert_eq!(
            r.shape(),
            [g.oh, g.ow, cout],
            "binary_conv2d: residual shape mismatch"
        );
    }

    let words = w.words;
    let tail = if w.cin.is_multiple_of(64) {
        !0u64
    } else {
        (1u64 << (w.cin % 64)) - 1
    };
    let mut pad_px = vec![0u64; words];
    if pad_value == PadValue::PlusOne {
        pad_px.fill(!0);
        pad_px[words - 1] = tail;
    }

    // Copy the input into a buffer padded with `pad_px` pixels so that every
    // tap of every output pixel is in bounds: the kernels then address tap
    // `k` of a pixel as `pixel_ptr + koff[k]`, with no per-tap branching.
    let ph = (g.oh - 1) * g.sh + g.kh;
    let pw = (g.ow - 1) * g.sw + g.kw;
    let mut padded = vec![0u64; ph * pw * words];
    for (py, prow) in padded.chunks_exact_mut(pw * words).enumerate() {
        let iy = py.checked_sub(g.pad_top).filter(|&y| y < g.ih);
        for (px, dst) in prow.chunks_exact_mut(words).enumerate() {
            let ix = px.checked_sub(g.pad_left).filter(|&x| x < g.iw);
            match (iy, ix) {
                (Some(iy), Some(ix)) => dst.copy_from_slice(input.pixel(iy, ix)),
                _ => dst.copy_from_slice(&pad_px),
            }
        }
    }
    let nk = w.kh * w.kw;
    let koff: Vec<usize> = (0..w.kh)
        .flat_map(|ky| (0..w.kw).map(move |kx| (ky * pw + kx) * words))
        .collect();
    let px_stride = g.sw * words;
    let n_total = (nk * w.cin) as f32;
    let nb = cout.div_ceil(BLOCK);
    let padded = &padded;
    let koff = &koff[..];

    let mut out = FeatureMap::zeros(g.oh, g.ow, cout);
    out.data
        .par_chunks_mut(g.ow * cout)
        .enumerate()
        .for_each(|(oy, row)| {
            let row_base = oy * g.sh * pw * words;
            let mut mis = [0u64; BLOCK];
            for b in 0..nb {
                let wb = w.block(b);
                let lanes = (cout - b * BLOCK).min(BLOCK);
                let mut ox = 0;
                while ox < g.ow {
                    let np = match kernel {
                        BinaryKernel::Avx512 if g.ow - ox >= PIX_TILE => PIX_TILE,
                        _ => 1,
                    };
                    // Pixel `ox + p` reads `padded[pix + p * px_stride + koff[k] ..][..words]`
                    // for all k < nk, p < np: in bounds because
                    // (ox+p)*sw + kx < pw and oy*sh + ky < ph.
                    let pix = row_base + ox * px_stride;
                    let dst = row[ox * cout + b * BLOCK..].as_mut_ptr();
                    // SAFETY (all arms): each kernel is only selected when the
                    // CPU supports it (asserted above); the input addressing is
                    // in bounds as argued above; `wb` holds `nk * words * BLOCK`
                    // words; the AVX-512 tile writes `lanes` f32s at
                    // `dst + p * cout` for p < np, inside `row` because
                    // `ox + np <= ow` and `b * BLOCK + lanes <= cout`.
                    let direct = match kernel {
                        #[cfg(target_arch = "x86_64")]
                        BinaryKernel::Avx512 if np == PIX_TILE => unsafe {
                            tile_avx512::<PIX_TILE>(
                                padded, pix, px_stride, koff, words, wb, dst, cout, lanes, n_total,
                            );
                            true
                        },
                        #[cfg(target_arch = "x86_64")]
                        BinaryKernel::Avx512 => unsafe {
                            tile_avx512::<1>(
                                padded, pix, px_stride, koff, words, wb, dst, cout, lanes, n_total,
                            );
                            true
                        },
                        #[cfg(target_arch = "x86_64")]
                        BinaryKernel::Avx2 => unsafe {
                            tile_avx2(padded, pix, koff, words, wb, &mut mis);
                            false
                        },
                        _ => {
                            tile_scalar(padded, pix, koff, words, wb, &mut mis);
                            false
                        }
                    };
                    if !direct {
                        let d = &mut row[ox * cout + b * BLOCK..][..lanes];
                        for (d, &mm) in d.iter_mut().zip(mis.iter()) {
                            *d = n_total - 2.0 * mm as f32;
                        }
                    }
                    ox += np;
                }
            }
            for ox in 0..g.ow {
                let r = residual.map(|r| r.pixel(oy, ox));
                epi.apply(&mut row[ox * cout..(ox + 1) * cout], r);
            }
        });
    out
}

/// Portable kernel: mismatch counts of one pixel against one weight block.
/// Panics (bounds checks) rather than reading out of range.
fn tile_scalar(
    input: &[u64],
    pix: usize,
    koff: &[usize],
    words: usize,
    wb: &[u64],
    mis: &mut [u64; BLOCK],
) {
    mis.fill(0);
    for (k, &off) in koff.iter().enumerate() {
        let a_words = &input[pix + off..pix + off + words];
        for (j, &a) in a_words.iter().enumerate() {
            let wv = &wb[(k * words + j) * BLOCK..][..BLOCK];
            for (m, &x) in mis.iter_mut().zip(wv) {
                *m += (a ^ x).count_ones() as u64;
            }
        }
    }
}

/// # Safety
/// CPU must support `avx512f` and `avx512vpopcntdq`. For every `p < P`,
/// `k < koff.len()`: `pix + p * px_stride + koff[k] + words <= input.len()`.
/// `wb.len() >= koff.len() * words * BLOCK`; `1 <= lanes <= BLOCK`; for
/// every `p < P`, `out + p * out_stride` is valid for writing `lanes` f32s.
///
/// Writes `n_total - 2 * popcount(a ^ w)` (the exact ±1 dot product) for
/// the `lanes` output channels of each of the `P` pixels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn tile_avx512<const P: usize>(
    input: &[u64],
    pix: usize,
    px_stride: usize,
    koff: &[usize],
    words: usize,
    wb: &[u64],
    out: *mut f32,
    out_stride: usize,
    lanes: usize,
    n_total: f32,
) {
    let nk = koff.len();
    debug_assert!(wb.len() >= nk * words * BLOCK && (1..=BLOCK).contains(&lanes));
    debug_assert!(
        koff.iter()
            .all(|&o| pix + (P - 1) * px_stride + o + words <= input.len())
    );
    let ip = input.as_ptr().add(pix);
    let mut acc = [[_mm512_setzero_si512(); 4]; P];
    let mut wp = wb.as_ptr() as *const __m512i;
    for &off in koff {
        let tap = ip.add(off);
        for j in 0..words {
            let w0 = _mm512_loadu_si512(wp);
            let w1 = _mm512_loadu_si512(wp.add(1));
            let w2 = _mm512_loadu_si512(wp.add(2));
            let w3 = _mm512_loadu_si512(wp.add(3));
            wp = wp.add(4);
            for p in 0..P {
                let a = _mm512_set1_epi64(*tap.add(p * px_stride + j) as i64);
                acc[p][0] =
                    _mm512_add_epi64(acc[p][0], _mm512_popcnt_epi64(_mm512_xor_si512(a, w0)));
                acc[p][1] =
                    _mm512_add_epi64(acc[p][1], _mm512_popcnt_epi64(_mm512_xor_si512(a, w1)));
                acc[p][2] =
                    _mm512_add_epi64(acc[p][2], _mm512_popcnt_epi64(_mm512_xor_si512(a, w2)));
                acc[p][3] =
                    _mm512_add_epi64(acc[p][3], _mm512_popcnt_epi64(_mm512_xor_si512(a, w3)));
            }
        }
    }
    let lo_mask: __mmask16 = if lanes >= 16 {
        0xffff
    } else {
        ((1u32 << lanes) - 1) as u16
    };
    let hi_mask: __mmask16 = if lanes >= 32 {
        0xffff
    } else if lanes > 16 {
        ((1u32 << (lanes - 16)) - 1) as u16
    } else {
        0
    };
    let n = _mm512_set1_ps(n_total);
    let minus2 = _mm512_set1_ps(-2.0);
    for p in 0..P {
        // Counts are < 2^31, so narrowing u64 -> i32 is exact.
        let lo = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtepi64_epi32(acc[p][0])),
            _mm512_cvtepi64_epi32(acc[p][1]),
        );
        let hi = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtepi64_epi32(acc[p][2])),
            _mm512_cvtepi64_epi32(acc[p][3]),
        );
        let dst = out.add(p * out_stride);
        _mm512_mask_storeu_ps(
            dst,
            lo_mask,
            _mm512_fmadd_ps(minus2, _mm512_cvtepi32_ps(lo), n),
        );
        _mm512_mask_storeu_ps(
            dst.add(16),
            hi_mask,
            _mm512_fmadd_ps(minus2, _mm512_cvtepi32_ps(hi), n),
        );
    }
}

/// # Safety
/// CPU must support `avx2`. For every `k`, `pix + koff[k] + words <=
/// input.len()`; `wb.len() >= koff.len() * words * BLOCK`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_avx2(
    input: &[u64],
    pix: usize,
    koff: &[usize],
    words: usize,
    wb: &[u64],
    mis: &mut [u64; BLOCK],
) {
    debug_assert!(wb.len() >= koff.len() * words * BLOCK);
    debug_assert!(koff.iter().all(|&o| pix + o + words <= input.len()));
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let mut acc = [_mm256_setzero_si256(); BLOCK / 4];
    let ip = input.as_ptr().add(pix);
    let mut wp = wb.as_ptr() as *const __m256i;
    for &off in koff {
        for j in 0..words {
            let a = _mm256_set1_epi64x(*ip.add(off + j) as i64);
            for (q, accq) in acc.iter_mut().enumerate() {
                let x = _mm256_xor_si256(a, _mm256_loadu_si256(wp.add(q)));
                let lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, low));
                let hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), low));
                *accq = _mm256_add_epi64(*accq, _mm256_sad_epu8(_mm256_add_epi8(lo, hi), zero));
            }
            wp = wp.add(BLOCK / 4);
        }
    }
    let dst = mis.as_mut_ptr() as *mut __m256i;
    for (q, accq) in acc.iter().enumerate() {
        _mm256_storeu_si256(dst.add(q), *accq);
    }
}

// ---------------------------------------------------------------------------
// Float reference
// ---------------------------------------------------------------------------

/// Straightforward float reference of a larq binary conv: `sign` the input
/// and the HWIO weights, pad the *binarised* input with `pad_value`, and
/// accumulate the products in `f32` (exact: all partial sums are small
/// integers). Shares no code with the packed kernels.
#[allow(clippy::too_many_arguments)]
pub fn binary_conv2d_reference(
    input: &FeatureMap,
    w_hwio: &[f32],
    kernel: [usize; 2],
    cout: usize,
    stride: [usize; 2],
    padding: Padding,
    pad_value: f32,
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
) -> FeatureMap {
    let sign = |x: f32| if x >= 0.0 { 1.0f32 } else { -1.0 };
    let cin = input.c;
    assert_eq!(w_hwio.len(), kernel[0] * kernel[1] * cin * cout);
    let g = Geometry::new(input.h, input.w, kernel, stride, padding).expect("kernel does not fit");
    let mut out = FeatureMap::zeros(g.oh, g.ow, cout);
    for oy in 0..g.oh {
        for ox in 0..g.ow {
            let mut acc = vec![0.0f32; cout];
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let src = match (g.in_y(oy, ky), g.in_x(ox, kx)) {
                        (Some(iy), Some(ix)) => Some(input.pixel(iy, ix)),
                        _ => None,
                    };
                    for ci in 0..cin {
                        let a = src.map_or(pad_value, |p| sign(p[ci]));
                        for (o, accv) in acc.iter_mut().enumerate() {
                            *accv += a * sign(w_hwio[((ky * g.kw + kx) * cin + ci) * cout + o]);
                        }
                    }
                }
            }
            epi.apply(&mut acc, residual.map(|r| r.pixel(oy, ox)));
            let o = (oy * g.ow + ox) * cout;
            out.data[o..o + cout].copy_from_slice(&acc);
        }
    }
    out
}
