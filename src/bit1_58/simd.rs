//! Vectorised dot products over 1.58-bit ternary weights.
//!
//! BitNet stores each weight as a 2-bit code in `{-1, 0, +1}`, packed 4 per
//! byte (`pack4`): within one `u32` weight word the codes are the little-endian
//! 2-bit fields `p0 | p1<<2 | p2<<4 | p3<<6`. Because the activation side is
//! quantised to `i8`, the whole matmul reduces to a per-row dot product
//! `Σ a_i * w_i` where each `w_i ∈ {-1,0,+1}`.
//!
//! The hot loop decodes 16 packed bytes (= 64 weights) per 512-bit lane by
//! splatting the word into four 2-bit fields, looking each up in an 8-bit LUT
//! (0→-1, 1→0, 2→+1, 3→0) to get signed activation weights `w`, then doing the
//! dot product with the i8 activations:
//! - pre-VNNI: `_mm512_maddubs_epi16` (u8=1 × i8, horizontally summed to i16)
//!   followed by `_mm512_madd_epi16` (i16 pairs summed to i32);
//! - VNNI: the LUT yields the unsigned `w + 1 ∈ {0, 1, 2}` instead, and two
//!   `_mm512_dpbusd_epi32` accumulate `Σ (w+1)·a` and `Σ a` (no sign fix-up
//!   or mask ops at all); the result is their difference.
//!
//! Multiple independent accumulators are kept live so the ~5-cycle multiply
//! latency is hidden by instruction-level parallelism.

//!
//! # Activation range precondition
//!
//! All SIMD kernels form `±a` with a wrapping 8-bit negate (`0 - a` or
//! `sign_epi8`), so an activation of `-128` paired with a `-1` weight yields
//! `-128` instead of `+128`. Activations must therefore lie in `[-127, 127]`,
//! which is exactly what [`quantize_f32_to_i8`] produces (it clamps
//! symmetrically). The safe entry points `debug_assert!` this; within that
//! range every kernel is bit-exact with [`ternary_dot_product_pack4_scalar`].
//!
//! [`quantize_f32_to_i8`]: crate::bit1_58::quantization::quantize_f32_to_i8
//!
//! # Weight codes
//!
//! pack4 codes are `0b00 => -1`, `0b01 => 0`, `0b10 => +1`; the unused code
//! `0b11` decodes to `0` in every kernel.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Decode one 2-bit pack4 code (`0b00 => -1`, `0b10 => +1`, else `0`).
#[inline(always)]
pub(crate) fn decode_pack4_code(code: u8) -> i32 {
    match code & 0b11 {
        0b00 => -1,
        0b10 => 1,
        _ => 0,
    }
}

/// Scalar tail shared by the SIMD kernels: elements `start..total_elems`.
#[inline(always)]
pub(crate) fn pack4_tail(a_i8: &[i8], b_pack4: &[u8], start: usize, total_elems: usize) -> i32 {
    let mut total = 0i32;
    for j in start..total_elems {
        let code = b_pack4[j / 4] >> ((j % 4) * 2);
        total += a_i8[j] as i32 * decode_pack4_code(code);
    }
    total
}

#[inline(always)]
fn check_pack4_args(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) {
    assert!(
        a_i8.len() >= total_elems,
        "ternary pack4: activation slice too short ({} < {})",
        a_i8.len(),
        total_elems
    );
    assert!(
        b_pack4.len() >= total_elems.div_ceil(4),
        "ternary pack4: weight slice too short ({} < {})",
        b_pack4.len(),
        total_elems.div_ceil(4)
    );
    debug_assert!(
        a_i8[..total_elems].iter().all(|&x| x != i8::MIN),
        "ternary pack4: activations must be in [-127, 127]"
    );
}

/// Reference scalar pack4 dot product `Σ a_i · w_i` (exact for any `i8`).
///
/// Panics if `a_i8.len() < total_elems` or `b_pack4.len() < total_elems.div_ceil(4)`.
pub fn ternary_dot_product_pack4_scalar(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    assert!(a_i8.len() >= total_elems && b_pack4.len() >= total_elems.div_ceil(4));
    pack4_tail(a_i8, b_pack4, 0, total_elems)
}

/// AVX-512BW pack4 kernel (pre-VNNI: `maddubs` + `madd`).
///
/// # Safety
/// The CPU must support `avx512f` and `avx512bw`; `a_i8.len() >= total_elems`
/// and `b_pack4.len() >= total_elems.div_ceil(4)`. Activations should be in
/// `[-127, 127]` (see the module docs) for an exact result.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_product_pack4_avx512(
    a_i8: &[i8],
    b_pack4: &[u8],
    total_elems: usize,
) -> i32 {
    let a_ptr = a_i8.as_ptr() as *const __m512i;
    let b_ptr = b_pack4.as_ptr();

    let zero = _mm512_setzero_si512();
    // 4 independent accumulators hide the ~5-cycle add_epi32 latency
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;
    let ones_u8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let mask3 = _mm512_set1_epi32(0x03);
    let lut = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, -1,
    );

    let chunks64 = total_elems / 64;
    let step4 = chunks64 / 4 * 4;
    let mut idx = 0usize;

    while idx < step4 {
        // Prefetch addresses may run past the end of the weights; a prefetch
        // never faults, but `add` past the allocation would be UB, so wrap.
        _mm_prefetch(b_ptr.wrapping_add(idx * 16 + 64) as *const i8, _MM_HINT_NTA);
        _mm_prefetch(
            b_ptr.wrapping_add(idx * 16 + 128) as *const i8,
            _MM_HINT_NTA,
        );

        macro_rules! block {
            ($acc:expr, $i:expr) => {{
                let acts = _mm512_loadu_si512(a_ptr.add(idx + $i));
                let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                    b_ptr.add((idx + $i) * 16) as *const __m128i
                ));
                let p0 = _mm512_and_si512(w32, mask3);
                let p1 = _mm512_and_si512(_mm512_srli_epi32(w32, 2), mask3);
                let p2 = _mm512_and_si512(_mm512_srli_epi32(w32, 4), mask3);
                let p3 = _mm512_and_si512(_mm512_srli_epi32(w32, 6), mask3);
                let packed = _mm512_or_si512(
                    _mm512_or_si512(p0, _mm512_slli_epi32(p1, 8)),
                    _mm512_or_si512(_mm512_slli_epi32(p2, 16), _mm512_slli_epi32(p3, 24)),
                );
                let w = _mm512_shuffle_epi8(lut, packed);
                let pos_mask = _mm512_cmp_epi8_mask(w, zero, 6);
                let zero_mask = _mm512_cmp_epi8_mask(w, zero, 0);
                let neg_a = _mm512_sub_epi8(zero, acts);
                let blend = _mm512_mask_blend_epi8(pos_mask, neg_a, acts);
                let sa = _mm512_mask_mov_epi8(blend, zero_mask, zero);
                $acc = _mm512_add_epi32(
                    $acc,
                    _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8, sa), ones_i16),
                );
            }};
        }

        block!(acc0, 0);
        block!(acc1, 1);
        block!(acc2, 2);
        block!(acc3, 3);

        idx += 4;
    }

    // Merge 4 accumulators
    let mut acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));

    // Cleanup: remaining chunks (< 4)
    while idx < chunks64 {
        _mm_prefetch(
            b_ptr.wrapping_add(idx * 16 + 256) as *const i8,
            _MM_HINT_NTA,
        );
        let acts = _mm512_loadu_si512(a_ptr.add(idx));
        let packed_w = _mm_loadu_si128(b_ptr.add(idx * 16) as *const __m128i);
        let w32 = _mm512_cvtepu8_epi32(packed_w);
        let p0 = _mm512_and_si512(w32, mask3);
        let p1 = _mm512_and_si512(_mm512_srli_epi32(w32, 2), mask3);
        let p2 = _mm512_and_si512(_mm512_srli_epi32(w32, 4), mask3);
        let p3 = _mm512_and_si512(_mm512_srli_epi32(w32, 6), mask3);
        let packed = _mm512_or_si512(
            _mm512_or_si512(p0, _mm512_slli_epi32(p1, 8)),
            _mm512_or_si512(_mm512_slli_epi32(p2, 16), _mm512_slli_epi32(p3, 24)),
        );
        let w = _mm512_shuffle_epi8(lut, packed);
        let pos_mask = _mm512_cmp_epi8_mask(w, zero, 6);
        let zero_mask = _mm512_cmp_epi8_mask(w, zero, 0);
        let neg_a = _mm512_sub_epi8(zero, acts);
        let blend = _mm512_mask_blend_epi8(pos_mask, neg_a, acts);
        let signed_acts = _mm512_mask_mov_epi8(blend, zero_mask, zero);
        acc = _mm512_add_epi32(
            acc,
            _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8, signed_acts), ones_i16),
        );
        idx += 1;
    }

    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
    let mut total = tmp.iter().sum::<i32>();
    total += pack4_tail(a_i8, b_pack4, chunks64 * 64, total_elems);
    total
}

/// AVX2 pack4 kernel (`cvtepu8` + shift decode, `sign_epi8` + `maddubs`).
///
/// # Safety
/// The CPU must support `avx2`; `a_i8.len() >= total_elems` and
/// `b_pack4.len() >= total_elems.div_ceil(4)`. Activations should be in
/// `[-127, 127]` (see the module docs) for an exact result.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_product_pack4_avx2(
    a_i8: &[i8],
    b_pack4: &[u8],
    total_elems: usize,
) -> i32 {
    unsafe {
        let a_ptr = a_i8.as_ptr() as *const __m256i;
        let b_ptr = b_pack4.as_ptr();

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);
        let mask3 = _mm256_set1_epi32(0x03);
        let lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        );

        let chunks32 = total_elems / 32;
        let step = chunks32 / 2 * 2;
        let mut i = 0;
        while i < step {
            _mm_prefetch(b_ptr.wrapping_add(i * 8 + 16) as *const i8, _MM_HINT_NTA);

            let acts0 = _mm256_loadu_si256(a_ptr.add(i));
            let packed_w0 = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let x0 = _mm256_cvtepu8_epi32(packed_w0);

            let p00 = _mm256_and_si256(x0, mask3);
            let p01 = _mm256_and_si256(_mm256_srli_epi32(x0, 2), mask3);
            let p02 = _mm256_and_si256(_mm256_srli_epi32(x0, 4), mask3);
            let p03 = _mm256_and_si256(_mm256_srli_epi32(x0, 6), mask3);

            let w_i8_0 = _mm256_shuffle_epi8(
                lut,
                _mm256_or_si256(
                    _mm256_or_si256(p00, _mm256_slli_epi32(p01, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(p02, 16), _mm256_slli_epi32(p03, 24)),
                ),
            );
            let sums0 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts0, w_i8_0)),
                ones_i16,
            );
            acc0 = _mm256_add_epi32(acc0, sums0);

            let acts1 = _mm256_loadu_si256(a_ptr.add(i + 1));
            let packed_w1 = _mm_loadl_epi64(b_ptr.add((i + 1) * 8) as *const __m128i);
            let x1 = _mm256_cvtepu8_epi32(packed_w1);

            let p10 = _mm256_and_si256(x1, mask3);
            let p11 = _mm256_and_si256(_mm256_srli_epi32(x1, 2), mask3);
            let p12 = _mm256_and_si256(_mm256_srli_epi32(x1, 4), mask3);
            let p13 = _mm256_and_si256(_mm256_srli_epi32(x1, 6), mask3);

            let w_i8_1 = _mm256_shuffle_epi8(
                lut,
                _mm256_or_si256(
                    _mm256_or_si256(p10, _mm256_slli_epi32(p11, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(p12, 16), _mm256_slli_epi32(p13, 24)),
                ),
            );
            let sums1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts1, w_i8_1)),
                ones_i16,
            );
            acc1 = _mm256_add_epi32(acc1, sums1);

            i += 2;
        }

        let mut acc32 = _mm256_add_epi32(acc0, acc1);

        for i in i..chunks32 {
            let acts = _mm256_loadu_si256(a_ptr.add(i));
            let packed_w = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let x = _mm256_cvtepu8_epi32(packed_w);

            let p0 = _mm256_and_si256(x, mask3);
            let p1 = _mm256_and_si256(_mm256_srli_epi32(x, 2), mask3);
            let p2 = _mm256_and_si256(_mm256_srli_epi32(x, 4), mask3);
            let p3 = _mm256_and_si256(_mm256_srli_epi32(x, 6), mask3);

            let w_i8 = _mm256_shuffle_epi8(
                lut,
                _mm256_or_si256(
                    _mm256_or_si256(p0, _mm256_slli_epi32(p1, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(p2, 16), _mm256_slli_epi32(p3, 24)),
                ),
            );
            let sums = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts, w_i8)),
                ones_i16,
            );
            acc32 = _mm256_add_epi32(acc32, sums);
        }

        let mut sums = [0i32; 8];
        _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc32);
        let mut total_sum = sums.iter().sum::<i32>();

        total_sum += pack4_tail(a_i8, b_pack4, chunks32 * 32, total_elems);
        total_sum
    }
}

/// Tunable weight prefetch look-ahead (in bytes) for the VNNI GEMV kernel.
/// Set via `XORNET_VNNI_PREFETCH` (0 = uninitialized → default 256). Larger
/// values issue weight prefetches further ahead to hide DDR5 latency; the
/// optimum depends on the memory controller and how many concurrent weight
/// streams the surrounding rayon fan-out issues.
///
/// Weight prefetches use the non-temporal hint (`_MM_HINT_NTA`) because each
/// weight byte is read exactly once: pulling it into L1/L2 would only evict the
/// reused activation vector and RoPE tables, so we keep the weights out of the
/// low-level caches. (A true streaming `vmovntdqa` load would need 16-byte
/// aligned weight buffers, which the current `Vec<u8>` does not guarantee.)
#[cfg(target_arch = "x86_64")]
static VNNI_PREFETCH_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_arch = "x86_64")]
#[inline]
fn vnni_prefetch_bytes() -> usize {
    let mut v = VNNI_PREFETCH_BYTES.load(Ordering::Relaxed);
    if v == 0 {
        v = std::env::var("XORNET_VNNI_PREFETCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&x| x > 0)
            .unwrap_or(256);
        VNNI_PREFETCH_BYTES.store(v, Ordering::Relaxed);
    }
    v
}

/// AVX-512 VNNI pack4 kernel (`vpdpbusd`, 16 accumulator chains). The decode GEMV
/// hot loop on Zen 4/5 and Ice Lake+.
///
/// Instead of building signed `±a` bytes (two compares, a masked subtract
/// and a masked move per 64 weights), the codes are mapped to the unsigned
/// "weight + 1" `u ∈ {0, 1, 2}` and fed straight into `vpdpbusd` as the
/// unsigned operand: `Σ a·w = Σ a·u − Σ a`. `Σ a` is a second `vpdpbusd`
/// against all-ones. Both sums are exact in `i32` (no saturation; `vpdpbusd`
/// does not saturate), so the result is bit-identical to the scalar kernel
/// for *any* `i8` activation, `-128` included.
///
/// # Safety
/// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`;
/// `a_i8.len() >= total_elems` and `b_pack4.len() >= total_elems.div_ceil(4)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_dot_product_pack4_avx512_vnni(
    a_i8: &[i8],
    b_pack4: &[u8],
    total_elems: usize,
) -> i32 {
    let a_ptr = a_i8.as_ptr() as *const __m512i;
    let b_ptr = b_pack4.as_ptr();
    let pf = vnni_prefetch_bytes();

    let zero = _mm512_setzero_si512();
    let ones_u8 = _mm512_set1_epi8(1);
    let mask3 = _mm512_set1_epi32(0x03);
    // Code -> weight + 1: 0b00 (-1) -> 0, 0b01 (0) -> 1, 0b10 (+1) -> 2,
    // 0b11 (unused, decodes to 0) -> 1. Only indices 0..=3 occur.
    let lut = _mm512_set4_epi32(0, 0, 0, 0x0102_0100);

    // One 64-weight chunk: `acc += Σ u·a`, `asum += Σ a` (per i32 lane).
    macro_rules! chunk {
        ($i:expr, $acc:expr, $asum:expr) => {{
            let i = $i;
            let acts = _mm512_loadu_si512(a_ptr.add(i));
            let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(b_ptr.add(i * 16) as *const __m128i));
            let p0 = _mm512_and_si512(w32, mask3);
            let p1 = _mm512_and_si512(_mm512_srli_epi32(w32, 2), mask3);
            let p2 = _mm512_and_si512(_mm512_srli_epi32(w32, 4), mask3);
            let p3 = _mm512_and_si512(_mm512_srli_epi32(w32, 6), mask3);
            let packed = _mm512_or_si512(
                _mm512_or_si512(p0, _mm512_slli_epi32(p1, 8)),
                _mm512_or_si512(_mm512_slli_epi32(p2, 16), _mm512_slli_epi32(p3, 24)),
            );
            let u = _mm512_shuffle_epi8(lut, packed);
            $acc = _mm512_dpbusd_epi32($acc, u, acts);
            $asum = _mm512_dpbusd_epi32($asum, ones_u8, acts);
        }};
    }

    let chunks64 = total_elems / 64;
    let mut idx = 0usize;

    // 8 chunks per step with 8 product + 8 sum accumulators: 16 independent
    // `vpdpbusd` chains hide its latency.
    let mut acc = [zero; 8];
    let mut asum = [zero; 8];
    while idx + 8 <= chunks64 {
        _mm_prefetch(b_ptr.wrapping_add(idx * 16 + pf) as *const i8, _MM_HINT_NTA);
        chunk!(idx, acc[0], asum[0]);
        chunk!(idx + 1, acc[1], asum[1]);
        chunk!(idx + 2, acc[2], asum[2]);
        chunk!(idx + 3, acc[3], asum[3]);
        chunk!(idx + 4, acc[4], asum[4]);
        chunk!(idx + 5, acc[5], asum[5]);
        chunk!(idx + 6, acc[6], asum[6]);
        chunk!(idx + 7, acc[7], asum[7]);
        idx += 8;
    }
    // Remaining chunks (< 8) still spread over independent accumulators.
    if idx + 4 <= chunks64 {
        _mm_prefetch(b_ptr.wrapping_add(idx * 16 + pf) as *const i8, _MM_HINT_NTA);
        chunk!(idx, acc[0], asum[0]);
        chunk!(idx + 1, acc[1], asum[1]);
        chunk!(idx + 2, acc[2], asum[2]);
        chunk!(idx + 3, acc[3], asum[3]);
        idx += 4;
    }
    if idx + 2 <= chunks64 {
        chunk!(idx, acc[4], asum[4]);
        chunk!(idx + 1, acc[5], asum[5]);
        idx += 2;
    }
    if idx < chunks64 {
        chunk!(idx, acc[6], asum[6]);
    }

    let a01 = _mm512_add_epi32(acc[0], acc[1]);
    let a23 = _mm512_add_epi32(acc[2], acc[3]);
    let a45 = _mm512_add_epi32(acc[4], acc[5]);
    let a67 = _mm512_add_epi32(acc[6], acc[7]);
    let s01 = _mm512_add_epi32(asum[0], asum[1]);
    let s23 = _mm512_add_epi32(asum[2], asum[3]);
    let s45 = _mm512_add_epi32(asum[4], asum[5]);
    let s67 = _mm512_add_epi32(asum[6], asum[7]);
    let prod = _mm512_add_epi32(_mm512_add_epi32(a01, a23), _mm512_add_epi32(a45, a67));
    let sum = _mm512_add_epi32(_mm512_add_epi32(s01, s23), _mm512_add_epi32(s45, s67));
    // No lane can overflow: each chunk adds at most 4 · 2 · 128 per lane.
    let mut total = _mm512_reduce_add_epi32(_mm512_sub_epi32(prod, sum));
    total += pack4_tail(a_i8, b_pack4, chunks64 * 64, total_elems);
    total
}

/// Ternary pack4 dot product `Σ a_i · w_i` over the first `total_elems`
/// elements, dispatched to the best kernel for this CPU.
///
/// Panics if `a_i8.len() < total_elems` or
/// `b_pack4.len() < total_elems.div_ceil(4)`. Activations must be in
/// `[-127, 127]` (checked in debug builds; see the module docs).
#[inline]
pub fn ternary_dot_product_pack4(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    check_pack4_args(a_i8, b_pack4, total_elems);
    #[cfg(target_arch = "x86_64")]
    {
        let f = crate::util::cpu_features();
        // SAFETY (all three): the required CPU features were detected and
        // the slice lengths were checked by `check_pack4_args`.
        if f.avx512_vnni() {
            return unsafe { ternary_dot_product_pack4_avx512_vnni(a_i8, b_pack4, total_elems) };
        }
        if f.avx512bw() {
            return unsafe { ternary_dot_product_pack4_avx512(a_i8, b_pack4, total_elems) };
        }
        if f.avx2 {
            return unsafe {
                crate::bit1_58::lut::ternary_dot_product_pack4_avx2_hybrid(
                    a_i8,
                    b_pack4,
                    total_elems,
                )
            };
        }
    }
    pack4_tail(a_i8, b_pack4, 0, total_elems)
}

pub fn ternary_dot_product_pack5(a_i8: &[i8], b_pack5: &[u8], total_elems: usize) -> i32 {
    ternary_dot_product_pack5_scalar(a_i8, b_pack5, total_elems)
}

pub fn ternary_dot_product_pack5_scalar(a_i8: &[i8], b_pack5: &[u8], total_elems: usize) -> i32 {
    let mut total_sum = 0i32;

    let chunks = total_elems / 5;
    for i in 0..chunks {
        let mut b = b_pack5[i];
        for j in 0..5 {
            let val = b % 3;
            b /= 3;
            let w = if val == 0 {
                -1
            } else if val == 2 {
                1
            } else {
                0
            };
            total_sum += a_i8[i * 5 + j] as i32 * w;
        }
    }

    let remainder_start = chunks * 5;
    if remainder_start < total_elems {
        let mut b = b_pack5[chunks];
        for j in 0..(total_elems - remainder_start) {
            let val = b % 3;
            b /= 3;
            let w = if val == 0 {
                -1
            } else if val == 2 {
                1
            } else {
                0
            };
            total_sum += a_i8[remainder_start + j] as i32 * w;
        }
    }

    total_sum
}

// ===========================================================================
// XorSparse lossless sparse ternary dot product
// ===========================================================================

#[inline(always)]
fn check_sparse_args(a_i8: &[i8], masks: &[u64], signs: &[u64], num_blocks: usize, in_dim: usize) {
    assert_eq!(
        num_blocks,
        in_dim.div_ceil(64),
        "ternary sparse: num_blocks must be ceil(in_dim / 64)"
    );
    assert!(
        masks.len() >= num_blocks && signs.len() >= num_blocks,
        "ternary sparse: masks/signs shorter than num_blocks"
    );
    assert!(
        a_i8.len() >= in_dim,
        "ternary sparse: activation slice too short ({} < {})",
        a_i8.len(),
        in_dim
    );
    debug_assert!(
        a_i8[..in_dim].iter().all(|&x| x != i8::MIN),
        "ternary sparse: activations must be in [-127, 127]"
    );
}

/// Scalar XorSparse row dot product. Consumes the decoded per-block lane-order
/// `masks`/`signs` (no parsing, no set-order expansion). Exact for any `i8`.
///
/// Panics unless `num_blocks == in_dim.div_ceil(64)`, `masks`/`signs` hold at
/// least `num_blocks` entries and `a_i8.len() >= in_dim`. Mask bits at lanes
/// `>= in_dim` in the final block are ignored.
pub fn ternary_dot_product_sparse_scalar(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    assert_eq!(num_blocks, in_dim.div_ceil(64));
    assert!(masks.len() >= num_blocks && signs.len() >= num_blocks && a_i8.len() >= in_dim);
    let mut total = 0i32;
    for b in 0..num_blocks {
        let start = b * 64;
        let end = (start + 64).min(in_dim);
        total += dot_block_sparse_scalar(&a_i8[start..end], masks[b], signs[b]);
    }
    total
}

/// One block of up to 64 lanes, scalar. `k2` is the lane-order sign mask (bit
/// *k* set => weight at lane *k* is `-1`). Only lanes `< a.len()` are used.
#[inline]
fn dot_block_sparse_scalar(a: &[i8], mask: u64, k2: u64) -> i32 {
    let valid = if a.len() >= 64 {
        u64::MAX
    } else {
        (1u64 << a.len()) - 1
    };
    let mut acc = 0i32;
    let mut m = mask & valid;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        let sign = if (k2 >> bit) & 1 != 0 { -1 } else { 1 };
        acc += a[bit] as i32 * sign;
        m &= m - 1;
    }
    acc
}

/// One full 64-lane block with AVX-512BW mask ops. Returns 16 i32 partials.
///
/// # Safety
/// The CPU must support `avx512f` and `avx512bw`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_block_avx512(a: __m512i, mask: u64, k2: u64) -> __m512i {
    let ones_u8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let zero = _mm512_setzero_si512();

    // `k2` is the lane-order sign mask (bit *k* set => weight at lane *k* is `-1`).
    let k1 = _load_mask64(&mask);
    let k2m = _load_mask64(&k2);
    let k_pos = _kandn_mask64(k2m, k1); // +1 positions
    let k_neg = _kand_mask64(k1, k2m); // -1 positions

    // Signed product sa = a*w in one fused step: -a at neg lanes, +a at pos
    // lanes, 0 elsewhere. `mask_sub(src, k, a, b) = k ? a - b : src` and
    // `mask_add(src, k, a, b) = k ? a + b : src`.
    let t = _mm512_mask_sub_epi8(zero, k_neg, zero, a); // -a at neg, 0 else
    let sa = _mm512_mask_add_epi8(t, k_pos, zero, a); // +a at pos, else t
    _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8, sa), ones_i16)
}

/// AVX-512BW XorSparse row dot product.
///
/// # Safety
/// The CPU must support `avx512f` and `avx512bw`; `num_blocks ==
/// in_dim.div_ceil(64)`, `masks.len() >= num_blocks`,
/// `signs.len() >= num_blocks` and `a_i8.len() >= in_dim`. Activations should
/// be in `[-127, 127]` (see the module docs) for an exact result.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_product_sparse_avx512(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    let a_ptr = a_i8.as_ptr();
    let zero = _mm512_setzero_si512();
    let mut acc = [zero; 16];
    let mut tail = 0i32;

    for b in 0..num_blocks {
        let mask = masks[b];
        let k2 = signs[b];
        let start = b * 64;
        let n = (start + 64).min(in_dim) - start;
        // A short final block would over-read the activations: scalar.
        if n < 64 {
            tail = dot_block_sparse_scalar(&a_i8[start..start + n], mask, k2);
            break;
        }

        let a = _mm512_loadu_si512(a_ptr.add(start) as *const __m512i);
        let partial = ternary_dot_block_avx512(a, mask, k2);
        acc[b % 16] = _mm512_add_epi32(acc[b % 16], partial);
    }

    let mut merged = acc[0];
    for v in &acc[1..] {
        merged = _mm512_add_epi32(merged, *v);
    }
    _mm512_reduce_add_epi32(merged) + tail
}

/// XorSparse row dot product. Consumes the decoded per-block lane-order
/// `masks`/`signs` (built once at load). Dispatches to the AVX-512 kernel when
/// available, otherwise scalar.
///
/// Panics unless `num_blocks == in_dim.div_ceil(64)`, `masks`/`signs` hold at
/// least `num_blocks` entries and `a_i8.len() >= in_dim`. Activations must be
/// in `[-127, 127]` (checked in debug builds).
pub fn ternary_dot_product_sparse(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    check_sparse_args(a_i8, masks, signs, num_blocks, in_dim);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::util::cpu_features().avx512bw() {
            // SAFETY: features detected, lengths checked above.
            return unsafe {
                ternary_dot_product_sparse_avx512(a_i8, masks, signs, num_blocks, in_dim)
            };
        }
    }
    ternary_dot_product_sparse_scalar(a_i8, masks, signs, num_blocks, in_dim)
}

#[cfg(test)]
mod sparse_correctness_tests {
    use crate::bit1_58::quantization::pack_1_58bit_4pack;
    use crate::bit1_58::sparse::{SparseTernary, encode_sparse_tensor};

    // Simple deterministic LCG so the test needs no external deps.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_dense(in_dim: usize, out_dim: usize, rng: &mut Lcg) -> Vec<f32> {
        (0..out_dim * in_dim)
            .map(|_| match rng.below(3) {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect()
    }

    fn random_act(n: usize, rng: &mut Lcg) -> Vec<i8> {
        (0..n).map(|_| (rng.below(7) as i8) - 3).collect()
    }

    #[test]
    fn dense_and_sparse_dot_match() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for (in_dim, out_dim) in [(2560, 2560), (6912, 2560), (2560, 640), (2560, 6912)] {
            let w = random_dense(in_dim, out_dim, &mut rng);
            let packed = pack_1_58bit_4pack(&w, 1.0);
            let act = random_act(in_dim, &mut rng);

            let bpr = in_dim.div_ceil(4);
            let dense: Vec<i32> = (0..out_dim)
                .map(|r| {
                    super::ternary_dot_product_pack4(&act, &packed[r * bpr..(r + 1) * bpr], in_dim)
                })
                .collect();

            let w_i8: Vec<i8> = w.iter().map(|&v| v as i8).collect();
            let (blob, _offs) = encode_sparse_tensor(&w_i8, in_dim, out_dim);
            let st = SparseTernary::from_blob(blob, out_dim);
            let sparse: Vec<i32> = (0..out_dim)
                .map(|r| {
                    let (m, sg) = st.blocks(r);
                    super::ternary_dot_product_sparse(&act, m, sg, st.num_blocks, in_dim)
                })
                .collect();
            let sparse_scalar: Vec<i32> = (0..out_dim)
                .map(|r| {
                    let (m, sg) = st.blocks(r);
                    super::ternary_dot_product_sparse_scalar(&act, m, sg, st.num_blocks, in_dim)
                })
                .collect();

            for r in 0..out_dim {
                assert_eq!(
                    dense[r], sparse_scalar[r],
                    "SCALAR row {} mismatch in_dim={} out_dim={} (dense={}, scalar={})",
                    r, in_dim, out_dim, dense[r], sparse_scalar[r]
                );
                assert_eq!(
                    dense[r], sparse[r],
                    "row {} mismatch in_dim={} out_dim={} (dense={}, sparse={})",
                    r, in_dim, out_dim, dense[r], sparse[r]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_block_matches_scalar() {
        use core::arch::x86_64::*;
        if !crate::util::cpu_features().avx512bw() {
            eprintln!("skipping avx_block_matches_scalar: no AVX-512BW");
            return;
        }
        let a: Vec<i8> = (0..64).map(|i| ((i as i8) % 5) - 2).collect();
        let avx_a = unsafe { _mm512_loadu_si512(a.as_ptr() as *const __m512i) };
        let mask: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        // signs in set-order
        let mut sb = 0u64;
        let mut si = 0u32;
        for k in 0..64u32 {
            if (mask >> k) & 1 != 0 {
                if k % 3 == 0 {
                    sb |= 1u64 << si;
                }
                si += 1;
            }
        }
        // expand to lane-order sign mask k2 (mirrors SparseTernary::from_blob)
        let mut k2 = 0u64;
        let mut s2 = 0u32;
        let mut m = mask;
        let mut lane = 0u32;
        while m != 0 {
            if m & 1 != 0 {
                if (sb >> s2) & 1 != 0 {
                    k2 |= 1u64 << lane;
                }
                s2 += 1;
            }
            m >>= 1;
            lane += 1;
        }
        let scalar = super::dot_block_sparse_scalar(&a, mask, k2);
        let avx = unsafe { super::ternary_dot_block_avx512(avx_a, mask, k2) };
        let mut tmp = [0i32; 16];
        unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, avx) };
        let avx_sum = tmp.iter().sum::<i32>();
        assert_eq!(
            avx_sum, scalar,
            "avx block {} != scalar {}",
            avx_sum, scalar
        );
    }

    #[test]
    fn sparse_decode_roundtrip() {
        let mut rng = Lcg(0xdead_beef_cafe_babe);
        let in_dim = 2560;
        let out_dim = 2560;
        let w = random_dense(in_dim, out_dim, &mut rng);
        let w_i8: Vec<i8> = w.iter().map(|&v| v as i8).collect();
        let (blob, _offs) = encode_sparse_tensor(&w_i8, in_dim, out_dim);
        let st = SparseTernary::from_blob(blob, out_dim);
        for r in 0..out_dim {
            let decoded = st.decode_row(r, in_dim);
            for i in 0..in_dim {
                assert_eq!(
                    decoded[i],
                    w_i8[r * in_dim + i],
                    "decode mismatch r={} i={}",
                    r,
                    i
                );
            }
        }
    }
}
