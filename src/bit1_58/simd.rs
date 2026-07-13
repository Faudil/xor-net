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
//! - VNNI: a single `_mm512_dpbusd_epi32` does `acc += Σ (u8=1)·(i8)` directly.
//!
//! Multiple independent accumulators are kept live so the ~5-cycle multiply
//! latency is hidden by instruction-level parallelism.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_product_pack4_avx512(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
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
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
    );

    let chunks64 = total_elems / 64;
    let step4 = chunks64 / 4 * 4;
    let mut idx = 0usize;

    while idx < step4 {
        _mm_prefetch(b_ptr.add(idx * 16 + 64) as *const i8, _MM_HINT_NTA);
        _mm_prefetch(b_ptr.add(idx * 16 + 128) as *const i8, _MM_HINT_NTA);

        macro_rules! block {
            ($acc:expr, $i:expr) => {{
                let acts = _mm512_loadu_si512(a_ptr.add(idx + $i));
                let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(b_ptr.add((idx + $i) * 16) as *const __m128i));
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
                $acc = _mm512_add_epi32($acc, _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8, sa), ones_i16));
            }};
        }

        block!(acc0, 0);
        block!(acc1, 1);
        block!(acc2, 2);
        block!(acc3, 3);

        idx += 4;
    }

    // Merge 4 accumulators
    let mut acc = _mm512_add_epi32(
        _mm512_add_epi32(acc0, acc1),
        _mm512_add_epi32(acc2, acc3),
    );

    // Cleanup: remaining chunks (< 4)
    while idx < chunks64 {
        _mm_prefetch(b_ptr.add(idx * 16 + 256) as *const i8, _MM_HINT_NTA);
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
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(ones_u8, signed_acts), ones_i16));
        idx += 1;
    }

    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
    let mut total = tmp.iter().sum::<i32>();

    let remainder_start = chunks64 * 64;
    for j in remainder_start..total_elems {
        let byte_idx = j / 4;
        let bit_shift = (j % 4) * 2;
        let val = (b_pack4[byte_idx] >> bit_shift) & 0b11;
        let w = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
        total += a_i8[j] as i32 * w as i32;
    }

    total
}


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_product_pack4_avx2(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    unsafe {
        let a_ptr = a_i8.as_ptr() as *const __m256i;
        let b_ptr = b_pack4.as_ptr();

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);
        let mask3 = _mm256_set1_epi32(0x03);
        let lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );

        let chunks32 = total_elems / 32;
        let step = chunks32 / 2 * 2;
        let mut i = 0;
        while i < step {
            _mm_prefetch(b_ptr.add(i * 8 + 16) as *const i8, _MM_HINT_NTA);

            let acts0 = _mm256_loadu_si256(a_ptr.add(i));
            let packed_w0 = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let x0 = _mm256_cvtepu8_epi32(packed_w0);

            let p00 = _mm256_and_si256(x0, mask3);
            let p01 = _mm256_and_si256(_mm256_srli_epi32(x0, 2), mask3);
            let p02 = _mm256_and_si256(_mm256_srli_epi32(x0, 4), mask3);
            let p03 = _mm256_and_si256(_mm256_srli_epi32(x0, 6), mask3);

            let w_i8_0 = _mm256_shuffle_epi8(lut, _mm256_or_si256(
                _mm256_or_si256(p00, _mm256_slli_epi32(p01, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p02, 16), _mm256_slli_epi32(p03, 24))
            ));
            let sums0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts0, w_i8_0)), ones_i16);
            acc0 = _mm256_add_epi32(acc0, sums0);

            let acts1 = _mm256_loadu_si256(a_ptr.add(i + 1));
            let packed_w1 = _mm_loadl_epi64(b_ptr.add((i + 1) * 8) as *const __m128i);
            let x1 = _mm256_cvtepu8_epi32(packed_w1);

            let p10 = _mm256_and_si256(x1, mask3);
            let p11 = _mm256_and_si256(_mm256_srli_epi32(x1, 2), mask3);
            let p12 = _mm256_and_si256(_mm256_srli_epi32(x1, 4), mask3);
            let p13 = _mm256_and_si256(_mm256_srli_epi32(x1, 6), mask3);

            let w_i8_1 = _mm256_shuffle_epi8(lut, _mm256_or_si256(
                _mm256_or_si256(p10, _mm256_slli_epi32(p11, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p12, 16), _mm256_slli_epi32(p13, 24))
            ));
            let sums1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts1, w_i8_1)), ones_i16);
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

            let w_i8 = _mm256_shuffle_epi8(lut, _mm256_or_si256(
                _mm256_or_si256(p0, _mm256_slli_epi32(p1, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p2, 16), _mm256_slli_epi32(p3, 24))
            ));
            let sums = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts, w_i8)), ones_i16);
            acc32 = _mm256_add_epi32(acc32, sums);
        }

        let mut sums = [0i32; 8];
        _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc32);
        let mut total_sum = sums.iter().sum::<i32>();

        let remainder_start = chunks32 * 32;
        for i in remainder_start..total_elems {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let val = (b_pack4[byte_idx] >> bit_shift) & 0b11;
            let w = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
            total_sum += a_i8[i] as i32 * w as i32;
        }

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
static VNNI_PREFETCH_BYTES: AtomicUsize = AtomicUsize::new(0);

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_dot_product_pack4_avx512_vnni(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    let a_ptr = a_i8.as_ptr() as *const __m512i;
    let b_ptr = b_pack4.as_ptr();
    let pf = vnni_prefetch_bytes();

    let zero = _mm512_setzero_si512();
    // 16 independent accumulators hide the ~5-cycle dpbusd latency.
    let mut acc = [zero; 16];
    let ones_u8 = _mm512_set1_epi8(1);
    let mask3 = _mm512_set1_epi32(0x03);
    let lut = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,
    );

    let chunks64 = total_elems / 64;
    let step16 = chunks64 / 16 * 16;
    let mut idx = 0usize;

    while idx < step16 {
        _mm_prefetch(b_ptr.add(idx * 16 + pf) as *const i8, _MM_HINT_NTA);

        macro_rules! block {
            ($k:expr) => {{
                let acts = _mm512_loadu_si512(a_ptr.add(idx + $k));
                let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(b_ptr.add((idx + $k) * 16) as *const __m128i));
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
                // dpbusd: acc += Σ (u8=1) * (i8=sa) = Σ sa, one i32 per 4 bytes.
                acc[$k] = _mm512_dpbusd_epi32(acc[$k], ones_u8, sa);
            }};
        }

        block!(0); block!(1); block!(2); block!(3);
        block!(4); block!(5); block!(6); block!(7);
        block!(8); block!(9); block!(10); block!(11);
        block!(12); block!(13); block!(14); block!(15);

        idx += 16;
    }

    // Merge the 16 accumulators.
    let mut merged = acc[0];
    for k in 1..16 {
        merged = _mm512_add_epi32(merged, acc[k]);
    }

    // Cleanup: remaining chunks (< 16) accumulate into `merged`.
    while idx < chunks64 {
        _mm_prefetch(b_ptr.add(idx * 16 + pf) as *const i8, _MM_HINT_NTA);
        let acts = _mm512_loadu_si512(a_ptr.add(idx));
        let w32 = _mm512_cvtepu8_epi32(_mm_loadu_si128(b_ptr.add(idx * 16) as *const __m128i));
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
        merged = _mm512_dpbusd_epi32(merged, ones_u8, sa);
        idx += 1;
    }

    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, merged);
    let mut total = tmp.iter().sum::<i32>();

    let remainder_start = chunks64 * 64;
    for j in remainder_start..total_elems {
        let byte_idx = j / 4;
        let bit_shift = (j % 4) * 2;
        let val = (b_pack4[byte_idx] >> bit_shift) & 0b11;
        let w = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
        total += a_i8[j] as i32 * w as i32;
    }

    total
}

pub fn ternary_dot_product_pack4(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512vnni") {
            return unsafe { ternary_dot_product_pack4_avx512_vnni(a_i8, b_pack4, total_elems) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { ternary_dot_product_pack4_avx512(a_i8, b_pack4, total_elems) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { crate::bit1_58::lut::ternary_dot_product_pack4_avx2_hybrid(a_i8, b_pack4, total_elems) };
        }
    }
    
    let mut total_sum = 0i32;
    for i in 0..total_elems {
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        let val = (b_pack4[byte_idx] >> bit_shift) & 0b11;
        let w = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
        total_sum += a_i8[i] as i32 * w as i32;
    }
    
    total_sum
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
            let w = if val == 0 { -1 } else if val == 2 { 1 } else { 0 };
            total_sum += a_i8[i * 5 + j] as i32 * w as i32;
        }
    }
    
    let remainder_start = chunks * 5;
    if remainder_start < total_elems {
        let mut b = b_pack5[chunks];
        for j in 0..(total_elems - remainder_start) {
            let val = b % 3;
            b /= 3;
            let w = if val == 0 { -1 } else if val == 2 { 1 } else { 0 };
            total_sum += a_i8[remainder_start + j] as i32 * w as i32;
        }
    }
    
    total_sum
}

// ===========================================================================
// XorSparse lossless sparse ternary dot product
// ===========================================================================

/// Dot product of one block of 64 activations against the ternary weight signs
/// described by `mask` (non-zero positions) and `sb` (sign bits: 1 => -1).
/// `a` holds the 64 activations (i8); only the low `a.len()` lanes are valid
/// Scalar XorSparse row dot product. Consumes the decoded per-block lane-order
/// `masks`/`signs` (no parsing, no set-order expansion).
pub fn ternary_dot_product_sparse_scalar(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    let mut total = 0i32;
    for b in 0..num_blocks {
        let mask = masks[b];
        let k2 = signs[b];
        let start = b * 64;
        let end = (start + 64).min(in_dim);
        total += dot_block_sparse_scalar(&a_i8[start..end], mask, k2);
    }
    total
}

/// One 64-lane block, scalar fallback. `k2` is the lane-order sign mask (bit *k*
/// set ⇒ weight at lane *k* is `-1`).
#[inline]
fn dot_block_sparse_scalar(a: &[i8], mask: u64, k2: u64) -> i32 {
    let mut acc = 0i32;
    let mut bit = 0;
    let mut m = mask;
    while m != 0 {
        if m & 1 != 0 {
            let sign = if (k2 >> bit) & 1 != 0 { -1 } else { 1 };
            acc += a[bit] as i32 * sign;
        }
        m >>= 1;
        bit += 1;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn ternary_dot_block_avx512(a: __m512i, mask: u64, k2: u64) -> __m512i {
    use core::arch::x86_64::*;
    let ones_u8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let zero = _mm512_setzero_si512();

    // `k2` is the lane-order sign mask (bit *k* set ⇒ weight at lane *k* is `-1`).
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn ternary_dot_product_sparse_avx512(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    use core::arch::x86_64::*;
    let a_ptr = a_i8.as_ptr();
    let zero = _mm512_setzero_si512();
    let mut acc = [zero; 16];
    let mut block_idx = 0usize;

    for b in 0..num_blocks {
        let mask = masks[b];
        let k2 = signs[b];
        let start = b * 64;
        let n = (start + 64).min(in_dim) - start;
        // The final block may be short; fall back to scalar for it.
        if b + 1 == num_blocks && n < 64 {
            let mut tmp = [0i8; 64];
            tmp[..n].copy_from_slice(&a_i8[start..start + n]);
            let d = dot_block_sparse_scalar(&tmp[..n], mask, k2);
            let mut merged = acc[block_idx % 16];
            merged = _mm512_add_epi32(merged, _mm512_set1_epi32(d));
            acc[block_idx % 16] = merged;
            break;
        }

        let a = _mm512_loadu_si512(a_ptr.add(start) as *const __m512i);
        let partial = ternary_dot_block_avx512(a, mask, k2);
        acc[block_idx % 16] = _mm512_add_epi32(acc[block_idx % 16], partial);
        block_idx += 1;
    }

    let mut merged = acc[0];
    for k in 1..16 {
        merged = _mm512_add_epi32(merged, acc[k]);
    }
    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, merged);
    tmp.iter().sum::<i32>()
}

/// XorSparse row dot product. Consumes the decoded per-block lane-order
/// `masks`/`signs` (built once at load). Dispatches to the AVX-512 kernel when
/// available, otherwise scalar.
pub fn ternary_dot_product_sparse(
    a_i8: &[i8],
    masks: &[u64],
    signs: &[u64],
    num_blocks: usize,
    in_dim: usize,
) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bw") {
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
    use crate::bit1_58::sparse::{encode_sparse_tensor, SparseTernary};

    // Simple deterministic LCG so the test needs no external deps.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

            let bpr = (in_dim + 3) / 4;
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

    #[test]
    fn avx_block_matches_scalar() {
        use core::arch::x86_64::*;
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
        assert_eq!(avx_sum, scalar, "avx block {} != scalar {}", avx_sum, scalar);
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
                assert_eq!(decoded[i], w_i8[r * in_dim + i], "decode mismatch r={} i={}", r, i);
            }
        }
    }
}
