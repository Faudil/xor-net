//! XNOR-popcount dot product for 1-bit (`{-1,+1}`) vectors packed 8 per byte.
//!
//! Bit `i` of byte `j` holds element `8*j + i` (`1` => `+1`, `0` => `-1`).
//! `Σ a_i·b_i = matches - mismatches = 2·matches - total_bits`, where
//! `matches = popcount(!(a ^ b))` over the `total_bits` valid bits.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Number of XNOR matches in the padding bits of the final byte (bits at
/// positions `>= total_bits`), which the full-byte popcount over-counts.
#[inline(always)]
fn padding_matches(a: &[u8], b: &[u8], total_bits: usize) -> u64 {
    let rem = total_bits % 8;
    if rem == 0 {
        return 0;
    }
    let last = total_bits / 8;
    let pad_mask: u8 = !((1u8 << rem) - 1);
    ((!(a[last] ^ b[last])) & pad_mask).count_ones() as u64
}

#[inline(always)]
fn check_xnor_args(a: &[u8], b: &[u8], total_bits: usize) {
    assert_eq!(a.len(), b.len(), "xnor_dot_product: a/b length mismatch");
    assert_eq!(
        a.len(),
        total_bits.div_ceil(8),
        "xnor_dot_product: slices must hold exactly ceil(total_bits/8) bytes"
    );
}

/// AVX2 XNOR-popcount kernel (nibble-LUT popcount + `vpsadbw`).
///
/// # Safety
/// The CPU must support `avx2`, and `a.len() == b.len()`. Passing
/// `a.len() == total_bits.div_ceil(8)` gives the exact result; prefer the safe
/// [`xnor_dot_product`], which checks both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn xnor_dot_product_avx2(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut matches = 0;
    let chunks = a.len() / 32;
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let va = _mm256_loadu_si256(a_ptr.add(i));
        let vb = _mm256_loadu_si256(b_ptr.add(i));

        let xored = _mm256_xor_si256(va, vb);
        let ones = _mm256_set1_epi32(-1);
        let xnored = _mm256_xor_si256(xored, ones);

        let lo = _mm256_and_si256(xnored, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xnored, 4), low_mask);
        let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);

        let popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
        let zero = _mm256_setzero_si256();
        let sum64 = _mm256_sad_epu8(popcnt, zero);
        acc = _mm256_add_epi64(acc, sum64);
    }

    let mut sums = [0u64; 4];
    _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc);
    matches += sums[0] + sums[1] + sums[2] + sums[3];

    let remainder_start = chunks * 32;
    for i in remainder_start..a.len() {
        matches += (!(a[i] ^ b[i])).count_ones() as u64;
    }

    finish(matches, a, b, total_bits)
}

/// Turn the full-byte match count into the signed dot product.
#[inline(always)]
fn finish(matches: u64, a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    // Exact when the slices hold ceil(total_bits/8) bytes (the safe API
    // enforces it); otherwise the result is meaningless but never panics or
    // wraps around.
    let pad = if a.len() == b.len() && a.len() == total_bits.div_ceil(8) {
        padding_matches(a, b, total_bits)
    } else {
        0
    };
    let valid = matches.saturating_sub(pad);
    (2.0 * valid as f32) - total_bits as f32
}

/// Portable scalar XNOR-popcount (8 bytes at a time).
pub fn xnor_dot_product_scalar(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    check_xnor_args(a, b, total_bits);
    let mut matches = 0u64;
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        let va = u64::from_le_bytes(*x);
        let vb = u64::from_le_bytes(*y);
        matches += (!(va ^ vb)).count_ones() as u64;
    }
    for (&x, &y) in ra.iter().zip(rb) {
        matches += (!(x ^ y)).count_ones() as u64;
    }
    finish(matches, a, b, total_bits)
}

/// Signed dot product of two packed 1-bit vectors of `total_bits` elements.
///
/// Panics unless `a.len() == b.len() == total_bits.div_ceil(8)`. Padding bits
/// in the final byte are ignored whatever their value.
pub fn xnor_dot_product(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    check_xnor_args(a, b, total_bits);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::util::cpu_features().avx2 {
            // SAFETY: avx2 detected; lengths checked above.
            return unsafe { xnor_dot_product_avx2(a, b, total_bits) };
        }
    }
    xnor_dot_product_scalar(a, b, total_bits)
}
