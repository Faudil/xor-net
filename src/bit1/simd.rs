#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "popcnt")]
pub unsafe fn xnor_dot_product_avx2(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    let mut matches = 0;
    let chunks = a.len() / 32;
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
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

    let padding_bits = (a.len() * 8) - total_bits;
    let valid_matches = matches as usize - padding_bits;
    (2.0 * valid_matches as f32) - total_bits as f32
}

pub fn xnor_dot_product(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { xnor_dot_product_avx2(a, b, total_bits) };
        }
    }
    
    let mut matches = 0;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let chunks = a.len() / 8;
    for i in 0..chunks {
        unsafe {
            let va = std::ptr::read_unaligned(a_ptr.add(i * 8) as *const u64);
            let vb = std::ptr::read_unaligned(b_ptr.add(i * 8) as *const u64);
            matches += (!(va ^ vb)).count_ones() as u64;
        }
    }
    
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        matches += (!(a[i] ^ b[i])).count_ones() as u64;
    }
    
    let padding_bits = (a.len() * 8) - total_bits;
    let valid_matches = matches as usize - padding_bits;
    (2.0 * valid_matches as f32) - total_bits as f32
}
