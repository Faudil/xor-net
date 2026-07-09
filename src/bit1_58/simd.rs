#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_product_pack4_avx2(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    let chunks = a_i8.len() / 32;
    let a_ptr = a_i8.as_ptr() as *const __m256i;
    let b_ptr = b_pack4.as_ptr();

    let mut acc32 = _mm256_setzero_si256();
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);

    for i in 0..chunks {
        let acts = _mm256_loadu_si256(a_ptr.add(i));
        
        let packed_w = std::ptr::read_unaligned(b_ptr.add(i * 8) as *const u64);
        
        let mut unpacked = [0i8; 32];
        let mut current_packed = packed_w;
        for j in 0..32 {
            let val = (current_packed & 0b11) as u8;
            unpacked[j] = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
            current_packed >>= 2;
        }
        
        let w_i8 = _mm256_loadu_si256(unpacked.as_ptr() as *const __m256i);
        
        let prod = _mm256_sign_epi8(acts, w_i8);
        
        let sums_i16 = _mm256_maddubs_epi16(ones_u8, prod);
        
        let sums_i32 = _mm256_madd_epi16(sums_i16, ones_i16);
        
        acc32 = _mm256_add_epi32(acc32, sums_i32);
    }
    
    let mut sums = [0i32; 8];
    _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc32);
    let mut total_sum = sums.iter().sum::<i32>();
    
    let remainder_start = chunks * 32;
    for i in remainder_start..total_elems {
        let byte_idx = i / 4;
        let bit_shift = (i % 4) * 2;
        let val = (b_pack4[byte_idx] >> bit_shift) & 0b11;
        let w = if val == 0b00 { -1 } else if val == 0b10 { 1 } else { 0 };
        total_sum += a_i8[i] as i32 * w as i32;
    }
    
    total_sum
}

pub fn ternary_dot_product_pack4(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { ternary_dot_product_pack4_avx2(a_i8, b_pack4, total_elems) };
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
