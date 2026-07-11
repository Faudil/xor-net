#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::OnceLock;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn ternary_dot_product_pack4_avx512(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 { unsafe {
    let a_ptr = a_i8.as_ptr() as *const __m512i;
    let b_ptr = b_pack4.as_ptr();

    let zero = _mm512_setzero_si512();
    let mut acc = zero;
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
    let mut idx = 0usize;
    while idx < chunks64 {
        _mm_prefetch(b_ptr.add(idx * 16 + 256) as *const i8, _MM_HINT_T0);

        let acts = _mm512_loadu_si512(a_ptr.add(idx));
        let packed_w = _mm_loadu_si128(b_ptr.add(idx * 16) as *const __m128i);
        let w32 = _mm512_cvtepu8_epi32(packed_w);

        let p0 = _mm512_and_si512(w32, mask3);
        let p1 = _mm512_and_si512(_mm512_srli_epi32(w32, 2), mask3);
        let p2 = _mm512_and_si512(_mm512_srli_epi32(w32, 4), mask3);
        let p3 = _mm512_and_si512(_mm512_srli_epi32(w32, 6), mask3);

        let packed = _mm512_or_si512(
            _mm512_or_si512(p0, _mm512_slli_epi32(p1, 8)),
            _mm512_or_si512(_mm512_slli_epi32(p2, 16), _mm512_slli_epi32(p3, 24))
        );

        let w = _mm512_shuffle_epi8(lut, packed);

        let pos_mask = _mm512_cmp_epi8_mask(w, zero, 6);
        let zero_mask = _mm512_cmp_epi8_mask(w, zero, 0);
        let neg_a = _mm512_sub_epi8(zero, acts);
        let blend = _mm512_mask_blend_epi8(pos_mask, neg_a, acts);
        let signed_acts = _mm512_mask_mov_epi8(blend, zero_mask, zero);

        let sums_i16 = _mm512_maddubs_epi16(ones_u8, signed_acts);
        let sums_i32 = _mm512_madd_epi16(sums_i16, ones_i16);
        acc = _mm512_add_epi32(acc, sums_i32);

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
}}

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
            _mm_prefetch(b_ptr.add(i * 8 + 16) as *const i8, _MM_HINT_T0);

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

pub fn ternary_dot_product_pack4(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { ternary_dot_product_pack4_avx512(a_i8, b_pack4, total_elems) };
        }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn pack5_gather_data() -> &'static ([[[u8; 64]; 5]; 5], [[u64; 5]; 5]) {
    static DATA: OnceLock<([[[u8; 64]; 5]; 5], [[u64; 5]; 5])> = OnceLock::new();
    DATA.get_or_init(|| {
        let mut idx = [[[0u8; 64]; 5]; 5];
        let mut masks = [[0u64; 5]; 5];
        for k in 0..5 {
            let mut counts = [0u32; 5];
            let mut start_pos = [0u32; 5];
            for j in 0..64 {
                counts[(j * 5 + k) / 64] += 1;
            }
            let mut pos = 0u32;
            for m in 0..5 {
                start_pos[m] = pos;
                pos += counts[m];
            }
            let mut counters = [0u32; 5];
            for j in 0..64 {
                let global = j * 5 + k;
                let win = global / 64;
                let off = (global % 64) as u8;
                let p = start_pos[win] + counters[win];
                idx[k][win][p as usize] = off;
                counters[win] += 1;
            }
            for m in 0..5 {
                if counts[m] > 0 {
                    masks[k][m] = ((1u64 << counts[m]) - 1) << start_pos[m];
                }
            }
        }
        (idx, masks)
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn ternary_dot_product_pack5_avx512(a_i8: &[i8], b_pack5: &[u8], total_elems: usize) -> i32 { unsafe {
    let num_groups = total_elems / 5;
    let a_ptr = a_i8.as_ptr();
    let b_ptr = b_pack5.as_ptr();
    let (gi, gm) = &*pack5_gather_data();

    let zero = _mm512_setzero_si512();
    let mut acc = zero;
    let ones_u8 = _mm512_set1_epi8(1);
    let ones_i16 = _mm512_set1_epi16(1);
    let inv3 = _mm512_set1_epi16(0x5556i16);
    let three = _mm512_set1_epi16(3i16);
    let one_i16 = _mm512_set1_epi16(1i16);

    let mut total = 0i32;
    let num_chunks = num_groups / 64;

    for ch in 0..num_chunks {
        let w_base = b_ptr.add(ch * 64);
        let a_base = a_ptr.add(ch * 320);

        let packed_lo = _mm256_loadu_si256(w_base as *const __m256i);
        let packed_hi = _mm256_loadu_si256(w_base.add(32) as *const __m256i);
        let mut cur_lo = _mm512_cvtepu8_epi16(packed_lo);
        let mut cur_hi = _mm512_cvtepu8_epi16(packed_hi);

        let a0 = _mm512_loadu_si512(a_base as *const __m512i);
        let a1 = _mm512_loadu_si512(a_base.add(64) as *const __m512i);
        let a2 = _mm512_loadu_si512(a_base.add(128) as *const __m512i);
        let a3 = _mm512_loadu_si512(a_base.add(192) as *const __m512i);
        let a4 = _mm512_loadu_si512(a_base.add(256) as *const __m512i);

        for tk in 0..5 {
            let q_lo = _mm512_mulhi_epu16(cur_lo, inv3);
            let q_hi = _mm512_mulhi_epu16(cur_hi, inv3);
            let r3_lo = _mm512_mullo_epi16(q_lo, three);
            let r3_hi = _mm512_mullo_epi16(q_hi, three);
            let rem_lo = _mm512_sub_epi16(cur_lo, r3_lo);
            let rem_hi = _mm512_sub_epi16(cur_hi, r3_hi);
            let w_lo16 = _mm512_sub_epi16(rem_lo, one_i16);
            let w_hi16 = _mm512_sub_epi16(rem_hi, one_i16);

            let w_lo8 = _mm512_cvtepi16_epi8(w_lo16);
            let w_hi8 = _mm512_cvtepi16_epi8(w_hi16);
            let w_all = _mm512_inserti32x8(_mm512_castsi256_si512(w_lo8), w_hi8, 1);

            let i0 = _mm512_loadu_si512(gi[tk][0].as_ptr() as *const __m512i);
            let i1 = _mm512_loadu_si512(gi[tk][1].as_ptr() as *const __m512i);
            let i2 = _mm512_loadu_si512(gi[tk][2].as_ptr() as *const __m512i);
            let i3 = _mm512_loadu_si512(gi[tk][3].as_ptr() as *const __m512i);
            let i4 = _mm512_loadu_si512(gi[tk][4].as_ptr() as *const __m512i);

            let g0 = _mm512_mask_permutexvar_epi8(zero, gm[tk][0], i0, a0);
            let g1 = _mm512_mask_permutexvar_epi8(zero, gm[tk][1], i1, a1);
            let g2 = _mm512_mask_permutexvar_epi8(zero, gm[tk][2], i2, a2);
            let g3 = _mm512_mask_permutexvar_epi8(zero, gm[tk][3], i3, a3);
            let g4 = _mm512_mask_permutexvar_epi8(zero, gm[tk][4], i4, a4);
            let acts = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(_mm512_or_si512(g0, g1), g2), g3), g4);

            let pos_mask = _mm512_cmp_epi8_mask(w_all, zero, 6);
            let zero_mask = _mm512_cmp_epi8_mask(w_all, zero, 0);
            let neg_a = _mm512_sub_epi8(zero, acts);
            let blend = _mm512_mask_blend_epi8(pos_mask, neg_a, acts);
            let signed_acts = _mm512_mask_mov_epi8(blend, zero_mask, zero);

            let s16 = _mm512_maddubs_epi16(ones_u8, signed_acts);
            let s32 = _mm512_madd_epi16(s16, ones_i16);
            acc = _mm512_add_epi32(acc, s32);

            cur_lo = q_lo;
            cur_hi = q_hi;
        }
    }

    let mut tmp = [0i32; 16];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
    total += tmp.iter().sum::<i32>();

    let processed = num_chunks * 320;
    for i in processed..total_elems {
        let byte_idx = i / 5;
        let trit_pos = i % 5;
        let mut b = b_pack5[byte_idx];
        for _ in 0..trit_pos { b /= 3; }
        let val = b % 3;
        let w = if val == 0 { -1 } else if val == 2 { 1 } else { 0 };
        total += a_i8[i] as i32 * w as i32;
    }

    total
}}

pub fn ternary_dot_product_pack5(a_i8: &[i8], b_pack5: &[u8], total_elems: usize) -> i32 {
    // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // {
    //     if is_x86_feature_detected!("avx512vbmi") {
    //         return unsafe { ternary_dot_product_pack5_avx512(a_i8, b_pack5, total_elems) };
    //     }
    // }

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
