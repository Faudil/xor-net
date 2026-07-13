#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Pure LUT-based AVX2 decode: broadcastq → shuffle → blend → mask → shuffle.
/// Fewer instructions overall but shuffle-port heavy.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_product_pack4_avx2_lut(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    unsafe {
        let a_ptr = a_i8.as_ptr() as *const __m256i;
        let b_ptr = b_pack4.as_ptr();

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);

        let dup_mask = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
        );
        let mask_0c03 = _mm256_set1_epi32(0x0C030C03u32 as i32);
        let decode_lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        );

        let chunks32 = total_elems / 32;
        let step = chunks32 / 2 * 2;
        let mut i = 0;
        while i < step {
            _mm_prefetch(b_ptr.add(i * 8 + 16) as *const i8, _MM_HINT_T0);

            let acts0 = _mm256_loadu_si256(a_ptr.add(i));
            let pw0 = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let bcast0 = _mm256_broadcastq_epi64(pw0);
            let v0 = _mm256_shuffle_epi8(bcast0, dup_mask);
            let vs0 = _mm256_srli_epi32(v0, 4);
            let vb0 = _mm256_blend_epi16(v0, vs0, 0xAA);
            let idx0 = _mm256_and_si256(vb0, mask_0c03);
            let w0 = _mm256_shuffle_epi8(decode_lut, idx0);
            let s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts0, w0)), ones_i16);
            acc0 = _mm256_add_epi32(acc0, s0);

            let acts1 = _mm256_loadu_si256(a_ptr.add(i + 1));
            let pw1 = _mm_loadl_epi64(b_ptr.add((i + 1) * 8) as *const __m128i);
            let bcast1 = _mm256_broadcastq_epi64(pw1);
            let v1 = _mm256_shuffle_epi8(bcast1, dup_mask);
            let vs1 = _mm256_srli_epi32(v1, 4);
            let vb1 = _mm256_blend_epi16(v1, vs1, 0xAA);
            let idx1 = _mm256_and_si256(vb1, mask_0c03);
            let w1 = _mm256_shuffle_epi8(decode_lut, idx1);
            let s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts1, w1)), ones_i16);
            acc1 = _mm256_add_epi32(acc1, s1);

            i += 2;
        }

        let mut acc32 = _mm256_add_epi32(acc0, acc1);
        for i in i..chunks32 {
            let acts = _mm256_loadu_si256(a_ptr.add(i));
            let pw = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let bcast = _mm256_broadcastq_epi64(pw);
            let v = _mm256_shuffle_epi8(bcast, dup_mask);
            let vs = _mm256_srli_epi32(v, 4);
            let vb = _mm256_blend_epi16(v, vs, 0xAA);
            let idx = _mm256_and_si256(vb, mask_0c03);
            let w = _mm256_shuffle_epi8(decode_lut, idx);
            let s = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts, w)), ones_i16);
            acc32 = _mm256_add_epi32(acc32, s);
        }

        let mut acc_arr = [0i32; 8];
        _mm256_storeu_si256(acc_arr.as_mut_ptr() as *mut __m256i, acc32);
        let mut sum: i32 = acc_arr.iter().sum();

        let mut rem_idx = chunks32 * 32;
        let mut w_idx = chunks32 * 8;
        while rem_idx < total_elems {
            let w_byte = b_pack4[w_idx];
            let w0 = (w_byte & 0x03) as i8 - 1;
            let w1 = ((w_byte >> 2) & 0x03) as i8 - 1;
            let w2 = ((w_byte >> 4) & 0x03) as i8 - 1;
            let w3 = ((w_byte >> 6) & 0x03) as i8 - 1;
            if rem_idx < total_elems { sum += a_i8[rem_idx] as i32 * w0 as i32; }
            if rem_idx + 1 < total_elems { sum += a_i8[rem_idx + 1] as i32 * w1 as i32; }
            if rem_idx + 2 < total_elems { sum += a_i8[rem_idx + 2] as i32 * w2 as i32; }
            if rem_idx + 3 < total_elems { sum += a_i8[rem_idx + 3] as i32 * w3 as i32; }
            rem_idx += 4;
            w_idx += 1;
        }

        sum
    }
}

/// Hybrid AVX2 decode: 4-way unroll alternating LUT decode (blocks 0,2)
/// and original cvtepu8 decode (blocks 1,3) to maximise port diversity.
///
/// LUT path is shuffle-port heavy (port 5 on Intel / port 1 on AMD).
/// Original path is shift/ALU heavy (ports 0,1 on Intel / ports 0,3 on AMD).
/// Interleaving them lets the OoO engine overlap both pipelines.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_product_pack4_avx2_hybrid(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    unsafe {
        let a_ptr = a_i8.as_ptr() as *const __m256i;
        let b_ptr = b_pack4.as_ptr();

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);

        // ---- LUT decode constants ----
        let dup_mask = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
        );
        let mask_0c03 = _mm256_set1_epi32(0x0C030C03u32 as i32);
        let decode_lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        );

        // ---- Original decode constants ----
        let mask3 = _mm256_set1_epi32(0x03);
        let orig_lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );

        let chunks32 = total_elems / 32;
        let step4 = chunks32 / 4 * 4;
        let mut i = 0;

        while i < step4 {
            // Prefetch weight bytes for upcoming iterations
            _mm_prefetch(b_ptr.add(i * 8 + 32) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_i8.as_ptr().add((i + 4) * 32) as *const i8, _MM_HINT_T0);

            // =========== Block 0: LUT decode ===========
            let acts0 = _mm256_loadu_si256(a_ptr.add(i));
            let pw0 = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let bcast0 = _mm256_broadcastq_epi64(pw0);
            let v0 = _mm256_shuffle_epi8(bcast0, dup_mask);
            let vs0 = _mm256_srli_epi32(v0, 4);
            let vb0 = _mm256_blend_epi16(v0, vs0, 0xAA);
            let idx0 = _mm256_and_si256(vb0, mask_0c03);
            let w_i8_0 = _mm256_shuffle_epi8(decode_lut, idx0);
            let s0 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts0, w_i8_0)),
                ones_i16,
            );
            acc0 = _mm256_add_epi32(acc0, s0);

            // =========== Block 1: Original decode ===========
            let acts1 = _mm256_loadu_si256(a_ptr.add(i + 1));
            let pw1 = _mm_loadl_epi64(b_ptr.add((i + 1) * 8) as *const __m128i);
            let x1 = _mm256_cvtepu8_epi32(pw1);
            let p10 = _mm256_and_si256(x1, mask3);
            let p11 = _mm256_and_si256(_mm256_srli_epi32(x1, 2), mask3);
            let p12 = _mm256_and_si256(_mm256_srli_epi32(x1, 4), mask3);
            let p13 = _mm256_and_si256(_mm256_srli_epi32(x1, 6), mask3);
            let w_i8_1 = _mm256_shuffle_epi8(
                orig_lut,
                _mm256_or_si256(
                    _mm256_or_si256(p10, _mm256_slli_epi32(p11, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(p12, 16), _mm256_slli_epi32(p13, 24)),
                ),
            );
            let s1 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts1, w_i8_1)),
                ones_i16,
            );
            acc1 = _mm256_add_epi32(acc1, s1);

            // =========== Block 2: LUT decode ===========
            let acts2 = _mm256_loadu_si256(a_ptr.add(i + 2));
            let pw2 = _mm_loadl_epi64(b_ptr.add((i + 2) * 8) as *const __m128i);
            let bcast2 = _mm256_broadcastq_epi64(pw2);
            let v2 = _mm256_shuffle_epi8(bcast2, dup_mask);
            let vs2 = _mm256_srli_epi32(v2, 4);
            let vb2 = _mm256_blend_epi16(v2, vs2, 0xAA);
            let idx2 = _mm256_and_si256(vb2, mask_0c03);
            let w_i8_2 = _mm256_shuffle_epi8(decode_lut, idx2);
            let s2 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts2, w_i8_2)),
                ones_i16,
            );
            acc2 = _mm256_add_epi32(acc2, s2);

            // =========== Block 3: Original decode ===========
            let acts3 = _mm256_loadu_si256(a_ptr.add(i + 3));
            let pw3 = _mm_loadl_epi64(b_ptr.add((i + 3) * 8) as *const __m128i);
            let x3 = _mm256_cvtepu8_epi32(pw3);
            let p30 = _mm256_and_si256(x3, mask3);
            let p31 = _mm256_and_si256(_mm256_srli_epi32(x3, 2), mask3);
            let p32 = _mm256_and_si256(_mm256_srli_epi32(x3, 4), mask3);
            let p33 = _mm256_and_si256(_mm256_srli_epi32(x3, 6), mask3);
            let w_i8_3 = _mm256_shuffle_epi8(
                orig_lut,
                _mm256_or_si256(
                    _mm256_or_si256(p30, _mm256_slli_epi32(p31, 8)),
                    _mm256_or_si256(_mm256_slli_epi32(p32, 16), _mm256_slli_epi32(p33, 24)),
                ),
            );
            let s3 = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts3, w_i8_3)),
                ones_i16,
            );
            acc3 = _mm256_add_epi32(acc3, s3);

            i += 4;
        }

        // Merge the 4 accumulators
        let mut acc32 = _mm256_add_epi32(
            _mm256_add_epi32(acc0, acc1),
            _mm256_add_epi32(acc2, acc3),
        );

        // Handle remainder chunks (< 4) with LUT decode
        for i in i..chunks32 {
            let acts = _mm256_loadu_si256(a_ptr.add(i));
            let pw = _mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i);
            let bcast = _mm256_broadcastq_epi64(pw);
            let v = _mm256_shuffle_epi8(bcast, dup_mask);
            let vs = _mm256_srli_epi32(v, 4);
            let vb = _mm256_blend_epi16(v, vs, 0xAA);
            let idx = _mm256_and_si256(vb, mask_0c03);
            let w = _mm256_shuffle_epi8(decode_lut, idx);
            let s = _mm256_madd_epi16(
                _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts, w)),
                ones_i16,
            );
            acc32 = _mm256_add_epi32(acc32, s);
        }

        // Horizontal sum
        let mut acc_arr = [0i32; 8];
        _mm256_storeu_si256(acc_arr.as_mut_ptr() as *mut __m256i, acc32);
        let mut sum: i32 = acc_arr.iter().sum();

        // Scalar remainder
        let mut rem_idx = chunks32 * 32;
        let mut w_idx = chunks32 * 8;
        while rem_idx < total_elems {
            let w_byte = b_pack4[w_idx];
            let w0 = (w_byte & 0x03) as i8 - 1;
            let w1 = ((w_byte >> 2) & 0x03) as i8 - 1;
            let w2 = ((w_byte >> 4) & 0x03) as i8 - 1;
            let w3 = ((w_byte >> 6) & 0x03) as i8 - 1;
            if rem_idx < total_elems { sum += a_i8[rem_idx] as i32 * w0 as i32; }
            if rem_idx + 1 < total_elems { sum += a_i8[rem_idx + 1] as i32 * w1 as i32; }
            if rem_idx + 2 < total_elems { sum += a_i8[rem_idx + 2] as i32 * w2 as i32; }
            if rem_idx + 3 < total_elems { sum += a_i8[rem_idx + 3] as i32 * w3 as i32; }
            rem_idx += 4;
            w_idx += 1;
        }

        sum
    }
}
