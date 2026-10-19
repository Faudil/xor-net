use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bench_old(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    let a_ptr = a_i8.as_ptr() as *const __m256i;
    let b_ptr = b_pack4.as_ptr();

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    let mask3 = _mm256_set1_epi32(0x03);
    let lut = _mm256_setr_epi8(
        -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    );

    let chunks32 = total_elems / 32;
    let step = chunks32 / 2 * 2;
    let mut i = 0;
    while i < step {
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

    let mut sums = [0i32; 8];
    _mm256_storeu_si256(
        sums.as_mut_ptr() as *mut __m256i,
        _mm256_add_epi32(acc0, acc1),
    );
    sums.iter().sum::<i32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bench_unrolled4(a_i8: &[i8], b_pack4: &[u8], total_elems: usize) -> i32 {
    let a_ptr = a_i8.as_ptr() as *const __m256i;
    let b_ptr = b_pack4.as_ptr();

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    let mask3 = _mm256_set1_epi32(0x03);
    let lut = _mm256_setr_epi8(
        -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    );

    let chunks32 = total_elems / 32;
    let step = chunks32 / 4 * 4;
    let mut i = 0;
    while i < step {
        let acts0 = _mm256_loadu_si256(a_ptr.add(i));
        let acts1 = _mm256_loadu_si256(a_ptr.add(i + 1));
        let acts2 = _mm256_loadu_si256(a_ptr.add(i + 2));
        let acts3 = _mm256_loadu_si256(a_ptr.add(i + 3));

        let w0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_ptr.add(i * 8) as *const __m128i));
        let w1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_ptr.add((i + 1) * 8) as *const __m128i));
        let w2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_ptr.add((i + 2) * 8) as *const __m128i));
        let w3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_ptr.add((i + 3) * 8) as *const __m128i));

        let p00 = _mm256_and_si256(w0, mask3);
        let p01 = _mm256_and_si256(_mm256_srli_epi32(w0, 2), mask3);
        let p02 = _mm256_and_si256(_mm256_srli_epi32(w0, 4), mask3);
        let p03 = _mm256_and_si256(_mm256_srli_epi32(w0, 6), mask3);
        let w_i8_0 = _mm256_shuffle_epi8(
            lut,
            _mm256_or_si256(
                _mm256_or_si256(p00, _mm256_slli_epi32(p01, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p02, 16), _mm256_slli_epi32(p03, 24)),
            ),
        );

        let p10 = _mm256_and_si256(w1, mask3);
        let p11 = _mm256_and_si256(_mm256_srli_epi32(w1, 2), mask3);
        let p12 = _mm256_and_si256(_mm256_srli_epi32(w1, 4), mask3);
        let p13 = _mm256_and_si256(_mm256_srli_epi32(w1, 6), mask3);
        let w_i8_1 = _mm256_shuffle_epi8(
            lut,
            _mm256_or_si256(
                _mm256_or_si256(p10, _mm256_slli_epi32(p11, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p12, 16), _mm256_slli_epi32(p13, 24)),
            ),
        );

        let p20 = _mm256_and_si256(w2, mask3);
        let p21 = _mm256_and_si256(_mm256_srli_epi32(w2, 2), mask3);
        let p22 = _mm256_and_si256(_mm256_srli_epi32(w2, 4), mask3);
        let p23 = _mm256_and_si256(_mm256_srli_epi32(w2, 6), mask3);
        let w_i8_2 = _mm256_shuffle_epi8(
            lut,
            _mm256_or_si256(
                _mm256_or_si256(p20, _mm256_slli_epi32(p21, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p22, 16), _mm256_slli_epi32(p23, 24)),
            ),
        );

        let p30 = _mm256_and_si256(w3, mask3);
        let p31 = _mm256_and_si256(_mm256_srli_epi32(w3, 2), mask3);
        let p32 = _mm256_and_si256(_mm256_srli_epi32(w3, 4), mask3);
        let p33 = _mm256_and_si256(_mm256_srli_epi32(w3, 6), mask3);
        let w_i8_3 = _mm256_shuffle_epi8(
            lut,
            _mm256_or_si256(
                _mm256_or_si256(p30, _mm256_slli_epi32(p31, 8)),
                _mm256_or_si256(_mm256_slli_epi32(p32, 16), _mm256_slli_epi32(p33, 24)),
            ),
        );

        let sums0 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts0, w_i8_0)),
            ones_i16,
        );
        let sums1 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts1, w_i8_1)),
            ones_i16,
        );
        let sums2 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts2, w_i8_2)),
            ones_i16,
        );
        let sums3 = _mm256_madd_epi16(
            _mm256_maddubs_epi16(ones_u8, _mm256_sign_epi8(acts3, w_i8_3)),
            ones_i16,
        );

        acc0 = _mm256_add_epi32(acc0, sums0);
        acc1 = _mm256_add_epi32(acc1, sums1);
        acc2 = _mm256_add_epi32(acc2, sums2);
        acc3 = _mm256_add_epi32(acc3, sums3);

        i += 4;
    }

    let mut sums = [0i32; 8];
    let acc_total = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc_total);
    sums.iter().sum::<i32>()
}

fn main() {
    let size = 14336;
    let a = vec![1i8; size];
    let mut b = vec![0u8; size / 4];
    for i in 0..b.len() {
        b[i] = 0b10010010; // 1, 0, -1, 1
    }

    unsafe {
        let mut t1 = 0;
        let mut t2 = 0;

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            t1 += bench_old(&a, &b, size);
        }
        let elapsed_old = start.elapsed();

        let start = Instant::now();
        for _ in 0..iters {
            t2 += bench_unrolled4(&a, &b, size);
        }
        let elapsed_new = start.elapsed();

        println!("Old: {:.2?}, Output: {}", elapsed_old, t1);
        println!("Unrolled 4: {:.2?}, Output: {}", elapsed_new, t2);
    }
}
