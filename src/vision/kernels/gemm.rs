//! Small single-threaded SGEMM used by the real-valued convolutions.
//!
//! `C[m x n] = A[m x k] * B[k x n]`, all row-major and densely packed.
//! Callers parallelise over independent row blocks (output image rows).
//!
//! * AVX-512F + FMA: register-blocked micro-kernel, 6 rows x 32 columns
//!   (12 zmm accumulators), masked loads/stores for the column tail.
//! * Fallback: `axpy`-style loops that LLVM vectorises for the build target.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Which SGEMM implementation [`sgemm`] uses on this CPU.
pub fn has_avx512_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let f = crate::util::cpu_features();
        f.avx512f && f.fma
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `c = a * b` (overwrites `c`). Panics on inconsistent slice lengths.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "sgemm: a.len() != m*k");
    assert_eq!(b.len(), k * n, "sgemm: b.len() != k*n");
    assert_eq!(c.len(), m * n, "sgemm: c.len() != m*n");
    #[cfg(target_arch = "x86_64")]
    if has_avx512_fma() {
        // SAFETY: features checked; lengths asserted above.
        unsafe { sgemm_avx512(a, b, c, m, k, n) };
        return;
    }
    sgemm_portable(a, b, c, m, k, n);
}

/// Portable reference/fallback. Same contract as [`sgemm`].
pub fn sgemm_portable(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert!(
        a.len() == m * k && b.len() == k * n && c.len() == m * n,
        "sgemm: bad lengths"
    );
    for i in 0..m {
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(0.0);
        for (p, &av) in a[i * k..(i + 1) * k].iter().enumerate() {
            let brow = &b[p * n..(p + 1) * n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    }
}

/// # Safety
/// CPU must support `avx512f` and `fma`; `a.len() >= m*k`, `b.len() >= k*n`,
/// `c.len() >= m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sgemm_avx512(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    const MR: usize = 6;
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    let mut i = 0;
    while i < m {
        let mr = (m - i).min(MR);
        let mut j = 0;
        while j < n {
            let nr = (n - j).min(32);
            let m0: __mmask16 = if nr >= 16 {
                0xffff
            } else {
                (1u32 << nr) as u16 - 1
            };
            let m1: __mmask16 = if nr >= 32 {
                0xffff
            } else if nr > 16 {
                ((1u32 << (nr - 16)) - 1) as u16
            } else {
                0
            };
            let (a_blk, b_blk, c_blk) = (ap.add(i * k), bp.add(j), cp.add(i * n + j));
            if nr <= 16 {
                // Narrow tail (e.g. a 16-channel stem): one zmm per row.
                match mr {
                    6 => tile16::<6>(a_blk, b_blk, c_blk, k, n, m0),
                    5 => tile16::<5>(a_blk, b_blk, c_blk, k, n, m0),
                    4 => tile16::<4>(a_blk, b_blk, c_blk, k, n, m0),
                    3 => tile16::<3>(a_blk, b_blk, c_blk, k, n, m0),
                    2 => tile16::<2>(a_blk, b_blk, c_blk, k, n, m0),
                    _ => tile16::<1>(a_blk, b_blk, c_blk, k, n, m0),
                }
                j += 32;
                continue;
            }
            match mr {
                6 => tile::<6>(a_blk, b_blk, c_blk, k, n, m0, m1),
                5 => tile::<5>(a_blk, b_blk, c_blk, k, n, m0, m1),
                4 => tile::<4>(a_blk, b_blk, c_blk, k, n, m0, m1),
                3 => tile::<3>(a_blk, b_blk, c_blk, k, n, m0, m1),
                2 => tile::<2>(a_blk, b_blk, c_blk, k, n, m0, m1),
                _ => tile::<1>(a_blk, b_blk, c_blk, k, n, m0, m1),
            }
            j += 32;
        }
        i += MR;
    }
}

/// `R x 32` output tile.
///
/// # Safety
/// avx512f + fma; `a` addresses `R` rows of stride `k`; `b` addresses `k`
/// rows of stride `n` with the lanes enabled in `m0`/`m1` in bounds; `c`
/// addresses `R` rows of stride `n` (same lanes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline]
#[allow(clippy::needless_range_loop)]
unsafe fn tile<const R: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    m0: __mmask16,
    m1: __mmask16,
) {
    let mut acc0 = [_mm512_setzero_ps(); R];
    let mut acc1 = [_mm512_setzero_ps(); R];
    for p in 0..k {
        let brow = b.add(p * n);
        let b0 = _mm512_maskz_loadu_ps(m0, brow);
        let b1 = _mm512_maskz_loadu_ps(m1, brow.add(16));
        for r in 0..R {
            let av = _mm512_set1_ps(*a.add(r * k + p));
            acc0[r] = _mm512_fmadd_ps(av, b0, acc0[r]);
            acc1[r] = _mm512_fmadd_ps(av, b1, acc1[r]);
        }
    }
    for r in 0..R {
        _mm512_mask_storeu_ps(c.add(r * n), m0, acc0[r]);
        _mm512_mask_storeu_ps(c.add(r * n + 16), m1, acc1[r]);
    }
}

/// `R x 16` output tile; same contract as [`tile`] with only `m0` lanes.
///
/// # Safety
/// See [`tile`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline]
#[allow(clippy::needless_range_loop)]
unsafe fn tile16<const R: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    m0: __mmask16,
) {
    let mut acc = [_mm512_setzero_ps(); R];
    for p in 0..k {
        let b0 = _mm512_maskz_loadu_ps(m0, b.add(p * n));
        for r in 0..R {
            acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(r * k + p)), b0, acc[r]);
        }
    }
    for r in 0..R {
        _mm512_mask_storeu_ps(c.add(r * n), m0, acc[r]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512_matches_portable() {
        let mut s = 1u32;
        let mut r = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        for &(m, k, n) in &[
            (1, 1, 1),
            (7, 27, 16),
            (13, 64, 128),
            (5, 3, 33),
            (6, 512, 1000),
            (2, 9, 47),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
            let mut c1 = vec![9.0; m * n];
            let mut c2 = vec![9.0; m * n];
            sgemm_portable(&a, &b, &mut c1, m, k, n);
            sgemm(&a, &b, &mut c2, m, k, n);
            for (x, y) in c1.iter().zip(&c2) {
                assert!(
                    (x - y).abs() < 1e-4 * (1.0 + x.abs()),
                    "{m}x{k}x{n}: {x} vs {y}"
                );
            }
        }
    }
}
