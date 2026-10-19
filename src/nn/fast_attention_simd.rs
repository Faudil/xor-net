//! SIMD kernels for attention and normalisation (f32).
//!
//! Every operation has a portable scalar implementation (`*_scalar`), and the
//! safe dispatchers validate slice lengths once per call before entering an
//! `unsafe` x86_64 kernel. The kernels are only compiled on `x86_64`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::util::cpu_features;

// ───────────────────────────── dot product ─────────────────────────────

/// AVX2+FMA f32 dot product (4 accumulators).
///
/// # Safety
/// The CPU must support `avx2` and `fma`, and `b.len() >= a.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(b.len() >= a.len());
    let n = a.len();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);

        i += 32;
    }

    let mut sum8 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum8 = _mm256_fmadd_ps(va, vb, sum8);
        i += 8;
    }

    let mut total = hsum256(sum8);
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// AVX-512F f32 dot product.
///
/// # Safety
/// The CPU must support `avx512f`, and `b.len() >= a.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(b.len() >= a.len());
    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Reference scalar f32 dot product. Panics if `a.len() != b.len()`.
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product_f32: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// f32 dot product. Panics if `a.len() != b.len()`.
#[inline(always)]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product_f32: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx512f {
            return unsafe { dot_product_f32_avx512(a, b) };
        }
        if f.avx2_fma() {
            return unsafe { dot_product_f32_avx2(a, b) };
        }
    }
    dot_product_f32_scalar(a, b)
}

/// Horizontal sum of an `__m256`.
///
/// # Safety
/// The CPU must support `avx`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256(v: __m256) -> f32 {
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), v);
    t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]
}

// ───────────────────────────── weighted sum ─────────────────────────────

/// Validates the ring-buffer V access pattern of [`weighted_sum`]: rows
/// `h_kv * max_seq_len + (t % max_seq_len)` for `t < total_kv_len`, each
/// `head_dim` long.
#[inline(always)]
fn check_weighted_sum_args(
    out: &[f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
) {
    assert!(
        out.len() >= head_dim,
        "weighted_sum: out shorter than head_dim"
    );
    assert!(
        scores.len() >= total_kv_len,
        "weighted_sum: scores shorter than total_kv_len"
    );
    if total_kv_len > 0 && head_dim > 0 {
        assert!(max_seq_len > 0, "weighted_sum: max_seq_len is 0");
        let need = (h_kv + 1)
            .checked_mul(max_seq_len)
            .and_then(|x| x.checked_mul(head_dim))
            .expect("weighted_sum: V buffer extent overflows usize");
        assert!(
            v_buf.len() >= need,
            "weighted_sum: v_buf too short ({} < {})",
            v_buf.len(),
            need
        );
    }
}

/// AVX2+FMA `out[i] = Σ_t scores[t] · V[t][i]`.
///
/// # Safety
/// The CPU must support `avx2` and `fma`, and the arguments must satisfy the
/// length checks done by [`weighted_sum`] (`out.len() >= head_dim`,
/// `scores.len() >= total_kv_len`, `v_buf.len() >= (h_kv+1)·max_seq_len·head_dim`
/// whenever `total_kv_len > 0`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn weighted_sum_avx2(
    out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
) {
    let mut i = 0;
    while i + 8 <= head_dim {
        let mut sum = _mm256_setzero_ps();
        for (t_k, &s) in scores[..total_kv_len].iter().enumerate() {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            let v_vec = _mm256_loadu_ps(v_buf.as_ptr().add(v_offset));
            sum = _mm256_fmadd_ps(_mm256_set1_ps(s), v_vec, sum);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), sum);
        i += 8;
    }
    weighted_sum_cols(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
        i,
    );
}

/// AVX-512F `out[i] = Σ_t scores[t] · V[t][i]`.
///
/// # Safety
/// The CPU must support `avx512f`; same length requirements as
/// [`weighted_sum_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn weighted_sum_avx512(
    out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
) {
    let mut i = 0;
    while i + 16 <= head_dim {
        let mut sum = _mm512_setzero_ps();
        for (t_k, &s) in scores[..total_kv_len].iter().enumerate() {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            let v_vec = _mm512_loadu_ps(v_buf.as_ptr().add(v_offset));
            sum = _mm512_fmadd_ps(_mm512_set1_ps(s), v_vec, sum);
        }
        _mm512_storeu_ps(out.as_mut_ptr().add(i), sum);
        i += 16;
    }
    weighted_sum_cols(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
        i,
    );
}

/// Scalar weighted sum for columns `from..head_dim` (bounds-checked).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn weighted_sum_cols(
    out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
    from: usize,
) {
    for (i, o) in out[..head_dim].iter_mut().enumerate().skip(from) {
        let mut s = 0.0f32;
        for (t_k, &w) in scores[..total_kv_len].iter().enumerate() {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            s += w * v_buf[v_offset];
        }
        *o = s;
    }
}

/// Reference scalar implementation of [`weighted_sum`].
pub fn weighted_sum_scalar(
    out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
) {
    check_weighted_sum_args(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
    );
    weighted_sum_cols(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
        0,
    );
}

/// `out[i] = Σ_{t < total_kv_len} scores[t] · V[h_kv][t % max_seq_len][i]` for
/// `i < head_dim`, where `v_buf` is a `[kv_heads][max_seq_len][head_dim]` ring
/// buffer.
///
/// Panics if `out.len() < head_dim`, `scores.len() < total_kv_len`, or (when
/// `total_kv_len > 0`) `v_buf.len() < (h_kv + 1) · max_seq_len · head_dim`.
#[inline(always)]
pub fn weighted_sum(
    out: &mut [f32],
    scores: &[f32],
    v_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    total_kv_len: usize,
) {
    check_weighted_sum_args(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
    );
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx512f {
            return unsafe {
                weighted_sum_avx512(
                    out,
                    scores,
                    v_buf,
                    h_kv,
                    max_seq_len,
                    head_dim,
                    total_kv_len,
                )
            };
        }
        if f.avx2_fma() {
            return unsafe {
                weighted_sum_avx2(
                    out,
                    scores,
                    v_buf,
                    h_kv,
                    max_seq_len,
                    head_dim,
                    total_kv_len,
                )
            };
        }
    }
    weighted_sum_cols(
        out,
        scores,
        v_buf,
        h_kv,
        max_seq_len,
        head_dim,
        total_kv_len,
        0,
    );
}

// ─────────────────────────────── softmax ───────────────────────────────

/// AVX2 softmax (vector max/sum, libm `exp`).
///
/// # Safety
/// The CPU must support `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_avx2(scores: &mut [f32]) {
    let len = scores.len();
    let mut i = 0;

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    while i + 8 <= len {
        max_vec = _mm256_max_ps(max_vec, _mm256_loadu_ps(scores.as_ptr().add(i)));
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), max_vec);
    let mut max_val = f32::NEG_INFINITY;
    for &t in lanes.iter().chain(&scores[i..]) {
        if t > max_val {
            max_val = t;
        }
    }

    let mut sum_exp = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum_exp += *s;
    }

    let inv = 1.0 / sum_exp;
    let inv_sum = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(scores.as_ptr().add(i));
        _mm256_storeu_ps(scores.as_mut_ptr().add(i), _mm256_mul_ps(v, inv_sum));
        i += 8;
    }
    for s in &mut scores[i..] {
        *s *= inv;
    }
}

/// AVX-512F softmax (vector max/sum, libm `exp`).
///
/// # Safety
/// The CPU must support `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_avx512(scores: &mut [f32]) {
    let len = scores.len();
    let mut i = 0;

    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    while i + 16 <= len {
        max_vec = _mm512_max_ps(max_vec, _mm512_loadu_ps(scores.as_ptr().add(i)));
        i += 16;
    }
    let mut max_val = _mm512_reduce_max_ps(max_vec);
    for &t in &scores[i..] {
        if t > max_val {
            max_val = t;
        }
    }

    let mut sum_exp = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum_exp += *s;
    }

    let inv = 1.0 / sum_exp;
    let inv_sum = _mm512_set1_ps(inv);
    i = 0;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(scores.as_ptr().add(i));
        _mm512_storeu_ps(scores.as_mut_ptr().add(i), _mm512_mul_ps(v, inv_sum));
        i += 16;
    }
    for s in &mut scores[i..] {
        *s *= inv;
    }
}

/// Reference scalar softmax.
pub fn softmax_scalar(scores: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_val {
            max_val = s;
        }
    }

    let mut sum_exp = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum_exp += *s;
    }

    let inv_sum = 1.0 / sum_exp;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

/// In-place numerically stable softmax.
#[inline(always)]
pub fn softmax_inplace(scores: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; the kernels only touch `scores`.
        if f.avx512f {
            return unsafe { softmax_avx512(scores) };
        }
        if f.avx2 {
            return unsafe { softmax_avx2(scores) };
        }
    }
    softmax_scalar(scores)
}

// ──────────────────────────────── RoPE ────────────────────────────────

/// Rotary embedding on one head vector (`x[..2·half_dim]`), "rotate half"
/// layout. Bounds-checked scalar code; the compiler auto-vectorises it.
#[inline(always)]
fn rope_scalar(x: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    let (lo, hi) = x.split_at_mut(half_dim);
    for (((x1, x2), &c), &s) in lo
        .iter_mut()
        .zip(hi[..half_dim].iter_mut())
        .zip(&cos[..half_dim])
        .zip(&sin[..half_dim])
    {
        let (a, b) = (*x1, *x2);
        *x1 = a * c - b * s;
        *x2 = a * s + b * c;
    }
}

/// RoPE compiled with AVX2 enabled (the body is bounds-checked scalar code
/// that the compiler vectorises).
///
/// # Safety
/// The CPU must support `avx2`. Slice bounds are checked (panics on misuse).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn rope_avx2(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    _index_pos: usize,
    _head_dim: usize,
    half_dim: usize,
) {
    rope_scalar(x, cos, sin, half_dim)
}

/// Apply rotary embedding to one `head_dim` vector.
///
/// Panics if `x.len() < head_dim` or `cos`/`sin` are shorter than `head_dim / 2`.
pub fn rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32], index_pos: usize, head_dim: usize) {
    let half_dim = head_dim / 2;
    #[cfg(target_arch = "x86_64")]
    {
        if cpu_features().avx2 {
            // SAFETY: avx2 detected; the body is bounds-checked.
            return unsafe { rope_avx2(x, cos, sin, index_pos, head_dim, half_dim) };
        }
    }
    let _ = index_pos;
    rope_scalar(x, cos, sin, half_dim)
}

// ────────────────────────────── RMS norm ──────────────────────────────

#[inline(always)]
fn check_rms_args(x: &[f32], weight: &[f32]) {
    assert!(
        weight.len() >= x.len(),
        "rms_norm: weight shorter than x ({} < {})",
        weight.len(),
        x.len()
    );
}

/// AVX2 RMS norm: `x = x / sqrt(mean(x²) + eps) * weight`.
///
/// # Safety
/// The CPU must support `avx2`, and `weight.len() >= x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn rms_norm_avx2(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert!(weight.len() >= x.len());
    let n = x.len();
    let mut i = 0;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(v1, v1));
        i += 16;
    }
    if i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v, v));
        i += 8;
    }
    let mut sum_sq = hsum256(_mm256_add_ps(acc0, acc1));
    for &v in &x[i..] {
        sum_sq += v * v;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let inv_rms_vec = _mm256_set1_ps(inv_rms);
    i = 0;
    while i + 8 <= n {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let w_vec = _mm256_loadu_ps(weight.as_ptr().add(i));
        let out = _mm256_mul_ps(_mm256_mul_ps(x_vec, inv_rms_vec), w_vec);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), out);
        i += 8;
    }
    while i < n {
        x[i] = x[i] * inv_rms * weight[i];
        i += 1;
    }
}

/// AVX-512F RMS norm: `x = x / sqrt(mean(x²) + eps) * weight`.
///
/// # Safety
/// The CPU must support `avx512f`, and `weight.len() >= x.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rms_norm_avx512(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert!(weight.len() >= x.len());
    let n = x.len();
    let mut i = 0;

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    while i + 32 <= n {
        let v0 = _mm512_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm512_loadu_ps(x.as_ptr().add(i + 16));
        acc0 = _mm512_fmadd_ps(v0, v0, acc0);
        acc1 = _mm512_fmadd_ps(v1, v1, acc1);
        i += 32;
    }
    if i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(v, v, acc0);
        i += 16;
    }
    let mut sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for &v in &x[i..] {
        sum_sq += v * v;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let inv_rms_vec = _mm512_set1_ps(inv_rms);
    i = 0;
    while i + 16 <= n {
        let x_vec = _mm512_loadu_ps(x.as_ptr().add(i));
        let w_vec = _mm512_loadu_ps(weight.as_ptr().add(i));
        let out = _mm512_mul_ps(_mm512_mul_ps(x_vec, inv_rms_vec), w_vec);
        _mm512_storeu_ps(x.as_mut_ptr().add(i), out);
        i += 16;
    }
    while i < n {
        x[i] = x[i] * inv_rms * weight[i];
        i += 1;
    }
}

/// Reference scalar RMS norm. Panics if `weight.len() < x.len()`.
pub fn rms_norm_scalar(x: &mut [f32], weight: &[f32], eps: f32) {
    check_rms_args(x, weight);
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = *v * inv_rms * w;
    }
}

/// In-place RMS norm `x = x / sqrt(mean(x²) + eps) * weight`.
///
/// Panics if `weight.len() < x.len()`.
#[inline(always)]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    check_rms_args(x, weight);
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx512f {
            return unsafe { rms_norm_avx512(x, weight, eps) };
        }
        if f.avx2 {
            return unsafe { rms_norm_avx2(x, weight, eps) };
        }
    }
    rms_norm_scalar(x, weight, eps)
}

// ───────────────────────── batched Q·K dot products ─────────────────────────

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn check_batched_args(
    q: &[f32],
    k_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    t_k_start: usize,
    t_k_end: usize,
    out: &[f32],
) {
    assert!(
        t_k_end >= t_k_start,
        "batched_dot_product: t_k_end < t_k_start"
    );
    assert!(
        q.len() >= head_dim,
        "batched_dot_product: q shorter than head_dim"
    );
    assert!(
        out.len() >= t_k_end - t_k_start,
        "batched_dot_product: out shorter than the t_k range"
    );
    if t_k_end > t_k_start && head_dim > 0 {
        assert!(max_seq_len > 0, "batched_dot_product: max_seq_len is 0");
        let need = (h_kv + 1)
            .checked_mul(max_seq_len)
            .and_then(|x| x.checked_mul(head_dim))
            .expect("batched_dot_product: K buffer extent overflows usize");
        assert!(k_buf.len() >= need, "batched_dot_product: k_buf too short");
    }
}

/// AVX2+FMA batched `q · K[t]` for `t in t_k_start..t_k_end`.
///
/// # Safety
/// The CPU must support `avx2` and `fma`; `q.len() >= head_dim` and
/// `t_k_end >= t_k_start` (K rows and `out` are bounds-checked).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn batched_dot_product_avx2(
    q: &[f32],
    k_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    t_k_start: usize,
    t_k_end: usize,
    out: &mut [f32],
) {
    let k_stride = max_seq_len * head_dim;
    let h_offset = h_kv * k_stride;

    for t_k in t_k_start..t_k_end {
        let k_offset = h_offset + (t_k % max_seq_len) * head_dim;
        let k_vec = &k_buf[k_offset..k_offset + head_dim];
        out[t_k - t_k_start] = dot_product_f32_avx2(&q[..head_dim], k_vec);
    }
}

/// AVX-512F batched `q · K[t]` for `t in t_k_start..t_k_end`.
///
/// # Safety
/// The CPU must support `avx512f`; `q.len() >= head_dim` and
/// `t_k_end >= t_k_start` (K rows and `out` are bounds-checked).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
pub unsafe fn batched_dot_product_avx512(
    q: &[f32],
    k_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    t_k_start: usize,
    t_k_end: usize,
    out: &mut [f32],
) {
    let k_stride = max_seq_len * head_dim;
    let h_offset = h_kv * k_stride;

    for t_k in t_k_start..t_k_end {
        let k_offset = h_offset + (t_k % max_seq_len) * head_dim;
        let k_vec = &k_buf[k_offset..k_offset + head_dim];
        out[t_k - t_k_start] = dot_product_f32_avx512(&q[..head_dim], k_vec);
    }
}

/// Reference scalar implementation of [`batched_dot_product`].
#[allow(clippy::too_many_arguments)]
pub fn batched_dot_product_scalar(
    q: &[f32],
    k_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    t_k_start: usize,
    t_k_end: usize,
    out: &mut [f32],
) {
    check_batched_args(
        q,
        k_buf,
        h_kv,
        max_seq_len,
        head_dim,
        t_k_start,
        t_k_end,
        out,
    );
    let k_stride = max_seq_len * head_dim;
    let h_offset = h_kv * k_stride;
    for t_k in t_k_start..t_k_end {
        let k_offset = h_offset + (t_k % max_seq_len) * head_dim;
        let k_vec = &k_buf[k_offset..k_offset + head_dim];
        out[t_k - t_k_start] = q[..head_dim].iter().zip(k_vec).map(|(a, b)| a * b).sum();
    }
}

/// `out[t - t_k_start] = q · K[h_kv][t % max_seq_len]` for
/// `t in t_k_start..t_k_end`, with `k_buf` a `[kv_heads][max_seq_len][head_dim]`
/// ring buffer.
///
/// Panics if `t_k_end < t_k_start`, `q.len() < head_dim`,
/// `out.len() < t_k_end - t_k_start`, or `k_buf` is too short.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn batched_dot_product(
    q: &[f32],
    k_buf: &[f32],
    h_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    t_k_start: usize,
    t_k_end: usize,
    out: &mut [f32],
) {
    check_batched_args(
        q,
        k_buf,
        h_kv,
        max_seq_len,
        head_dim,
        t_k_start,
        t_k_end,
        out,
    );
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx512f {
            return unsafe {
                batched_dot_product_avx512(
                    q,
                    k_buf,
                    h_kv,
                    max_seq_len,
                    head_dim,
                    t_k_start,
                    t_k_end,
                    out,
                )
            };
        }
        if f.avx2_fma() {
            return unsafe {
                batched_dot_product_avx2(
                    q,
                    k_buf,
                    h_kv,
                    max_seq_len,
                    head_dim,
                    t_k_start,
                    t_k_end,
                    out,
                )
            };
        }
    }
    batched_dot_product_scalar(
        q,
        k_buf,
        h_kv,
        max_seq_len,
        head_dim,
        t_k_start,
        t_k_end,
        out,
    )
}

// ──────────────────────────── f16 <-> f32 slices ────────────────────────────
//
// `half`'s slice conversions detect F16C at run time but then call a
// non-inlinable `#[target_feature]` helper per 8 elements, which costs more
// than the conversion itself unless the whole crate is built with
// `target-cpu=native`. These convert a whole slice inside one kernel.

/// AVX-512F `f16 -> f32`.
///
/// # Safety
/// The CPU must support `avx512f`; `dst.len() == src.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn f16_to_f32_slice_avx512(src: &[half::f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let (s, d) = (src.as_ptr() as *const u16, dst.as_mut_ptr());
    let mut i = 0;
    while i + 16 <= n {
        let h = _mm256_loadu_si256(s.add(i) as *const __m256i);
        _mm512_storeu_ps(d.add(i), _mm512_cvtph_ps(h));
        i += 16;
    }
    while i + 8 <= n {
        let h = _mm_loadu_si128(s.add(i) as *const __m128i);
        _mm256_storeu_ps(d.add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..n {
        dst[j] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(src[j].to_bits() as i32)));
    }
}

/// F16C `f16 -> f32`.
///
/// # Safety
/// The CPU must support `avx` and `f16c`; `dst.len() == src.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn f16_to_f32_slice_f16c(src: &[half::f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let (s, d) = (src.as_ptr() as *const u16, dst.as_mut_ptr());
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(s.add(i) as *const __m128i);
        _mm256_storeu_ps(d.add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..n {
        dst[j] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(src[j].to_bits() as i32)));
    }
}

/// `dst[i] = src[i] as f32` (exact). Panics if the lengths differ.
#[inline]
pub fn f16_to_f32_slice(src: &[half::f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "f16_to_f32_slice: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx512f {
            return unsafe { f16_to_f32_slice_avx512(src, dst) };
        }
        if f.avx2 && f.f16c {
            return unsafe { f16_to_f32_slice_f16c(src, dst) };
        }
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

/// F16C `f32 -> f16`, round to nearest even.
///
/// # Safety
/// The CPU must support `avx` and `f16c`; `dst.len() == src.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn f32_to_f16_slice_f16c(src: &[f32], dst: &mut [half::f16]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let (s, d) = (src.as_ptr(), dst.as_mut_ptr() as *mut u16);
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(s.add(i)));
        _mm_storeu_si128(d.add(i) as *mut __m128i, h);
        i += 8;
    }
    for j in i..n {
        let h = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set_ss(src[j]));
        dst[j] = half::f16::from_bits(_mm_extract_epi16::<0>(h) as u16);
    }
}

/// `dst[i] = f16::from_f32(src[i])` (round to nearest even, like `half`).
/// Panics if the lengths differ.
#[inline]
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [half::f16]) {
    assert_eq!(src.len(), dst.len(), "f32_to_f16_slice: length mismatch");
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpu_features();
        // SAFETY: features detected; lengths checked above.
        if f.avx2 && f.f16c {
            return unsafe { f32_to_f16_slice_f16c(src, dst) };
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_f32(s);
    }
}
