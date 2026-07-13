use core::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
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
    
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum8);
    let mut total = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_product_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
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

#[inline(always)]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { dot_product_f32_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_f32_avx2(a, b) };
        }
    }
    
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
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
        for t_k in 0..total_kv_len {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            let v_vec = _mm256_loadu_ps(v_buf.as_ptr().add(v_offset));
            let w = _mm256_set1_ps(scores[t_k]);
            sum = _mm256_fmadd_ps(w, v_vec, sum);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), sum);
        i += 8;
    }
    
    while i < head_dim {
        let mut s = 0.0f32;
        for t_k in 0..total_kv_len {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            s += scores[t_k] * v_buf[v_offset];
        }
        out[i] = s;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
        for t_k in 0..total_kv_len {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            let v_vec = _mm512_loadu_ps(v_buf.as_ptr().add(v_offset));
            let w = _mm512_set1_ps(scores[t_k]);
            sum = _mm512_fmadd_ps(w, v_vec, sum);
        }
        _mm512_storeu_ps(out.as_mut_ptr().add(i), sum);
        i += 16;
    }
    
    while i < head_dim {
        let mut s = 0.0f32;
        for t_k in 0..total_kv_len {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            s += scores[t_k] * v_buf[v_offset];
        }
        out[i] = s;
        i += 1;
    }
}

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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { weighted_sum_avx512(out, scores, v_buf, h_kv, max_seq_len, head_dim, total_kv_len) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { weighted_sum_avx2(out, scores, v_buf, h_kv, max_seq_len, head_dim, total_kv_len) };
        }
    }
    
    for i in 0..head_dim {
        let mut s = 0.0f32;
        for t_k in 0..total_kv_len {
            let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim + i;
            s += scores[t_k] * v_buf[v_offset];
        }
        out[i] = s;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_avx2(scores: &mut [f32]) {
    let len = scores.len();
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;
    
    while i + 8 <= len {
        let v = _mm256_loadu_ps(scores.as_ptr().add(i));
        let mut max_vec = v;
        if i == 0 {
            max_vec = _mm256_set1_ps(max_val);
        }
        max_vec = _mm256_max_ps(max_vec, v);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);
        for &t in &tmp {
            if t > max_val { max_val = t; }
        }
        i += 8;
    }
    
    while i < len {
        if scores[i] > max_val { max_val = scores[i]; }
        i += 1;
    }
    
    let max_vec = _mm256_set1_ps(max_val);
    let mut sum_exp = 0.0f32;
    i = 0;
    
    while i + 8 <= len {
        let v = _mm256_loadu_ps(scores.as_ptr().add(i));
        let exp_v = _mm256_exp_ps_fallback(_mm256_sub_ps(v, max_vec));
        _mm256_storeu_ps(scores.as_mut_ptr().add(i), exp_v);
        
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), exp_v);
        for &t in &tmp {
            sum_exp += t;
        }
        i += 8;
    }
    
    while i < len {
        scores[i] = (scores[i] - max_val).exp();
        sum_exp += scores[i];
        i += 1;
    }
    
    let inv_sum = _mm256_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(scores.as_ptr().add(i));
        let scaled = _mm256_mul_ps(v, inv_sum);
        _mm256_storeu_ps(scores.as_mut_ptr().add(i), scaled);
        i += 8;
    }
    
    while i < len {
        scores[i] *= 1.0 / sum_exp;
        i += 1;
    }
}

#[inline(always)]
unsafe fn _mm256_exp_ps_fallback(v: __m256) -> __m256 {
    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    for i in 0..8 {
        arr[i] = arr[i].exp();
    }
    _mm256_loadu_ps(arr.as_ptr())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_avx512(scores: &mut [f32]) {
    let len = scores.len();
    let mut max_val = f32::NEG_INFINITY;
    let mut i = 0;
    
    while i + 16 <= len {
        let v = _mm512_loadu_ps(scores.as_ptr().add(i));
        let max_vec = _mm512_max_ps(_mm512_set1_ps(max_val), v);
        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), max_vec);
        for &t in &tmp {
            if t > max_val { max_val = t; }
        }
        i += 16;
    }
    
    while i < len {
        if scores[i] > max_val { max_val = scores[i]; }
        i += 1;
    }
    
    let max_vec = _mm512_set1_ps(max_val);
    let mut sum_exp = 0.0f32;
    i = 0;
    
    while i + 16 <= len {
        let v = _mm512_loadu_ps(scores.as_ptr().add(i));
        let exp_v = _mm512_exp_ps_fallback(_mm512_sub_ps(v, max_vec));
        _mm512_storeu_ps(scores.as_mut_ptr().add(i), exp_v);
        
        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), exp_v);
        for &t in &tmp {
            sum_exp += t;
        }
        i += 16;
    }
    
    while i < len {
        scores[i] = (scores[i] - max_val).exp();
        sum_exp += scores[i];
        i += 1;
    }
    
    let inv_sum = _mm512_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(scores.as_ptr().add(i));
        let scaled = _mm512_mul_ps(v, inv_sum);
        _mm512_storeu_ps(scores.as_mut_ptr().add(i), scaled);
        i += 16;
    }
    
    while i < len {
        scores[i] *= 1.0 / sum_exp;
        i += 1;
    }
}

#[inline(always)]
unsafe fn _mm512_exp_ps_fallback(v: __m512) -> __m512 {
    let mut arr = [0.0f32; 16];
    _mm512_storeu_ps(arr.as_mut_ptr(), v);
    for i in 0..16 {
        arr[i] = arr[i].exp();
    }
    _mm512_loadu_ps(arr.as_ptr())
}

#[inline(always)]
pub fn softmax_inplace(scores: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { softmax_avx512(scores) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { softmax_avx2(scores) };
        }
    }
    
    let mut max_val = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_val { max_val = s; }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn rope_avx2(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    _index_pos: usize,
    _head_dim: usize,
    half_dim: usize,
) {
    for i in 0..half_dim {
        let cos_val = cos[i];
        let sin_val = sin[i];
        let x1 = x[i];
        let x2 = x[i + half_dim];
        x[i] = x1 * cos_val - x2 * sin_val;
        x[i + half_dim] = x1 * sin_val + x2 * cos_val;
    }
}

pub fn rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32], index_pos: usize, head_dim: usize) {
    let half_dim = head_dim / 2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { rope_avx2(x, cos, sin, index_pos, head_dim, half_dim) };
        }
    }
    
    for i in 0..half_dim {
        let cos_val = cos[i];
        let sin_val = sin[i];
        let x1 = x[i];
        let x2 = x[i + half_dim];
        x[i] = x1 * cos_val - x2 * sin_val;
        x[i + half_dim] = x1 * sin_val + x2 * cos_val;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn rms_norm_avx2(
    x: &mut [f32],
    weight: &[f32],
    eps: f32,
) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let sq = _mm256_mul_ps(v, v);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), sq);
        for &t in &tmp {
            sum_sq += t;
        }
        i += 8;
    }
    
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }
    
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_vec = _mm256_set1_ps(inv_rms);
    i = 0;
    
    while i + 8 <= n {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let w_vec = _mm256_loadu_ps(weight.as_ptr().add(i));
        let normalized = _mm256_mul_ps(x_vec, inv_rms_vec);
        let out = _mm256_mul_ps(normalized, w_vec);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), out);
        i += 8;
    }
    
    while i < n {
        x[i] = x[i] * inv_rms * weight[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn rms_norm_avx512(
    x: &mut [f32],
    weight: &[f32],
    eps: f32,
) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0;
    
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        let sq = _mm512_mul_ps(v, v);
        let mut tmp = [0.0f32; 16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), sq);
        for &t in &tmp {
            sum_sq += t;
        }
        i += 16;
    }
    
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }
    
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_vec = _mm512_set1_ps(inv_rms);
    i = 0;
    
    while i + 16 <= n {
        let x_vec = _mm512_loadu_ps(x.as_ptr().add(i));
        let w_vec = _mm512_loadu_ps(weight.as_ptr().add(i));
        let normalized = _mm512_mul_ps(x_vec, inv_rms_vec);
        let out = _mm512_mul_ps(normalized, w_vec);
        _mm512_storeu_ps(x.as_mut_ptr().add(i), out);
        i += 16;
    }
    
    while i < n {
        x[i] = x[i] * inv_rms * weight[i];
        i += 1;
    }
}

#[inline(always)]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { rms_norm_avx512(x, weight, eps) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { rms_norm_avx2(x, weight, eps) };
        }
    }
    
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        sum_sq += x[i] * x[i];
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
        
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut i = 0;
        
        while i + 32 <= head_dim {
            let vq0 = _mm256_loadu_ps(q.as_ptr().add(i));
            let vk0 = _mm256_loadu_ps(k_vec.as_ptr().add(i));
            let vq1 = _mm256_loadu_ps(q.as_ptr().add(i + 8));
            let vk1 = _mm256_loadu_ps(k_vec.as_ptr().add(i + 8));
            let vq2 = _mm256_loadu_ps(q.as_ptr().add(i + 16));
            let vk2 = _mm256_loadu_ps(k_vec.as_ptr().add(i + 16));
            let vq3 = _mm256_loadu_ps(q.as_ptr().add(i + 24));
            let vk3 = _mm256_loadu_ps(k_vec.as_ptr().add(i + 24));
            
            sum0 = _mm256_fmadd_ps(vq0, vk0, sum0);
            sum1 = _mm256_fmadd_ps(vq1, vk1, sum1);
            sum2 = _mm256_fmadd_ps(vq2, vk2, sum2);
            sum3 = _mm256_fmadd_ps(vq3, vk3, sum3);
            
            i += 32;
        }
        
        let mut sum8 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        
        while i + 8 <= head_dim {
            let vq = _mm256_loadu_ps(q.as_ptr().add(i));
            let vk = _mm256_loadu_ps(k_vec.as_ptr().add(i));
            sum8 = _mm256_fmadd_ps(vq, vk, sum8);
            i += 8;
        }
        
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum8);
        let mut total = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        
        while i < head_dim {
            total += q[i] * k_vec[i];
            i += 1;
        }
        
        out[t_k - t_k_start] = total;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
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
        
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        
        while i + 16 <= head_dim {
            let vq = _mm512_loadu_ps(q.as_ptr().add(i));
            let vk = _mm512_loadu_ps(k_vec.as_ptr().add(i));
            acc = _mm512_fmadd_ps(vq, vk, acc);
            i += 16;
        }
        
        let mut sum = _mm512_reduce_add_ps(acc);
        
        while i < head_dim {
            sum += q[i] * k_vec[i];
            i += 1;
        }
        
        out[t_k - t_k_start] = sum;
    }
}

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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { batched_dot_product_avx512(q, k_buf, h_kv, max_seq_len, head_dim, t_k_start, t_k_end, out) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { batched_dot_product_avx2(q, k_buf, h_kv, max_seq_len, head_dim, t_k_start, t_k_end, out) };
        }
    }
    
    let k_stride = max_seq_len * head_dim;
    let h_offset = h_kv * k_stride;
    for t_k in t_k_start..t_k_end {
        let k_offset = h_offset + (t_k % max_seq_len) * head_dim;
        let k_vec = &k_buf[k_offset..k_offset + head_dim];
        let mut sum = 0.0f32;
        for i in 0..head_dim {
            sum += q[i] * k_vec[i];
        }
        out[t_k - t_k_start] = sum;
    }
}