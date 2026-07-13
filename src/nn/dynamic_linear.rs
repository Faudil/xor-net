use rayon::prelude::*;
use crate::tensor::FastTensor;
use crate::bit1::layers::BitLinear;
use crate::bit1_58::layers::TernaryLinear;
use crate::bit1_58::quantization::TernaryPackType;
use crate::loader::SafeTensorLoader;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
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

#[inline(always)]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
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
pub unsafe fn dot_product_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i));

        let s0 = _mm256_madd_epi16(a0, b0);
        let s1 = _mm256_madd_epi16(a1, b1);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(s0, s1));
        i += 32;
    }

    let mut sums = [0i32; 8];
    _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, acc);
    let mut total = sums.iter().sum::<i32>();

    for j in i..n {
        total += a[j] as i32 * b[j] as i32;
    }
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_product_i8_avx512(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    while i + 64 <= n {
        let a0 = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(a.as_ptr().add(i + 32) as *const __m256i);
        let b1 = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);

        let s0 = _mm512_madd_epi16(
            _mm512_cvtepi8_epi16(a0),
            _mm512_cvtepi8_epi16(b0),
        );
        let s1 = _mm512_madd_epi16(
            _mm512_cvtepi8_epi16(a1),
            _mm512_cvtepi8_epi16(b1),
        );
        acc = _mm512_add_epi32(acc, _mm512_add_epi32(s0, s1));
        i += 64;
    }
    if i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i));
        let s = _mm256_add_epi32(
            _mm256_madd_epi16(a0, b0),
            _mm256_madd_epi16(a1, b1),
        );
        acc = _mm512_add_epi32(
            acc,
            _mm512_inserti32x8(_mm512_setzero_si512(), s, 0)
        );
        i += 32;
    }

    let mut sums = [0i32; 16];
    _mm512_storeu_si512(sums.as_mut_ptr() as *mut __m512i, acc);
    let mut total = sums.iter().sum::<i32>();

    for j in i..n {
        total += a[j] as i32 * b[j] as i32;
    }
    total
}

#[inline(always)]
pub fn dot_product_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { dot_product_i8_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_i8_avx2(a, b) };
        }
    }
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum += a[i] as i32 * b[i] as i32;
    }
    sum
}

/// VNNI (dpbusd) int8 GEMV. `a_u8` must be the activation quantized to i8 then
/// re-interpreted as u8 with a +128 zero-point (`byte ^ 0x80`); `b_i8` is the
/// signed int8 weight row. `w_row_sum` is the precomputed Σ of the signed
/// weight row. Returns the true signed dot product Σ a_i8·b_i8.
///
/// dpbusd(u8, i8) = Σ (a_i8 + 128)·b_i8 = true_dot + 128·Σ b_i8, so we recover
/// true_dot = dpbusd - 128·w_row_sum. Runs ~4× faster than the pre-VNNI
/// `madd_epi16` path used by `dot_product_i8_avx512`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn dot_product_i8_avx512_vnni(a_u8: &[u8], b_i8: &[i8], w_row_sum: i32) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = a_u8.len();
    let mut acc = [_mm512_setzero_si512(); 8];
    let a_ptr = a_u8.as_ptr() as *const __m512i;
    let b_ptr = b_i8.as_ptr() as *const __m512i;
    let groups = n / 64;
    let mut g = 0;
    while g + 8 <= groups {
        for k in 0..8 {
            let av = _mm512_loadu_si512(a_ptr.add(g + k));
            let bv = _mm512_loadu_si512(b_ptr.add(g + k));
            acc[k] = _mm512_dpbusd_epi32(acc[k], av, bv);
        }
        g += 8;
    }
    while g < groups {
        let av = _mm512_loadu_si512(a_ptr.add(g));
        let bv = _mm512_loadu_si512(b_ptr.add(g));
        acc[0] = _mm512_dpbusd_epi32(acc[0], av, bv);
        g += 1;
    }

    let mut merged = acc[0];
    for k in 1..8 {
        merged = _mm512_add_epi32(merged, acc[k]);
    }
    let mut sums = [0i32; 16];
    _mm512_storeu_si512(sums.as_mut_ptr() as *mut __m512i, merged);
    let dp = sums.iter().sum::<i32>();

    let mut tail_true = 0i32;
    let mut tail_b = 0i32;
    let r = groups * 64;
    for j in r..n {
        let ai = a_u8[j] as i32 - 128;
        let bi = b_i8[j] as i32;
        tail_true += ai * bi;
        tail_b += bi;
    }
    let covered_b = w_row_sum - tail_b;
    dp - 128 * covered_b + tail_true
}

#[derive(Debug, Clone)]
pub struct Int8Linear {
    pub weight_i8: Vec<i8>,
    pub scales: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    /// Σ of each signed weight row; used to correct the zero-point bias when
    /// computing the dot product with the VNNI `dpbusd` path.
    pub weight_row_sum: Vec<i32>,
}

impl Int8Linear {
    pub fn new(weights_f32: &[f32], out_dim: usize, in_dim: usize) -> Self {
        let mut weight_i8 = Vec::with_capacity(weights_f32.len());
        let mut scales = Vec::with_capacity(out_dim);
        let mut weight_row_sum = Vec::with_capacity(out_dim);
        for r in 0..out_dim {
            let row = &weights_f32[r * in_dim .. (r + 1) * in_dim];
            let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            let mut ssum = 0i32;
            for &w in row {
                let q = (w / scale).round().max(-127.0).min(127.0) as i8;
                ssum += q as i32;
                weight_i8.push(q);
            }
            weight_row_sum.push(ssum);
        }
        Self { weight_i8, scales, in_dim, out_dim, weight_row_sum }
    }

    pub fn forward(&self, xs: &FastTensor) -> anyhow::Result<FastTensor> {
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        if xs.shape[rank - 1] != self.in_dim {
            anyhow::bail!("Int8Linear shape mismatch");
        }
        let b_size: usize = xs.shape[..rank - 1].iter().product();
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = out_dim;
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(b_size * out_dim);

        if b_size == 1 {
            let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
            let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(&xs.data[..in_dim], &mut quantized_in);

            let use_vnni = is_x86_feature_detected!("avx512vnni");
            // Reinterpret the i8 activation as u8 with a +128 zero-point so it
            // can feed the unsigned operand of `vpdpbusd`.
            let mut act_u8 = vec![0u8; in_dim];
            if use_vnni {
                for i in 0..in_dim {
                    act_u8[i] = (quantized_in[i] as u8) ^ 0x80;
                }
            }

            let num_threads = crate::util::get_num_threads();
            let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);

            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.weight_i8[o * in_dim .. (o + 1) * in_dim];
                    let dot = if use_vnni {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(
                                w_row.as_ptr().add(64) as *const i8,
                                std::arch::x86_64::_MM_HINT_NTA,
                            );
                        }
                        unsafe { dot_product_i8_avx512_vnni(&act_u8, w_row, self.weight_row_sum[o]) }
                    } else {
                        dot_product_i8(&quantized_in, w_row)
                    };
                    *out_val = dot as f32 * inv_scale * self.scales[o];
                }
            });

            crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
        } else {
            out_data.par_chunks_mut(out_dim).enumerate().for_each(|(b, out_row)| {
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
                let in_row = &xs.data[b * in_dim .. (b + 1) * in_dim];
                let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(in_row, &mut quantized_in);
                for o in 0..out_dim {
                    let w_row = &self.weight_i8[o * in_dim .. (o + 1) * in_dim];
                    let dot = dot_product_i8(&quantized_in, w_row);
                    out_row[o] = dot as f32 * inv_scale * self.scales[o];
                }
                crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
            });
        }

        Ok(FastTensor::new(out_data, out_shape))
    }

    pub fn forward_with_quantized(&self, xs: &FastTensor, quantized_in: &[i8], inv_scale: f32) -> anyhow::Result<FastTensor> {
        let mut out_shape = xs.shape.clone();
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let last_dim = &mut out_shape[rank - 1];
        if *last_dim != self.in_dim {
            anyhow::bail!("Int8Linear shape mismatch");
        }
        *last_dim = self.out_dim;
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;

        let use_vnni = is_x86_feature_detected!("avx512vnni");
        let mut act_u8 = vec![0u8; in_dim];
        if use_vnni {
            for i in 0..in_dim {
                act_u8[i] = (quantized_in[i] as u8) ^ 0x80;
            }
        }

        let mut out_data = crate::tensor::workspace::get_pooled_buffer(out_dim);
        let num_threads = crate::util::get_num_threads();
        let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);

        out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let start_o = chunk_idx * chunk_size;
            for (i, out_val) in out_chunk.iter_mut().enumerate() {
                let o = start_o + i;
                let w_row = &self.weight_i8[o * in_dim .. (o + 1) * in_dim];
                let dot = if use_vnni {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            w_row.as_ptr().add(64) as *const i8,
                            std::arch::x86_64::_MM_HINT_NTA,
                        );
                    }
                    unsafe { dot_product_i8_avx512_vnni(&act_u8, w_row, self.weight_row_sum[o]) }
                } else {
                    dot_product_i8(quantized_in, w_row)
                };
                *out_val = dot as f32 * inv_scale * self.scales[o];
            }
        });

        Ok(FastTensor::new(out_data, out_shape))
    }
}

#[derive(Debug, Clone)]
pub struct Int4Linear {
    pub weight_i4: Vec<u8>,
    pub scales: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Int4Linear {
    pub fn new(weights_f32: &[f32], out_dim: usize, in_dim: usize) -> Self {
        let mut weight_i4 = Vec::with_capacity((weights_f32.len() + 1) / 2);
        let mut scales = Vec::with_capacity(out_dim);
        for r in 0..out_dim {
            let row = &weights_f32[r * in_dim .. (r + 1) * in_dim];
            let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
            
            for c in (0..in_dim).step_by(2) {
                let w0 = row[c];
                let w1 = if c + 1 < in_dim { row[c + 1] } else { 0.0 };
                
                let q0 = (w0 / scale).round().max(-8.0).min(7.0) as i8;
                let q1 = (w1 / scale).round().max(-8.0).min(7.0) as i8;
                
                let packed = ((q0 as u8) & 0x0F) | ((q1 as u8) << 4);
                weight_i4.push(packed);
            }
        }
        Self { weight_i4, scales, in_dim, out_dim }
    }

    pub fn forward(&self, xs: &FastTensor) -> anyhow::Result<FastTensor> {
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        if xs.shape[rank - 1] != self.in_dim {
            anyhow::bail!("Int4Linear shape mismatch");
        }
        let b_size: usize = xs.shape[..rank - 1].iter().product();
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = out_dim;
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(b_size * out_dim);

        if b_size == 1 {
            let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
            let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(&xs.data[..in_dim], &mut quantized_in);

            let num_threads = crate::util::get_num_threads();
            let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);

            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row_start = o * ((in_dim + 1) / 2);
                    let w_row_end = w_row_start + ((in_dim + 1) / 2);
                    let w_packed = &self.weight_i4[w_row_start .. w_row_end];
                    
                    let mut dot = 0i32;
                    let q_in_chunks = quantized_in.chunks_exact(2);
                    for (&b, in_chunk) in w_packed.iter().zip(q_in_chunks) {
                        let q0 = ((b & 0x0F) as i8) << 4 >> 4;
                        let q1 = ((b & 0xF0) as i8) >> 4;
                        
                        dot += (in_chunk[0] as i32) * (q0 as i32) + (in_chunk[1] as i32) * (q1 as i32);
                    }
                    *out_val = dot as f32 * inv_scale * self.scales[o];
                }
            });
            crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
        } else {
            out_data.par_chunks_mut(out_dim).enumerate().for_each(|(b, out_row)| {
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
                let in_row = &xs.data[b * in_dim .. (b + 1) * in_dim];
                let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(in_row, &mut quantized_in);

                for o in 0..out_dim {
                    let w_row_start = o * ((in_dim + 1) / 2);
                    let w_row_end = w_row_start + ((in_dim + 1) / 2);
                    let w_packed = &self.weight_i4[w_row_start .. w_row_end];
                    
                    let mut dot = 0i32;
                    let q_in_chunks = quantized_in.chunks_exact(2);
                    for (&w_b, in_chunk) in w_packed.iter().zip(q_in_chunks) {
                        let q0 = ((w_b & 0x0F) as i8) << 4 >> 4;
                        let q1 = ((w_b & 0xF0) as i8) >> 4;
                        
                        dot += (in_chunk[0] as i32) * (q0 as i32) + (in_chunk[1] as i32) * (q1 as i32);
                    }
                    out_row[o] = dot as f32 * inv_scale * self.scales[o];
                }
                crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
            });
        }

        Ok(FastTensor::new(out_data, out_shape))
    }

    pub fn forward_with_quantized(&self, xs: &FastTensor, quantized_in: &[i8], inv_scale: f32) -> anyhow::Result<FastTensor> {
        let mut out_shape = xs.shape.clone();
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let last_dim = &mut out_shape[rank - 1];
        if *last_dim != self.in_dim {
            anyhow::bail!("Int4Linear shape mismatch");
        }
        *last_dim = self.out_dim;
        let out_dim = self.out_dim;
        let in_dim = self.in_dim;

        let mut out_data = crate::tensor::workspace::get_pooled_buffer(out_dim);
        let num_threads = crate::util::get_num_threads();
        let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);

        out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let start_o = chunk_idx * chunk_size;
            for (i, out_val) in out_chunk.iter_mut().enumerate() {
                let o = start_o + i;
                let w_row_start = o * ((in_dim + 1) / 2);
                let w_row_end = w_row_start + ((in_dim + 1) / 2);
                let w_packed = &self.weight_i4[w_row_start .. w_row_end];
                
                let mut dot = 0i32;
                let q_in_chunks = quantized_in.chunks_exact(2);
                for (&b, in_chunk) in w_packed.iter().zip(q_in_chunks) {
                    let q0 = ((b & 0x0F) as i8) << 4 >> 4;
                    let q1 = ((b & 0xF0) as i8) >> 4;
                    
                    dot += (in_chunk[0] as i32) * (q0 as i32) + (in_chunk[1] as i32) * (q1 as i32);
                }
                *out_val = dot as f32 * inv_scale * self.scales[o];
            }
        });

        Ok(FastTensor::new(out_data, out_shape))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmHeadConfig {
    F32,
    F16,
    Int8,
    Int4,
    /// Route the LM head through the fast VNNI ternary (`dpbusd`) path by
    /// ternarizing its FP32 weights on load (like the transformer layers).
    Ternary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationConfig {
    None,
    Bit1(LmHeadConfig),
    Bit1_58(TernaryPackType, LmHeadConfig, bool),
}

#[derive(Debug, Clone)]
pub struct F32Linear {
    pub weight: FastTensor, // shape [out_features, in_features]
}

impl F32Linear {
    pub fn new(weight: FastTensor) -> Self {
        Self { weight }
    }
    
    pub fn forward(&self, xs: &FastTensor) -> anyhow::Result<FastTensor> {
        let in_features = self.weight.shape[1];
        let out_features = self.weight.shape[0];
        
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dim");
        }
        if xs.shape[rank - 1] != in_features {
            anyhow::bail!("F32Linear shape mismatch: input last dim {}, expected {}", xs.shape[rank - 1], in_features);
        }
        
        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = out_features;
        
        let b_size: usize = xs.shape[..rank - 1].iter().product();
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(b_size * out_features);
        
        if b_size == 1 {
            let in_row = &xs.data[0 .. in_features];
            let num_threads = crate::util::get_num_threads();
            let chunk_size = ((out_features + num_threads - 1) / num_threads).max(128);
            
            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.weight.data[o * in_features .. (o + 1) * in_features];
                    *out_val = dot_product_f32(in_row, w_row);
                }
            });
        } else {
            out_data.par_chunks_mut(out_features).enumerate().for_each(|(b, out_row)| {
                let in_row = &xs.data[b * in_features .. (b + 1) * in_features];
                for o in 0..out_features {
                    let w_row = &self.weight.data[o * in_features .. (o + 1) * in_features];
                    out_row[o] = dot_product_f32(in_row, w_row);
                }
            });
        }
        
        Ok(FastTensor::new(out_data, out_shape))
    }
}

#[derive(Debug, Clone)]
pub enum LinearKind {
    Standard(F32Linear),
    Int8(Int8Linear),
    Int4(Int4Linear),
    Ternary(TernaryLinear),
    Bit(BitLinear),
}

#[derive(Debug, Clone)]
pub struct DynamicLinear {
    pub inner: LinearKind,
}

impl DynamicLinear {
    pub fn forward(&self, xs: &FastTensor) -> anyhow::Result<FastTensor> {
        match &self.inner {
            LinearKind::Standard(l) => l.forward(xs),
            LinearKind::Int8(l) => l.forward(xs),
            LinearKind::Int4(l) => l.forward(xs),
            LinearKind::Ternary(l) => l.forward(xs),
            LinearKind::Bit(l) => l.forward(xs),
        }
    }

    pub fn forward_with_quantized(&self, xs: &FastTensor, quantized_in: &[i8], inv_scale: f32) -> anyhow::Result<FastTensor> {
        match &self.inner {
            LinearKind::Standard(l) => l.forward(xs),
            LinearKind::Int8(l) => l.forward_with_quantized(xs, quantized_in, inv_scale),
            LinearKind::Int4(l) => l.forward_with_quantized(xs, quantized_in, inv_scale),
            LinearKind::Ternary(l) => l.forward_with_quantized(xs, quantized_in, inv_scale),
            LinearKind::Bit(_) => anyhow::bail!("forward_with_quantized not supported for BitLinear"),
        }
    }

    pub fn new_standard(weight: FastTensor) -> Self {
        Self {
            inner: LinearKind::Standard(F32Linear::new(weight)),
        }
    }

    pub fn new_int8(weights_f32: &[f32], out_dim: usize, in_dim: usize) -> Self {
        Self {
            inner: LinearKind::Int8(Int8Linear::new(weights_f32, out_dim, in_dim)),
        }
    }

    pub fn new_int4(weights_f32: &[f32], out_dim: usize, in_dim: usize) -> Self {
        Self {
            inner: LinearKind::Int4(Int4Linear::new(weights_f32, out_dim, in_dim)),
        }
    }

    pub fn new_ternary(in_dim: usize, out_dim: usize, weights_f32: &[f32], pack_type: TernaryPackType, provided_scale: Option<f32>) -> anyhow::Result<Self> {
        let l = TernaryLinear::new(in_dim, out_dim, weights_f32, pack_type, provided_scale)?;
        Ok(Self {
            inner: LinearKind::Ternary(l),
        })
    }

    pub fn new_ternary_direct(packed_weights: Vec<u8>, in_dim: usize, out_dim: usize, pack_type: TernaryPackType, w_scales: Vec<f32>) -> Self {
        Self {
            inner: LinearKind::Ternary(TernaryLinear {
                packed_weights,
                in_dim,
                out_dim,
                pack_type,
                w_scales,
            })
        }
    }

    pub fn new_bit(in_dim: usize, out_dim: usize, weights_f32: &[f32]) -> anyhow::Result<Self> {
        let l = BitLinear::new(in_dim, out_dim, weights_f32)?;
        Ok(Self {
            inner: LinearKind::Bit(l),
        })
    }

    pub fn load(
        in_dim: usize,
        out_dim: usize,
        loader: &SafeTensorLoader,
        name: &str,
        config: QuantizationConfig,
    ) -> anyhow::Result<Self> {
        match config {
            QuantizationConfig::None => {
                let weight = loader.get(&[out_dim, in_dim], name)?;
                Ok(Self::new_standard(weight))
            }
            QuantizationConfig::Bit1(_lm_head_cfg) => {
                let weight = loader.get(&[out_dim, in_dim], name)?;
                Self::new_bit(in_dim, out_dim, &weight.data)
            }
            QuantizationConfig::Bit1_58(pack_type, lm_head_cfg, is_inverted_scale) => {
                if let Ok((packed_weights, w_scales, p_in, p_out)) =
                    loader.get_prepacked_ternary(&[out_dim, in_dim], name, pack_type, is_inverted_scale)
                {
                    return Ok(Self::new_ternary_direct(
                        packed_weights, p_in, p_out, pack_type, w_scales,
                    ));
                }
                
                let weight = loader.get(&[out_dim, in_dim], name)?;
                // Only apply lm_head quantization to the LM head layer.
                // For transformer layers, ternarize the FP32 weights on-the-fly.
                // (Models like 1bitLLM/bitnet_b1_58-3B store FP32 pre-ternarization weights
                //  that must be ternarized at load time to match the training behaviour.)
                if loader.prefix_ends_with("lm_head") {
                    match lm_head_cfg {
                        LmHeadConfig::Int4 => Ok(Self::new_int4(&weight.data, out_dim, in_dim)),
                        LmHeadConfig::Int8 => Ok(Self::new_int8(&weight.data, out_dim, in_dim)),
                        LmHeadConfig::F16 | LmHeadConfig::F32 => Ok(Self::new_standard(weight)),
                        LmHeadConfig::Ternary => {
                            let scale_name = format!("{}_scale", name);
                            let provided_scale =
                                loader.get(&[1], &scale_name).map(|s| s.data[0]).ok();
                            Self::new_ternary(in_dim, out_dim, &weight.data, pack_type, provided_scale)
                        }
                    }
                } else {
                    let scale_name = format!("{}_scale", name);
                    let provided_scale = loader.get(&[1], &scale_name).map(|s| s.data[0]).ok();
                    eprintln!("Loaded scale for {}: {:?}", name, provided_scale);
                    Self::new_ternary(in_dim, out_dim, &weight.data, pack_type, provided_scale)
                }
            }
        }
    }
}
