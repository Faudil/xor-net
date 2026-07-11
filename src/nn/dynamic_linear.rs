use rayon::prelude::*;
use crate::tensor::FastTensor;
use crate::bit1::layers::BitLinear;
use crate::bit1_58::layers::TernaryLinear;
use crate::bit1_58::quantization::TernaryPackType;
use crate::loader::SafeTensorLoader;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

fn get_num_threads() -> usize {
    let mut val = NUM_THREADS.load(Ordering::Relaxed);
    if val == 0 {
        val = rayon::current_num_threads().max(1);
        NUM_THREADS.store(val, Ordering::Relaxed);
    }
    val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
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
    unsafe {
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
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_product_i8_avx512(a: &[i8], b: &[i8]) -> i32 {
    unsafe {
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

#[derive(Debug, Clone)]
pub struct Int8Linear {
    pub weight_i8: Vec<i8>,
    pub scales: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Int8Linear {
    pub fn new(weights_f32: &[f32], out_dim: usize, in_dim: usize) -> Self {
        let mut weight_i8 = Vec::with_capacity(weights_f32.len());
        let mut scales = Vec::with_capacity(out_dim);
        for r in 0..out_dim {
            let row = &weights_f32[r * in_dim .. (r + 1) * in_dim];
            let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            for &w in row {
                let q = (w / scale).round().max(-127.0).min(127.0) as i8;
                weight_i8.push(q);
            }
        }
        Self { weight_i8, scales, in_dim, out_dim }
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
        let mut out_data = crate::tensor::uninit_vec(b_size * out_dim);

        if b_size == 1 {
            let (quantized_in, inv_scale) = crate::bit1_58::quantization::quantize_f32_to_i8(&xs.data[..in_dim]);

            let num_threads = get_num_threads();
            let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);

            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.weight_i8[o * in_dim .. (o + 1) * in_dim];
                    let dot = dot_product_i8(&quantized_in, w_row);
                    *out_val = dot as f32 * inv_scale * self.scales[o];
                }
            });
        } else {
            anyhow::bail!("Int8Linear batched not implemented");
        }

        Ok(FastTensor::new(out_data, out_shape))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationConfig {
    None,
    Bit1,
    Bit1_58(TernaryPackType),
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
        let mut out_data = crate::tensor::uninit_vec(b_size * out_features);
        
        if b_size == 1 {
            let in_row = &xs.data[0 .. in_features];
            let num_threads = get_num_threads();
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
            LinearKind::Ternary(l) => l.forward(xs),
            LinearKind::Bit(l) => l.forward(xs),
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

    pub fn new_ternary(in_dim: usize, out_dim: usize, weights_f32: &[f32], pack_type: TernaryPackType) -> anyhow::Result<Self> {
        let l = TernaryLinear::new(in_dim, out_dim, weights_f32, pack_type)?;
        Ok(Self {
            inner: LinearKind::Ternary(l),
        })
    }

    pub fn new_ternary_direct(packed_weights: Vec<u8>, in_dim: usize, out_dim: usize, pack_type: TernaryPackType, w_scale: f32) -> Self {
        Self {
            inner: LinearKind::Ternary(TernaryLinear {
                packed_weights,
                in_dim,
                out_dim,
                pack_type,
                w_scale,
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
            QuantizationConfig::Bit1 => {
                let weight = loader.get(&[out_dim, in_dim], name)?;
                Self::new_bit(in_dim, out_dim, &weight.data)
            }
            QuantizationConfig::Bit1_58(pack_type) => {
                // Try direct pre-packed path first (preserves original w_scale)
                if let Ok((packed_weights, w_scale, p_in, p_out)) =
                    loader.get_prepacked_ternary(&[out_dim, in_dim], name, pack_type)
                {
                    return Ok(Self::new_ternary_direct(
                        packed_weights, p_in, p_out, pack_type, w_scale,
                    ));
                }
                // Not pre-packed (e.g., BF16 weight like lm_head): keep F32 precision
                let weight = loader.get(&[out_dim, in_dim], name)?;
                Ok(Self::new_int8(&weight.data, out_dim, in_dim))
            }
        }
    }
}
