use rayon::prelude::*;
use crate::nn::fast_attention_simd::{rms_norm, rope_inplace as rope_simd_inplace};
pub mod workspace;
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct FastTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Debug for FastTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastTensor")
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl FastTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        debug_assert_eq!(
            data.len(),
            expected_size,
            "FastTensor::new shape mismatch: data len is {}, expected {}",
            data.len(),
            expected_size
        );
        Self { data, shape }
    }
}

impl Drop for FastTensor {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.data);
        crate::tensor::workspace::return_pooled_buffer(buf);
    }
}

impl FastTensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; size],
            shape,
        }
    }

    pub fn into_data(mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> anyhow::Result<Self> {
        let size: usize = shape.iter().product();
        if size != self.data.len() {
            anyhow::bail!(
                "FastTensor::reshape: cannot reshape from {:?} to {:?}",
                self.shape,
                shape
            );
        }
        Ok(Self {
            data: self.data.clone(),
            shape,
        })
    }

    pub fn add(&self, other: &Self) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            anyhow::bail!(
                "FastTensor::add shape mismatch: {:?} vs {:?}",
                self.shape,
                other.shape
            );
        }
        let mut out_data = crate::util::uninit_vec(self.data.len());
        if self.data.len() >= crate::util::PARALLEL_THRESHOLD {
            out_data.par_chunks_mut(crate::util::CHUNK_SIZE)
                .zip(self.data.par_chunks(crate::util::CHUNK_SIZE))
                .zip(other.data.par_chunks(crate::util::CHUNK_SIZE))
                .for_each(|((out_chunk, self_chunk), other_chunk)| {
                    for i in 0..out_chunk.len() {
                        out_chunk[i] = self_chunk[i] + other_chunk[i];
                    }
                });
        } else {
            for i in 0..out_data.len() {
                out_data[i] = self.data[i] + other.data[i];
            }
        }
        Ok(Self::new(out_data, self.shape.clone()))
    }

    pub fn silu_mul(&self, other: &Self) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            anyhow::bail!(
                "FastTensor::silu_mul shape mismatch: {:?} vs {:?}",
                self.shape,
                other.shape
            );
        }
        let mut out_data = crate::util::uninit_vec(self.data.len());
        if self.data.len() >= crate::util::PARALLEL_THRESHOLD {
            out_data.par_chunks_mut(crate::util::CHUNK_SIZE)
                .zip(self.data.par_chunks(crate::util::CHUNK_SIZE))
                .zip(other.data.par_chunks(crate::util::CHUNK_SIZE))
                .for_each(|((out_chunk, self_chunk), other_chunk)| {
                    for i in 0..out_chunk.len() {
                        let x = self_chunk[i];
                        let silu = x / (1.0f32 + (-x).exp());
                        out_chunk[i] = silu * other_chunk[i];
                    }
                });
        } else {
            for i in 0..out_data.len() {
                let x = self.data[i];
                let silu = x / (1.0f32 + (-x).exp());
                out_data[i] = silu * other.data[i];
            }
        }
        Ok(Self::new(out_data, self.shape.clone()))
    }
    
    pub fn add_inplace(mut self, other: &Self) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            anyhow::bail!(
                "FastTensor::add_inplace shape mismatch: {:?} != {:?}",
                self.shape,
                other.shape
            );
        }
        
        if self.data.len() >= crate::util::PARALLEL_THRESHOLD {
            self.data.par_chunks_mut(1024)
                .zip(other.data.par_chunks(1024))
                .for_each(|(out_chunk, other_chunk)| {
                    for i in 0..out_chunk.len() {
                        debug_assert!(i < out_chunk.len() && i < other_chunk.len());
                        unsafe {
                            *out_chunk.get_unchecked_mut(i) += *other_chunk.get_unchecked(i);
                        }
                    }
                });
        } else {
            for i in 0..self.data.len() {
                debug_assert!(i < self.data.len() && i < other.data.len());
                unsafe {
                    *self.data.get_unchecked_mut(i) += *other.data.get_unchecked(i);
                }
            }
        }
            
        Ok(self)
    }

    pub fn silu_mul_inplace(mut self, other: &Self) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            anyhow::bail!(
                "FastTensor::silu_mul_inplace shape mismatch: {:?} != {:?}",
                self.shape,
                other.shape
            );
        }
        
        if self.data.len() >= crate::util::PARALLEL_THRESHOLD {
            self.data.par_chunks_mut(1024)
                .zip(other.data.par_chunks(1024))
                .for_each(|(s1_chunk, s2_chunk)| {
                    for i in 0..s1_chunk.len() {
                        debug_assert!(i < s1_chunk.len() && i < s2_chunk.len());
                        unsafe {
                            let x = *s1_chunk.get_unchecked(i);
                            *s1_chunk.get_unchecked_mut(i) =
                                (x / (1.0 + (-x).exp())) * *s2_chunk.get_unchecked(i);
                        }
                    }
                });
        } else {
            for i in 0..self.data.len() {
                debug_assert!(i < self.data.len() && i < other.data.len());
                unsafe {
                    let x = *self.data.get_unchecked(i);
                    *self.data.get_unchecked_mut(i) =
                        (x / (1.0 + (-x).exp())) * *other.data.get_unchecked(i);
                }
            }
        }
            
        Ok(self)
    }

    pub fn relu2_mul_inplace(mut self, other: &Self) -> anyhow::Result<Self> {
        if self.shape != other.shape {
            anyhow::bail!(
                "FastTensor::relu2_mul_inplace shape mismatch: {:?} != {:?}",
                self.shape,
                other.shape
            );
        }

        if self.data.len() >= crate::util::PARALLEL_THRESHOLD {
            self.data.par_chunks_mut(1024)
                .zip(other.data.par_chunks(1024))
                .for_each(|(s1_chunk, s2_chunk)| {
                    for i in 0..s1_chunk.len() {
                        debug_assert!(i < s1_chunk.len() && i < s2_chunk.len());
                        unsafe {
                            let x = *s1_chunk.get_unchecked(i);
                            let relu = if x > 0.0 { x } else { 0.0 };
                            *s1_chunk.get_unchecked_mut(i) = relu * relu * *s2_chunk.get_unchecked(i);
                        }
                    }
                });
        } else {
            for i in 0..self.data.len() {
                debug_assert!(i < self.data.len() && i < other.data.len());
                unsafe {
                    let x = *self.data.get_unchecked(i);
                    let relu = if x > 0.0 { x } else { 0.0 };
                    *self.data.get_unchecked_mut(i) = relu * relu * *other.data.get_unchecked(i);
                }
            }
        }

        Ok(self)
    }

    pub fn rmsnorm(&self, weight: &Self, eps: f32) -> anyhow::Result<Self> {
        let hidden_size = *self.shape.last().ok_or_else(|| {
            anyhow::anyhow!("FastTensor::rmsnorm: empty shape")
        })?;
        if weight.data.len() != hidden_size {
            anyhow::bail!(
                "FastTensor::rmsnorm weight size mismatch: expected {}, got {}",
                hidden_size,
                weight.data.len()
            );
        }
        
        let mut out_data = crate::util::uninit_vec(self.data.len());
        
        out_data.par_chunks_mut(hidden_size)
            .zip(self.data.par_chunks(hidden_size))
            .for_each(|(out_row, in_row)| {
                let mut row_copy: Vec<f32> = in_row.to_vec();
                rms_norm(&mut row_copy, &weight.data, eps);
                out_row.copy_from_slice(&row_copy);
            });
        
        Ok(Self::new(out_data, self.shape.clone()))
    }

    pub fn embedding(ids: &[u32], weight: &Self) -> anyhow::Result<Self> {
        if weight.shape.len() != 2 {
            anyhow::bail!(
                "FastTensor::embedding weight must be 2D, got shape {:?}",
                weight.shape
            );
        }
        let vocab_size = weight.shape[0];
        let hidden_size = weight.shape[1];
        
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(ids.len() * hidden_size);
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab_size {
                anyhow::bail!(
                    "FastTensor::embedding index out of bounds: id {} >= vocab_size {}",
                    id,
                    vocab_size
                );
            }
            let src = &weight.data[id * hidden_size .. (id + 1) * hidden_size];
            let dest = &mut out_data[i * hidden_size .. (i + 1) * hidden_size];
            dest.copy_from_slice(src);
        }
        
        Ok(Self::new(out_data, vec![1, ids.len(), hidden_size]))
    }

    pub fn rope_inplace(mut self, cos: &Self, sin: &Self, index_pos: usize) -> anyhow::Result<Self> {
        let (_b_sz, num_heads, seq_len, head_dim) = match self.shape.as_slice() {
            &[b_sz, num_heads, seq_len, head_dim] => (b_sz, num_heads, seq_len, head_dim),
            _ => anyhow::bail!("rope_inplace: input must be 4D shape [b_sz, num_heads, seq_len, head_dim], got {:?}", self.shape),
        };
        
        self.data.par_chunks_mut(seq_len * head_dim)
            .enumerate()
            .for_each(|(bh, head_out)| {
                let _b = bh / num_heads;
                for t in 0..seq_len {
                    let pos = index_pos + t;
                    let cos_pos = &cos.data[pos * (head_dim / 2) .. (pos + 1) * (head_dim / 2)];
                    let sin_pos = &sin.data[pos * (head_dim / 2) .. (pos + 1) * (head_dim / 2)];
                    
                    let token_vec = &mut head_out[t * head_dim .. (t + 1) * head_dim];
                    rope_simd_inplace(token_vec, cos_pos, sin_pos, pos, head_dim);
                }
            });
            
        Ok(self)
    }

    pub fn slice_last_token(&self) -> anyhow::Result<Self> {
        let (b_sz, seq_len, hidden_size) = match self.shape.as_slice() {
            &[b_sz, seq_len, hidden_size] => (b_sz, seq_len, hidden_size),
            _ => anyhow::bail!("slice_last_token: input must be 3D [b_sz, seq_len, hidden_size], got {:?}", self.shape),
        };
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(b_sz * hidden_size);
        for b in 0..b_sz {
            let src = &self.data[b * seq_len * hidden_size + (seq_len - 1) * hidden_size .. b * seq_len * hidden_size + seq_len * hidden_size];
            let dest = &mut out_data[b * hidden_size .. (b + 1) * hidden_size];
            dest.copy_from_slice(src);
        }
        Ok(Self::new(out_data, vec![b_sz, 1, hidden_size]))
    }

    pub fn transpose_seq_to_heads(&self, num_heads: usize, head_dim: usize) -> anyhow::Result<Self> {
        let (b_sz, seq_len, hidden_size) = match self.shape.as_slice() {
            &[b_sz, seq_len, hidden_size] => (b_sz, seq_len, hidden_size),
            _ => anyhow::bail!("transpose_seq_to_heads: input must be 3D, got {:?}", self.shape),
        };
        if hidden_size != num_heads * head_dim {
            anyhow::bail!("transpose_seq_to_heads: hidden_size {} must equal num_heads {} * head_dim {}", hidden_size, num_heads, head_dim);
        }
        let mut out_data = crate::util::uninit_vec(self.data.len());
        
        out_data.par_chunks_mut(seq_len * head_dim)
            .enumerate()
            .for_each(|(h, head_out)| {
                for t in 0..seq_len {
                    let in_offset = t * num_heads * head_dim + h * head_dim;
                    let out_offset = t * head_dim;
                    head_out[out_offset .. out_offset + head_dim]
                        .copy_from_slice(&self.data[in_offset .. in_offset + head_dim]);
                }
            });
        
        Ok(Self::new(out_data, vec![b_sz, num_heads, seq_len, head_dim]))
    }

    pub fn transpose_heads_to_seq(&self) -> anyhow::Result<Self> {
        let (b_sz, num_heads, seq_len, head_dim) = match self.shape.as_slice() {
            &[b_sz, num_heads, seq_len, head_dim] => (b_sz, num_heads, seq_len, head_dim),
            _ => anyhow::bail!("transpose_heads_to_seq: input must be 4D, got {:?}", self.shape),
        };
        let hidden_size = num_heads * head_dim;
        let mut out_data = crate::util::uninit_vec(self.data.len());
        
        out_data.par_chunks_mut(hidden_size)
            .enumerate()
            .for_each(|(t, seq_out)| {
                for h in 0..num_heads {
                    let in_offset = h * seq_len * head_dim + t * head_dim;
                    let out_offset = h * head_dim;
                    seq_out[out_offset .. out_offset + head_dim]
                        .copy_from_slice(&self.data[in_offset .. in_offset + head_dim]);
                }
            });
        
        Ok(Self::new(out_data, vec![b_sz, seq_len, hidden_size]))
    }

    pub fn gelu(&self) -> anyhow::Result<Self> {
        let mut out_data = crate::util::uninit_vec(self.data.len());
        out_data.par_iter_mut().zip(self.data.par_iter()).for_each(|(out, &x)| {
            let x_cube = x * x * x;
            let inner = 0.79788456f32 * (x + 0.044715f32 * x_cube);
            *out = 0.5f32 * x * (1.0f32 + inner.tanh());
        });
        Ok(Self::new(out_data, self.shape.clone()))
    }
}
