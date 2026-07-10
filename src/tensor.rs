use rayon::prelude::*;

#[inline(always)]
pub fn uninit_vec<T>(size: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(size);
    unsafe {
        v.set_len(size);
    }
    v
}
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
        assert_eq!(
            data.len(),
            expected_size,
            "FastTensor::new shape mismatch: data len is {}, expected {}",
            data.len(),
            expected_size
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0f32; size],
            shape,
        }
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
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        if self.data.len() >= 32768 {
            out_data.par_chunks_mut(4096)
                .zip(self.data.par_chunks(4096))
                .zip(other.data.par_chunks(4096))
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
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        if self.data.len() >= 32768 {
            out_data.par_chunks_mut(4096)
                .zip(self.data.par_chunks(4096))
                .zip(other.data.par_chunks(4096))
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
        
        if self.data.len() >= 32768 {
            self.data.par_chunks_mut(1024)
                .zip(other.data.par_chunks(1024))
                .for_each(|(out_chunk, other_chunk)| {
                    for i in 0..out_chunk.len() {
                        out_chunk[i] += other_chunk[i];
                    }
                });
        } else {
            for i in 0..self.data.len() {
                self.data[i] += other.data[i];
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
        
        if self.data.len() >= 32768 {
            self.data.par_chunks_mut(1024)
                .zip(other.data.par_chunks(1024))
                .for_each(|(s1_chunk, s2_chunk)| {
                    for i in 0..s1_chunk.len() {
                        let x = s1_chunk[i];
                        s1_chunk[i] = (x / (1.0 + (-x).exp())) * s2_chunk[i];
                    }
                });
        } else {
            for i in 0..self.data.len() {
                let x = self.data[i];
                self.data[i] = (x / (1.0 + (-x).exp())) * other.data[i];
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
        
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        if self.data.len() >= 32768 {
            out_data.par_chunks_mut(hidden_size)
                .zip(self.data.par_chunks(hidden_size))
                .for_each(|(out_row, in_row)| {
                    let mut sum_sq = 0.0f32;
                    for &x in in_row.iter() {
                        sum_sq += x * x;
                    }
                    let inv_std = 1.0f32 / (sum_sq / hidden_size as f32 + eps).sqrt();
                    for i in 0..hidden_size {
                        out_row[i] = in_row[i] * inv_std * weight.data[i];
                    }
                });
        } else {
            for (out_row, in_row) in out_data.chunks_mut(hidden_size).zip(self.data.chunks(hidden_size)) {
                let mut sum_sq = 0.0f32;
                for &x in in_row.iter() {
                    sum_sq += x * x;
                }
                let inv_std = 1.0f32 / (sum_sq / hidden_size as f32 + eps).sqrt();
                for i in 0..hidden_size {
                    out_row[i] = in_row[i] * inv_std * weight.data[i];
                }
            }
        }
        
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
        
        let mut out_data = crate::tensor::uninit_vec(ids.len() * hidden_size);
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
        
        // Output shape is [b_sz, seq_len, hidden_size] or just [1, seq_len, hidden_size]
        // Usually, b_sz = 1. Let's make it [1, ids.len(), hidden_size] for consistency with llama.
        Ok(Self::new(out_data, vec![1, ids.len(), hidden_size]))
    }

    pub fn rope_inplace(mut self, cos: &Self, sin: &Self, index_pos: usize) -> anyhow::Result<Self> {
        let (_b_sz, num_heads, seq_len, head_dim) = match self.shape.as_slice() {
            &[b_sz, num_heads, seq_len, head_dim] => (b_sz, num_heads, seq_len, head_dim),
            _ => anyhow::bail!("rope_inplace: input must be 4D shape [b_sz, num_heads, seq_len, head_dim], got {:?}", self.shape),
        };
        let half_dim = head_dim / 2;
        
        if self.data.len() >= 32768 {
            self.data.par_chunks_mut(seq_len * head_dim)
                .enumerate()
                .for_each(|(bh, head_out)| {
                    let _b = bh / num_heads;
                    for t in 0..seq_len {
                        let pos = index_pos + t;
                        let cos_pos = &cos.data[pos * half_dim .. (pos + 1) * half_dim];
                        let sin_pos = &sin.data[pos * half_dim .. (pos + 1) * half_dim];
                        
                        let token_vec = &mut head_out[t * head_dim .. (t + 1) * head_dim];
                        for d in 0..half_dim {
                            let v_real = token_vec[d];
                            let v_imag = token_vec[d + half_dim];
                            
                            let c = cos_pos[d];
                            let s = sin_pos[d];
                            
                            token_vec[d] = v_real * c - v_imag * s;
                            token_vec[d + half_dim] = v_real * s + v_imag * c;
                        }
                    }
                });
        } else {
            for (bh, head_out) in self.data.chunks_mut(seq_len * head_dim).enumerate() {
                let _b = bh / num_heads;
                for t in 0..seq_len {
                    let pos = index_pos + t;
                    let cos_pos = &cos.data[pos * half_dim .. (pos + 1) * half_dim];
                    let sin_pos = &sin.data[pos * half_dim .. (pos + 1) * half_dim];
                    
                    let token_vec = &mut head_out[t * head_dim .. (t + 1) * head_dim];
                    for d in 0..half_dim {
                        let v_real = token_vec[d];
                        let v_imag = token_vec[d + half_dim];
                        
                        let c = cos_pos[d];
                        let s = sin_pos[d];
                        
                        token_vec[d] = v_real * c - v_imag * s;
                        token_vec[d + half_dim] = v_real * s + v_imag * c;
                    }
                }
            }
        }
            
        Ok(self)
    }

    pub fn slice_last_token(&self) -> anyhow::Result<Self> {
        let (b_sz, seq_len, hidden_size) = match self.shape.as_slice() {
            &[b_sz, seq_len, hidden_size] => (b_sz, seq_len, hidden_size),
            _ => anyhow::bail!("slice_last_token: input must be 3D [b_sz, seq_len, hidden_size], got {:?}", self.shape),
        };
        let mut out_data = crate::tensor::uninit_vec(b_sz * hidden_size);
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
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        
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
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        
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
        let mut out_data = crate::tensor::uninit_vec(self.data.len());
        out_data.par_iter_mut().zip(self.data.par_iter()).for_each(|(out, &x)| {
            let x_cube = x * x * x;
            let inner = 0.79788456f32 * (x + 0.044715f32 * x_cube);
            *out = 0.5f32 * x * (1.0f32 + inner.tanh());
        });
        Ok(Self::new(out_data, self.shape.clone()))
    }
}
