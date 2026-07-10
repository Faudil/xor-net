use crate::tensor::FastTensor;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct CpuRingCache {
    pub k_buffers: Vec<Vec<f32>>,
    pub v_buffers: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl CpuRingCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let size = num_kv_heads * max_seq_len * head_dim;
        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_buffers.push(vec![0.0f32; size]);
            v_buffers.push(vec![0.0f32; size]);
        }
        Self {
            k_buffers,
            v_buffers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }
}

pub fn fast_attention(
    q: &FastTensor,
    k: &FastTensor,
    v: &FastTensor,
    layer_idx: usize,
    index_pos: usize,
    cache: &mut CpuRingCache,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> anyhow::Result<FastTensor> {
    let (b_sz, _q_heads, seq_len, _q_head_dim) = match q.shape.as_slice() {
        &[b_sz, q_heads, seq_len, q_head_dim] => (b_sz, q_heads, seq_len, q_head_dim),
        _ => anyhow::bail!("fast_attention: Q shape must be 4D"),
    };
    if b_sz != 1 {
        anyhow::bail!("fast_attention currently only supports batch size 1, got {}", b_sz);
    }
    
    let q_data = &q.data;
    let k_data = &k.data;
    let v_data = &v.data;
    
    let k_buf = &mut cache.k_buffers[layer_idx];
    let v_buf = &mut cache.v_buffers[layer_idx];
    let max_seq_len = cache.max_seq_len;
    
    // 1. Update the cache with new K/V tokens
    for h_kv in 0..num_kv_heads {
        for t_new in 0..seq_len {
            let t_abs = index_pos + t_new;
            if t_abs >= max_seq_len {
                // Ring buffer wrapping if we exceed max position embeddings
                let wrapped_t = t_abs % max_seq_len;
                let dest_offset = h_kv * (max_seq_len * head_dim) + wrapped_t * head_dim;
                let src_offset = h_kv * (seq_len * head_dim) + t_new * head_dim;
                k_buf[dest_offset .. dest_offset + head_dim]
                    .copy_from_slice(&k_data[src_offset .. src_offset + head_dim]);
                v_buf[dest_offset .. dest_offset + head_dim]
                    .copy_from_slice(&v_data[src_offset .. src_offset + head_dim]);
            } else {
                let dest_offset = h_kv * (max_seq_len * head_dim) + t_abs * head_dim;
                let src_offset = h_kv * (seq_len * head_dim) + t_new * head_dim;
                k_buf[dest_offset .. dest_offset + head_dim]
                    .copy_from_slice(&k_data[src_offset .. src_offset + head_dim]);
                v_buf[dest_offset .. dest_offset + head_dim]
                    .copy_from_slice(&v_data[src_offset .. src_offset + head_dim]);
            }
        }
    }
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_kv_len = index_pos + seq_len;
    
    let mut out_data = crate::tensor::uninit_vec(num_heads * seq_len * head_dim);
    
    // 2. Parallel Attention computation over Query heads
    out_data.par_chunks_mut(seq_len * head_dim)
        .enumerate()
        .for_each(|(h, head_out)| {
            let h_kv = h / (num_heads / num_kv_heads);
            let k_buf = &cache.k_buffers[layer_idx];
            let v_buf = &cache.v_buffers[layer_idx];
            let max_seq_len = cache.max_seq_len;
            
            // Score vector: [seq_len * total_kv_len]
            let mut scores = crate::tensor::uninit_vec(seq_len * total_kv_len);
            
            for t_q in 0..seq_len {
                let q_offset = h * (seq_len * head_dim) + t_q * head_dim;
                let q_vec = &q_data[q_offset .. q_offset + head_dim];
                
                let limit = index_pos + t_q;
                
                for t_k in 0..=limit {
                    let k_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim;
                    let k_vec = &k_buf[k_offset .. k_offset + head_dim];
                    
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        sum += q_vec[d] * k_vec[d];
                    }
                    scores[t_q * total_kv_len + t_k] = sum * scale;
                }
                
                // Mask future tokens
                for t_k in (limit + 1)..total_kv_len {
                    scores[t_q * total_kv_len + t_k] = f32::NEG_INFINITY;
                }
                
                // Softmax
                let row_scores = &mut scores[t_q * total_kv_len .. (t_q + 1) * total_kv_len];
                let mut max_score = f32::NEG_INFINITY;
                for &s in row_scores.iter() {
                    if s > max_score { max_score = s; }
                }
                
                let mut sum_exp = 0.0f32;
                for s in row_scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                
                let inv_sum_exp = 1.0 / sum_exp;
                for s in row_scores.iter_mut() {
                    *s *= inv_sum_exp;
                }
                
                // Weighted sum over V
                let out_vec = &mut head_out[t_q * head_dim .. (t_q + 1) * head_dim];
                out_vec.fill(0.0f32);
                for t_k in 0..total_kv_len {
                    let attn_w = row_scores[t_k];
                    if attn_w < 1e-6 { continue; } // Skip negligible weights
                    
                    let v_offset = h_kv * (max_seq_len * head_dim) + (t_k % max_seq_len) * head_dim;
                    let v_vec = &v_buf[v_offset .. v_offset + head_dim];
                    for d in 0..head_dim {
                        out_vec[d] += attn_w * v_vec[d];
                    }
                }
            }
        });
        
    Ok(FastTensor::new(out_data, vec![b_sz, num_heads, seq_len, head_dim]))
}
