use crate::tensor::FastTensor;
use crate::nn::fast_attention_simd::{softmax_inplace, weighted_sum, batched_dot_product};
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    static SCORE_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn get_score_buffer(size: usize) -> Vec<f32> {
    SCORE_BUFFER.with(|buf| {
        let mut b = buf.borrow_mut();
        if b.capacity() < size {
            b.clear();
            b.reserve(size);
        }
        b.resize(size, 0.0);
        let result = b.clone();
        b.clear();
        result
    })
}

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
    
    // 2. Parallel Attention computation over KV heads
    let heads_per_kv = num_heads / num_kv_heads;
    let mut out_data = vec![0.0f32; num_heads * seq_len * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Parallel over KV heads - each KV head writes to disjoint output regions
    out_data.par_chunks_mut(heads_per_kv * seq_len * head_dim)
        .enumerate()
        .for_each(|(h_kv, kv_head_out)| {
            let k_buf = &cache.k_buffers[layer_idx];
            let v_buf = &cache.v_buffers[layer_idx];
            let max_seq_len = cache.max_seq_len;
            
            // Process all Q heads that map to this KV head
            let q_head_start = h_kv * heads_per_kv;
            let q_head_end = q_head_start + heads_per_kv;
            
            // Thread-local score buffer (reused across Q heads and query positions)
            let max_kv_len = index_pos + seq_len;
            let mut scores = get_score_buffer(max_kv_len * seq_len);
            
            for h in q_head_start..q_head_end {
                let local_h = h - q_head_start;
                let head_out_base = local_h * seq_len * head_dim;
                
                for t_q in 0..seq_len {
                    let q_offset = h * (seq_len * head_dim) + t_q * head_dim;
                    let q_vec = &q_data[q_offset .. q_offset + head_dim];
                    let out_vec = &mut kv_head_out[head_out_base + t_q * head_dim .. head_out_base + (t_q + 1) * head_dim];
                    
                    let limit = index_pos + t_q;
                    let total_kv_len = limit + 1;
                    
                    // Compute Q * K^T using batched SIMD dot products
                    let batch_size = 16;
                    let mut t_k = 0;
                    while t_k <= limit {
                        let batch_end = (t_k + batch_size).min(limit + 1);
                        let batch_len = batch_end - t_k;
                        let mut batch_scores = [0.0f32; 16];
                        
                        batched_dot_product(
                            q_vec, k_buf, h_kv, max_seq_len, head_dim,
                            t_k, batch_end, &mut batch_scores
                        );
                        
                        for i in 0..batch_len {
                            scores[t_k + i] = batch_scores[i] * scale;
                        }
                        t_k = batch_end;
                    }
                    
                    // Softmax (SIMD-optimized)
                    let row_scores = &mut scores[..total_kv_len];
                    softmax_inplace(row_scores);
                    
                    // Weighted sum over V using SIMD
                    out_vec.fill(0.0f32);
                    weighted_sum(
                        out_vec,
                        row_scores,
                        v_buf,
                        h_kv,
                        max_seq_len,
                        head_dim,
                        total_kv_len,
                    );
                }
            }
        });
    
    Ok(FastTensor::new(out_data, vec![b_sz, num_heads, seq_len, head_dim]))
}