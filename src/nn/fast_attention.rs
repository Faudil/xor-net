use crate::tensor::FastTensor;
use crate::nn::fast_attention_simd::{softmax_inplace, weighted_sum, batched_dot_product};
use rayon::prelude::*;

/// Per-layer U8 KV cache buffers.
#[derive(Debug, Clone)]
pub struct U8CacheLayer {
    /// Quantized K values: [num_kv_heads * max_seq_len * head_dim]
    pub u8_k: Vec<u8>,
    /// Quantized V values: [num_kv_heads * max_seq_len * head_dim]
    pub u8_v: Vec<u8>,
    /// Per-token K scales: [num_kv_heads * max_seq_len]
    pub k_scales: Vec<f32>,
    /// Per-token V scales: [num_kv_heads * max_seq_len]
    pub v_scales: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CpuRingCache {
    pub k_buffers: Vec<Vec<f32>>,
    pub v_buffers: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Optional U8-persistent cache for memory-efficient long-context.
    /// When set, K/V are stored in u8 and lazily dequantized into k_buffers/v_buffers.
    pub u8_buffers: Vec<U8CacheLayer>,
}

impl CpuRingCache {
    pub fn new(
        num_layers: usize, num_kv_heads: usize, max_seq_len: usize, head_dim: usize,
        use_u8: bool,
    ) -> Self {
        let size = num_kv_heads * max_seq_len * head_dim;
        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);
        let mut u8_buffers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_buffers.push(vec![0.0f32; size]);
            v_buffers.push(vec![0.0f32; size]);
            if use_u8 {
                let scale_size = num_kv_heads * max_seq_len;
                u8_buffers.push(U8CacheLayer {
                    u8_k: vec![128u8; size],
                    u8_v: vec![128u8; size],
                    k_scales: vec![0.0f32; scale_size],
                    v_scales: vec![0.0f32; scale_size],
                });
            }
        }
        Self {
            k_buffers,
            v_buffers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            u8_buffers,
        }
    }

    /// Quantize an f32 slice to u8 and store in the U8 buffer.
    fn quantize_to_u8(data: &[f32]) -> (Vec<u8>, f32) {
        let max_abs = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        let quantized: Vec<u8> = data.iter().map(|&x| {
            ((x * inv_scale) + 128.0).round().clamp(0.0, 255.0) as u8
        }).collect();
        (quantized, scale)
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
    let use_u8 = layer_idx < cache.u8_buffers.len();
    
    // 1. Update the cache with new K/V tokens
    for h_kv in 0..num_kv_heads {
        for t_new in 0..seq_len {
            let t_abs = index_pos + t_new;
            let wrapped_t = t_abs % max_seq_len;
            let dest_offset = h_kv * (max_seq_len * head_dim) + wrapped_t * head_dim;
            let src_offset = h_kv * (seq_len * head_dim) + t_new * head_dim;
            let k_slice = &k_data[src_offset .. src_offset + head_dim];
            let v_slice = &v_data[src_offset .. src_offset + head_dim];
            
            // Always store in F32 working buffer
            k_buf[dest_offset .. dest_offset + head_dim].copy_from_slice(k_slice);
            v_buf[dest_offset .. dest_offset + head_dim].copy_from_slice(v_slice);
            
            // Also quantize to U8 if persistent cache is enabled
            if use_u8 {
                let u8 = &mut cache.u8_buffers[layer_idx];
                let scale_offset = h_kv * max_seq_len + wrapped_t;
                let (u8_k_slice, k_scale) = CpuRingCache::quantize_to_u8(k_slice);
                let (u8_v_slice, v_scale) = CpuRingCache::quantize_to_u8(v_slice);
                let u8_offset = h_kv * (max_seq_len * head_dim) + wrapped_t * head_dim;
                u8.u8_k[u8_offset .. u8_offset + head_dim].copy_from_slice(&u8_k_slice);
                u8.u8_v[u8_offset .. u8_offset + head_dim].copy_from_slice(&u8_v_slice);
                u8.k_scales[scale_offset] = k_scale;
                u8.v_scales[scale_offset] = v_scale;
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
            
            // Score buffer for this KV head
            let max_kv_len = index_pos + seq_len;
            let mut scores = vec![0.0f32; max_kv_len];
            
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