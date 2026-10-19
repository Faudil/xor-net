use crate::nn::fast_attention_simd::{batched_dot_product, softmax_inplace, weighted_sum};
use crate::tensor::FastTensor;
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

/// Per-layer K/V storage `[num_kv_heads][max_seq_len][head_dim]`.
///
/// `max_seq_len` is the *current* capacity (and the per-head row stride).
/// A cache built with [`CpuRingCache::new`] is allocated at full size and
/// wraps positions modulo its capacity (ring). One built with
/// [`CpuRingCache::growable`] starts empty and doubles on demand (amortised
/// copy) up to `max_capacity` before it would ever wrap, so a short sequence
/// only pays for the positions it uses.
#[derive(Debug, Clone)]
pub struct CpuRingCache {
    pub k_buffers: Vec<Vec<f32>>,
    pub v_buffers: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    /// Upper bound for growth (`== max_seq_len` for a fixed-size cache).
    pub max_capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Optional U8-persistent cache for memory-efficient long-context.
    /// When set, K/V are stored in u8 and lazily dequantized into k_buffers/v_buffers.
    pub u8_buffers: Vec<U8CacheLayer>,
}

/// First allocation of a growable cache, in positions.
pub const RING_CACHE_INITIAL_CAPACITY: usize = 256;

impl CpuRingCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        use_u8: bool,
    ) -> Self {
        let mut c = Self::growable(num_layers, num_kv_heads, max_seq_len, head_dim, use_u8);
        c.grow_to(max_seq_len);
        c
    }

    /// An empty cache that grows on demand up to `max_capacity` positions.
    pub fn growable(
        num_layers: usize,
        num_kv_heads: usize,
        max_capacity: usize,
        head_dim: usize,
        use_u8: bool,
    ) -> Self {
        let empty_u8 = || U8CacheLayer {
            u8_k: Vec::new(),
            u8_v: Vec::new(),
            k_scales: Vec::new(),
            v_scales: Vec::new(),
        };
        Self {
            k_buffers: vec![Vec::new(); num_layers],
            v_buffers: vec![Vec::new(); num_layers],
            max_seq_len: 0,
            max_capacity,
            num_kv_heads,
            head_dim,
            u8_buffers: if use_u8 {
                (0..num_layers).map(|_| empty_u8()).collect()
            } else {
                Vec::new()
            },
        }
    }

    /// Make room for positions `< len` (geometric growth, capped at
    /// `max_capacity`; beyond that the cache wraps as a ring).
    pub fn reserve(&mut self, len: usize) {
        if len > self.max_seq_len && self.max_seq_len < self.max_capacity {
            let cap = len
                .max(self.max_seq_len * 2)
                .max(RING_CACHE_INITIAL_CAPACITY)
                .min(self.max_capacity);
            self.grow_to(cap);
        }
    }

    /// Re-lay every buffer out with capacity `new_cap`, keeping the stored
    /// positions of each head.
    fn grow_to(&mut self, new_cap: usize) {
        let (old_cap, hd, heads) = (self.max_seq_len, self.head_dim, self.num_kv_heads);
        if new_cap <= old_cap {
            return;
        }
        fn regrow<T: Copy>(
            buf: &mut Vec<T>,
            fill: T,
            heads: usize,
            old: usize,
            new: usize,
            w: usize,
        ) {
            let mut out = vec![fill; heads * new * w];
            if !buf.is_empty() {
                for h in 0..heads {
                    out[h * new * w..(h * new + old) * w]
                        .copy_from_slice(&buf[h * old * w..(h + 1) * old * w]);
                }
            }
            *buf = out;
        }
        for b in self.k_buffers.iter_mut().chain(self.v_buffers.iter_mut()) {
            regrow(b, 0.0f32, heads, old_cap, new_cap, hd);
        }
        for u in &mut self.u8_buffers {
            regrow(&mut u.u8_k, 128u8, heads, old_cap, new_cap, hd);
            regrow(&mut u.u8_v, 128u8, heads, old_cap, new_cap, hd);
            regrow(&mut u.k_scales, 0.0f32, heads, old_cap, new_cap, 1);
            regrow(&mut u.v_scales, 0.0f32, heads, old_cap, new_cap, 1);
        }
        self.max_seq_len = new_cap;
    }

    /// Bytes currently allocated for K/V (f32 plus the optional u8 copy).
    pub fn bytes(&self) -> usize {
        let f: usize = self
            .k_buffers
            .iter()
            .chain(&self.v_buffers)
            .map(|b| b.len() * 4)
            .sum();
        let u: usize = self
            .u8_buffers
            .iter()
            .map(|u| u.u8_k.len() + u.u8_v.len() + (u.k_scales.len() + u.v_scales.len()) * 4)
            .sum();
        f + u
    }

    /// Quantize an f32 slice to u8 and store in the U8 buffer.
    fn quantize_to_u8(data: &[f32]) -> (Vec<u8>, f32) {
        let max_abs = data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| ((x * inv_scale) + 128.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        (quantized, scale)
    }
}

#[allow(clippy::too_many_arguments)]
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
        anyhow::bail!(
            "fast_attention currently only supports batch size 1, got {}",
            b_sz
        );
    }

    cache.reserve(index_pos + seq_len);
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
            let k_slice = &k_data[src_offset..src_offset + head_dim];
            let v_slice = &v_data[src_offset..src_offset + head_dim];

            // Always store in F32 working buffer
            k_buf[dest_offset..dest_offset + head_dim].copy_from_slice(k_slice);
            v_buf[dest_offset..dest_offset + head_dim].copy_from_slice(v_slice);

            // Also quantize to U8 if persistent cache is enabled
            if use_u8 {
                let u8 = &mut cache.u8_buffers[layer_idx];
                let scale_offset = h_kv * max_seq_len + wrapped_t;
                let (u8_k_slice, k_scale) = CpuRingCache::quantize_to_u8(k_slice);
                let (u8_v_slice, v_scale) = CpuRingCache::quantize_to_u8(v_slice);
                let u8_offset = h_kv * (max_seq_len * head_dim) + wrapped_t * head_dim;
                u8.u8_k[u8_offset..u8_offset + head_dim].copy_from_slice(&u8_k_slice);
                u8.u8_v[u8_offset..u8_offset + head_dim].copy_from_slice(&u8_v_slice);
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
    out_data
        .par_chunks_mut(heads_per_kv * seq_len * head_dim)
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
                    let q_vec = &q_data[q_offset..q_offset + head_dim];
                    let out_vec = &mut kv_head_out
                        [head_out_base + t_q * head_dim..head_out_base + (t_q + 1) * head_dim];

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
                            q_vec,
                            k_buf,
                            h_kv,
                            max_seq_len,
                            head_dim,
                            t_k,
                            batch_end,
                            &mut batch_scores,
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

    Ok(FastTensor::new(
        out_data,
        vec![b_sz, num_heads, seq_len, head_dim],
    ))
}
