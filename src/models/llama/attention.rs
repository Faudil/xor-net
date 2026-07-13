//! Multi-head causal self-attention with fused ternary QKV projection and a
//! ring KV-cache (`fast_attention`).

use crate::tensor::FastTensor;
use crate::nn::{DynamicLinear, FastRmsNorm};
use crate::nn::dynamic_linear::LinearKind;
use crate::loader::{SafeTensorLoader, sparse_loader::SparseFile};
use crate::models::llama::{Cache, Config};
use anyhow::Result;

/// Issue a non-temporal prefetch for the leading cache line of a projection's
/// packed ternary weights (no-op for non-ternary projections).
fn prefetch_proj(lin: &DynamicLinear) {
    if let LinearKind::Ternary(t) = &lin.inner {
        if let Some(ptr) = t.prefetch_ptr() {
            unsafe { crate::models::llama::prefetch_weight(ptr); }
        }
    }
}

/// Bytes of packed ternary weight memory for a projection (0 if not ternary).
fn proj_bytes(lin: &DynamicLinear) -> usize {
    match &lin.inner {
        LinearKind::Ternary(t) => t.packed_bytes(),
        _ => 0,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CausalSelfAttention {
    q_proj: DynamicLinear,
    k_proj: DynamicLinear,
    v_proj: DynamicLinear,
    o_proj: DynamicLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    inner_attn_ln: Option<FastRmsNorm>,
}

impl CausalSelfAttention {
    pub(crate) fn forward(
        &self,
        x: &FastTensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<FastTensor> {
        let rank = x.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let b_size: usize = x.shape[..rank - 1].iter().product();

        // Fast path: when Q/K/V are ternary and we decode a single token we can
        // quantize the input once and compute all three projections in a single
        // fused call (no per-projection DynamicLinear dispatch / re-quant).
        let (q, k, v) = match (&self.q_proj.inner, &self.k_proj.inner, &self.v_proj.inner) {
            (LinearKind::Ternary(q_lin), LinearKind::Ternary(k_lin), LinearKind::Ternary(v_lin))
                if b_size == 1 =>
            {
                let in_row = &x.data[0..q_lin.in_dim];
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(q_lin.in_dim);
                let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(in_row, &mut quantized_in);
                let res = crate::bit1_58::layers::TernaryLinear::fused_forward_qkv(
                    x,
                    &quantized_in,
                    inv_scale,
                    q_lin,
                    k_lin,
                    v_lin,
                );
                crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
                res
            }
            _ => {
                let q = self.q_proj.forward(x)?;
                let k = self.k_proj.forward(x)?;
                let v = self.v_proj.forward(x)?;
                (q, k, v)
            }
        };

        let q = q.transpose_seq_to_heads(self.num_attention_heads, self.head_dim)?;
        let k = k.transpose_seq_to_heads(self.num_key_value_heads, self.head_dim)?;
        let v = v.transpose_seq_to_heads(self.num_key_value_heads, self.head_dim)?;

        let q = q.rope_inplace(&cache.cos, &cache.sin, index_pos)?;
        let k = k.rope_inplace(&cache.cos, &cache.sin, index_pos)?;

        if let Some(ref mut ring_cache) = cache.ring_cache {
            let y = crate::nn::fast_attention(
                &q,
                &k,
                &v,
                block_idx,
                index_pos,
                ring_cache,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )?;
            let y = y.transpose_heads_to_seq()?;
            let y = match &self.inner_attn_ln {
                Some(ln) => ln.forward(&y)?,
                None => y,
            };
            let y = self.o_proj.forward(&y)?;
            return Ok(y);
        }

        anyhow::bail!("KV Cache is required for FastTensor inference");
    }

    /// Prefetch the leading cache lines of all four projections' weights so the
    /// next block's weight stream can begin while this block is still computing.
    pub(crate) fn prefetch_weights(&self) {
        prefetch_proj(&self.q_proj);
        prefetch_proj(&self.k_proj);
        prefetch_proj(&self.v_proj);
        prefetch_proj(&self.o_proj);
    }

    /// Bytes of packed ternary weight memory owned by this attention block
    /// (used for the per-token bandwidth estimate).
    pub(crate) fn weight_bytes(&self) -> usize {
        proj_bytes(&self.q_proj)
            + proj_bytes(&self.k_proj)
            + proj_bytes(&self.v_proj)
            + proj_bytes(&self.o_proj)
    }

    pub(crate) fn load(loader: SafeTensorLoader, cfg: &Config, sparse: Option<&SparseFile>) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let q_proj =
            DynamicLinear::load(size_in, size_q, &loader.pp("q_proj"), "weight", cfg.quantization_config, sparse)?;
        let k_proj =
            DynamicLinear::load(size_in, size_kv, &loader.pp("k_proj"), "weight", cfg.quantization_config, sparse)?;
        let v_proj =
            DynamicLinear::load(size_in, size_kv, &loader.pp("v_proj"), "weight", cfg.quantization_config, sparse)?;
        let o_proj =
            DynamicLinear::load(size_q, size_in, &loader.pp("o_proj"), "weight", cfg.quantization_config, sparse)?;

        let inner_attn_ln = if loader.pp("inner_attn_ln").has_tensor("weight") {
            Some(FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("inner_attn_ln"))?)
        } else if loader.pp("attn_sub_norm").has_tensor("weight") {
            Some(FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("attn_sub_norm"))?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            inner_attn_ln,
        })
    }
}
