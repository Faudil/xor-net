//! Reference (oracle) implementation of the dense Bonsai / Qwen3 model in
//! candle: every projection dequantizes its GGUF Q1_0 weights to f32 and
//! runs an f32 matmul. Very slow (~0.15 tok/s on Bonsai-1.7B) but simple;
//! used only to validate `xor_net::models::q1::qwen3` (tests/bonsai_reference.rs).
//!
//! Formerly `src/models/bonsai.rs`, with two fixes so it follows HF Qwen3
//! exactly: Q/K RMS norms are applied *before* RoPE, and YaRN rope scaling
//! from the GGUF metadata is honoured.
#![allow(dead_code)]

use std::sync::Arc;

use candle_core::Result as CandleResult;
use candle_core::shape::D;
use candle_core::{Device, Tensor};

use xor_net::loader::gguf;
use xor_net::loader::gguf::{BonsaiConfig, GgufFile};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let x_dtype = x.dtype();
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(candle_core::DType::F32)?;
    let norm_x = (x.powf(2.0)?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&norm_x.affine(1.0, eps)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)
}

fn softmax(x: &Tensor, dim: usize) -> CandleResult<Tensor> {
    let max = x.max_keepdim(dim)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    Ok(exp.broadcast_div(&sum)?)
}

fn silu(x: &Tensor) -> CandleResult<Tensor> {
    x.silu()
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, seq_pos: usize) -> CandleResult<Tensor> {
    let (b_sz, seq_len, n_head, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos_slice = cos
        .narrow(0, seq_pos, seq_len)?
        .reshape((seq_len, 1, half))?;
    let sin_slice = sin
        .narrow(0, seq_pos, seq_len)?
        .reshape((seq_len, 1, half))?;
    let m1 = x1.broadcast_mul(&cos_slice)?;
    let m2 = x2.broadcast_mul(&sin_slice)?;
    let m3 = x1.broadcast_mul(&sin_slice)?;
    let m4 = x2.broadcast_mul(&cos_slice)?;
    let part0 = (m1 - m2)?;
    let part1 = (m3 + m4)?;
    let rotated = Tensor::cat(&[&part0, &part1], D::Minus1)?;
    rotated.reshape((b_sz, seq_len, n_head, head_dim))
}

fn matmul_proj(
    hidden: &Tensor,
    weight_f32: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
) -> CandleResult<Tensor> {
    let device = hidden.device();
    let w = Tensor::from_vec(weight_f32, (out_dim, in_dim), device)?;
    let wt = w.t()?;
    hidden.broadcast_matmul(&wt)
}

// ---------------------------------------------------------------------------
// Projection handle: reads + dequants from GGUF on demand
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ProjWeights {
    gguf: Arc<GgufFile>,
    name: String,
    in_dim: usize,
    out_dim: usize,
}

impl ProjWeights {
    fn load_f32(&self) -> CandleResult<Vec<f32>> {
        let name = &self.name;
        let bytes = self
            .gguf
            .tensor_bytes(name)
            .map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?;
        let info = self
            .gguf
            .tensor_info(name)
            .map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?;
        gguf::dequantize_data(bytes, info.dtype_u32, self.out_dim * self.in_dim)
            .map_err(|e| candle_core::Error::Msg(format!("{e:?}")))
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Embed {
    weight: Tensor,
}

impl Embed {
    fn forward(&self, tokens: &[u32]) -> CandleResult<Tensor> {
        let device = self.weight.device();
        let ids_len = tokens.len();
        let ids = match ids_len {
            0 => Tensor::from_vec(Vec::<u32>::new(), 0, device),
            _ => Tensor::from_vec(tokens.to_vec(), ids_len, device),
        }?;
        let out = self.weight.embedding(&ids)?;
        out.unsqueeze(0) // [1, seq, hidden]
    }
}

// ---------------------------------------------------------------------------
// Causal self-attention (standard full attention)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CausalSelfAttention {
    q_proj: ProjWeights,
    k_proj: ProjWeights,
    v_proj: ProjWeights,
    o_proj: ProjWeights,
    q_norm: ProjWeights,
    k_norm: ProjWeights,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_norm_eps: f64,
}

impl CausalSelfAttention {
    fn forward(
        &self,
        x: &Tensor,
        seq_pos: usize,
        _cos: &Tensor,
        _sin: &Tensor,
        cache: &mut (Tensor, Tensor),
    ) -> CandleResult<Tensor> {
        let b_sz = 1;
        let in_dim = x.dim(D::Minus1)?;
        let seq_len = x.dim(1)?;

        // Project
        let q = matmul_proj(
            x,
            self.q_proj.load_f32()?,
            self.n_head * self.head_dim,
            in_dim,
        )?;
        let k = matmul_proj(
            x,
            self.k_proj.load_f32()?,
            self.n_kv_head * self.head_dim,
            in_dim,
        )?;
        let v = matmul_proj(
            x,
            self.v_proj.load_f32()?,
            self.n_kv_head * self.head_dim,
            in_dim,
        )?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;

        // Q/K norms (RMS norm over head_dim), then RoPE (HF Qwen3Attention).
        let q_norm_f32 = self.q_norm.load_f32()?;
        let k_norm_f32 = self.k_norm.load_f32()?;
        let q_norm_w = Tensor::from_vec(q_norm_f32, self.head_dim, x.device())?;
        let k_norm_w = Tensor::from_vec(k_norm_f32, self.head_dim, x.device())?;
        let q = rms_norm(&q, &q_norm_w, self.rms_norm_eps)?;
        let k = rms_norm(&k, &k_norm_w, self.rms_norm_eps)?;

        let q = apply_rotary(&q, _cos, _sin, seq_pos)?;
        let k = apply_rotary(&k, _cos, _sin, seq_pos)?;

        // Transpose to [batch, n_heads, seq, head_dim] for cache and attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // KV cache: concatenate on seq dimension
        let k = Tensor::cat(&[&cache.0, &k], 2)?;
        let v = Tensor::cat(&[&cache.1, &v], 2)?;
        cache.0 = k.clone();
        cache.1 = v.clone();

        let kv_len = k.dim(2)?;

        // Scaled dot-product attention with GQA
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_out = if self.n_head == self.n_kv_head {
            let scores = q.matmul(&k.transpose(2, 3)?)?;
            let scores = scores.affine(scale, 0.0)?;
            let mask = make_causal_mask(seq_len, kv_len, seq_pos, q.device())?;
            let scores = scores.broadcast_add(&mask)?;
            let attn = softmax(&scores, 3)?;
            attn.matmul(&v)?
        } else {
            let n_groups = self.n_head / self.n_kv_head;
            let k = k
                .unsqueeze(2)?
                .expand((b_sz, self.n_kv_head, n_groups, kv_len, self.head_dim))?
                .reshape((b_sz, self.n_head, kv_len, self.head_dim))?;
            let v = v
                .unsqueeze(2)?
                .expand((b_sz, self.n_kv_head, n_groups, kv_len, self.head_dim))?
                .reshape((b_sz, self.n_head, kv_len, self.head_dim))?;
            let scores = q.matmul(&k.transpose(2, 3)?)?;
            let scores = scores.affine(scale, 0.0)?;
            let mask = make_causal_mask(seq_len, kv_len, seq_pos, q.device())?;
            let scores = scores.broadcast_add(&mask)?;
            let attn = softmax(&scores, 3)?;
            attn.matmul(&v)?
        };

        attn_out =
            attn_out
                .transpose(1, 2)?
                .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        matmul_proj(
            &attn_out,
            self.o_proj.load_f32()?,
            in_dim,
            self.n_head * self.head_dim,
        )
    }
}

fn make_causal_mask(
    seq_len: usize,
    kv_len: usize,
    _seq_pos: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    if seq_len == 1 {
        return Tensor::zeros((1, 1, 1, kv_len), candle_core::DType::F32, device);
    }
    let prior_count = kv_len - seq_len;
    let data: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let max_visible = prior_count + i;
            (0..kv_len).map(move |j| {
                if j <= max_visible {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, seq_len, kv_len), device)
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Mlp {
    gate_proj: ProjWeights,
    up_proj: ProjWeights,
    down_proj: ProjWeights,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let in_dim = x.dim(D::Minus1)?;
        let intermediate = self.gate_proj.out_dim;

        let gate = matmul_proj(x, self.gate_proj.load_f32()?, intermediate, in_dim)?;
        let up = matmul_proj(x, self.up_proj.load_f32()?, intermediate, in_dim)?;
        let gate = silu(&gate)?;
        let hidden = (gate * up)?;
        matmul_proj(&hidden, self.down_proj.load_f32()?, in_dim, intermediate)
    }
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Block {
    attn_norm: ProjWeights,
    ffn_norm: ProjWeights,
    attention: CausalSelfAttention,
    mlp: Mlp,
    rms_norm_eps: f64,
    hidden_size: usize,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        seq_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut (Tensor, Tensor),
    ) -> CandleResult<Tensor> {
        let device = x.device();

        // Pre-attention norm
        let attn_norm_w = Tensor::from_vec(self.attn_norm.load_f32()?, self.hidden_size, device)?;
        let residual = x.clone();
        let x = rms_norm(x, &attn_norm_w, self.rms_norm_eps)?;

        let attn_out = self.attention.forward(&x, seq_pos, cos, sin, cache)?;
        let x = (residual + attn_out)?;

        // Pre-FFN norm
        let ffn_norm_w = Tensor::from_vec(self.ffn_norm.load_f32()?, self.hidden_size, device)?;
        let residual = x.clone();
        let x = rms_norm(&x, &ffn_norm_w, self.rms_norm_eps)?;

        let mlp_out = self.mlp.forward(&x)?;
        Ok((residual + mlp_out)?)
    }
}

// ---------------------------------------------------------------------------
// Top-level Bonsai model
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct BonsaiModel {
    #[allow(dead_code)]
    gguf: Arc<GgufFile>,
    pub config: BonsaiConfig,
    token_embd: Embed,
    output_norm: ProjWeights,
    lm_head: ProjWeights,
    blocks: Vec<Block>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl BonsaiModel {
    pub fn from_gguf(
        gguf: Arc<GgufFile>,
        max_seq_len: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let config = BonsaiConfig::from_gguf_metadata(&gguf.metadata.metadata)?;

        let emb_info = gguf.tensor_info("token_embd.weight")?;
        let emb_f32 = gguf::dequantize_data(
            gguf.tensor_bytes("token_embd.weight")?,
            emb_info.dtype_u32,
            emb_info.elem_count,
        )?;
        let emb_weight = Tensor::from_vec(emb_f32, emb_info.shape.clone(), device)?;
        let token_embd = Embed { weight: emb_weight };

        let output_norm = ProjWeights {
            gguf: gguf.clone(),
            name: "output_norm.weight".to_string(),
            in_dim: config.hidden_size,
            out_dim: config.hidden_size,
        };

        let lm_head = if gguf.has_tensor("output.weight") {
            ProjWeights {
                gguf: gguf.clone(),
                name: "output.weight".to_string(),
                in_dim: config.hidden_size,
                out_dim: config.vocab_size,
            }
        } else {
            ProjWeights {
                gguf: gguf.clone(),
                name: "token_embd.weight".to_string(),
                in_dim: config.hidden_size,
                out_dim: config.vocab_size,
            }
        };

        let head_dim = config.head_dim;
        let q_out = config.num_attention_heads * head_dim;
        let k_out = config.num_key_value_heads * head_dim;
        let v_out = config.num_key_value_heads * head_dim;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("blk.{i}");
            blocks.push(Block {
                attn_norm: ProjWeights {
                    gguf: gguf.clone(),
                    name: format!("{prefix}.attn_norm.weight"),
                    in_dim: config.hidden_size,
                    out_dim: config.hidden_size,
                },
                ffn_norm: ProjWeights {
                    gguf: gguf.clone(),
                    name: format!("{prefix}.ffn_norm.weight"),
                    in_dim: config.hidden_size,
                    out_dim: config.hidden_size,
                },
                attention: CausalSelfAttention {
                    q_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_q.weight"),
                        in_dim: config.hidden_size,
                        out_dim: q_out,
                    },
                    k_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_k.weight"),
                        in_dim: config.hidden_size,
                        out_dim: k_out,
                    },
                    v_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_v.weight"),
                        in_dim: config.hidden_size,
                        out_dim: v_out,
                    },
                    o_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_output.weight"),
                        in_dim: q_out,
                        out_dim: config.hidden_size,
                    },
                    q_norm: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_q_norm.weight"),
                        in_dim: head_dim,
                        out_dim: head_dim,
                    },
                    k_norm: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.attn_k_norm.weight"),
                        in_dim: head_dim,
                        out_dim: head_dim,
                    },
                    n_head: config.num_attention_heads,
                    n_kv_head: config.num_key_value_heads,
                    head_dim,
                    rms_norm_eps: config.rms_norm_eps,
                },
                mlp: Mlp {
                    gate_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.ffn_gate.weight"),
                        in_dim: config.hidden_size,
                        out_dim: config.intermediate_size,
                    },
                    up_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.ffn_up.weight"),
                        in_dim: config.hidden_size,
                        out_dim: config.intermediate_size,
                    },
                    down_proj: ProjWeights {
                        gguf: gguf.clone(),
                        name: format!("{prefix}.ffn_down.weight"),
                        in_dim: config.intermediate_size,
                        out_dim: config.hidden_size,
                    },
                },
                rms_norm_eps: config.rms_norm_eps,
                hidden_size: config.hidden_size,
            });
        }

        let yarn = yarn_params(&gguf, &config.architecture);
        let (cos, sin) = precompute_rope(
            config.head_dim * config.num_attention_heads,
            config.num_attention_heads,
            max_seq_len.min(config.max_position_embeddings),
            config.rope_theta as f64,
            yarn,
            device,
        )?;

        Ok(BonsaiModel {
            gguf,
            config,
            token_embd,
            output_norm,
            lm_head,
            blocks,
            cos,
            sin,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        index_pos: usize,
        cache: &mut BonsaiCache,
    ) -> CandleResult<Tensor> {
        let x = self.token_embd.forward(tokens)?;
        self.forward_embed(&x, index_pos, cache)
    }

    pub fn forward_embed(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut BonsaiCache,
    ) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, &self.cos, &self.sin, &mut cache.kv_caches[i])?;
        }

        let norm_w = Tensor::from_vec(
            self.output_norm.load_f32()?,
            self.config.hidden_size,
            &self.device,
        )?;
        x = rms_norm(&x, &norm_w, self.config.rms_norm_eps)?;

        let last = x.narrow(1, x.dim(1)? - 1, 1)?.squeeze(1)?;

        matmul_proj(
            &last,
            self.lm_head.load_f32()?,
            self.config.vocab_size,
            self.config.hidden_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Pre-compute RoPE
// ---------------------------------------------------------------------------

/// YaRN `(factor, original_context)` from `{arch}.rope.scaling.*`.
fn yarn_params(g: &GgufFile, arch: &str) -> Option<(f64, f64)> {
    let ty = g
        .get_metadata(&format!("{arch}.rope.scaling.type"))?
        .to_string()?;
    if ty != "yarn" {
        return None;
    }
    let factor = g
        .get_metadata(&format!("{arch}.rope.scaling.factor"))?
        .to_f32()? as f64;
    let orig = g
        .get_metadata(&format!("{arch}.rope.scaling.original_context_length"))?
        .to_u32()? as f64;
    Some((factor, orig))
}

/// cos/sin tables `[max_seq_len, head_dim/2]`, following HF
/// `_compute_default_rope_parameters` / `_compute_yarn_parameters`
/// (beta_fast = 32, beta_slow = 1, truncate = true).
fn precompute_rope(
    hidden_size: usize,
    n_head: usize,
    max_seq_len: usize,
    theta: f64,
    yarn: Option<(f64, f64)>,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let head_dim = hidden_size / n_head;
    let half = head_dim / 2;
    let dim = head_dim as f64;
    let pos_freqs: Vec<f64> = (0..half)
        .map(|i| theta.powf((2 * i) as f64 / dim))
        .collect();
    let (inv_freq, attention_factor): (Vec<f64>, f64) = match yarn {
        Some((factor, orig)) if factor > 1.0 => {
            let corr = |rot: f64| {
                dim * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln())
            };
            let low = corr(32.0).floor().max(0.0);
            let high = corr(1.0).ceil().min(dim - 1.0);
            let high = if low == high { high + 0.001 } else { high };
            let inv = (0..half)
                .map(|i| {
                    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrapolation_factor = 1.0 - ramp;
                    let interp = 1.0 / (factor * pos_freqs[i]);
                    let extrap = 1.0 / pos_freqs[i];
                    interp * (1.0 - extrapolation_factor) + extrap * extrapolation_factor
                })
                .collect();
            (inv, 0.1 * factor.ln() + 1.0)
        }
        _ => (pos_freqs.iter().map(|p| 1.0 / p).collect(), 1.0),
    };

    let mut cos_data = vec![0.0f32; max_seq_len * half];
    let mut sin_data = vec![0.0f32; max_seq_len * half];
    for pos in 0..max_seq_len {
        for i in 0..half {
            let freq = pos as f64 * inv_freq[i];
            cos_data[pos * half + i] = (freq.cos() * attention_factor) as f32;
            sin_data[pos * half + i] = (freq.sin() * attention_factor) as f32;
        }
    }

    let cos = Tensor::from_vec(cos_data, (max_seq_len, half), device)?;
    let sin = Tensor::from_vec(sin_data, (max_seq_len, half), device)?;
    Ok((cos, sin))
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

pub struct BonsaiCache {
    pub kv_caches: Vec<(Tensor, Tensor)>,
}

impl BonsaiCache {
    pub fn new(config: &BonsaiConfig, device: &Device) -> CandleResult<Self> {
        let head_dim = config.head_dim;
        let kv_caches = (0..config.num_hidden_layers)
            .map(|_| {
                let k = Tensor::zeros(
                    (1, config.num_key_value_heads, 0, head_dim),
                    candle_core::DType::F32,
                    device,
                )?;
                let v = Tensor::zeros(
                    (1, config.num_key_value_heads, 0, head_dim),
                    candle_core::DType::F32,
                    device,
                )?;
                Ok((k, v))
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(BonsaiCache { kv_caches })
    }
}

// ---------------------------------------------------------------------------
// Non-regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_make_causal_mask_prefill() {
        let device = &Device::Cpu;
        // seq_len=5, kv_len=5 (prefill with no prior cache)
        let mask = make_causal_mask(5, 5, 0, device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 5, 5]);
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..5 {
            for j in 0..5 {
                let val = data[i * 5 + j];
                if j <= i {
                    assert_eq!(val, 0.0, "mask[{i},{j}] should be 0.0 for prefill");
                } else {
                    assert!(
                        val.is_infinite() && val < 0.0,
                        "mask[{i},{j}] should be -inf for prefill, got {val}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_make_causal_mask_decode() {
        let device = &Device::Cpu;
        // seq_len=1, kv_len=6 (decode step with 5 cached + 1 new)
        let mask = make_causal_mask(1, 6, 5, device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 6]);
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Decode: all positions visible
        for j in 0..6 {
            assert_eq!(data[j], 0.0, "decode mask[{j}] should be 0.0");
        }
    }

    #[test]
    fn test_make_causal_mask_extend() {
        let device = &Device::Cpu;
        // seq_len=3, kv_len=5 (extending prompt with 2 prior tokens)
        let mask = make_causal_mask(3, 5, 2, device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 3, 5]);
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Position 0 of new tokens (cache pos 2): can see 0..2 (prior 0,1 + itself at 2)
        // Position 1 of new tokens (cache pos 3): can see 0..3
        // Position 2 of new tokens (cache pos 4): can see 0..4
        let expectations: [[f32; 5]; 3] = [
            [0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY], // i=0 sees 0..2
            [0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY],               // i=1 sees 0..3
            [0.0, 0.0, 0.0, 0.0, 0.0],                             // i=2 sees all 0..4
        ];
        for i in 0..3 {
            for j in 0..5 {
                let val = data[i * 5 + j];
                let exp = expectations[i][j];
                if exp == 0.0 {
                    assert_eq!(val, 0.0, "extend mask[{i},{j}] should be 0.0");
                } else {
                    assert!(
                        val.is_infinite() && val < 0.0,
                        "extend mask[{i},{j}] should be -inf, got {val}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_rms_norm() {
        let device = &Device::Cpu;
        // Simple 2D case: [1, 4] -> normalize over last dim
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), device).unwrap();
        let weight = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], 4, device).unwrap();
        let eps = 1e-6_f64;
        let result = rms_norm(&x, &weight, eps).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // mean = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5 + 1e-6) ≈ 2.7386
        // normalized = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386] * 2
        let rms = (7.5f64 + eps).sqrt();
        for (i, &v) in data.iter().enumerate() {
            let expected = ((i as f64 + 1.0) / rms * 2.0) as f32;
            assert!(
                (v - expected).abs() < 1e-3,
                "rms_norm[{i}]: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_rms_norm_broadcast() {
        let device = &Device::Cpu;
        // 3D case: [1, 2, 3] -> normalize over last dim
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3), device).unwrap();
        let weight = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], 3, device).unwrap();
        let eps = 1e-6_f64;
        let result = rms_norm(&x, &weight, eps).unwrap();
        let data: Vec<Vec<f32>> = result.squeeze(0).unwrap().to_vec2().unwrap();
        // Position 0: [1,2,3], mean=(1+4+9)/3 = 14/3 ≈ 4.6667, rms ≈ 2.1602
        // normalized ≈ [0.4629, 0.9258, 1.3887]
        let rms0 = ((14.0f64 / 3.0) + eps).sqrt();
        let expected0: Vec<f32> = [1.0, 2.0, 3.0]
            .iter()
            .map(|&v| (v as f64 / rms0) as f32)
            .collect();
        // Position 1: [4,5,6], mean=(16+25+36)/3 = 77/3 ≈ 25.6667, rms ≈ 5.0662
        // normalized ≈ [0.7895, 0.9869, 1.1843]
        let rms1 = ((77.0f64 / 3.0) + eps).sqrt();
        let expected1: Vec<f32> = [4.0, 5.0, 6.0]
            .iter()
            .map(|&v| (v as f64 / rms1) as f32)
            .collect();
        for j in 0..3 {
            assert!(
                (data[0][j] - expected0[j]).abs() < 1e-3,
                "rms_norm bcast pos0[{j}]: got {}, expected {}",
                data[0][j],
                expected0[j]
            );
            assert!(
                (data[1][j] - expected1[j]).abs() < 1e-3,
                "rms_norm bcast pos1[{j}]: got {}, expected {}",
                data[1][j],
                expected1[j]
            );
        }
    }

    #[test]
    fn test_softmax() {
        let device = &Device::Cpu;
        // 2D case: [1, 4] -> softmax over dim 1
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), device).unwrap();
        let result = softmax(&x, 1).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Manual softmax: e^1/(e^1+e^2+e^3+e^4) ≈ 0.0321, 0.0871, 0.2369, 0.6439
        let exp_sum = 1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp() + 4.0f64.exp();
        for (i, &v) in data.iter().enumerate() {
            let expected = ((i as f64 + 1.0).exp() / exp_sum) as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "softmax[{i}]: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_masked() {
        let device = &Device::Cpu;
        // 2D case with -inf masking: first position is the only valid one
        let x = Tensor::from_vec(
            vec![
                5.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            (1, 4),
            device,
        )
        .unwrap();
        let result = softmax(&x, 1).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Only position 0 is valid → softmax = [1, 0, 0, 0]
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "softmax masked[0]: got {}",
            data[0]
        );
        for i in 1..4 {
            assert_eq!(
                data[i], 0.0,
                "softmax masked[{i}]: should be 0, got {}",
                data[i]
            );
        }
    }

    #[test]
    fn test_dequant_upstream_roundtrip() {
        // Verify upstream Q1_0 dequant produces expected block values
        let elem_count = 256; // 2 blocks of 128
        let mut data = vec![0u8; (elem_count / 128) * 18];
        // Block 0: scale=1.0 (0x3c00 LE), first 8 bits=0x55 (10101010...)
        data[0] = 0x00;
        data[1] = 0x3c; // fp16 1.0
        data[2] = 0x55; // bits: 01010101...
        // Block 1: scale=2.0 (0x4000 LE), first 8 bits=0xFF (11111111...)
        data[18] = 0x00;
        data[19] = 0x40; // fp16 2.0
        data[20] = 0xFF; // bits: 11111111...

        let out = xor_net::loader::gguf::dequantize_q1_0_upstream(&data, elem_count);
        assert_eq!(out.len(), elem_count);

        // Block 0: scale=1.0, bit=1 → +1.0, bit=0 → -1.0 (ggml convention)
        // 0x55 = 01010101, LSB first: bit0=1, bit1=0, bit2=1, bit3=0, bit4=1, bit5=0, bit6=1, bit7=0
        // → +1, -1, +1, -1, +1, -1, +1, -1
        for i in 0..8 {
            let expected = if (i % 2) == 0 { 1.0 } else { -1.0 };
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "block0[{i}]: got {}, expected {}",
                out[i],
                expected
            );
        }

        // Block 1: scale=2.0, 0xFF on first byte → all bits=1 → all +2.0
        for i in 128..136 {
            assert!(
                (out[i] - 2.0).abs() < 1e-6,
                "block1[{i}]: got {}, expected 2.0",
                out[i]
            );
        }
    }

    #[test]
    fn test_precompute_rope_inv_freq() {
        let device = &Device::Cpu;
        let (cos, sin) = precompute_rope(2048, 16, 10, 1_000_000.0, None, device).unwrap();
        // cos shape: [10, 64] (max_seq_len=10, half=64)
        assert_eq!(cos.dims(), &[10, 64]);
        assert_eq!(sin.dims(), &[10, 64]);

        // Position 0: cos=1, sin=0 for all pairs
        let cos0: Vec<f32> = cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let sin0: Vec<f32> = sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for k in 0..64 {
            assert!(
                (cos0[k] - 1.0).abs() < 1e-5,
                "cos[0,{k}] should be ~1.0, got {}",
                cos0[k]
            );
            assert!(
                (sin0[k] - 0.0).abs() < 1e-5,
                "sin[0,{k}] should be ~0.0, got {}",
                sin0[k]
            );
        }

        // Position 1: verify theta computation: theta^(-2k/d) for k=0..63
        let theta = 1_000_000.0f64;
        let head_dim = 128.0;
        let cos1: Vec<f32> = cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let sin1: Vec<f32> = sin
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for k in 0..64 {
            let inv_freq = 1.0 / theta.powf((2 * k) as f64 / head_dim);
            let expected_cos = (1.0f64 * inv_freq).cos() as f32;
            let expected_sin = (1.0f64 * inv_freq).sin() as f32;
            assert!(
                (cos1[k] - expected_cos).abs() < 1e-5,
                "cos[1,{k}]: got {}, expected {}",
                cos1[k],
                expected_cos
            );
            assert!(
                (sin1[k] - expected_sin).abs() < 1e-5,
                "sin[1,{k}]: got {}, expected {}",
                sin1[k],
                expected_sin
            );
        }
    }
}
