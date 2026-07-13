//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! The transformer is split into focused submodules:
//! - [`attention`] — `CausalSelfAttention` (QKV/O projections + KV-cache attention)
//! - [`mlp`] — `Mlp` (gate/up/down projections, fused forward selection)
//! - [`block`] — `Block` (pre-norm residual wrapper around attention + MLP)

pub mod attention;
pub mod block;
pub mod mlp;

use crate::tensor::FastTensor;
use crate::nn::{DynamicLinear, QuantizationConfig, FastRmsNorm, CpuRingCache};
use crate::loader::SafeTensorLoader;
use std::f32::consts::PI;

use crate::nn::dynamic_linear::LinearKind;
use rayon::prelude::*;

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

// ---------------------------------------------------------------------------
// RoPE configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Silu,
    Relu2,
}

impl Default for Activation {
    fn default() -> Self { Self::Silu }
}

impl Activation {
    pub fn from_str(s: &str) -> Self {
        match s {
            "relu2" => Activation::Relu2,
            _ => Activation::Silu,
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            quantization_config: QuantizationConfig::None,
            hidden_act: self.hidden_act.as_deref().map(Activation::from_str).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub quantization_config: QuantizationConfig,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub hidden_act: Activation,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            quantization_config: QuantizationConfig::None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
            hidden_act: Activation::default(),
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            quantization_config: QuantizationConfig::None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
            hidden_act: Activation::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// KV-cache + RoPE tables
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Cache {
    pub use_kv_cache: bool,
    pub ring_cache: Option<CpuRingCache>,
    pub cos: FastTensor,
    pub sin: FastTensor,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, config: &Config) -> anyhow::Result<Self> {
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let head_dim = config.hidden_size / config.num_attention_heads;
        let half_dim = head_dim / 2;
        let mut idx_theta_data = vec![0.0f32; config.max_position_embeddings * half_dim];
        for pos in 0..config.max_position_embeddings {
            for i in 0..half_dim {
                idx_theta_data[pos * half_dim + i] = pos as f32 * theta[i];
            }
        }

        let cos_data = idx_theta_data.iter().map(|&x| x.cos()).collect::<Vec<_>>();
        let sin_data = idx_theta_data.iter().map(|&x| x.sin()).collect::<Vec<_>>();

        let cos = FastTensor::new(cos_data, vec![config.max_position_embeddings, half_dim]);
        let sin = FastTensor::new(sin_data, vec![config.max_position_embeddings, half_dim]);

        let ring_cache = if use_kv_cache {
            let num_kv_heads = config.num_key_value_heads;
            let head_dim = config.hidden_size / config.num_attention_heads;
            Some(CpuRingCache::new(
                config.num_hidden_layers,
                num_kv_heads,
                config.max_position_embeddings,
                head_dim,
                true,
            ))
        } else {
            None
        };

        Ok(Self {
            use_kv_cache,
            ring_cache,
            cos,
            sin,
        })
    }
}

// ---------------------------------------------------------------------------
// Token embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FastEmbedding {
    pub weight: FastTensor,
}

impl FastEmbedding {
    pub fn new(weight: FastTensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, x: &[u32]) -> anyhow::Result<FastTensor> {
        FastTensor::embedding(x, &self.weight)
    }
}

// ---------------------------------------------------------------------------
// Per-block profiling counters (read by the chat example breakdown)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};

pub static TIME_BLOCKS: AtomicU64 = AtomicU64::new(0);
pub static TIME_LM_HEAD: AtomicU64 = AtomicU64::new(0);
pub static TIME_OTHER: AtomicU64 = AtomicU64::new(0);
/// Full `CausalSelfAttention::forward` time.
pub static TIME_ATTN_GEMV: AtomicU64 = AtomicU64::new(0);
/// Reserved (unused) — kept for API stability.
pub static TIME_ATTN_MATH: AtomicU64 = AtomicU64::new(0);
/// `gate+up+silu+down` total (fused path).
pub static TIME_MLP_GEMV: AtomicU64 = AtomicU64::new(0);
/// Just `down_proj` (the prefill / non-fused fallback path).
pub static TIME_MLP_DOWN: AtomicU64 = AtomicU64::new(0);
/// RMS norms + residual additions.
pub static TIME_NORM: AtomicU64 = AtomicU64::new(0);

/// Human-readable tag for a `LinearKind`, used by the `XORNET_DEBUG` diagnostic.
pub(crate) fn linear_kind_tag(k: &LinearKind) -> &'static str {
    match k {
        LinearKind::Standard(_) => "F32",
        LinearKind::Int8(_) => "Int8",
        LinearKind::Int4(_) => "Int4",
        LinearKind::Ternary(_) => "Ternary",
        LinearKind::Bit(_) => "Bit",
    }
}

pub fn get_mlp_silu_time() -> u64 {
    crate::bit1_58::layers::TIME_SILU.load(Ordering::Relaxed)
}

pub fn get_profiling_stats() -> (u64, u64, u64) {
    (
        TIME_BLOCKS.load(Ordering::Relaxed),
        TIME_LM_HEAD.load(Ordering::Relaxed),
        TIME_OTHER.load(Ordering::Relaxed),
    )
}

pub fn get_detailed_stats() -> (u64, u64, u64, u64, u64) {
    (
        TIME_ATTN_GEMV.load(Ordering::Relaxed),
        TIME_ATTN_MATH.load(Ordering::Relaxed),
        TIME_MLP_GEMV.load(Ordering::Relaxed),
        TIME_MLP_DOWN.load(Ordering::Relaxed),
        TIME_NORM.load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

use self::block::Block;

#[derive(Debug, Clone)]
pub struct Llama {
    wte: FastEmbedding,
    blocks: Vec<Block>,
    ln_f: FastRmsNorm,
    lm_head: DynamicLinear,
}

impl Llama {
    pub fn embed(&self, x: &[u32]) -> anyhow::Result<FastTensor> {
        self.wte.forward(x)
    }

    pub fn forward_input_embed(
        &self,
        input_embed: &FastTensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<FastTensor> {
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let last_token_x = x.slice_last_token()?;
        let logits = self.lm_head.forward(&last_token_x)?;
        Ok(logits)
    }

    pub fn forward(&self, x: &[u32], index_pos: usize, cache: &mut Cache) -> anyhow::Result<FastTensor> {
        let start_other = std::time::Instant::now();
        let mut x = self.wte.forward(x)?;
        let other_elapsed1 = start_other.elapsed().as_micros() as u64;
        TIME_OTHER.fetch_add(other_elapsed1, Ordering::Relaxed);

        let start_blocks = std::time::Instant::now();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let blocks_elapsed = start_blocks.elapsed().as_micros() as u64;
        TIME_BLOCKS.fetch_add(blocks_elapsed, Ordering::Relaxed);

        let start_other2 = std::time::Instant::now();
        let x = self.ln_f.forward(&x)?;
        let last_token_x = x.slice_last_token()?;
        let other_elapsed2 = start_other2.elapsed().as_micros() as u64;
        TIME_OTHER.fetch_add(other_elapsed2, Ordering::Relaxed);

        let start_lm_head = std::time::Instant::now();
        let logits = self.lm_head.forward(&last_token_x)?;
        let lm_head_elapsed = start_lm_head.elapsed().as_micros() as u64;
        TIME_LM_HEAD.fetch_add(lm_head_elapsed, Ordering::Relaxed);

        Ok(logits)
    }

    /// Layer-skipping draft forward for self-speculative decoding. Runs only the
    /// specified transformer blocks (e.g. every other layer) and returns logits
    /// for the last token. Cheaper than a full forward; used to propose tokens.
    ///
    /// NOTE: experimental / gated behind `XORNET_SPEC` in the chat example. On a
    /// bandwidth-bound engine the verify pass dominates, so this is slower than
    /// plain greedy decoding (see OPTIMISATION.md).
    pub fn forward_layers(
        &self,
        x: &[u32],
        index_pos: usize,
        cache: &mut Cache,
        layers: &[usize],
    ) -> anyhow::Result<FastTensor> {
        let mut x = self.wte.forward(x)?;
        for &bi in layers {
            x = self.blocks[bi].forward(&x, index_pos, bi, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let last = x.slice_last_token()?;
        let logits = self.lm_head.forward(&last)?;
        Ok(logits)
    }

    /// Full forward returning logits for ALL input positions, shape `[seq, vocab]`.
    /// Used by speculative decoding to verify a drafted sequence in one pass.
    pub fn forward_all(
        &self,
        x: &[u32],
        index_pos: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<FastTensor> {
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn load(loader: &SafeTensorLoader, cfg: &Config) -> anyhow::Result<Self> {
        let wte_weight = loader.get(&[cfg.vocab_size, cfg.hidden_size], "model.embed_tokens.weight")?;
        let wte = FastEmbedding::new(wte_weight);
        let lm_head = if cfg.tie_word_embeddings {
            DynamicLinear::new_int8(&wte.weight.data, cfg.vocab_size, cfg.hidden_size)
        } else {
            DynamicLinear::load(
                cfg.hidden_size,
                cfg.vocab_size,
                &loader.pp("lm_head"),
                "weight",
                cfg.quantization_config,
            )?
        };
        let ln_f = FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("model.norm"))?;
        let blocks: Result<Vec<_>, _> = (0..cfg.num_hidden_layers)
            .into_par_iter()
            .map(|i| Block::load(loader.pp(format!("model.layers.{i}")), cfg))
            .collect();
        let blocks = blocks?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}
