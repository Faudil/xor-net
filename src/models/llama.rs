//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)

use crate::tensor::FastTensor;
use crate::nn::{DynamicLinear, QuantizationConfig, FastRmsNorm, CpuRingCache};
use crate::loader::SafeTensorLoader;
use std::f32::consts::PI;

use crate::nn::dynamic_linear::LinearKind;
use rayon::prelude::*;

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

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

#[derive(Debug, Clone)]
struct CausalSelfAttention {
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
    fn forward(
        &self,
        x: &FastTensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<FastTensor> {
        let rank = x.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let b_size: usize = x.shape[..rank - 1].iter().product();

        let (q, k, v) = match (&self.q_proj.inner, &self.k_proj.inner, &self.v_proj.inner) {
            (LinearKind::Ternary(q_lin), LinearKind::Ternary(k_lin), LinearKind::Ternary(v_lin)) if b_size == 1 => {
                let in_row = &x.data[0 .. q_lin.in_dim];
                let (quantized_in, inv_scale) = crate::bit1_58::quantization::quantize_f32_to_i8(in_row);
                crate::bit1_58::layers::TernaryLinear::fused_forward_qkv(
                    x, &quantized_in, inv_scale, q_lin, k_lin, v_lin,
                )
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
                None => y 
            };
            let y = self.o_proj.forward(&y)?;
            return Ok(y);
        }

        anyhow::bail!("KV Cache is required for FastTensor inference");
    }

    fn load(loader: SafeTensorLoader, cfg: &Config) -> anyhow::Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        
        let q_proj = DynamicLinear::load(size_in, size_q, &loader.pp("q_proj"), "weight", cfg.quantization_config)?;
        let k_proj = DynamicLinear::load(size_in, size_kv, &loader.pp("k_proj"), "weight", cfg.quantization_config)?;
        let v_proj = DynamicLinear::load(size_in, size_kv, &loader.pp("v_proj"), "weight", cfg.quantization_config)?;
        let o_proj = DynamicLinear::load(size_q, size_in, &loader.pp("o_proj"), "weight", cfg.quantization_config)?;
        
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

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: DynamicLinear,
    c_fc2: DynamicLinear,
    c_proj: DynamicLinear,
    ffn_layernorm: Option<FastRmsNorm>,
    hidden_act: Activation,
}

impl Mlp {
    fn forward(&self, x: &FastTensor) -> anyhow::Result<FastTensor> {
        let rank = x.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let b_size: usize = x.shape[..rank - 1].iter().product();

        let (h1, h2) = match (&self.c_fc1.inner, &self.c_fc2.inner) {
            (LinearKind::Ternary(fc1_lin), LinearKind::Ternary(fc2_lin)) if b_size == 1 => {
                let in_row = &x.data[0 .. fc1_lin.in_dim];
                let (quantized_in, inv_scale) = crate::bit1_58::quantization::quantize_f32_to_i8(in_row);
                crate::bit1_58::layers::TernaryLinear::fused_forward_mlp(
                    x, &quantized_in, inv_scale, fc1_lin, fc2_lin,
                )
            }
            _ => {
                let h1 = self.c_fc1.forward(x)?;
                let h2 = self.c_fc2.forward(x)?;
                (h1, h2)
            }
        };

        let x_mul = match self.hidden_act {
            Activation::Relu2 => h1.relu2_mul_inplace(&h2)?,
            Activation::Silu => h1.silu_mul_inplace(&h2)?,
        };
        let x_norm = match &self.ffn_layernorm { 
            Some(ln) => ln.forward(&x_mul)?, 
            None => x_mul 
        };
        self.c_proj.forward(&x_norm)
    }

    fn load(loader: SafeTensorLoader, cfg: &Config) -> anyhow::Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = DynamicLinear::load(h_size, i_size, &loader.pp("gate_proj"), "weight", cfg.quantization_config)?;
        let c_fc2 = DynamicLinear::load(h_size, i_size, &loader.pp("up_proj"), "weight", cfg.quantization_config)?;
        let c_proj = DynamicLinear::load(i_size, h_size, &loader.pp("down_proj"), "weight", cfg.quantization_config)?;
        let ffn_layernorm = if loader.pp("ffn_layernorm").has_tensor("weight") {
            Some(FastRmsNorm::load(i_size, cfg.rms_norm_eps as f32, &loader.pp("ffn_layernorm"))?)
        } else if loader.pp("ffn_sub_norm").has_tensor("weight") {
            Some(FastRmsNorm::load(i_size, cfg.rms_norm_eps as f32, &loader.pp("ffn_sub_norm"))?)
        } else {
            None
        };
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            ffn_layernorm,
            hidden_act: cfg.hidden_act,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: FastRmsNorm,
    attn: CausalSelfAttention,
    rms_2: FastRmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &FastTensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<FastTensor> {
        let x_norm1 = self.rms_1.forward(x)?;
        let attn_out = self.attn.forward(&x_norm1, index_pos, block_idx, cache)?;
        let x_add = attn_out.add_inplace(x)?;
        
        let x_norm2 = self.rms_2.forward(&x_add)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        x_add.add_inplace(&mlp_out)
    }

    fn load(loader: SafeTensorLoader, cfg: &Config) -> anyhow::Result<Self> {
        let attn = CausalSelfAttention::load(loader.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(loader.pp("mlp"), cfg)?;
        let rms_1 = FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("input_layernorm"))?;
        let rms_2 = FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("post_attention_layernorm"))?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

pub static TIME_BLOCKS: AtomicU64 = AtomicU64::new(0);
pub static TIME_LM_HEAD: AtomicU64 = AtomicU64::new(0);
pub static TIME_OTHER: AtomicU64 = AtomicU64::new(0);

pub fn get_profiling_stats() -> (u64, u64, u64) {
    (
        TIME_BLOCKS.load(Ordering::Relaxed),
        TIME_LM_HEAD.load(Ordering::Relaxed),
        TIME_OTHER.load(Ordering::Relaxed),
    )
}

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
