//! Pre-norm transformer block: `x + attn(rms_1(x))` then `x + mlp(rms_2(x))`,
//! with per-stage timing fed into the profiling counters in `mod`.

use crate::tensor::FastTensor;
use crate::nn::FastRmsNorm;
use crate::loader::SafeTensorLoader;
use crate::models::llama::{
    attention::CausalSelfAttention, mlp::Mlp, Cache, Config, TIME_ATTN_GEMV, TIME_MLP_GEMV, TIME_NORM,
};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone)]
pub(crate) struct Block {
    rms_1: FastRmsNorm,
    attn: CausalSelfAttention,
    rms_2: FastRmsNorm,
    mlp: Mlp,
}

impl Block {
    pub(crate) fn forward(
        &self,
        x: &FastTensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<FastTensor> {
        let t0 = std::time::Instant::now();
        let x_norm1 = self.rms_1.forward(x)?;
        TIME_NORM.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t1 = std::time::Instant::now();
        let attn_out = self.attn.forward(&x_norm1, index_pos, block_idx, cache)?;
        TIME_ATTN_GEMV.fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t2 = std::time::Instant::now();
        let x_add = attn_out.add_inplace(x)?;
        TIME_NORM.fetch_add(t2.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t3 = std::time::Instant::now();
        let x_norm2 = self.rms_2.forward(&x_add)?;
        TIME_NORM.fetch_add(t3.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t4 = std::time::Instant::now();
        let mlp_out = self.mlp.forward(&x_norm2)?;
        TIME_MLP_GEMV.fetch_add(t4.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t5 = std::time::Instant::now();
        let result = x_add.add_inplace(&mlp_out);
        TIME_NORM.fetch_add(t5.elapsed().as_micros() as u64, Ordering::Relaxed);
        result
    }

    pub(crate) fn load(loader: SafeTensorLoader, cfg: &Config) -> anyhow::Result<Self> {
        let attn = CausalSelfAttention::load(loader.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(loader.pp("mlp"), cfg)?;
        let rms_1 = FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("input_layernorm"))?;
        let rms_2 =
            FastRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps as f32, &loader.pp("post_attention_layernorm"))?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}
