//! SwiGLU/ReLU² feed-forward MLP. Selects a fused ternary fast path at runtime
//! (see `forward` for the dispatch logic) to avoid extra quantize/allocate work.

use crate::tensor::FastTensor;
use crate::nn::{DynamicLinear, FastRmsNorm};
use crate::nn::dynamic_linear::LinearKind;
use crate::loader::{SafeTensorLoader, sparse_loader::SparseFile};
use crate::models::llama::{Activation, Config, TIME_MLP_DOWN};
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Monotonic block index used to label per-block MLP-kind diagnostics.
pub(crate) static MLP_DBG_IDX: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    c_fc1: DynamicLinear,
    c_fc2: DynamicLinear,
    c_proj: DynamicLinear,
    ffn_layernorm: Option<FastRmsNorm>,
    hidden_act: Activation,
}

impl Mlp {
    pub(crate) fn forward(&self, x: &FastTensor) -> anyhow::Result<FastTensor> {
        let rank = x.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let b_size: usize = x.shape[..rank - 1].iter().product();

        // Fast path: all three projections are ternary and batch size is 1.
        // Quantize the input once and run gate+up+silu+down with no intermediate
        // DynamicLinear dispatch or extra allocations.
        if b_size == 1 {
            if let (LinearKind::Ternary(gate_lin), LinearKind::Ternary(up_lin)) =
                (&self.c_fc1.inner, &self.c_fc2.inner)
            {
                let in_row = &x.data[0..gate_lin.in_dim];
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(gate_lin.in_dim);
                let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(in_row, &mut quantized_in);
                let use_silu = matches!(self.hidden_act, Activation::Silu);

                if let LinearKind::Ternary(down_lin) = &self.c_proj.inner {
                    // All-ternary: fully fused gate+up+silu+down (VNNI kernel).
                    let result = crate::bit1_58::layers::TernaryLinear::fused_mlp_all(
                        x,
                        &quantized_in,
                        inv_scale,
                        gate_lin,
                        up_lin,
                        down_lin,
                        self.ffn_layernorm.as_ref(),
                        use_silu,
                    );
                    crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
                    return result;
                }

                if let LinearKind::Int8(_) = &self.c_proj.inner {
                    // Non-ternary down (e.g. Int8): still fuse gate+up+silu and
                    // feed the single quantized activation to down_proj, avoiding
                    // the separate c_proj.forward re-quantization fallback.
                    let result = crate::bit1_58::layers::TernaryLinear::fused_mlp_gate_up_down(
                        x,
                        &quantized_in,
                        inv_scale,
                        gate_lin,
                        up_lin,
                        &self.c_proj,
                        self.ffn_layernorm.as_ref(),
                        use_silu,
                    );
                    crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
                    return result;
                }
            }
        }

        // Fallback: non-ternary or batched (b_size > 1) path.
        let (h1, h2) = match (&self.c_fc1.inner, &self.c_fc2.inner) {
            (LinearKind::Ternary(fc1_lin), LinearKind::Ternary(fc2_lin)) if b_size == 1 => {
                let in_row = &x.data[0..fc1_lin.in_dim];
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(fc1_lin.in_dim);
                let inv_scale = crate::bit1_58::quantization::quantize_f32_to_i8(in_row, &mut quantized_in);
                let res = crate::bit1_58::layers::TernaryLinear::fused_forward_mlp(
                    x,
                    &quantized_in,
                    inv_scale,
                    fc1_lin,
                    fc2_lin,
                );
                crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
                res
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
            None => x_mul,
        };
        let t_down = std::time::Instant::now();
        let result = self.c_proj.forward(&x_norm);
        TIME_MLP_DOWN.fetch_add(t_down.elapsed().as_micros() as u64, Ordering::Relaxed);
        result
    }

    /// Prefetch the leading cache lines of gate/up/down weights so the next
    /// block's weight stream can begin while this block is still computing.
    pub(crate) fn prefetch_weights(&self) {
        prefetch_proj(&self.c_fc1);
        prefetch_proj(&self.c_fc2);
        prefetch_proj(&self.c_proj);
    }

    /// Bytes of packed ternary weight memory owned by this MLP (bandwidth estimate).
    pub(crate) fn weight_bytes(&self) -> usize {
        proj_bytes(&self.c_fc1) + proj_bytes(&self.c_fc2) + proj_bytes(&self.c_proj)
    }

    pub(crate) fn load(loader: SafeTensorLoader, cfg: &Config, sparse: Option<&SparseFile>) -> anyhow::Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 =
            DynamicLinear::load(h_size, i_size, &loader.pp("gate_proj"), "weight", cfg.quantization_config, sparse)?;
        let c_fc2 =
            DynamicLinear::load(h_size, i_size, &loader.pp("up_proj"), "weight", cfg.quantization_config, sparse)?;
        let c_proj =
            DynamicLinear::load(i_size, h_size, &loader.pp("down_proj"), "weight", cfg.quantization_config, sparse)?;
        let ffn_layernorm = if loader.pp("ffn_layernorm").has_tensor("weight") {
            Some(FastRmsNorm::load(i_size, cfg.rms_norm_eps as f32, &loader.pp("ffn_layernorm"))?)
        } else if loader.pp("ffn_sub_norm").has_tensor("weight") {
            Some(FastRmsNorm::load(i_size, cfg.rms_norm_eps as f32, &loader.pp("ffn_sub_norm"))?)
        } else {
            None
        };
        let mlp = Self {
            c_fc1,
            c_fc2,
            c_proj,
            ffn_layernorm,
            hidden_act: cfg.hidden_act,
        };
        mlp.debug_print_kinds();
        Ok(mlp)
    }

    /// When `XORNET_DEBUG` is set, print this block's MLP projection kinds so
    /// the user can see exactly which blocks are non-ternary (and therefore
    /// force the fused-MLP fallback path).
    fn debug_print_kinds(&self) {
        if std::env::var("XORNET_DEBUG").is_err() {
            return;
        }
        let idx = MLP_DBG_IDX.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[xor-net] MLP[{}] gate={} up={} down={} ffn_norm={}",
            idx,
            super::linear_kind_tag(&self.c_fc1.inner),
            super::linear_kind_tag(&self.c_fc2.inner),
            super::linear_kind_tag(&self.c_proj.inner),
            if self.ffn_layernorm.is_some() { "Some" } else { "None" },
        );
    }
}
