use candle_core::{Tensor, Result, Error, Module};
use super::ops::TernaryMatMulOp;
use super::quantization::{TernaryPackType, pack_1_58bit_4pack, pack_1_58bit_5pack};

#[derive(Debug, Clone)]
pub struct TernaryLinear {
    pub weight_op: TernaryMatMulOp,
}

impl TernaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, weights_f32: &[f32], pack_type: TernaryPackType) -> Result<Self> {
        if weights_f32.len() != in_dim * out_dim {
            return Err(Error::Msg("Weight length must match in_dim * out_dim".to_string()));
        }
        
        let sum_abs: f32 = weights_f32.iter().map(|x| x.abs()).sum();
        let w_scale = if weights_f32.len() > 0 { sum_abs / weights_f32.len() as f32 } else { 1.0 };
        
        let mut packed_weights = Vec::new();
        for row in weights_f32.chunks(in_dim) {
            match pack_type {
                TernaryPackType::Pack4 => packed_weights.extend(pack_1_58bit_4pack(row, w_scale)),
                TernaryPackType::Pack5 => packed_weights.extend(pack_1_58bit_5pack(row, w_scale)),
            }
        }
        
        Ok(Self {
            weight_op: TernaryMatMulOp {
                packed_weights,
                in_dim,
                out_dim,
                pack_type,
                w_scale,
            },
        })
    }
}

impl Module for TernaryLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(&self.weight_op)
    }
}
