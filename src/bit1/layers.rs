use candle_core::{Tensor, Result, Error, Module};
use super::ops::BitMatMulOp;
use super::quantization::pack_1bit;

pub struct BitLinear {
    weight_op: BitMatMulOp,
}

impl BitLinear {
    pub fn new(in_dim: usize, out_dim: usize, weights_f32: &[f32]) -> Result<Self> {
        if weights_f32.len() != in_dim * out_dim {
            return Err(Error::Msg("Weight length must match in_dim * out_dim".to_string()));
        }
        
        let mut packed_weights = Vec::new();
        for row in weights_f32.chunks(in_dim) {
            packed_weights.extend(pack_1bit(row));
        }
        
        Ok(Self {
            weight_op: BitMatMulOp {
                packed_weights,
                in_dim,
                out_dim,
            },
        })
    }
}

impl Module for BitLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(&self.weight_op)
    }
}
