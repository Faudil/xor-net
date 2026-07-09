use candle_core::{CustomOp1, Error, Layout, Result, Shape, CpuStorage};
use super::quantization::pack_1bit;
use super::simd::xnor_dot_product;

pub struct BitMatMulOp {
    pub packed_weights: Vec<u8>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl CustomOp1 for BitMatMulOp {
    fn name(&self) -> &'static str {
        "bit-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let input = match storage {
            CpuStorage::F32(slice) => slice,
            _ => return Err(Error::Msg("BitMatMul only supports f32 input".to_string())),
        };

        if !layout.is_contiguous() {
            return Err(Error::Msg("BitMatMul input must be contiguous".to_string()));
        }

        let start = layout.start_offset();
        let input_slice = &input[start..start + layout.shape().elem_count()];

        let out_shape = {
            let mut dims = layout.shape().dims().to_vec();
            if let Some(last) = dims.last_mut() {
                if *last != self.in_dim {
                    return Err(Error::Msg(format!("Input dimension mismatch: expected {}, got {}", self.in_dim, last)));
                }
                *last = self.out_dim;
            } else {
                return Err(Error::Msg("Input must have at least 1 dimension".to_string()));
            }
            Shape::from(dims)
        };
        
        let b_size: usize = layout.shape().dims()[..layout.shape().rank() - 1].iter().product();
        let mut out_data = vec![0.0; b_size * self.out_dim];
        let bytes_per_row = (self.in_dim + 7) / 8;
        
        use rayon::prelude::*;
        
        out_data.par_chunks_mut(self.out_dim).enumerate().for_each(|(b, out_row)| {
            let in_row = &input_slice[b * self.in_dim .. (b + 1) * self.in_dim];
            let packed_in = pack_1bit(in_row);
            
            out_row.par_iter_mut().enumerate().for_each(|(o, out_val)| {
                let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                *out_val = xnor_dot_product(&packed_in, w_row, self.in_dim);
            });
        });
        
        Ok((CpuStorage::F32(out_data), out_shape))
    }
}
