use candle_core::{CustomOp1, Error, Layout, Result, Shape, CpuStorage};
use super::quantization::{TernaryPackType, quantize_f32_to_i8};
use super::simd::{ternary_dot_product_pack4, ternary_dot_product_pack5};

#[derive(Debug, Clone)]
pub struct TernaryMatMulOp {
    pub packed_weights: Vec<u8>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub pack_type: TernaryPackType,
    pub w_scales: Vec<f32>,
}

impl CustomOp1 for TernaryMatMulOp {
    fn name(&self) -> &'static str {
        "ternary-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let input = match storage {
            CpuStorage::F32(slice) => slice,
            _ => return Err(Error::Msg("TernaryMatMul only supports f32 input".to_string())),
        };

        if !layout.is_contiguous() {
            return Err(Error::Msg("TernaryMatMul input must be contiguous".to_string()));
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
        
        let bytes_per_row = match self.pack_type {
            TernaryPackType::Pack4 => (self.in_dim + 3) / 4,
            TernaryPackType::Pack5 => (self.in_dim + 4) / 5,
        };
        
        use rayon::prelude::*;
        
        out_data.par_chunks_mut(self.out_dim).enumerate().for_each(|(b, out_row)| {
            let in_row = &input_slice[b * self.in_dim .. (b + 1) * self.in_dim];
            
            let (quantized_in, inv_scale) = quantize_f32_to_i8(in_row);
            
            
            let chunk_size = (self.out_dim / rayon::current_num_threads().max(1)).max(128);
            out_row.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    
                    let dot_i32 = match self.pack_type {
                        TernaryPackType::Pack4 => ternary_dot_product_pack4(&quantized_in, w_row, self.in_dim),
                        TernaryPackType::Pack5 => ternary_dot_product_pack5(&quantized_in, w_row, self.in_dim),
                    };
                    
                    *out_val = dot_i32 as f32 * inv_scale * self.w_scales[o];
                }
            });
        });
        
        Ok((CpuStorage::F32(out_data), out_shape))
    }
}
