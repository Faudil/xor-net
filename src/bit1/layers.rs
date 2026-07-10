use rayon::prelude::*;
use crate::tensor::FastTensor;
use super::quantization::pack_1bit;
use super::simd::xnor_dot_product;
use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

fn get_num_threads() -> usize {
    let mut val = NUM_THREADS.load(Ordering::Relaxed);
    if val == 0 {
        val = rayon::current_num_threads().max(1);
        NUM_THREADS.store(val, Ordering::Relaxed);
    }
    val
}

#[derive(Debug, Clone)]
pub struct BitLinear {
    pub packed_weights: Vec<u8>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl BitLinear {
    pub fn new(in_dim: usize, out_dim: usize, weights_f32: &[f32]) -> anyhow::Result<Self> {
        if weights_f32.len() != in_dim * out_dim {
            anyhow::bail!("Weight length must match in_dim * out_dim");
        }
        
        let packed_weights: Vec<u8> = weights_f32.par_chunks(in_dim)
            .flat_map_iter(|row| pack_1bit(row))
            .collect();
        
        Ok(Self {
            packed_weights,
            in_dim,
            out_dim,
        })
    }

    pub fn forward(&self, xs: &FastTensor) -> anyhow::Result<FastTensor> {
        let input_slice = &xs.data;
        let mut out_shape = xs.shape.clone();
        let rank = xs.shape.len();
        if rank == 0 {
            anyhow::bail!("Input must have at least 1 dimension");
        }
        let last_dim = &mut out_shape[rank - 1];
        if *last_dim != self.in_dim {
            anyhow::bail!("Input dimension mismatch: expected {}, got {}", self.in_dim, last_dim);
        }
        *last_dim = self.out_dim;
        
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let b_size: usize = xs.shape[..rank - 1].iter().product();
        
        let mut out_data = crate::tensor::uninit_vec(b_size * out_dim);
        let bytes_per_row = (in_dim + 7) / 8;
        
        if b_size == 1 {
            let in_row = &input_slice[0 .. in_dim];
            let packed_in = pack_1bit(in_row);
            let num_threads = get_num_threads();
            let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);
            
            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    *out_val = xnor_dot_product(&packed_in, w_row, in_dim);
                }
            });
        } else {
            out_data.par_chunks_mut(out_dim).zip(input_slice.par_chunks(in_dim)).for_each(|(out_row, in_row)| {
                let packed_in = pack_1bit(in_row);
                for o in 0..out_dim {
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    out_row[o] = xnor_dot_product(&packed_in, w_row, in_dim);
                }
            });
        }
        
        Ok(FastTensor::new(out_data, out_shape))
    }
}
