use rayon::prelude::*;
use crate::tensor::FastTensor;
use super::quantization::{TernaryPackType, pack_1_58bit_4pack, pack_1_58bit_5pack, quantize_f32_to_i8};
use super::simd::{ternary_dot_product_pack4, ternary_dot_product_pack5};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct TernaryLinear {
    pub packed_weights: Vec<u8>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub pack_type: TernaryPackType,
    pub w_scales: Vec<f32>,
}

impl TernaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, weights_f32: &[f32], pack_type: TernaryPackType, provided_scale: Option<f32>) -> anyhow::Result<Self> {
        if weights_f32.len() != in_dim * out_dim {
            anyhow::bail!("Weight length must match in_dim * out_dim");
        }
        
        let already_ternary = weights_f32.iter().all(|&w| w == -1.0 || w == 0.0 || w == 1.0);
        let mut w_scales = Vec::with_capacity(out_dim);
        let mut packed_weights = Vec::with_capacity((out_dim * in_dim + 3) / 4);

        if already_ternary {
            let global_w_scale = provided_scale.unwrap_or(1.0);
            for row in weights_f32.chunks(in_dim) {
                w_scales.push(global_w_scale);
                match pack_type {
                    TernaryPackType::Pack4 => packed_weights.extend(pack_1_58bit_4pack(row, global_w_scale)),
                    TernaryPackType::Pack5 => packed_weights.extend(pack_1_58bit_5pack(row, global_w_scale)),
                }
            }
        } else {
            let global_sum_abs: f32 = weights_f32.iter().map(|x| x.abs()).sum();
            let global_w_scale = if weights_f32.is_empty() { 1.0 } else { global_sum_abs / weights_f32.len() as f32 };
            for row in weights_f32.chunks(in_dim) {
                w_scales.push(global_w_scale);
                match pack_type {
                    TernaryPackType::Pack4 => packed_weights.extend(pack_1_58bit_4pack(row, global_w_scale)),
                    TernaryPackType::Pack5 => packed_weights.extend(pack_1_58bit_5pack(row, global_w_scale)),
                }
            }
        }
        
        Ok(Self {
            packed_weights,
            in_dim,
            out_dim,
            pack_type,
            w_scales,
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
        
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(b_size * out_dim);
        
        let bytes_per_row = match self.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };
        
        if b_size == 1 {
            let in_row = &input_slice[0 .. in_dim];
            let (quantized_in, inv_scale) = quantize_f32_to_i8(in_row);
            
            
            let num_threads = crate::util::get_num_threads();
            let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);
            
            out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
                let start_o = chunk_idx * chunk_size;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let o = start_o + i;
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    let dot_i32 = match self.pack_type {
                        TernaryPackType::Pack4 => ternary_dot_product_pack4(&quantized_in, w_row, in_dim),
                        TernaryPackType::Pack5 => ternary_dot_product_pack5(&quantized_in, w_row, in_dim),
                    };
                    *out_val = dot_i32 as f32 * inv_scale * self.w_scales[o];
                }
            });
        } else {
            out_data.par_chunks_mut(out_dim).enumerate().for_each(|(b, out_row)| {
                let in_row = &input_slice[b * in_dim .. (b + 1) * in_dim];
                let (quantized_in, inv_scale) = quantize_f32_to_i8(in_row);
                
                
                for o in 0..out_dim {
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    let dot_i32 = match self.pack_type {
                        TernaryPackType::Pack4 => ternary_dot_product_pack4(&quantized_in, w_row, in_dim),
                        TernaryPackType::Pack5 => ternary_dot_product_pack5(&quantized_in, w_row, in_dim),
                    };
                    out_row[o] = dot_i32 as f32 * inv_scale * self.w_scales[o];
                }
            });
        }
        
        Ok(FastTensor::new(out_data, out_shape))
    }

    pub fn forward_with_quantized(&self, xs: &FastTensor, quantized_in: &[i8], inv_scale: f32) -> anyhow::Result<FastTensor> {
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
        
        let mut out_data = crate::tensor::workspace::get_pooled_buffer(out_dim);
        
        let bytes_per_row = match self.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };
        
        let num_threads = crate::util::get_num_threads();
        let chunk_size = ((out_dim + num_threads - 1) / num_threads).max(128);
        
        out_data.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let start_o = chunk_idx * chunk_size;
            for (i, out_val) in out_chunk.iter_mut().enumerate() {
                let o = start_o + i;
                let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                let dot_i32 = match self.pack_type {
                    TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                    TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                };
                *out_val = dot_i32 as f32 * inv_scale * self.w_scales[o];
            }
        });
        
        Ok(FastTensor::new(out_data, out_shape))
    }

    pub fn fused_forward_qkv(
        xs: &FastTensor,
        quantized_in: &[i8],
        inv_scale: f32,
        q_lin: &TernaryLinear,
        k_lin: &TernaryLinear,
        v_lin: &TernaryLinear,
    ) -> (FastTensor, FastTensor, FastTensor) {
        let rank = xs.shape.len();
        let in_dim = q_lin.in_dim;

        let mut q_out = crate::tensor::workspace::get_pooled_buffer(q_lin.out_dim);
        let mut k_out = crate::tensor::workspace::get_pooled_buffer(k_lin.out_dim);
        let mut v_out = crate::tensor::workspace::get_pooled_buffer(v_lin.out_dim);

        let bytes_per_row = match q_lin.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };

        let q_out_dim = q_lin.out_dim;
        let k_out_dim = k_lin.out_dim;
        let total_rows = q_out_dim + k_out_dim + v_lin.out_dim;

        let num_threads = crate::util::get_num_threads();
        let chunk_size = ((total_rows + num_threads - 1) / num_threads).max(128);

        let q_ptr: usize = q_out.as_mut_ptr() as usize;
        let k_ptr: usize = k_out.as_mut_ptr() as usize;
        let v_ptr: usize = v_out.as_mut_ptr() as usize;
        let q_packed = &q_lin.packed_weights;
        let k_packed = &k_lin.packed_weights;
        let v_packed = &v_lin.packed_weights;
        let q_w = &q_lin.w_scales;
        let k_w = &k_lin.w_scales;
        let v_w = &v_lin.w_scales;

        let next_row = AtomicU64::new(0);
        let num_threads = rayon::current_num_threads();
        let pack_type = q_lin.pack_type;
        rayon::scope(|s| {
            for _ in 1..num_threads {
                s.spawn(|_| loop {
                    let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                    if start >= total_rows { break; }
                    let end = (start + chunk_size).min(total_rows);
                    for row_idx in start..end {
                        if row_idx < q_out_dim {
                            let w_row = &q_packed[row_idx * bytes_per_row .. (row_idx + 1) * bytes_per_row];
                            let dot = match pack_type {
                                TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                                TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                            };
                            unsafe { *(q_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * q_w[row_idx]; }
                        } else if row_idx < q_out_dim + k_out_dim {
                            let kr = row_idx - q_out_dim;
                            let w_row = &k_packed[kr * bytes_per_row .. (kr + 1) * bytes_per_row];
                            let dot = match pack_type {
                                TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                                TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                            };
                            unsafe { *(k_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * k_w[kr]; }
                        } else {
                            let vr = row_idx - q_out_dim - k_out_dim;
                            let w_row = &v_packed[vr * bytes_per_row .. (vr + 1) * bytes_per_row];
                            let dot = match pack_type {
                                TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                                TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                            };
                            unsafe { *(v_ptr as *mut f32).add(vr) = dot as f32 * inv_scale * v_w[vr]; }
                        }
                    }
                });
            }
            loop {
                let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                if start >= total_rows { break; }
                let end = (start + chunk_size).min(total_rows);
                for row_idx in start..end {
                    if row_idx < q_out_dim {
                        let w_row = &q_packed[row_idx * bytes_per_row .. (row_idx + 1) * bytes_per_row];
                        let dot = match pack_type {
                            TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                            TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                        };
                        unsafe { *(q_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * q_w[row_idx]; }
                    } else if row_idx < q_out_dim + k_out_dim {
                        let kr = row_idx - q_out_dim;
                        let w_row = &k_packed[kr * bytes_per_row .. (kr + 1) * bytes_per_row];
                        let dot = match pack_type {
                            TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                            TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                        };
                        unsafe { *(k_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * k_w[kr]; }
                    } else {
                        let vr = row_idx - q_out_dim - k_out_dim;
                        let w_row = &v_packed[vr * bytes_per_row .. (vr + 1) * bytes_per_row];
                        let dot = match pack_type {
                            TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                            TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                        };
                        unsafe { *(v_ptr as *mut f32).add(vr) = dot as f32 * inv_scale * v_w[vr]; }
                    }
                }
            }
        });

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = q_lin.out_dim;
        let q_tensor = FastTensor::new(q_out, out_shape.clone());
        out_shape[rank - 1] = k_lin.out_dim;
        let k_tensor = FastTensor::new(k_out, out_shape.clone());
        out_shape[rank - 1] = v_lin.out_dim;
        let v_tensor = FastTensor::new(v_out, out_shape);

        (q_tensor, k_tensor, v_tensor)
    }

    pub fn fused_forward_mlp(
        xs: &FastTensor,
        quantized_in: &[i8],
        inv_scale: f32,
        fc1_lin: &TernaryLinear,
        fc2_lin: &TernaryLinear,
    ) -> (FastTensor, FastTensor) {
        let rank = xs.shape.len();
        let in_dim = fc1_lin.in_dim;

        let mut fc1_out = crate::tensor::workspace::get_pooled_buffer(fc1_lin.out_dim);
        let mut fc2_out = crate::tensor::workspace::get_pooled_buffer(fc2_lin.out_dim);

        let bytes_per_row = match fc1_lin.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };

        let fc1_out_dim = fc1_lin.out_dim;
        let total_rows = fc1_out_dim + fc2_lin.out_dim;

        let num_threads = crate::util::get_num_threads();
        let chunk_size = ((total_rows + num_threads - 1) / num_threads).max(128);

        let fc1_ptr: usize = fc1_out.as_mut_ptr() as usize;
        let fc2_ptr: usize = fc2_out.as_mut_ptr() as usize;
        let fc1_packed = &fc1_lin.packed_weights;
        let fc2_packed = &fc2_lin.packed_weights;
        let fc1_w = &fc1_lin.w_scales;
        let fc2_w = &fc2_lin.w_scales;

        let next_row = AtomicU64::new(0);
        let num_threads = rayon::current_num_threads();
        let pack_type = fc1_lin.pack_type;
        rayon::scope(|s| {
            for _ in 1..num_threads {
                s.spawn(|_| loop {
                    let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                    if start >= total_rows { break; }
                    let end = (start + chunk_size).min(total_rows);
                    for row_idx in start..end {
                        if row_idx < fc1_out_dim {
                            let w_row = &fc1_packed[row_idx * bytes_per_row .. (row_idx + 1) * bytes_per_row];
                            let dot = match pack_type {
                                TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                                TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                            };
                            unsafe { *(fc1_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * fc1_w[row_idx]; }
                        } else {
                            let kr = row_idx - fc1_out_dim;
                            let w_row = &fc2_packed[kr * bytes_per_row .. (kr + 1) * bytes_per_row];
                            let dot = match pack_type {
                                TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                                TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                            };
                            unsafe { *(fc2_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * fc2_w[kr]; }
                        }
                    }
                });
            }
            loop {
                let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                if start >= total_rows { break; }
                let end = (start + chunk_size).min(total_rows);
                for row_idx in start..end {
                    if row_idx < fc1_out_dim {
                        let w_row = &fc1_packed[row_idx * bytes_per_row .. (row_idx + 1) * bytes_per_row];
                        let dot = match pack_type {
                            TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                            TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                        };
                        unsafe { *(fc1_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * fc1_w[row_idx]; }
                    } else {
                        let kr = row_idx - fc1_out_dim;
                        let w_row = &fc2_packed[kr * bytes_per_row .. (kr + 1) * bytes_per_row];
                        let dot = match pack_type {
                            TernaryPackType::Pack4 => ternary_dot_product_pack4(quantized_in, w_row, in_dim),
                            TernaryPackType::Pack5 => ternary_dot_product_pack5(quantized_in, w_row, in_dim),
                        };
                        unsafe { *(fc2_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * fc2_w[kr]; }
                    }
                }
            }
        });

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = fc1_lin.out_dim;
        let fc1_tensor = FastTensor::new(fc1_out, out_shape.clone());
        out_shape[rank - 1] = fc2_lin.out_dim;
        let fc2_tensor = FastTensor::new(fc2_out, out_shape);

        (fc1_tensor, fc2_tensor)
    }
}
