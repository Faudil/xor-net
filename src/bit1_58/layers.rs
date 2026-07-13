use rayon::prelude::*;
use crate::tensor::FastTensor;
use super::quantization::{TernaryPackType, pack_1_58bit_4pack, pack_1_58bit_5pack, quantize_f32_to_i8};
use super::simd::{ternary_dot_product_pack4, ternary_dot_product_pack5};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Dispatches the vectorized ternary dot product according to the packing scheme.
#[inline]
fn ternary_dot(pack_type: TernaryPackType, input: &[i8], weights: &[u8], n: usize) -> i32 {
    match pack_type {
        TernaryPackType::Pack4 => ternary_dot_product_pack4(input, weights, n),
        TernaryPackType::Pack5 => ternary_dot_product_pack5(input, weights, n),
    }
}

/// Time spent in the MLP activation (SiLU/ReLU2) + the silu->i8 quantization
/// pass, across all tokens. Used to confirm how much of the MLP budget the
/// scalar `exp()` loop accounts for (and to verify the vectorized replacement).
pub static TIME_SILU: AtomicU64 = AtomicU64::new(0);

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
            match provided_scale {
                Some(global_w_scale) => {
                    for row in weights_f32.chunks(in_dim) {
                        w_scales.push(global_w_scale);
                        match pack_type {
                            TernaryPackType::Pack4 => packed_weights.extend(pack_1_58bit_4pack(row, global_w_scale)),
                            TernaryPackType::Pack5 => packed_weights.extend(pack_1_58bit_5pack(row, global_w_scale)),
                        }
                    }
                }
                None => {
                    // Per-row scale (BitNet γ): each output neuron gets its own
                    // scale. Used when no precomputed `_scale` tensor is
                    // supplied (e.g. a tied/embedded LM head). A single global
                    // scale is far too coarse for the logits projection and
                    // destroys accuracy.
                    for row in weights_f32.chunks(in_dim) {
                        let sum_abs: f32 = row.iter().map(|x| x.abs()).sum();
                        let s = if row.is_empty() { 1.0 } else { sum_abs / row.len() as f32 };
                        w_scales.push(s);
                        match pack_type {
                            TernaryPackType::Pack4 => packed_weights.extend(pack_1_58bit_4pack(row, s)),
                            TernaryPackType::Pack5 => packed_weights.extend(pack_1_58bit_5pack(row, s)),
                        }
                    }
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
            let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
            let inv_scale = quantize_f32_to_i8(in_row, &mut quantized_in);
            
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
            crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
        } else {
            out_data.par_chunks_mut(out_dim).enumerate().for_each(|(b, out_row)| {
                let mut quantized_in = crate::tensor::workspace::get_pooled_buffer_i8(in_dim);
                let in_row = &input_slice[b * in_dim .. (b + 1) * in_dim];
                let inv_scale = quantize_f32_to_i8(in_row, &mut quantized_in);
                
                for o in 0..out_dim {
                    let w_row = &self.packed_weights[o * bytes_per_row .. (o + 1) * bytes_per_row];
                    let dot_i32 = match self.pack_type {
                        TernaryPackType::Pack4 => ternary_dot_product_pack4(&quantized_in, w_row, in_dim),
                        TernaryPackType::Pack5 => ternary_dot_product_pack5(&quantized_in, w_row, in_dim),
                    };
                    out_row[o] = dot_i32 as f32 * inv_scale * self.w_scales[o];
                }
                crate::tensor::workspace::return_pooled_buffer_i8(quantized_in);
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

        // Each output row index is claimed exactly once via the atomic counter,
        // so the parallel workers below can write to disjoint f32 slots through
        // raw pointers without any lock.
        let q_ptr: usize = q_out.as_mut_ptr() as usize;
        let k_ptr: usize = k_out.as_mut_ptr() as usize;
        let v_ptr: usize = v_out.as_mut_ptr() as usize;
        let q_packed = &q_lin.packed_weights;
        let k_packed = &k_lin.packed_weights;
        let v_packed = &v_lin.packed_weights;
        let q_w = &q_lin.w_scales;
        let k_w = &k_lin.w_scales;
        let v_w = &v_lin.w_scales;
        let pack_type = q_lin.pack_type;

        let compute_row = |row_idx: usize| {
            if row_idx < q_out_dim {
                let w_row = &q_packed[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
                let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                unsafe { *(q_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * q_w[row_idx]; }
            } else if row_idx < q_out_dim + k_out_dim {
                let kr = row_idx - q_out_dim;
                let w_row = &k_packed[kr * bytes_per_row..(kr + 1) * bytes_per_row];
                let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                unsafe { *(k_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * k_w[kr]; }
            } else {
                let vr = row_idx - q_out_dim - k_out_dim;
                let w_row = &v_packed[vr * bytes_per_row..(vr + 1) * bytes_per_row];
                let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                unsafe { *(v_ptr as *mut f32).add(vr) = dot as f32 * inv_scale * v_w[vr]; }
            }
        };

        // Split the combined QKV row range across threads; the calling thread
        // also joins the pool once the scope closure returns, so we spawn one
        // work-stealing task per thread.
        let next_row = AtomicU64::new(0);
        rayon::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|_| {
                    loop {
                        let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                        if start >= total_rows {
                            break;
                        }
                        let end = (start + chunk_size).min(total_rows);
                        for row_idx in start..end {
                            compute_row(row_idx);
                        }
                    }
                });
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
        let pack_type = fc1_lin.pack_type;

        let compute_row = |row_idx: usize| {
            if row_idx < fc1_out_dim {
                let w_row = &fc1_packed[row_idx * bytes_per_row..(row_idx + 1) * bytes_per_row];
                let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                unsafe { *(fc1_ptr as *mut f32).add(row_idx) = dot as f32 * inv_scale * fc1_w[row_idx]; }
            } else {
                let kr = row_idx - fc1_out_dim;
                let w_row = &fc2_packed[kr * bytes_per_row..(kr + 1) * bytes_per_row];
                let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                unsafe { *(fc2_ptr as *mut f32).add(kr) = dot as f32 * inv_scale * fc2_w[kr]; }
            }
        };

        let next_row = AtomicU64::new(0);
        rayon::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|_| {
                    loop {
                        let start = next_row.fetch_add(chunk_size as u64, Ordering::Relaxed) as usize;
                        if start >= total_rows {
                            break;
                        }
                        let end = (start + chunk_size).min(total_rows);
                        for row_idx in start..end {
                            compute_row(row_idx);
                        }
                    }
                });
            }
        });

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = fc1_lin.out_dim;
        let fc1_tensor = FastTensor::new(fc1_out, out_shape.clone());
        out_shape[rank - 1] = fc2_lin.out_dim;
        let fc2_tensor = FastTensor::new(fc2_out, out_shape);

        (fc1_tensor, fc2_tensor)
    }

    /// Fused gate+up+silu+down in two tightly-coupled rayon scopes.
    ///
    /// Saves vs. the old path:
    ///  - 1 fewer `DynamicLinear::forward` dispatch (no virtual call + shape boxing)
    ///  - 1 fewer `get_pooled_buffer` / `return_pooled_buffer` round-trip for the down output
    ///  - 1 fewer standalone `quantize_f32_to_i8` pool acquire for the silu → i8 buffer
    ///  - Tighter spatial locality: silu buffer is quantized immediately while still L3-warm
    ///
    /// `use_silu`: true = SiLU activation (gate * σ(gate) * up), false = ReLU² (max(gate,0)² * up)
    pub fn fused_mlp_all(
        xs: &FastTensor,
        quantized_in: &[i8],      // pre-quantized input (hidden_size i8 values)
        inv_scale_in: f32,        // inverse scale for the input quantization
        gate_lin: &TernaryLinear, // gate_proj: hidden→intermediate
        up_lin: &TernaryLinear,   // up_proj:   hidden→intermediate
        down_lin: &TernaryLinear, // down_proj: intermediate→hidden
        ffn_norm: Option<&crate::nn::FastRmsNorm>,
        use_silu: bool,
    ) -> anyhow::Result<FastTensor> {
        let rank = xs.shape.len();
        let in_dim   = gate_lin.in_dim;    // hidden_size   e.g. 2048
        let inter    = gate_lin.out_dim;   // intermediate  e.g. 5504
        let out_dim  = down_lin.out_dim;   // hidden_size   e.g. 2048

        // ── Phase 1: gate + up in parallel ──────────────────────────────────
        let mut gate_buf = crate::tensor::workspace::get_pooled_buffer(inter);
        let mut up_buf   = crate::tensor::workspace::get_pooled_buffer(inter);

        let bytes_gate = match gate_lin.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };
        let bytes_down = match down_lin.pack_type {
            TernaryPackType::Pack4 => (inter + 3) / 4,
            TernaryPackType::Pack5 => (inter + 4) / 5,
        };

        let total_rows12 = inter + inter; // gate rows + up rows
        let num_threads  = rayon::current_num_threads();
        let chunk12      = ((total_rows12 + num_threads - 1) / num_threads).max(128);

        let gate_ptr: usize = gate_buf.as_mut_ptr() as usize;
        let up_ptr:   usize = up_buf.as_mut_ptr() as usize;

        let gate_packed = &gate_lin.packed_weights;
        let up_packed   = &up_lin.packed_weights;
        let gate_w      = &gate_lin.w_scales;
        let up_w        = &up_lin.w_scales;
        let pack_type   = gate_lin.pack_type;

        {
            let compute12 = |r: usize| {
                if r < inter {
                    let w_row = &gate_packed[r * bytes_gate..(r + 1) * bytes_gate];
                    let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                    unsafe { *(gate_ptr as *mut f32).add(r) = dot as f32 * inv_scale_in * gate_w[r]; }
                } else {
                    let ur = r - inter;
                    let w_row = &up_packed[ur * bytes_gate..(ur + 1) * bytes_gate];
                    let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                    unsafe { *(up_ptr as *mut f32).add(ur) = dot as f32 * inv_scale_in * up_w[ur]; }
                }
            };
            let next_row = AtomicU64::new(0);
            rayon::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|_| {
                        loop {
                            let start = next_row.fetch_add(chunk12 as u64, Ordering::Relaxed) as usize;
                            if start >= total_rows12 {
                                break;
                            }
                            let end = (start + chunk12).min(total_rows12);
                            for r in start..end {
                                compute12(r);
                            }
                        }
                    });
                }
            });
        }

        // ── Phase 2: activation element-wise (vectorized SiLU/ReLU²) ────────
        let t_act = Instant::now();
        crate::bit1_58::quantization::silu_inplace(&mut gate_buf, &up_buf, use_silu);
        crate::tensor::workspace::return_pooled_buffer(up_buf);

        // ── Phase 2b: optional ffn layernorm ────────────────────────────────
        if let Some(ln) = ffn_norm {
            let mut norm_shape = xs.shape.clone();
            norm_shape[rank - 1] = inter;
            let mid_tensor = FastTensor::new(gate_buf, norm_shape);
            let normed = ln.forward(&mid_tensor)?;
            // quantize the normed output
            let mut quantized_mid = crate::tensor::workspace::get_pooled_buffer_i8(inter);
            let inv_scale_mid = crate::bit1_58::quantization::quantize_f32_to_i8(&normed.data, &mut quantized_mid);

            // Phase 3 with normed data
            let mut down_buf = crate::tensor::workspace::get_pooled_buffer(out_dim);
            let chunk3 = ((out_dim + num_threads - 1) / num_threads).max(128);
            let down_ptr: usize = down_buf.as_mut_ptr() as usize;
            let down_packed = &down_lin.packed_weights;
            let down_w = &down_lin.w_scales;
            {
                let compute_down = |r: usize| {
                    let w_row = &down_packed[r * bytes_down..(r + 1) * bytes_down];
                    let dot = ternary_dot(pack_type, &quantized_mid, w_row, inter);
                    unsafe { *(down_ptr as *mut f32).add(r) = dot as f32 * inv_scale_mid * down_w[r]; }
                };
                let next_row3 = AtomicU64::new(0);
                rayon::scope(|s| {
                    for _ in 0..num_threads {
                        s.spawn(|_| {
                            loop {
                                let start = next_row3.fetch_add(chunk3 as u64, Ordering::Relaxed) as usize;
                                if start >= out_dim {
                                    break;
                                }
                                let end = (start + chunk3).min(out_dim);
                                for r in start..end {
                                    compute_down(r);
                                }
                            }
                        });
                    }
                });
            }
            crate::tensor::workspace::return_pooled_buffer_i8(quantized_mid);
            TIME_SILU.fetch_add(t_act.elapsed().as_micros() as u64, Ordering::Relaxed);
            let mut out_shape = xs.shape.clone();
            out_shape[rank - 1] = out_dim;
            return Ok(FastTensor::new(down_buf, out_shape));
        }

        // ── Phase 2c: quantize silu output (gate_buf is now silu result) ────
        let mut quantized_mid = crate::tensor::workspace::get_pooled_buffer_i8(inter);
        let inv_scale_mid = crate::bit1_58::quantization::quantize_f32_to_i8(&gate_buf, &mut quantized_mid);
        crate::tensor::workspace::return_pooled_buffer(gate_buf);
        TIME_SILU.fetch_add(t_act.elapsed().as_micros() as u64, Ordering::Relaxed);

        // ── Phase 3: down_proj ───────────────────────────────────────────────
        let mut down_buf = crate::tensor::workspace::get_pooled_buffer(out_dim);
        let chunk3    = ((out_dim + num_threads - 1) / num_threads).max(128);
        let down_ptr: usize = down_buf.as_mut_ptr() as usize;
        let down_packed = &down_lin.packed_weights;
        let down_w      = &down_lin.w_scales;
        {
            let compute_down = |r: usize| {
                let w_row = &down_packed[r * bytes_down..(r + 1) * bytes_down];
                let dot = ternary_dot(pack_type, &quantized_mid, w_row, inter);
                unsafe { *(down_ptr as *mut f32).add(r) = dot as f32 * inv_scale_mid * down_w[r]; }
            };
            let next_row3 = AtomicU64::new(0);
            rayon::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|_| {
                        loop {
                            let start = next_row3.fetch_add(chunk3 as u64, Ordering::Relaxed) as usize;
                            if start >= out_dim {
                                break;
                            }
                            let end = (start + chunk3).min(out_dim);
                            for r in start..end {
                                compute_down(r);
                            }
                        }
                    });
                }
            });
        }
        crate::tensor::workspace::return_pooled_buffer_i8(quantized_mid);

        let mut out_shape = xs.shape.clone();
        out_shape[rank - 1] = out_dim;
        Ok(FastTensor::new(down_buf, out_shape))
    }

    /// Fused MLP that always uses a single quantization of the activation, then
    /// runs `down` via its `forward_with_quantized` path. Unlike `fused_mlp_all`
    /// this does not require `down` to be ternary, so the fused path is taken
    /// even when `down_proj` is Int8/F32 — eliminating the separate
    /// `c_proj.forward` re-quantization fallback.
    ///
    /// `down` must be `Ternary` or `Int8` (both implement `forward_with_quantized`
    /// and ignore the f32 input tensor, using `quantized_mid` directly).
    pub fn fused_mlp_gate_up_down(
        xs: &FastTensor,
        quantized_in: &[i8],      // pre-quantized input (hidden_size i8 values)
        inv_scale_in: f32,        // inverse scale for the input quantization
        gate_lin: &TernaryLinear, // gate_proj: hidden→intermediate
        up_lin: &TernaryLinear,   // up_proj:   hidden→intermediate
        down: &crate::nn::dynamic_linear::DynamicLinear, // down_proj: intermediate→hidden
        ffn_norm: Option<&crate::nn::FastRmsNorm>,
        use_silu: bool,
    ) -> anyhow::Result<FastTensor> {
        let rank = xs.shape.len();
        let in_dim   = gate_lin.in_dim;    // hidden_size
        let inter    = gate_lin.out_dim;   // intermediate

        // ── Phase 1: gate + up in parallel ──────────────────────────────────
        let mut gate_buf = crate::tensor::workspace::get_pooled_buffer(inter);
        let mut up_buf   = crate::tensor::workspace::get_pooled_buffer(inter);

        let bytes_gate = match gate_lin.pack_type {
            TernaryPackType::Pack4 => (in_dim + 3) / 4,
            TernaryPackType::Pack5 => (in_dim + 4) / 5,
        };

        let total_rows12 = inter + inter; // gate rows + up rows
        let num_threads  = rayon::current_num_threads();
        let chunk12      = ((total_rows12 + num_threads - 1) / num_threads).max(128);

        let gate_ptr: usize = gate_buf.as_mut_ptr() as usize;
        let up_ptr:   usize = up_buf.as_mut_ptr() as usize;

        let gate_packed = &gate_lin.packed_weights;
        let up_packed   = &up_lin.packed_weights;
        let gate_w      = &gate_lin.w_scales;
        let up_w        = &up_lin.w_scales;
        let pack_type   = gate_lin.pack_type;

        {
            let compute12 = |r: usize| {
                if r < inter {
                    let w_row = &gate_packed[r * bytes_gate..(r + 1) * bytes_gate];
                    let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                    unsafe { *(gate_ptr as *mut f32).add(r) = dot as f32 * inv_scale_in * gate_w[r]; }
                } else {
                    let ur = r - inter;
                    let w_row = &up_packed[ur * bytes_gate..(ur + 1) * bytes_gate];
                    let dot = ternary_dot(pack_type, quantized_in, w_row, in_dim);
                    unsafe { *(up_ptr as *mut f32).add(ur) = dot as f32 * inv_scale_in * up_w[ur]; }
                }
            };
            let next_row = AtomicU64::new(0);
            rayon::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|_| {
                        loop {
                            let start = next_row.fetch_add(chunk12 as u64, Ordering::Relaxed) as usize;
                            if start >= total_rows12 {
                                break;
                            }
                            let end = (start + chunk12).min(total_rows12);
                            for r in start..end {
                                compute12(r);
                            }
                        }
                    });
                }
            });
        }

        // ── Phase 2: vectorized activation ──────────────────────────────────
        crate::bit1_58::quantization::silu_inplace(&mut gate_buf, &up_buf, use_silu);
        crate::tensor::workspace::return_pooled_buffer(up_buf);

        // ── Phase 2c + Phase 3: quantize activation, then down via the
        //     already-quantized buffer (no second quantization pass). ─────────
        // `forward_with_quantized` validates the input dim, so build a tensor
        // whose last dim is `inter` (= down.in_dim). The ternary/int8 kernels
        // ignore its data and use `quantized_mid` directly.
        let mut down_in_shape = xs.shape.clone();
        down_in_shape[rank - 1] = inter;
        let down_input = FastTensor::new(crate::tensor::workspace::get_pooled_buffer(inter), down_in_shape);

        if let Some(ln) = ffn_norm {
            let mut norm_shape = xs.shape.clone();
            norm_shape[rank - 1] = inter;
            // Clone gate_buf so the pooled buffer can be returned after.
            let mid = FastTensor::new(gate_buf.clone(), norm_shape);
            let normed = ln.forward(&mid)?;
            let mut quantized_mid = crate::tensor::workspace::get_pooled_buffer_i8(inter);
            let inv_scale_mid = crate::bit1_58::quantization::quantize_f32_to_i8(&normed.data, &mut quantized_mid);
            let result = down.forward_with_quantized(&down_input, &quantized_mid, inv_scale_mid)?;
            crate::tensor::workspace::return_pooled_buffer_i8(quantized_mid);
            crate::tensor::workspace::return_pooled_buffer(gate_buf);
            crate::tensor::workspace::return_pooled_buffer(down_input.into_data());
            return Ok(result);
        }

        let mut quantized_mid = crate::tensor::workspace::get_pooled_buffer_i8(inter);
        let inv_scale_mid = crate::bit1_58::quantization::quantize_f32_to_i8(&gate_buf, &mut quantized_mid);
        crate::tensor::workspace::return_pooled_buffer(gate_buf);
        let result = down.forward_with_quantized(&down_input, &quantized_mid, inv_scale_mid)?;
        crate::tensor::workspace::return_pooled_buffer_i8(quantized_mid);
        crate::tensor::workspace::return_pooled_buffer(down_input.into_data());

        Ok(result)
    }
}
