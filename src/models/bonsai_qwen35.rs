//! Bonsai-27B (qwen35 / Qwen3.5 hybrid architecture)
//!
//! Architecture:
//!   - 64 layers, blocks 3,7,11,... are full_attention, rest are linear_attention (SSM)
//!   - Full attention: gated multi-head attention with GQA + Q/K norms
//!   - Linear (SSM) attention: GatedDeltaNet (Mamba-2-style state-space model)
//!   - Both share: input_norm, post_attention_norm, SwiGLU MLP
//!
//! Optimization: all weights are pre-loaded (cached) at model creation.
//! No per-forward dequant.

use std::sync::OnceLock;
use std::sync::Arc;
use rayon::prelude::*;
use candle_core::{Device, Tensor, D};
use candle_core::Result as CandleResult;
use crate::loader::gguf::{GgufFile, self as gguf};

// ---------------------------------------------------------------------------
// Bitwise PRISM Q1_0 GEMV kernels (no dequant: dot product on packed signs)
// ---------------------------------------------------------------------------

/// 256-entry table: for each sign byte, the ±1.0 f32 lane values (bit j → lane j).
fn sign_table() -> &'static [[f32; 8]; 256] {
    static TABLE: OnceLock<[[f32; 8]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 256];
        for b in 0..256usize {
            for j in 0..8 {
                t[b][j] = if (b >> j) & 1 == 1 { -1.0 } else { 1.0 };
            }
        }
        t
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_block_avx2(x: &[f32], signs: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let table = sign_table();
    let mut acc = _mm256_setzero_ps();
    for i in 0..16 {
        let row = &table[signs[i] as usize];
        let xi = _mm256_loadu_ps(x.as_ptr().add(i * 8));
        let s = _mm256_loadu_ps(row.as_ptr());
        acc = _mm256_fmadd_ps(xi, s, acc);
    }
    let v = _mm256_hadd_ps(acc, acc);
    let v = _mm256_hadd_ps(v, v);
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_block_avx512(x: &[f32], signs: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let table = sign_table();
    let mut acc = _mm512_setzero_ps();
    for i in 0..8 {
        let b0 = signs[2 * i] as usize;
        let b1 = signs[2 * i + 1] as usize;
        let xi = _mm512_loadu_ps(x.as_ptr().add(i * 16));
        let s0 = _mm256_loadu_ps(table[b0].as_ptr());
        let s1 = _mm256_loadu_ps(table[b1].as_ptr());
        let s = _mm512_insertf32x8(_mm512_castps256_ps512(s0), s1, 1);
        acc = _mm512_fmadd_ps(xi, s, acc);
    }
    _mm512_reduce_add_ps(acc)
}

/// Dot product of a 128-elem activation slice with 16 sign bytes (bit=1 → −x).
fn dot_block(x: &[f32], signs: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { dot_block_avx512(x, signs) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { dot_block_avx2(x, signs) };
        }
    }
    let mut s = 0.0f32;
    for i in 0..16 {
        let byte = signs[i];
        for j in 0..8 {
            let xi = x[i * 8 + j];
            s += if (byte >> j) & 1 == 1 { -xi } else { xi };
        }
    }
    s
}

// ---------------------------------------------------------------------------
// LazyLinear — stores GGUF ref + name, dequantizes on demand during forward.
// Keeps RAM low: packed 1-bit weights stay in the mmap'd GGUF (~3.8 GB virtual),
// only one layer's f32 weights are materialized at a time (~0.5 GB peak).
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LazyLinear {
    gguf: Arc<GgufFile>,
    name: String,
    out_dim: usize,
    in_dim: usize,
}

impl LazyLinear {
    pub fn new_bench(gguf: &Arc<GgufFile>, name: &str) -> CandleResult<Self> {
        let info = gguf.tensor_info(name)
            .map_err(|_e| candle_core::Error::Msg("no tensor".into()))?;
        Ok(LazyLinear::new(gguf, name, info.shape[0], info.shape[1]))
    }
    pub fn in_dim(&self) -> usize { self.in_dim }
    pub fn out_dim(&self) -> usize { self.out_dim }
    pub fn bench_forward(&self, x: &[f32]) -> CandleResult<Vec<f32>> {
        let bytes = self.gguf.tensor_bytes(&self.name)
            .map_err(|_e| candle_core::Error::Msg("no bytes".into()))?;
        let blocks_per_row = self.in_dim.div_ceil(128);
        let row_bytes = blocks_per_row * 18;
        let chunks: Vec<(usize, usize)> = (0..self.out_dim)
            .step_by(512)
            .map(|s| (s, (s + 512).min(self.out_dim)))
            .collect();
        let out_chunks: Vec<Vec<f32>> = chunks
            .par_iter()
            .map(|&(lo, hi)| self.gemv_rows(x, bytes, lo, hi, self.in_dim, row_bytes, blocks_per_row))
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(out_chunks.into_iter().flatten().collect())
    }

    fn new(gguf: &Arc<GgufFile>, name: &str, out_dim: usize, in_dim: usize) -> Self {
        LazyLinear { gguf: gguf.clone(), name: name.to_string(), out_dim, in_dim }
    }

    /// Dequantize the full weight matrix to a row-major f32 Vec.
    fn load_f32(&self) -> CandleResult<Vec<f32>> {
        let info = self.gguf.tensor_info(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} not found", self.name)))?;
        let bytes = self.gguf.tensor_bytes(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} bytes", self.name)))?;
        gguf::dequantize_data(bytes, info.dtype_u32, info.elem_count)
            .map_err(|_e| candle_core::Error::Msg(format!("dequant {}", self.name)))
    }

    /// Dequantize rows [row_start, row_end) → f32 Vec of len rows*in_dim.
    /// Only the needed rows are unpacked (embedding + chunked matmul).
    fn load_rows(&self, row_start: usize, row_end: usize) -> CandleResult<Vec<f32>> {
        let info = self.gguf.tensor_info(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} not found", self.name)))?;
        let bytes = self.gguf.tensor_bytes(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} bytes", self.name)))?;

        match info.dtype_u32 {
            0 => { // F32: contiguous rows, in_dim*4 bytes each
                let row_bytes = self.in_dim * 4;
                let slice = &bytes[row_start * row_bytes..row_end * row_bytes];
                Ok(slice.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            41 | 43 => { // Upstream Q1_0 (128 elems / 18-byte block)
                let rows_per_block = 128usize;
                let blocks_per_row = self.in_dim.div_ceil(rows_per_block);
                let row_bytes = blocks_per_row * 18;
                let mut out = Vec::with_capacity((row_end - row_start) * self.in_dim);
                for row in row_start..row_end {
                    let base = row * row_bytes;
                    let mut row_vals = vec![0.0f32; self.in_dim];
                    let num_full_blocks = self.in_dim / rows_per_block;
                    for b in 0..num_full_blocks {
                        let bo = base + b * 18;
                        let scale = gguf::fp16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
                        let off = b * rows_per_block;
                        for i in 0..rows_per_block {
                            let bit = (bytes[bo + 2 + (i >> 3)] >> (i & 7)) & 1;
                            row_vals[off + i] = if bit == 0 { scale } else { -scale };
                        }
                    }
                    out.extend_from_slice(&row_vals);
                }
                Ok(out)
            }
            dt => Err(candle_core::Error::Msg(format!(
                "unsupported dtype {dt} for lazy row load"
            ))),
        }
    }

    fn forward_f32(&self, x: &Tensor) -> CandleResult<Tensor> {
        let w = self.load_f32()?;
        let wt = Tensor::from_vec(w, (self.out_dim, self.in_dim), x.device())?.t()?;
        x.broadcast_matmul(&wt)
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // For PRISM-packed weights use the bitwise GEMV (no dequant).
        // For F32 weights use the standard matmul.
        let info = self.gguf.tensor_info(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} not found", self.name)))?;
        match info.dtype_u32 {
            41 | 43 => self.forward_bitwise(x, 512),
            0 => self.forward_f32(x),
            dt => Err(candle_core::Error::Msg(format!("unsupported dtype {dt}"))),
        }
    }

    /// Bitwise GEMV for PRISM Q1_0 weights (type 41/43).
    /// W is packed as [out_dim, in_dim] in 128-element blocks:
    ///   per block: [f16 scale: 2B][128 sign bits: 16B] — bit 0 → +scale, bit 1 → -scale
    /// Computes out = x @ W^T without ever dequantizing to f32.
    /// Uses rayon to parallelize across output-row chunks.
    fn forward_bitwise(&self, x: &Tensor, chunk_rows: usize) -> CandleResult<Tensor> {
        let bytes = self.gguf.tensor_bytes(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} bytes", self.name)))?;
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let blocks_per_row = in_dim.div_ceil(128);
        let row_bytes = blocks_per_row * 18;
        let x_f32: Vec<f32> = x.flatten_all()?.to_vec1()?; // [seq*in_dim] or [in_dim]

        // Only supports the decode shape [1, in_dim] for now; prefill handled by f32 path.
        if x_f32.len() % in_dim != 0 {
            return Err(candle_core::Error::Msg("bitwise gemv: bad x shape".into()));
        }
        let rows_in_x = x_f32.len() / in_dim;
        if rows_in_x != 1 {
            // Fallback: prefill (seq_len > 1) uses the dequantized f32 matmul.
            return self.forward_f32(x);
        }

        // Rayon: parallelize over output row chunks.
        let chunks: Vec<(usize, usize)> = (0..out_dim)
            .step_by(chunk_rows)
            .map(|s| (s, (s + chunk_rows).min(out_dim)))
            .collect();
        let out_chunks: Vec<Vec<f32>> = chunks
            .par_iter()
            .map(|&(lo, hi)| self.gemv_rows(&x_f32, bytes, lo, hi, in_dim, row_bytes, blocks_per_row))
            .collect::<CandleResult<Vec<_>>>()?;
        let out: Vec<f32> = out_chunks.into_iter().flatten().collect();

        let t = Tensor::from_vec(out, (1, 1, out_dim), x.device())?;
        Ok(t)
    }

    /// Compute output rows [lo, hi) of y = x @ W^T via bitwise accumulation.
    #[inline(never)]
    fn gemv_rows(
        &self, x: &[f32], bytes: &[u8],
        lo: usize, hi: usize, in_dim: usize,
        row_bytes: usize, blocks_per_row: usize,
    ) -> CandleResult<Vec<f32>> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Ok(unsafe { gemv_rows_avx512(x, bytes, lo, hi, in_dim, row_bytes, blocks_per_row) });
            }
        }
        let mut out = vec![0.0f32; hi - lo];
        for (idx, row) in (lo..hi).enumerate() {
            let base = row * row_bytes;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let bo = base + b * 18;
                let scale = gguf::fp16_to_f32(u16::from_le_bytes([bytes[bo], bytes[bo + 1]]));
                let xo = b * 128;
                let signs = &bytes[bo + 2..bo + 18];
                acc += scale * dot_block(&x[xo..xo + 128], signs);
            }
            out[idx] = acc;
        }
        Ok(out)
    }

    /// Chunked matmul: x @ W^T computed in row-chunks to bound peak RAM.
    /// Uses the bitwise GEMV for PRISM-packed weights.
    fn forward_chunked(&self, x: &Tensor, chunk_rows: usize) -> CandleResult<Tensor> {
        let info = self.gguf.tensor_info(&self.name)
            .map_err(|_e| candle_core::Error::Msg(format!("tensor {} not found", self.name)))?;
        if matches!(info.dtype_u32, 41 | 43) {
            return self.forward_bitwise(x, chunk_rows);
        }
        let mut parts: Vec<Tensor> = Vec::new();
        for start in (0..self.out_dim).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(self.out_dim);
            let w_rows = self.load_rows(start, end)?; // (end-start) x in_dim
            let n_rows = end - start;
            let wt = Tensor::from_vec(w_rows, (n_rows, self.in_dim), x.device())?.t()?;
            let out = x.broadcast_matmul(&wt)?;
            parts.push(out);
        }
        Tensor::cat(&parts, D::Minus1)
    }
}

// ---------------------------------------------------------------------------
// RMS Norm
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
    dim: usize,
}

impl RmsNorm {
    fn from_gguf(gguf: &Arc<GgufFile>, name: &str, eps: f64) -> CandleResult<Self> {
        let info = gguf.tensor_info(name)
            .map_err(|_e| candle_core::Error::Msg("error".to_string()))?;
        let dim = info.shape.iter().product();
        let data = gguf.tensor_bytes(name)
            .map_err(|_e| candle_core::Error::Msg("error".to_string()))?;
        let f32_vec = gguf::dequantize_data(data, info.dtype_u32, info.elem_count)
            .map_err(|_e| candle_core::Error::Msg("error".to_string()))?;
        let weight = Tensor::from_vec(f32_vec, dim, &candle_core::Device::Cpu)?;
        Ok(RmsNorm { weight, eps, dim })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let norm = (x.powf(2.0)?.sum_keepdim(D::Minus1)? / self.dim as f64)?;
        let rms = norm.affine(1.0, self.eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&rms)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn silu(x: &Tensor) -> CandleResult<Tensor> { x.silu() }

fn sigmoid(x: &Tensor) -> CandleResult<Tensor> {
    let neg_x = x.affine(-1.0, 0.0)?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus = (exp_neg_x + 1.0f64)?;
    one_plus.recip()
}

fn softmax(x: &Tensor, dim: usize) -> CandleResult<Tensor> {
    let max = x.max_keepdim(dim)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    Ok(exp.broadcast_div(&sum)?)
}

fn precompute_rope(
    rotary_dim: usize,
    seq_len: usize,
    theta: f64,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        inv_freq.push(1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64));
    }
    let mut cos_data = vec![0.0f32; seq_len * half];
    let mut sin_data = vec![0.0f32; seq_len * half];
    for pos in 0..seq_len {
        for i in 0..half {
            let freq = pos as f64 * inv_freq[i];
            cos_data[pos * half + i] = freq.cos() as f32;
            sin_data[pos * half + i] = freq.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos_data, (seq_len, half), device)?;
    let sin = Tensor::from_vec(sin_data, (seq_len, half), device)?;
    Ok((cos, sin))
}

/// Apply RoPE to the first `rotary_dim` dims of the last axis; pass the rest through.
/// cos/sin: [seq_len, rotary_dim/2], x: [b, seq, n_head, head_dim]
fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor, seq_pos: usize, rotary_dim: usize) -> CandleResult<Tensor> {
    let (b_sz, seq_len, n_head, head_dim) = x.dims4()?;
    let half = rotary_dim / 2;
    let rot = x.narrow(D::Minus1, 0, rotary_dim)?;         // [b, s, h, rot]
    let passthrough = x.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?; // [b, s, h, rest]
    let x1 = rot.narrow(D::Minus1, 0, half)?;
    let x2 = rot.narrow(D::Minus1, half, half)?;
    let cos_slice = cos.narrow(0, seq_pos, seq_len)?.reshape((seq_len, 1, half))?;
    let sin_slice = sin.narrow(0, seq_pos, seq_len)?.reshape((seq_len, 1, half))?;
    let m1 = x1.broadcast_mul(&cos_slice)?;
    let m2 = x2.broadcast_mul(&sin_slice)?;
    let m3 = x1.broadcast_mul(&sin_slice)?;
    let m4 = x2.broadcast_mul(&cos_slice)?;
    let part0 = (m1 - m2)?;
    let part1 = (m3 + m4)?;
    let rotated = Tensor::cat(&[&part0, &part1], D::Minus1)?;
    let out = Tensor::cat(&[&rotated, &passthrough], D::Minus1)?;
    out.reshape((b_sz, seq_len, n_head, head_dim))
}

fn make_causal_mask(seq_len: usize, kv_len: usize, _seq_pos: usize, device: &Device) -> CandleResult<Tensor> {
    if seq_len == 1 {
        return Tensor::zeros((1, 1, 1, kv_len), candle_core::DType::F32, device);
    }
    let prior_count = kv_len - seq_len;
    let data: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let max_visible = prior_count + i;
            (0..kv_len).map(move |j| {
                if j <= max_visible { 0.0 } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, seq_len, kv_len), device)
}

// ---------------------------------------------------------------------------
// Full Attention block (blocks 3,7,11,...)
// Qwen3.5 gated attention: q_proj outputs 2× head_dim (Q + gate)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FullAttention {
    q_proj: LazyLinear,     // [12288, 5120] = 2*n_heads*head_dim (Q+gate)
    k_proj: LazyLinear,     // [1024,  5120] = n_kv_heads*head_dim
    v_proj: LazyLinear,     // [1024,  5120] = n_kv_heads*head_dim
    o_proj: LazyLinear,     // [5120,  6144]
    q_norm: RmsNorm,       // [256]
    k_norm: RmsNorm,       // [256]
    n_head: usize,         // 24
    n_kv_head: usize,    // 4
    head_dim: usize,     // 256
    rotary_dim: usize,   // 128 (partial RoPE)
}

impl FullAttention {
    fn load(gguf: &Arc<GgufFile>, pfx: &str, head_dim: usize, n_head: usize, n_kv: usize, eps: f64, rotary_dim: usize) -> CandleResult<Self> {
        let q_out = 2 * n_head * head_dim;
        let kv_out = n_kv * head_dim;
        let o_in = n_head * head_dim;
        Ok(FullAttention {
            q_proj: LazyLinear::new(gguf, &format!("{pfx}.attn_q.weight"), q_out, tensor_in_dim(gguf, &format!("{pfx}.attn_q.weight"))?),
            k_proj: LazyLinear::new(gguf, &format!("{pfx}.attn_k.weight"), kv_out, tensor_in_dim(gguf, &format!("{pfx}.attn_k.weight"))?),
            v_proj: LazyLinear::new(gguf, &format!("{pfx}.attn_v.weight"), kv_out, tensor_in_dim(gguf, &format!("{pfx}.attn_v.weight"))?),
            o_proj: LazyLinear::new(gguf, &format!("{pfx}.attn_output.weight"), tensor_out_dim(gguf, &format!("{pfx}.attn_output.weight"))?, o_in),
            q_norm: RmsNorm::from_gguf(gguf, &format!("{pfx}.attn_q_norm.weight"), eps)?,
            k_norm: RmsNorm::from_gguf(gguf, &format!("{pfx}.attn_k_norm.weight"), eps)?,
            n_head, n_kv_head: n_kv, head_dim, rotary_dim,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, seq_pos: usize, cache: &mut (Tensor, Tensor)) -> CandleResult<Tensor> {
        let b_sz = 1usize;
        let seq_len = x.dim(1)?;

        // Q (with gate): q_proj → [1, seq, 12288] = n_heads × [Q(256) | gate(256)]
        // NOTE: Q and gate are INTERLEAVED per head in the GGUF (matching HF's
        // view[.., -1, head_dim*2] + chunk(dim=-1)), not block-concatenated.
        let qg = self.q_proj.forward(x)?;
        let qg = qg.reshape((b_sz, seq_len, self.n_head, 2 * self.head_dim))?;
        let q = qg.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = qg.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        let k = self.k_proj.forward(x)?.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = self.v_proj.forward(x)?.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;

        // Q/K norms (applied BEFORE RoPE, matching the reference)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Partial RoPE (first rotary_dim=128 of each 256-dim head)
        let q = apply_rotary(&q, cos, sin, seq_pos, self.rotary_dim)?;
        let k = apply_rotary(&k, cos, sin, seq_pos, self.rotary_dim)?;

        // Transpose → [b, n_heads, seq, head_dim]
        let q = q.transpose(1,2)?;
        let k = k.transpose(1,2)?;
        let v = v.transpose(1,2)?;

        // KV cache
        let k = Tensor::cat(&[&cache.0, &k], 2)?;
        let v = Tensor::cat(&[&cache.1, &v], 2)?;
        cache.0 = k.clone();
        cache.1 = v.clone();
        let kv_len = k.dim(2)?;

        // GQA
        let n_groups = self.n_head / self.n_kv_head;
        let k = k.unsqueeze(2)?.expand((b_sz, self.n_kv_head, n_groups, kv_len, self.head_dim))?.reshape((b_sz, self.n_head, kv_len, self.head_dim))?;
        let v = v.unsqueeze(2)?.expand((b_sz, self.n_kv_head, n_groups, kv_len, self.head_dim))?.reshape((b_sz, self.n_head, kv_len, self.head_dim))?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2,3)?)?.affine(scale, 0.0)?;
        let mask = make_causal_mask(seq_len, kv_len, seq_pos, x.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let attn = softmax(&scores, 3)?;
        let attn_out = attn.matmul(&v)?;

        // Gate: sigmoid(gate) * attn_out
        let gate = gate.transpose(1,2)?;
        let gate = sigmoid(&gate)?;

        if std::env::var("BONSAI_DBG").is_ok() && seq_len == 1 {
            let vv: Vec<f32> = v.flatten_all()?.to_vec1()?;
            let gg: Vec<f32> = gate.flatten_all()?.to_vec1()?;
            eprintln!("  [attn] v[:4]={:?}", &vv[..4]);
            eprintln!("  [attn] gate[:4]={:?}", &gg[..4]);
        }

        let attn_out = (attn_out * gate)?;

        if std::env::var("BONSAI_DBG").is_ok() && seq_len == 1 {
            let attn_flat: Vec<f32> = attn_out.flatten_all()?.to_vec1()?;
            eprintln!("  [attn] gated_attn[:4]={:?}", &attn_flat[..4]);
        }

        let attn_out = attn_out.transpose(1,2)?.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;
        self.o_proj.forward(&attn_out)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet (SSM / linear attention) block
// Reference: transformers torch_recurrent_gated_delta_rule
//
// Per step t:
//   g_t = exp(sigmoid(alpha_t))                     [n_vh, 1, 1]
//   S   = S * g_t                                 [n_vh, k_hd, v_hd]
//   kv_mem = sum_k(S * k_t)                          [n_vh, v_hd]
//   delta = (v_t - kv_mem) * beta_t                    [n_vh, v_hd]
//   S   = S + outer(k_t, delta)                      [n_vh, k_hd, v_hd]
//   y   = sum_k(S * q_t_scaled)                      [n_vh, v_hd]
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[allow(dead_code)]
struct GatedDeltaNet {
    in_proj_qkv: LazyLinear,  // [10240, 5120] → 2*key_dim + value_dim
    in_proj_z: LazyLinear,    // [6144,  5120] → gate
    in_proj_a: LazyLinear,    // [48,   5120] → alpha
    in_proj_b: LazyLinear,    // [48,   5120] → beta
    conv1d_w: Tensor,        // [10240, 4]   → depthwise conv weights
    dt_bias_f32: Vec<f32>,   // [48]         → time step bias
    a_log_f32: Vec<f32>,     // [48]         → A_log (diagonal state decay)
    norm: RmsNorm,        // [128]        → per-v-head norm
    out_proj: LazyLinear,  // [5120, 6144] → output projection
    key_dim: usize,     // 2048
    value_dim: usize,   // 6144
    num_v_heads: usize, // 48
    num_k_heads: usize, // 16
    head_dim: usize,   // 128 (shared by k and v heads)
    conv_kernel: usize, // 4
}

impl GatedDeltaNet {
    fn load(gguf: &Arc<GgufFile>, pfx: &str, cfg: &BonsaiQwen35Config) -> CandleResult<Self> {
        let p = |n| format!("{pfx}.{n}");
        let conv_dim = cfg.ssm_key_dim * 2 + cfg.ssm_value_dim;
        Ok(GatedDeltaNet {
            in_proj_qkv: LazyLinear::new(gguf, &p("attn_qkv.weight"), conv_dim, cfg.hidden_size),
            in_proj_z: LazyLinear::new(gguf, &p("attn_gate.weight"), cfg.ssm_value_dim, cfg.hidden_size),
            in_proj_a: LazyLinear::new(gguf, &p("ssm_alpha.weight"), cfg.ssm_num_v_heads, cfg.hidden_size),
            in_proj_b: LazyLinear::new(gguf, &p("ssm_beta.weight"), cfg.ssm_num_v_heads, cfg.hidden_size),
            conv1d_w: Tensor::from_vec(load_f32_simple(gguf, &p("ssm_conv1d.weight"))?, (conv_dim, cfg.ssm_conv_kernel), &candle_core::Device::Cpu)?,
            dt_bias_f32: load_f32_simple(gguf, &p("ssm_dt.bias"))?,
            a_log_f32: load_f32_simple(gguf, &p("ssm_a"))?,
            norm: RmsNorm::from_gguf(gguf, &p("ssm_norm.weight"), cfg.rms_norm_eps)?,
            out_proj: LazyLinear::new(gguf, &p("ssm_out.weight"), cfg.hidden_size, cfg.ssm_value_dim),
            key_dim: cfg.ssm_key_dim,
            value_dim: cfg.ssm_value_dim,
            num_v_heads: cfg.ssm_num_v_heads,
            num_k_heads: cfg.ssm_num_k_heads,
            head_dim: cfg.ssm_head_v_dim,
            conv_kernel: cfg.ssm_conv_kernel,
        })
    }

    fn forward(&self, x: &Tensor, _seq_pos: usize, ssm_state: &mut SsmState) -> CandleResult<Tensor> {
        let seq_len = x.dim(1)?;

        // 1. Projections
        let qkv = self.in_proj_qkv.forward(x)?;   // [1, seq, 10240]
        let z = self.in_proj_z.forward(x)?;       // [1, seq, 6144]
        let a_coeff = self.in_proj_a.forward(x)?;   // [1, seq, 48]
        let b_coeff = self.in_proj_b.forward(x)?;   // [1, seq, 48]

        // 2. Causal conv1d (depthwise, along seq) + SiLU
        let qkv = apply_conv1d(&qkv, &self.conv1d_w, self.conv_kernel, ssm_state)?;

        // 3. Split QKV: Q[0:key_dim], K[key_dim:2*key_dim], V[2*key_dim:end]
        let q_raw = qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k_raw = qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_raw = qkv.narrow(D::Minus1, 2*self.key_dim, self.value_dim)?;

        // 4. Reshape per-head: Q/K→n_k_heads, V→n_v_heads
        let q = q_raw.reshape((1, seq_len, self.num_k_heads, self.head_dim))?;
        let k = k_raw.reshape((1, seq_len, self.num_k_heads, self.head_dim))?;
        let v = v_raw.reshape((1, seq_len, self.num_v_heads, self.head_dim))?;

        // 5. Delta rule per token (loop for prefill, single-step for decode)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let heads_per_k = self.num_v_heads / self.num_k_heads;  // 3

        if seq_len == 1 {
            // Decode: single-step
            let q_p = q.squeeze(1)?.squeeze(0)?;
            let k_p = k.squeeze(1)?.squeeze(0)?;
            let v_p = v.squeeze(1)?.squeeze(0)?;
            let z_p = z.squeeze(1)?.squeeze(0)?;
            let a_p = a_coeff.squeeze(1)?.squeeze(0)?;
            let b_p = b_coeff.squeeze(1)?.squeeze(0)?;
            return self.ssm_step(&q_p, &k_p, &v_p, &z_p, &a_p, &b_p, scale, heads_per_k, ssm_state);
        }

        // Prefill: per-token loop
        let mut parts: Vec<Tensor> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let q_p = q.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let k_p = k.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let v_p = v.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let z_p = z.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let a_p = a_coeff.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let b_p = b_coeff.narrow(1, pos,1)?.squeeze(1)?.squeeze(0)?;
            let out = self.ssm_step(&q_p, &k_p, &v_p, &z_p, &a_p, &b_p, scale, heads_per_k, ssm_state)?;
            parts.push(out);  // already [1, 1, hidden]
        }
        Tensor::cat(&parts, 1)
    }

    /// Delta rule for a single (q,k,v) step, all in plain f32 (no tiny tensors).
    fn ssm_step(
        &self, q: &Tensor, k: &Tensor, v: &Tensor, z: &Tensor,
        a: &Tensor, b: &Tensor, scale: f64,
        heads_per_k: usize, state: &mut SsmState,
    ) -> CandleResult<Tensor> {
        let hd = self.head_dim;
        let n_vh = self.num_v_heads;
        let n_kh = self.num_k_heads;

        let q: Vec<f32> = q.flatten_all()?.to_vec1()?;   // [n_kh*hd]
        let k: Vec<f32> = k.flatten_all()?.to_vec1()?;   // [n_kh*hd]
        let v: Vec<f32> = v.flatten_all()?.to_vec1()?;   // [n_vh*hd]
        let z: Vec<f32> = z.flatten_all()?.to_vec1()?;   // [value_dim]
        let a: Vec<f32> = a.flatten_all()?.to_vec1()?;   // [n_vh]
        let b: Vec<f32> = b.flatten_all()?.to_vec1()?;   // [n_vh]

        if std::env::var("BONSAI_DBG").is_ok() {
            eprintln!("  [ssm] q[:4]={:?}", &q[..4]);
            eprintln!("  [ssm] k[:4]={:?}", &k[..4]);
            eprintln!("  [ssm] v[:4]={:?}", &v[..4]);
            eprintln!("  [ssm] z[:4]={:?}", &z[..4]);
            eprintln!("  [ssm] a[:4]={:?}", &a[..4]);
            eprintln!("  [ssm] b[:4]={:?}", &b[..4]);
        }

        // g = -exp(A_log) * softplus(a + dt_bias)  [log space]
        // beta = sigmoid(b)
        let mut g = vec![0.0f32; n_vh];
        let mut beta = vec![0.0f32; n_vh];
        for h in 0..n_vh {
            let a_val = self.a_log_f32[h].exp();
            let softplus = (1.0f32 + (a[h] + self.dt_bias_f32[h]).exp()).ln();
            g[h] = -a_val * softplus;
            beta[h] = 1.0 / (1.0 + (-b[h]).exp());
        }
        if std::env::var("BONSAI_DBG").is_ok() {
            eprintln!("  [ssm] a_log[:4]={:?}", &self.a_log_f32[..4]);
            eprintln!("  [ssm] dt_bias[:4]={:?}", &self.dt_bias_f32[..4]);
            eprintln!("  [ssm] g[:4]={:?}", &g[..4]);
            eprintln!("  [ssm] beta[:4]={:?}", &beta[..4]);
        }

        // z gate: silu(z)
        let mut z_gate = vec![0.0f32; z.len()];
        for (i, &zi) in z.iter().enumerate() {
            let sig = 1.0 / (1.0 + (-zi).exp());
            z_gate[i] = zi * sig;
        }

        // L2-normalize q and k per key head (eps=1e-6)
        let mut qn = vec![0.0f32; n_kh * hd];
        let mut kn = vec![0.0f32; n_kh * hd];
        for h in 0..n_kh {
            let mut sq_q = 0.0f32;
            let mut sq_k = 0.0f32;
            for j in 0..hd {
                sq_q += q[h * hd + j] * q[h * hd + j];
                sq_k += k[h * hd + j] * k[h * hd + j];
            }
            let iq = (sq_q + 1e-6).sqrt().recip();
            let ik = (sq_k + 1e-6).sqrt().recip();
            for j in 0..hd {
                qn[h * hd + j] = q[h * hd + j] * iq;
                kn[h * hd + j] = k[h * hd + j] * ik;
            }
        }

        let q_scale = scale as f32;

        // Per-v-head delta rule in plain loops
        let mut out_heads = vec![0.0f32; n_vh * hd];
        for vh in 0..n_vh {
            let kh = vh / heads_per_k;
            let k_i = &kn[kh * hd..(kh + 1) * hd];
            let q_i = &qn[kh * hd..(kh + 1) * hd];
            let v_i = &v[vh * hd..(vh + 1) * hd];
            let decay = g[vh].exp();
            let beta_h = beta[vh];

            let s = &mut state.s[vh];  // [hd*hd] row-major: s[i*hd + j]

            // S = S * exp(g)
            for e in s.iter_mut() { *e *= decay; }

            // kv_mem[j] = sum_i S[i][j] * k_i[i]
            let mut kv_mem = vec![0.0f32; hd];
            for j in 0..hd {
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += s[i * hd + j] * k_i[i];
                }
                kv_mem[j] = acc;
            }

            // delta[j] = (v_i[j] - kv_mem[j]) * beta_h
            let mut delta = vec![0.0f32; hd];
            for j in 0..hd {
                delta[j] = (v_i[j] - kv_mem[j]) * beta_h;
            }

            // S[i][j] += k_i[i] * delta[j]
            for i in 0..hd {
                let ki = k_i[i];
                for j in 0..hd {
                    s[i * hd + j] += ki * delta[j];
                }
            }

            // y[j] = sum_i S[i][j] * q_i[i] * scale
            let out_base = vh * hd;
            for j in 0..hd {
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += s[i * hd + j] * q_i[i];
                }
                out_heads[out_base + j] = acc * q_scale;
            }
        }

        // Per-head RMS norm + gate (f32)
        let norm_w: Vec<f32> = self.norm.weight.flatten_all()?.to_vec1()?;
        let eps = self.norm.eps as f32;
        for vh in 0..n_vh {
            let base = vh * hd;
            let mut sq_sum = 0.0f32;
            for j in 0..hd {
                let x = out_heads[base + j];
                sq_sum += x * x;
            }
            let rms = (sq_sum / hd as f32 + eps).sqrt();
            for j in 0..hd {
                let idx = base + j;
                out_heads[idx] = out_heads[idx] / rms * norm_w[j] * z_gate[idx];
            }
        }

        let out = Tensor::from_vec(out_heads, (1, 1, self.value_dim), &candle_core::Device::Cpu)?;
        self.out_proj.forward(&out)
    }
}

/// AVX-512 row GEMV: fully inlined, 16 lanes, fast f16 scale decode.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemv_rows_avx512(
    x: &[f32], bytes: &[u8], lo: usize, hi: usize,
    _in_dim: usize, row_bytes: usize, blocks_per_row: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    let table = sign_table();
    let mut out = vec![0.0f32; hi - lo];
    for (idx, row) in (lo..hi).enumerate() {
        let base = row * row_bytes;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let bo = base + b * 18;
            let half = u16::from_le_bytes([*bytes.get_unchecked(bo), *bytes.get_unchecked(bo + 1)]);
            // Fast f16→f32: exp=0 or 31 handled; normal: (exp+112)<<23 | mant<<13
            let sign = ((half & 0x8000) as u32) << 16;
            let exp = ((half & 0x7c00) as u32) >> 10;
            let mant = (half & 0x03ff) as u32;
            let scale = if exp == 0 {
                if mant == 0 { f32::from_bits(sign) } else { 0.0 } // denormal ≈ 0 (rare)
            } else if exp == 31 {
                f32::from_bits(sign | 0x7f800000 | (mant << 13))
            } else {
                f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
            };
            let xo = b * 128;
            let signs = bytes.get_unchecked(bo + 2..bo + 18);
            let mut v = _mm512_setzero_ps();
            for i in 0..8 {
                let b0 = *signs.get_unchecked(2 * i) as usize;
                let b1 = *signs.get_unchecked(2 * i + 1) as usize;
                let xi = _mm512_loadu_ps(x.as_ptr().add(xo + i * 16));
                let s0 = _mm256_loadu_ps(table[b0].as_ptr());
                let s1 = _mm256_loadu_ps(table[b1].as_ptr());
                let s = _mm512_insertf32x8(_mm512_castps256_ps512(s0), s1, 1);
                v = _mm512_fmadd_ps(xi, s, v);
            }
            acc += scale * _mm512_reduce_add_ps(v);
        }
        out[idx] = acc;
    }
    out
}

/// Depthwise causal conv1d along sequence dim.
/// conv1d_w: [conv_dim, kernel_size], qkv: [1, seq, conv_dim]
/// For each pos t: out[c,t] = sum_k(conv_w[c,k] * qkv[c,t-k]) with left-zero-pad.
/// History from previous tokens is kept in state.conv_hist (pad rows).
fn apply_conv1d(
    qkv: &Tensor, conv1d_w: &Tensor, kernel_size: usize,
    state: &mut SsmState,
) -> CandleResult<Tensor> {
    let seq_len = qkv.dim(1)?;
    let conv_dim = qkv.dim(D::Minus1)?;
    let qkv_f32 = qkv.to_dtype(candle_core::DType::F32)?;

    // Build padded sequence: [hist (pad rows); qkv (seq rows)] → [1, pad+seq, dim]
    let pad = kernel_size - 1;
    let hist_flat: Vec<f32> = state.conv_hist.clone();
    let qkv_flat: Vec<f32> = qkv_f32.flatten_all()?.to_vec1()?;
    let mut padded_flat = Vec::with_capacity((pad + seq_len) * conv_dim);
    padded_flat.extend_from_slice(&hist_flat[..pad * conv_dim]);
    padded_flat.extend_from_slice(&qkv_flat);
    let padded = Tensor::from_vec(padded_flat, (1, pad + seq_len, conv_dim), qkv.device())?;

    // out[t] = sum_k conv_w[:,k] * padded[:, t+k, :]
    let mut acc: Option<Tensor> = None;
    for k in 0..kernel_size {
        let slice = padded.narrow(1, k, seq_len)?;       // [1, seq, dim]
        let w_k = conv1d_w.narrow(1, k, 1)?.squeeze(1)?; // [dim]
        let term = slice.broadcast_mul(&w_k)?;           // [1, seq, dim]
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    let out = acc.unwrap().silu()?;

    // Update history: last pad rows of the padded sequence.
    {
        let last = padded.narrow(1, seq_len, pad)?;
        let h: Vec<f32> = last.flatten_all()?.to_vec1()?;
        state.conv_hist = h;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SSM state: per-v-head state matrices [head_dim, head_dim] + conv1d history
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SsmState {
    /// State matrices: [num_v_heads] × [head_dim, head_dim] as flat f32 (row-major)
    pub s: Vec<Vec<f32>>,
    /// Conv1d history: [conv_dim, kernel_size-1] buffer (flat, row-major)
    pub conv_hist: Vec<f32>,
    pub conv_dim: usize,
    pub conv_pad: usize,
}

impl SsmState {
    fn new(num_v_heads: usize, head_dim: usize, conv_dim: usize, kernel_size: usize, _device: &Device) -> CandleResult<Self> {
        let s = (0..num_v_heads)
            .map(|_| vec![0.0f32; head_dim * head_dim])
            .collect::<Vec<_>>();
        let conv_pad = kernel_size.saturating_sub(1);
        let conv_hist = vec![0.0f32; conv_dim * conv_pad.max(1)];
        Ok(SsmState { s, conv_hist, conv_dim, conv_pad })
    }
}

// ---------------------------------------------------------------------------
// MLP (shared by both block types)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Mlp {
    gate_proj: LazyLinear,
    up_proj: LazyLinear,
    down_proj: LazyLinear,
}

impl Mlp {
    fn load(gguf: &Arc<GgufFile>, pfx: &str, hidden: usize, inter: usize) -> CandleResult<Self> {
        Ok(Mlp {
            gate_proj: LazyLinear::new(gguf, &format!("{pfx}.ffn_gate.weight"), inter, hidden),
            up_proj: LazyLinear::new(gguf, &format!("{pfx}.ffn_up.weight"), inter, hidden),
            down_proj: LazyLinear::new(gguf, &format!("{pfx}.ffn_down.weight"), hidden, inter),
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// Block: either FullAttention or GatedDeltaNet
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum BlockKind {
    FullAttention(FullAttention),
    GatedDeltaNet(GatedDeltaNet),
}

#[derive(Clone)]
struct Block {
    kind: BlockKind,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    mlp: Mlp,
    is_full_attn: bool,
}

#[allow(dead_code)]
impl Block {
    fn forward(
        &self, x: &Tensor, cos: &Tensor, sin: &Tensor,
        seq_pos: usize, attn_cache: &mut (Tensor, Tensor),
        ssm_state: &mut SsmState,
    ) -> CandleResult<Tensor> {
        let residual = x.clone();
        let x = self.input_norm.forward(x)?;

        let x = match &self.kind {
            BlockKind::FullAttention(attn) => attn.forward(&x, cos, sin, seq_pos, attn_cache)?,
            BlockKind::GatedDeltaNet(ssm) => ssm.forward(&x, seq_pos, ssm_state)?,
        };
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.post_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BonsaiQwen35Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_rotary_dim: usize,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_key_dim: usize,
    pub ssm_value_dim: usize,
    pub ssm_head_v_dim: usize,
    pub ssm_num_v_heads: usize,
    pub ssm_num_k_heads: usize,
    pub has_output_weight: bool,
}

impl BonsaiQwen35Config {
    pub fn from_gguf(gguf: &GgufFile) -> anyhow::Result<Self> {
        let meta = &gguf.metadata.metadata;
        let arch = meta.get("general.architecture")
            .and_then(|v| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))?;
        anyhow::ensure!(arch == "qwen35", "expected qwen35, got {arch}");

        let p = |k: &str| format!("{arch}.{k}");
        let get_u32 = |k: &str| -> anyhow::Result<usize> {
            meta.get(&p(k)).and_then(|v| v.to_u32()).map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("missing {}", p(k)))
        };
        let get_f32 = |k: &str| -> anyhow::Result<f32> {
            meta.get(&p(k)).and_then(|v| v.to_f32())
                .ok_or_else(|| anyhow::anyhow!("missing {}", p(k)))
        };

        let hidden = get_u32("embedding_length")?;
        let n_heads = get_u32("attention.head_count")?;
        let n_kv = get_u32("attention.head_count_kv")?;
        let head_dim = get_u32("attention.key_length")?;
        let n_layers = get_u32("block_count")?;
        let ctx = get_u32("context_length")?;
        let rms_eps = get_f32("attention.layer_norm_rms_epsilon")? as f64;
        let rope_theta = meta.get(&p("rope.freq_base")).and_then(|v| v.to_f32()).unwrap_or(10000000.0);
        // rope.dimension_count = number of inv_freq values → rotary dim = 2 × count
        let rope_dim_count = meta.get(&p("rope.dimension_count")).and_then(|v| v.to_u32()).unwrap_or((head_dim / 2) as u32) as usize;
        let rope_rotary_dim = 2 * rope_dim_count;
        let inter = get_u32("feed_forward_length")?;
        let fa_interval = get_u32("full_attention_interval")?;
        let ssm_state = get_u32("ssm.state_size")?;
        let ssm_groups = get_u32("ssm.group_count")?;
        let ssm_ts_rank = get_u32("ssm.time_step_rank")?;
        let ssm_inner = get_u32("ssm.inner_size")?;
        let ssm_conv = get_u32("ssm.conv_kernel")?;

        let num_v_heads = ssm_ts_rank;
        let value_dim = ssm_inner;
        let head_v_dim = value_dim / num_v_heads;
        let key_dim = head_v_dim * ssm_groups;
        let num_k_heads = ssm_groups;

        // Verify QKV split: 2*key_dim + value_dim = attn_qkv out_dim
        if let Some(info) = gguf.metadata.tensor_infos.get("blk.0.attn_qkv.weight") {
            let expected = 2 * key_dim + value_dim;
            anyhow::ensure!(info.shape[0] == expected,
                "attn_qkv mismatch: {} vs {}", info.shape[0], expected);
        }

        let vocab = meta.get("tokenizer.ggml.tokens").and_then(|v| v.array_len())
            .ok_or_else(|| anyhow::anyhow!("missing tokenizer.ggml.tokens"))?;
        let has_output = gguf.has_tensor("output.weight");

        Ok(BonsaiQwen35Config {
            hidden_size: hidden, intermediate_size: inter, vocab_size: vocab,
            num_hidden_layers: n_layers, num_attention_heads: n_heads,
            num_key_value_heads: n_kv, head_dim, rms_norm_eps: rms_eps,
            rope_theta, rope_rotary_dim, max_position_embeddings: ctx, full_attention_interval: fa_interval,
            ssm_conv_kernel: ssm_conv, ssm_state_size: ssm_state,
            ssm_group_count: ssm_groups, ssm_time_step_rank: ssm_ts_rank,
            ssm_key_dim: key_dim, ssm_value_dim: value_dim,
            ssm_head_v_dim: head_v_dim, ssm_num_v_heads: num_v_heads,
            ssm_num_k_heads: num_k_heads, has_output_weight: has_output,
        })
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct BonsaiQwen35Model {
    config: BonsaiQwen35Config,
    token_embd: LazyLinear,
    output_norm: RmsNorm,
    lm_head: LazyLinear,
    blocks: Vec<Block>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl BonsaiQwen35Model {
    pub fn from_gguf(gguf: Arc<GgufFile>, device: &Device) -> anyhow::Result<Self> {
        let config = BonsaiQwen35Config::from_gguf(&gguf)?;
        println!("Loading Bonsai-27B (qwen35): {} layers, {} hidden, {} heads/{} KV",
            config.num_hidden_layers, config.hidden_size,
            config.num_attention_heads, config.num_key_value_heads);

        let hidden = config.hidden_size;
        let token_embd = LazyLinear::new(&gguf, "token_embd.weight", config.vocab_size, hidden);
        let output_norm = RmsNorm::from_gguf(&gguf, "output_norm.weight", config.rms_norm_eps)?;
        let lm_head = if config.has_output_weight {
            LazyLinear::new(&gguf, "output.weight", config.vocab_size, hidden)
        } else {
            LazyLinear::new(&gguf, "token_embd.weight", config.vocab_size, hidden)
        };

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let pfx = format!("blk.{i}");
            let is_full = i % config.full_attention_interval == 3;
            let input_norm = RmsNorm::from_gguf(&gguf, &format!("{pfx}.attn_norm.weight"), config.rms_norm_eps)?;
            let post_norm = RmsNorm::from_gguf(&gguf, &format!("{pfx}.post_attention_norm.weight"), config.rms_norm_eps)?;
            let mlp = Mlp::load(&gguf, &pfx, hidden, config.intermediate_size)?;
            let kind = if is_full {
                BlockKind::FullAttention(FullAttention::load(&gguf, &pfx, config.head_dim, config.num_attention_heads, config.num_key_value_heads, config.rms_norm_eps, config.rope_rotary_dim)?)
            } else {
                BlockKind::GatedDeltaNet(GatedDeltaNet::load(&gguf, &pfx, &config)?)
            };
            blocks.push(Block { kind, input_norm, post_norm, mlp, is_full_attn: is_full });
        }

        let (cos, sin) = precompute_rope(config.rope_rotary_dim, config.max_position_embeddings, config.rope_theta as f64, device)?;
        Ok(BonsaiQwen35Model { config, token_embd, output_norm, lm_head, blocks, cos, sin, device: device.clone() })
    }

    pub fn forward(&self, tokens: &[u32], seq_pos: usize, cache: &mut BonsaiQwen35Cache) -> CandleResult<Tensor> {
        let x = self.embed(tokens)?;
        self.forward_embed(&x, seq_pos, cache)
    }

    /// Debug: run just block `idx` (SSM or attention) on an embedded input.
    pub fn debug_block(&self, idx: usize, x: &Tensor, seq_pos: usize, cache: &mut BonsaiQwen35Cache) -> CandleResult<Tensor> {
        let block = &self.blocks[idx];
        let attn_cache = &mut cache.attn_caches[idx];
        let ssm_state = &mut cache.ssm_states[idx];
        let x_norm = block.input_norm.forward(x)?;
        let out = match &block.kind {
            BlockKind::FullAttention(attn) => attn.forward(&x_norm, &self.cos, &self.sin, seq_pos, attn_cache)?,
            BlockKind::GatedDeltaNet(ssm) => ssm.forward(&x_norm, seq_pos, ssm_state)?,
        };
        Ok(out)
    }

    /// Debug: dump the embedded input for a token list.
    pub fn debug_embed(&self, tokens: &[u32]) -> CandleResult<Tensor> {
        self.embed(tokens)
    }

    /// Embedding via row lookup — only the needed rows are dequantized.
    fn embed(&self, tokens: &[u32]) -> CandleResult<Tensor> {
        let hidden = self.config.hidden_size;
        let mut rows: Vec<f32> = Vec::with_capacity(tokens.len() * hidden);
        let mut max_id = 0usize;
        for &t in tokens {
            max_id = max_id.max(t as usize);
        }
        // Load the needed rows in one contiguous call (rows are usually contiguous).
        let start = tokens.iter().map(|&t| t as usize).min().unwrap_or(0);
        let row_vals = self.token_embd.load_rows(start, max_id + 1)?;
        for &t in tokens {
            let off = (t as usize - start) * hidden;
            rows.extend_from_slice(&row_vals[off..off + hidden]);
        }
        let t = Tensor::from_vec(rows, (tokens.len(), hidden), &self.device)?;
        t.unsqueeze(0)
    }

    pub fn forward_embed(&self, x: &Tensor, seq_pos: usize, cache: &mut BonsaiQwen35Cache) -> CandleResult<Tensor> {
        let profile = std::env::var("BONSAI_PROFILE").is_ok();
        let mut t_ssm = 0.0f64;
        let mut t_attn = 0.0f64;
        let mut t_mlp = 0.0f64;
        let mut x = x.clone();
        for i in 0..self.blocks.len() {
            let block = &self.blocks[i];
            let attn_cache = &mut cache.attn_caches[i];
            let ssm_state = &mut cache.ssm_states[i];

            if block.is_full_attn {
                let residual = x.clone();
                let x_norm = block.input_norm.forward(&x)?;
                let t0 = std::time::Instant::now();
                if let BlockKind::FullAttention(ref attn) = block.kind {
                    let attn_out = attn.forward(&x_norm, &self.cos, &self.sin, seq_pos, attn_cache)?;
                    x = (residual + attn_out)?;
                }
                if profile { t_attn += t0.elapsed().as_secs_f64(); }
            } else {
                let residual = x.clone();
                let x_norm = block.input_norm.forward(&x)?;
                let t0 = std::time::Instant::now();
                if let BlockKind::GatedDeltaNet(ref ssm) = block.kind {
                    let ssm_out = ssm.forward(&x_norm, seq_pos, ssm_state)?;
                    x = (residual + ssm_out)?;
                }
                if profile { t_ssm += t0.elapsed().as_secs_f64(); }
            }

            let residual = x.clone();
            let x_norm = block.post_norm.forward(&x)?;
            let t0 = std::time::Instant::now();
            let mlp_out = block.mlp.forward(&x_norm)?;
            x = (residual + mlp_out)?;
            if profile { t_mlp += t0.elapsed().as_secs_f64(); }
        }

        if profile {
            eprintln!("  [profile] ssm={:.0}ms attn={:.0}ms mlp={:.0}ms", t_ssm*1000.0, t_attn*1000.0, t_mlp*1000.0);
        }

        x = self.output_norm.forward(&x)?;
        let last = x.narrow(1, x.dim(1)? - 1, 1)?.squeeze(1)?;
        // Chunked lm_head matmul (avoids materializing 5 GB of vocab×hidden f32).
        // Returns [1, vocab] (batch × vocab), matching the qwen3 model's convention.
        self.lm_head.forward_chunked(&last, 4096)?.squeeze(0)
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

pub struct BonsaiQwen35Cache {
    pub attn_caches: Vec<(Tensor, Tensor)>,
    pub ssm_states: Vec<SsmState>,
}

impl BonsaiQwen35Cache {
    pub fn new(config: &BonsaiQwen35Config, device: &Device) -> CandleResult<Self> {
        let attn_caches = (0..config.num_hidden_layers)
            .map(|_| {
                let k = Tensor::zeros((1, config.num_key_value_heads, 0, config.head_dim), candle_core::DType::F32, device)?;
                let v = Tensor::zeros((1, config.num_key_value_heads, 0, config.head_dim), candle_core::DType::F32, device)?;
                Ok((k, v))
            })
            .collect::<CandleResult<Vec<_>>>()?;

        let ssm_states = (0..config.num_hidden_layers)
            .map(|_| SsmState::new(
                config.ssm_num_v_heads, config.ssm_head_v_dim,
                config.ssm_key_dim * 2 + config.ssm_value_dim,
                config.ssm_conv_kernel, device,
            ))
            .collect::<CandleResult<Vec<_>>>()?;

        Ok(BonsaiQwen35Cache { attn_caches, ssm_states })
    }
}

// ---------------------------------------------------------------------------
// Helper: load f32 1D tensor
// ---------------------------------------------------------------------------

fn load_f32_simple(gguf: &Arc<GgufFile>, name: &str) -> CandleResult<Vec<f32>> {
    let info = gguf.tensor_info(name)
        .map_err(|_e| candle_core::Error::Msg("error".to_string()))?;
    let data = gguf.tensor_bytes(name)
        .map_err(|_e| candle_core::Error::Msg("error".to_string()))?;
    gguf::dequantize_data(data, info.dtype_u32, info.elem_count)
        .map_err(|_e| candle_core::Error::Msg("dequant error".to_string()))
}

/// in_dim of a [out, in] weight tensor.
fn tensor_in_dim(gguf: &Arc<GgufFile>, name: &str) -> CandleResult<usize> {
    let info = gguf.tensor_info(name)
        .map_err(|_e| candle_core::Error::Msg(format!("no tensor {name}")))?;
    Ok(info.shape[info.shape.len() - 1])
}

/// out_dim of a [out, in] weight tensor.
fn tensor_out_dim(gguf: &Arc<GgufFile>, name: &str) -> CandleResult<usize> {
    let info = gguf.tensor_info(name)
        .map_err(|_e| candle_core::Error::Msg(format!("no tensor {name}")))?;
    Ok(info.shape[0])
}
