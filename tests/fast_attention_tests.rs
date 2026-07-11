use candle_core::{DType, Device, Result, Tensor};
use xor_net::nn::{fast_attention, CpuRingCache};
use xor_net::tensor::FastTensor;

fn to_fast(t: &Tensor) -> Result<FastTensor> {
    let data = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(FastTensor::new(data, t.shape().dims().to_vec()))
}

fn to_candle(ft: &FastTensor, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(ft.data.clone(), ft.shape.clone(), dev)
}

// Helper to compute standard Candle attention with KV cache
fn candle_standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache_k: Option<&Tensor>,
    cache_v: Option<&Tensor>,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    index_pos: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (_b_sz, _, seq_len, _) = q.dims4()?;
    
    let mut full_k = k.clone();
    let mut full_v = v.clone();
    
    if let Some(ck) = cache_k {
        full_k = Tensor::cat(&[ck, k], 2)?;
    }
    if let Some(cv) = cache_v {
        full_v = Tensor::cat(&[cv, v], 2)?;
    }
    
    // Grouped Query Attention repeats KV
    let repeated_k = repeat_kv(full_k.clone(), num_heads / num_kv_heads)?;
    let repeated_v = repeat_kv(full_v.clone(), num_heads / num_kv_heads)?;
    
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = (q.matmul(&repeated_k.t()?)? * scale)?;
    
    let att = if seq_len == 1 {
        att
    } else {
        // Causal mask
        let mask = build_causal_mask(seq_len, index_pos, q.device())?.broadcast_as(att.shape())?;
        masked_fill(&att, &mask, f32::NEG_INFINITY)?
    };
    
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let out = att.matmul(&repeated_v.contiguous()?)?;
    
    Ok((out, full_k, full_v))
}

fn repeat_kv(x: Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x);
    }
    let (_b_sz, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((_b_sz, num_kv_heads, repeats, seq_len, head_dim))?
        .reshape((_b_sz, num_kv_heads * repeats, seq_len, head_dim))?;
    Ok(x)
}

fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| {
            (0..index_pos + seq_len).map(move |j| {
                if j > index_pos + i {
                    1
                } else {
                    0
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (seq_len, index_pos + seq_len), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[test]
fn test_fast_attention_parity_single_token() -> Result<()> {
    let device = Device::Cpu;
    let num_layers = 1;
    let num_heads = 4;
    let num_kv_heads = 2;
    let max_seq_len = 32;
    let head_dim = 8;
    
    let mut cache = CpuRingCache::new(num_layers, num_kv_heads, max_seq_len, head_dim);
    
    // Phase 1: Prompt phase (seq_len = 4)
    let prompt_len = 4;
    let q_prompt = Tensor::rand(-1.0f32, 1.0f32, (1, num_heads, prompt_len, head_dim), &device)?;
    let k_prompt = Tensor::rand(-1.0f32, 1.0f32, (1, num_kv_heads, prompt_len, head_dim), &device)?;
    let v_prompt = Tensor::rand(-1.0f32, 1.0f32, (1, num_kv_heads, prompt_len, head_dim), &device)?;
    
    let fast_q = to_fast(&q_prompt)?;
    let fast_k = to_fast(&k_prompt)?;
    let fast_v = to_fast(&v_prompt)?;
    
    let fast_out_prompt = fast_attention(
        &fast_q,
        &fast_k,
        &fast_v,
        0,
        0,
        &mut cache,
        num_heads,
        num_kv_heads,
        head_dim,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    let candle_out_prompt = to_candle(&fast_out_prompt, &device)?;
    
    let (ref_out_prompt, candle_k1, candle_v1) = candle_standard_attention(
        &q_prompt,
        &k_prompt,
        &v_prompt,
        None,
        None,
        head_dim,
        num_heads,
        num_kv_heads,
        0,
    )?;
    
    // Assert prompt outputs match
    let diff_prompt = (&candle_out_prompt - &ref_out_prompt)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
    for (i, &d) in diff_prompt.iter().enumerate() {
        assert!(d < 1e-5, "Prompt difference too large: {} at index {}", d, i);
    }
    
    // Phase 2: Token Gen phase (seq_len = 1, pos = 4)
    let q_token = Tensor::rand(-1.0f32, 1.0f32, (1, num_heads, 1, head_dim), &device)?;
    let k_token = Tensor::rand(-1.0f32, 1.0f32, (1, num_kv_heads, 1, head_dim), &device)?;
    let v_token = Tensor::rand(-1.0f32, 1.0f32, (1, num_kv_heads, 1, head_dim), &device)?;
    
    let fast_q_t = to_fast(&q_token)?;
    let fast_k_t = to_fast(&k_token)?;
    let fast_v_t = to_fast(&v_token)?;
    
    let fast_out_token = fast_attention(
        &fast_q_t,
        &fast_k_t,
        &fast_v_t,
        0,
        4,
        &mut cache,
        num_heads,
        num_kv_heads,
        head_dim,
    ).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    let candle_out_token = to_candle(&fast_out_token, &device)?;
    
    let (ref_out_token, _, _) = candle_standard_attention(
        &q_token,
        &k_token,
        &v_token,
        Some(&candle_k1),
        Some(&candle_v1),
        head_dim,
        num_heads,
        num_kv_heads,
        4,
    )?;
    
    // Assert token outputs match
    let diff_token = (&candle_out_token - &ref_out_token)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
    for (i, &d) in diff_token.iter().enumerate() {
        assert!(d < 1e-5, "Token difference too large: {} at index {}", d, i);
    }
    
    Ok(())
}

#[test]
fn test_ring_wrapping_correctness() -> Result<()> {
    let device = Device::Cpu;
    let num_layers = 1;
    let num_heads = 2;
    let num_kv_heads = 1;
    // Tiny cache to force wrapping quickly
    let max_seq_len = 4;
    let head_dim = 4;
    
    let mut cache = CpuRingCache::new(num_layers, num_kv_heads, max_seq_len, head_dim);
    
    // Write 4 tokens (completely filling the cache)
    for pos in 0..4 {
        let q = Tensor::ones((1, num_heads, 1, head_dim), DType::F32, &device)?;
        let k = Tensor::ones((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
        let v = Tensor::ones((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
        
        let fast_q = to_fast(&q)?;
        let fast_k = to_fast(&k)?;
        let fast_v = to_fast(&v)?;
        
        let _ = fast_attention(&fast_q, &fast_k, &fast_v, 0, pos, &mut cache, num_heads, num_kv_heads, head_dim)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    }
    
    // Verify cache has all ones
    assert_eq!(cache.k_buffers[0], vec![1.0; num_kv_heads * max_seq_len * head_dim]);
    
    // Write token at pos 4 (which wraps to index 0)
    // We pass a zero tensor to verify it overwrites pos 0
    let q = Tensor::zeros((1, num_heads, 1, head_dim), DType::F32, &device)?;
    let k = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
    let v = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &device)?;
    
    let fast_q = to_fast(&q)?;
    let fast_k = to_fast(&k)?;
    let fast_v = to_fast(&v)?;
    
    let _ = fast_attention(&fast_q, &fast_k, &fast_v, 0, 4, &mut cache, num_heads, num_kv_heads, head_dim)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    
    // Verify that index 0 of the cache is now 0.0, but index 1, 2, 3 remain 1.0
    // Layout: h_kv * max_seq_len * head_dim + t * head_dim + d
    let cache_k = &cache.k_buffers[0];
    for d in 0..head_dim {
        assert_eq!(cache_k[0 * head_dim + d], 0.0, "Index 0 should have wrapped and been overwritten with 0");
        assert_eq!(cache_k[1 * head_dim + d], 1.0, "Index 1 should still be 1.0");
    }
    
    Ok(())
}
