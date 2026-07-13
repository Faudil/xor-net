use candle_core::{Device, Result, Tensor};
use xor_net::tensor::FastTensor;

fn assert_parity(fast: &FastTensor, candle: &Tensor) -> Result<()> {
    assert_eq!(fast.shape, candle.shape().dims());
    let candle_vec = candle.flatten_all()?.to_vec1::<f32>()?;
    for (i, (&x_fast, &x_candle)) in fast.data.iter().zip(candle_vec.iter()).enumerate() {
        let diff = (x_fast - x_candle).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: FastTensor value {}, Candle value {}, diff {}",
            i,
            x_fast,
            x_candle,
            diff
        );
    }
    Ok(())
}

#[test]
fn test_fast_tensor_add() -> Result<()> {
    let device = Device::Cpu;
    let shape = vec![2, 3, 4];
    
    let a_data = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2,
    ];
    let b_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
    ];
    
    let fast_a = FastTensor::new(a_data.clone(), shape.clone());
    let fast_b = FastTensor::new(b_data.clone(), shape.clone());
    let fast_res = fast_a.add(&fast_b).unwrap();
    
    let candle_a = Tensor::new(&a_data[..], &device)?.reshape((2, 3, 4))?;
    let candle_b = Tensor::new(&b_data[..], &device)?.reshape((2, 3, 4))?;
    let candle_res = (&candle_a + &candle_b)?;
    
    assert_parity(&fast_res, &candle_res)?;
    Ok(())
}

#[test]
fn test_fast_tensor_silu_mul() -> Result<()> {
    let device = Device::Cpu;
    let shape = vec![1, 8];
    
    let a_data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let b_data = vec![1.5, 2.5, -0.5, 0.1, 10.0, 4.0, 0.0, -3.0];
    
    let fast_a = FastTensor::new(a_data.clone(), shape.clone());
    let fast_b = FastTensor::new(b_data.clone(), shape.clone());
    let fast_res = fast_a.silu_mul(&fast_b).unwrap();
    
    let candle_a = Tensor::new(&a_data[..], &device)?.reshape((1, 8))?;
    let candle_b = Tensor::new(&b_data[..], &device)?.reshape((1, 8))?;
    // Candle SiLU is x / (1.0 + exp(-x)), then multiply by b
    let candle_silu = (candle_a.clone() / (candle_a.neg()?.exp()? + 1.0)?)?;
    let candle_res = (candle_silu * candle_b)?;
    
    assert_parity(&fast_res, &candle_res)?;
    Ok(())
}

#[test]
fn test_fast_tensor_rmsnorm() -> Result<()> {
    let device = Device::Cpu;
    let shape = vec![2, 3, 4]; // hidden_size = 4
    
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        -1.0, 0.5, 1.5, -2.0,
        0.0, 0.0, 0.0, 1.0,
        
        2.0, -2.0, 2.0, -2.0,
        1.1, 2.2, 3.3, 4.4,
        -0.5, -0.5, -0.5, -0.5,
    ];
    let weight = vec![0.5, 1.0, 1.5, 2.0];
    
    let fast_x = FastTensor::new(data.clone(), shape.clone());
    let fast_w = FastTensor::new(weight.clone(), vec![4]);
    let fast_res = fast_x.rmsnorm(&fast_w, 1e-6).unwrap();
    
    let candle_x = Tensor::new(&data[..], &device)?.reshape((2, 3, 4))?;
    let candle_w = Tensor::new(&weight[..], &device)?;
    
    // Candle RMSNorm implementation
    let eps = 1e-6;
    let sum_sq = candle_x.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let mean_sq = (sum_sq / 4.0)?;
    let inv_std = (mean_sq + eps)?.sqrt()?.recip()?;
    let candle_res = candle_x.broadcast_mul(&inv_std)?.broadcast_mul(&candle_w)?;
    
    assert_parity(&fast_res, &candle_res)?;
    Ok(())
}

#[test]
fn test_fast_tensor_embedding() -> Result<()> {
    let device = Device::Cpu;
    let weight_data = vec![
        0.1, 0.2, 0.3,
        1.1, 1.2, 1.3,
        2.1, 2.2, 2.3,
        3.1, 3.2, 3.3,
    ]; // vocab_size = 4, hidden_size = 3
    let ids = vec![3, 0, 2, 1];
    
    let fast_w = FastTensor::new(weight_data.clone(), vec![4, 3]);
    let fast_res = FastTensor::embedding(&ids, &fast_w).unwrap();
    
    let candle_w = Tensor::new(&weight_data[..], &device)?.reshape((4, 3))?;
    let candle_ids = Tensor::new(&ids[..], &device)?;
    let candle_res = candle_w.index_select(&candle_ids, 0)?.unsqueeze(0)?;
    
    assert_parity(&fast_res, &candle_res)?;
    Ok(())
}

#[test]
fn test_fast_tensor_rope() -> Result<()> {
    let device = Device::Cpu;
    
    // shape: [b_sz, num_heads, seq_len, head_dim] = [1, 2, 3, 4]
    let x_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        
        -1.0, -2.0, -3.0, -4.0,
        -5.0, -6.0, -7.0, -8.0,
        -9.0, -10.0, -11.0, -12.0,
    ];
    let cos_data = vec![
        0.9, 0.8,
        0.7, 0.6,
        0.5, 0.4,
    ]; // shape [3, 2]
    let sin_data = vec![
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
    ]; // shape [3, 2]
    
    let fast_x = FastTensor::new(x_data.clone(), vec![1, 2, 3, 4]);
    let fast_cos = FastTensor::new(cos_data.clone(), vec![3, 2]);
    let fast_sin = FastTensor::new(sin_data.clone(), vec![3, 2]);
    
    let fast_res = fast_x.rope_inplace(&fast_cos, &fast_sin, 0).unwrap();
    
    let candle_x = Tensor::new(&x_data[..], &device)?.reshape((1, 2, 3, 4))?;
    let candle_cos = Tensor::new(&cos_data[..], &device)?.reshape((3, 2))?;
    let candle_sin = Tensor::new(&sin_data[..], &device)?.reshape((3, 2))?;
    
    let candle_res = candle_nn::rotary_emb::rope(&candle_x, &candle_cos, &candle_sin)?;
    
    assert_parity(&fast_res, &candle_res)?;
    Ok(())
}

#[test]
fn test_fast_tensor_transposes() -> Result<()> {
    let device = Device::Cpu;
    let b_sz = 1;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = 4;
    let hidden_size = num_heads * head_dim; // 8
    
    let x_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let fast_x = FastTensor::new(x_data.clone(), vec![b_sz, seq_len, hidden_size]);
    let candle_x = Tensor::new(&x_data[..], &device)?.reshape((b_sz, seq_len, hidden_size))?;
    
    let fast_heads = fast_x.transpose_seq_to_heads(num_heads, head_dim).unwrap();
    let candle_heads = candle_x
        .reshape((b_sz, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    assert_parity(&fast_heads, &candle_heads)?;
    
    let fast_seq = fast_heads.transpose_heads_to_seq().unwrap();
    let candle_seq = candle_heads
        .transpose(1, 2)?
        .reshape((b_sz, seq_len, hidden_size))?;
    assert_parity(&fast_seq, &candle_seq)?;
    
    Ok(())
}

// === Numerical correctness tests for core operations ===

#[test]
fn test_dot_product_f32_matches_naive() {
    // Test various sizes to stress remainder handling
    for len in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128] {
        let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.01).collect();
        
        let fast = xor_net::nn::dynamic_linear::dot_product_f32(&a, &b);
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        let diff = (fast - naive).abs();
        assert!(diff < 1e-4, "len={}: fast={}, naive={}, diff={}", len, fast, naive, diff);
    }
}

#[test]
fn test_dot_product_i8_matches_naive() {
    for len in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128] {
        let a: Vec<i8> = (0..len).map(|i| ((i * 13 + 5) % 127 - 63) as i8).collect();
        let b: Vec<i8> = (0..len).map(|i| ((i * 7 + 3) % 127 - 63) as i8).collect();
        
        let fast = xor_net::nn::dynamic_linear::dot_product_i8(&a, &b);
        let naive: i32 = a.iter().zip(b.iter()).map(|(x, y)| (*x as i32) * (*y as i32)).sum();
        
        assert_eq!(fast, naive, "len={}: fast={}, naive={}", len, fast, naive);
    }
}

#[test]
fn test_softmax_matches_naive() {
    for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 128, 256] {
        let mut scores: Vec<f32> = (0..len).map(|i| i as f32 - len as f32 / 2.0).collect();
        let mut reference = scores.clone();
        
        xor_net::nn::fast_attention_simd::softmax_inplace(&mut scores);
        
        // Manual softmax
        let max_val = reference.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0f32;
        for s in reference.iter_mut() {
            *s = (*s - max_val).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        for s in reference.iter_mut() {
            *s *= inv_sum;
        }
        
        for i in 0..len {
            let diff = (scores[i] - reference[i]).abs();
            assert!(diff < 1e-5, "len={}, idx={}: fast={}, ref={}, diff={}", len, i, scores[i], reference[i], diff);
        }
    }
}

#[test]
fn test_weighted_sum_matches_naive() {
    for head_dim in [4, 7, 8, 9, 15, 16, 17, 32, 64, 100, 128] {
        for total_kv in [1, 2, 3, 5, 10, 20] {
            let scores: Vec<f32> = (0..total_kv).map(|i| (i as f32 + 1.0) / total_kv as f32).collect();
            let v_buf: Vec<f32> = (0..2 * 64 * head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
            // Use h_kv=0, max_seq_len=64
            let mut fast_out = vec![0.0f32; head_dim];
            let mut ref_out = vec![0.0f32; head_dim];
            
            xor_net::nn::fast_attention_simd::weighted_sum(
                &mut fast_out, &scores, &v_buf, 0, 64, head_dim, total_kv,
            );
            
            // Naive
            for i in 0..head_dim {
                let mut s = 0.0f32;
                for t_k in 0..total_kv {
                    let v_offset = (t_k % 64) * head_dim + i;
                    s += scores[t_k] * v_buf[v_offset];
                }
                ref_out[i] = s;
            }
            
            for i in 0..head_dim {
                let diff = (fast_out[i] - ref_out[i]).abs();
                assert!(diff < 1e-4, "head_dim={}, total_kv={}, idx={}: fast={}, ref={}, diff={}",
                    head_dim, total_kv, i, fast_out[i], ref_out[i], diff);
            }
        }
    }
}

#[test]
fn test_batched_dot_product_matches_naive() {
    for head_dim in [4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100] {
        for total_kv in [1, 2, 3, 5, 10, 16, 20] {
            let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
            let k_buf: Vec<f32> = (0..2 * 64 * head_dim).map(|i| (i as f32 * 0.01).cos()).collect();
            let mut fast_out = vec![0.0f32; total_kv.max(16)];
            let mut ref_out = vec![0.0f32; total_kv];
            
            xor_net::nn::fast_attention_simd::batched_dot_product(
                &q, &k_buf, 0, 64, head_dim, 0, total_kv, &mut fast_out,
            );
            
            // Naive
            for t_k in 0..total_kv {
                let k_offset = (t_k % 64) * head_dim;
                let mut sum = 0.0f32;
                for i in 0..head_dim {
                    sum += q[i] * k_buf[k_offset + i];
                }
                ref_out[t_k] = sum;
            }
            
            for t_k in 0..total_kv {
                let diff = (fast_out[t_k] - ref_out[t_k]).abs();
                assert!(diff < 1e-4, "head_dim={}, total_kv={}, t_k={}: fast={}, ref={}, diff={}",
                    head_dim, total_kv, t_k, fast_out[t_k], ref_out[t_k], diff);
            }
        }
    }
}

#[test]
fn test_rmsnorm_matches_naive() {
    for n in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 128, 256, 3200] {
        if n > 3200 { continue; }
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - n as f32 / 2.0).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let eps = 1e-6;
        let mut reference = x.clone();
        
        xor_net::nn::fast_attention_simd::rms_norm(&mut x, &weight, eps);
        
        // Naive
        let sum_sq: f32 = reference.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..n {
            reference[i] = reference[i] * inv_rms * weight[i];
        }
        
        for i in 0..n {
            let diff = (x[i] - reference[i]).abs();
            assert!(diff < 1e-4, "n={}, idx={}: fast={}, ref={}, diff={}", n, i, x[i], reference[i], diff);
        }
    }
}
