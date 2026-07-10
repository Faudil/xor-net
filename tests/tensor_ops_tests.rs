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
