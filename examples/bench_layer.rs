use xor_net::bit1_58::layers::TernaryLinear;
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::tensor::FastTensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    xor_net::init_threads(0).unwrap();
    let in_dim = 4096;
    let out_dim = 4096;
    let weights = vec![0.0f32; in_dim * out_dim];
    let layer = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack4)?;
    
    let x = FastTensor::zeros(vec![1, in_dim]);
    
    // Warmup
    for _ in 0..10 {
        let _ = layer.forward(&x)?;
    }
    
    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        let _ = layer.forward(&x)?;
    }
    let elap = start.elapsed();
    
    println!("Time for {} passes: {:?}", iters, elap);
    println!("Time per pass: {:?}", elap / iters as u32);
    Ok(())
}
