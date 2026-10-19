//! Debug: dump layer outputs for correctness comparison with Python reference.
//! Usage: cargo run --example debug_bonsai27b --release -- <gguf>
use std::sync::Arc;
use xor_net::models::bonsai_qwen35::*;
use xor_net::*;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/bonsai-27b-q1_0.gguf".to_string());
    let device = candle_core::Device::Cpu;
    let gguf = Arc::new(GgufFile::open(&path)?);
    let cfg = BonsaiQwen35Config::from_gguf(&gguf)?;
    let model = BonsaiQwen35Model::from_gguf(gguf.clone(), &device)?;

    // Fixed input: embedding of tokens [561, 6511, 314, 9338, 369] (one token for the block test)
    let mut cache = BonsaiQwen35Cache::new(&cfg, &device)?;
    let x = model.debug_embed(&[561])?;
    let xe: Vec<f32> = x.flatten_all()?.to_vec1()?;
    println!("embed[561] shape={:?} first5={:?}", x.dims(), &xe[..5]);

    // Block 0 = SSM (linear attention)
    let out0 = model.debug_block(0, &x, 0, &mut cache)?;
    let o0: Vec<f32> = out0.flatten_all()?.to_vec1()?;
    println!("blk0(SSM) out: first10={:?}", &o0[..10]);
    println!(
        "blk0(SSM) out: min={} max={} mean={}",
        o0.iter().cloned().fold(f32::INFINITY, f32::min),
        o0.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        o0.iter().sum::<f32>() / o0.len() as f32
    );

    // Block 3 = full attention
    let mut cache2 = BonsaiQwen35Cache::new(&cfg, &device)?;
    let x2 = model.debug_embed(&[561])?;
    let out3 = model.debug_block(3, &x2, 0, &mut cache2)?;
    let o3: Vec<f32> = out3.flatten_all()?.to_vec1()?;
    println!("blk3(ATTN) out: first10={:?}", &o3[..10]);
    println!(
        "blk3(ATTN) out: min={} max={} mean={}",
        o3.iter().cloned().fold(f32::INFINITY, f32::min),
        o3.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        o3.iter().sum::<f32>() / o3.len() as f32
    );

    Ok(())
}
