//! Benchmark Bonsai-27B (qwen35) CPU inference speed.
//! Usage: cargo run --example bench_bonsai27b --release -- <gguf> [num_tokens]
use std::sync::Arc;
use std::time::Instant;
use xor_net::*;
use xor_net::models::bonsai_qwen35::*;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "/tmp/bonsai-27b-q1_0.gguf".to_string());
    let num_tokens: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let device = candle_core::Device::Cpu;

    let t0 = Instant::now();
    let gguf = Arc::new(GgufFile::open(&path)?);
    let cfg = BonsaiQwen35Config::from_gguf(&gguf)?;
    let model = BonsaiQwen35Model::from_gguf(gguf, &device)?;
    println!("Model loaded in {:.2}s (64 layers, hidden={})", t0.elapsed().as_secs_f64(), cfg.hidden_size);

    let mut cache = BonsaiQwen35Cache::new(&cfg, &device)?;
    let mut token = 0u32;
    let mut index_pos = 0usize;

    // Warmup + first token (prefill with 1 token)
    let t1 = Instant::now();
    let _ = model.forward(&[token], index_pos, &mut cache)?;
    index_pos += 1;
    let prefill_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("Prefill (1 token): {prefill_ms:.1} ms");

    // Decode steps
    let t2 = Instant::now();
    for i in 0..num_tokens {
        let _logits = model.forward(&[token], index_pos, &mut cache)?;
        // pick a simple token to continue
        token = (token + 1) % 1000;
        index_pos += 1;
    }
    let elapsed = t2.elapsed().as_secs_f64();
    let per_token = elapsed / num_tokens as f64;
    let per_token_ms = per_token * 1000.0;
    let tps = 1.0 / per_token;
    println!("Decode: {num_tokens} tokens in {elapsed:.2}s ({per_token_ms:.1} ms/token, {tps:.2} tok/s)");

    Ok(())
}
