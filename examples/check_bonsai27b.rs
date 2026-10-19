//! Verify Bonsai-27B correctness: tokenize prompt from GGUF tokenizer, run
//! greedy generation, print output + logit stats.
//! Usage: cargo run --example check_bonsai27b --release -- <gguf> [prompt]
use std::sync::Arc;
use std::time::Instant;
use xor_net::models::bonsai_qwen35::*;
use xor_net::*;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/bonsai-27b-q1_0.gguf".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let device = candle_core::Device::Cpu;

    let t0 = Instant::now();
    let gguf = Arc::new(GgufFile::open(&path)?);
    let cfg = BonsaiQwen35Config::from_gguf(&gguf)?;
    let model = BonsaiQwen35Model::from_gguf(gguf.clone(), &device)?;
    let tokenizer = xor_net::loader::gguf::build_tokenizer(&gguf)?;
    println!(
        "Model + tokenizer loaded in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let enc = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut tokens: Vec<u32> = enc.get_ids().to_vec();
    println!("Prompt: {prompt:?} → {tokens:?}");

    let mut cache = BonsaiQwen35Cache::new(&cfg, &device)?;
    let mut index_pos = 0usize;

    // Prefill
    let t1 = Instant::now();
    let _ = model.forward(&tokens, index_pos, &mut cache)?;
    index_pos += tokens.len();
    println!(
        "Prefill ({} tokens): {:.1} ms",
        tokens.len(),
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // Greedy decode
    let max_new = 32;
    let t2 = Instant::now();
    for i in 0..max_new {
        let logits = model.forward(&[tokens[tokens.len() - 1]], index_pos, &mut cache)?;
        index_pos += 1;
        let lv: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let next = lv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        tokens.push(next);
        if next == 248046 || next == 248044 {
            // eos/bos
            break;
        }
        if i % 8 == 0 {
            let text = tokenizer
                .decode(&tokens, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("  step {i}: ...{text:?}");
        }
    }
    let dt = t2.elapsed().as_secs_f64();
    let n = tokens.len() - enc.get_ids().len();
    println!(
        "Decode: {n} tokens in {dt:.2}s ({:.1} ms/tok)",
        dt / n as f64 * 1000.0
    );

    let output = tokenizer
        .decode(&tokens, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("\nOutput:\n{output}");
    Ok(())
}
