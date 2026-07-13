use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let model_id = std::env::args().nth(1).unwrap_or_else(|| "1bitLLM/bitnet_b1_58-3B".to_string());

    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::F32, true);

    println!("Loading {} with Bit1_58(Pack4, F32)...", model_id);
    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_pretrained(&model_id, quantization)?;
    println!("  Loaded in {:.2?}", load_start.elapsed());

    // Simple prompt tokens for comparison
    // "The capital of France is" with some tokenizer
    // For 1bitLLM tokenizer: token 1 = BOS
    let test_tokens: Vec<u32> = (1..=10).collect();
    println!("  Test tokens: {:?}", test_tokens);

    let mut cache = xor_net::Cache::new(true, &config)?;
    let logits = model.forward(&test_tokens, 0, &mut cache)?;

    let vocab_size = *logits.shape.last().unwrap_or(&1);
    let seq_len = logits.shape[logits.shape.len() - 2];
    let b_size: usize = logits.shape[..logits.shape.len() - 1].iter().product();
    let last_pos = (b_size - 1) * seq_len * vocab_size + (seq_len - 1) * vocab_size;

    println!("  Vocab size: {}", vocab_size);
    println!("  Last token logits (first 20): {:?}",
        &logits.data[last_pos..last_pos + 20.min(vocab_size)]);

    // Save logits for comparison
    let logits_data = &logits.data[last_pos..last_pos + vocab_size];
    println!("  Logits min: {:.4}, max: {:.4}, mean: {:.4}",
        logits_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        logits_data.iter().sum::<f32>() / logits_data.len() as f32
    );

    // Top-5 tokens
    let mut scored: Vec<_> = logits_data.iter().copied().enumerate().collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-5 tokens:");
    for (tok, score) in scored.iter().take(5) {
        println!("    token {:5}: {:.4}", tok, score);
    }

    Ok(())
}
