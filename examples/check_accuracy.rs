use xor_net::{AutoModelForCausalLM, QuantizationConfig, Cache};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let model_id = std::env::args().nth(1).unwrap_or_else(|| "1bitLLM/bitnet_b1_58-3B".to_string());
    let qtype = std::env::args().nth(2).unwrap_or_else(|| "none".to_string());

    let quantization = match qtype.to_lowercase().as_str() {
        "bit1_58_int4" | "int4" => QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true),
        "bit1_58_int8" | "int8" => QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int8, true),
        "bit1_58_f32" | "f32" => QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::F32, true),
        _ => QuantizationConfig::None,
    };

    println!("Loading {} with {:?}...", model_id, qtype);
    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_local(std::path::Path::new(&model_id), quantization)?;
    println!("  Loaded in {:.2?}", load_start.elapsed());

    // Test tokens
    let test_tokens: Vec<u32> = (1..=10).collect();
    println!("  Test tokens: {:?}", test_tokens);

    // Forward in one batch
    let mut cache = Cache::new(true, &config)?;
    let logits = model.forward(&test_tokens, 0, &mut cache)?;

    let vocab_size = *logits.shape.last().unwrap_or(&1);
    let seq_len = logits.shape[logits.shape.len() - 2];
    let b_size: usize = logits.shape[..logits.shape.len() - 1].iter().product();
    let last_pos = (b_size - 1) * seq_len * vocab_size + (seq_len - 1) * vocab_size;
    let last_slice = &logits.data[last_pos..last_pos + vocab_size.min(100)];

    println!("  Last token logits (first 20): {:?}", &last_slice[..20.min(last_slice.len())]);
    println!("  Logits min: {:.4}, max: {:.4}, mean: {:.4}",
        logits.data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        logits.data.iter().sum::<f32>() / logits.data.len() as f32
    );

    // Check top-5 tokens
    let mut scored: Vec<_> = logits.data.iter().copied().enumerate()
        .filter(|(i, _)| *i >= last_pos && *i < last_pos + vocab_size)
        .map(|(i, v)| (v, (i - last_pos) as u32))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-5 tokens:");
    for (score, tok) in scored.iter().take(5) {
        println!("    token {:5}: {:.4}", tok, score);
    }

    // Check for NaNs
    let nan_count = logits.data.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = logits.data.iter().filter(|&&x| x.is_infinite()).count();
    println!("  NaN count: {}, Inf count: {}", nan_count, inf_count);

    if nan_count > 0 || inf_count > 0 {
        println!("  ❌ FAIL: NaN or Inf detected in logits");
    } else {
        println!("  ✅ PASS: No NaN/Inf, logits look healthy");
    }

    Ok(())
}
