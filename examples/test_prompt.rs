//! Sampled completion with temperature, top-p and repetition penalty.
//!
//! XORNET_MODEL=models/bitnet-2b cargo run --release --example test_prompt

use std::path::Path;
use std::sync::Arc;

use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use xor_net::{GenerationParams, Generator, LlamaCausalLM, QuantizationConfig, SamplingParams};

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("XORNET_MODEL").unwrap_or_else(|_| "models/bitnet-2b".into());
    let model = LlamaCausalLM::from_local(
        Path::new(&dir),
        QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, false),
    )?;
    let mut generator = Generator::from_dir(Arc::new(model), &dir)?;

    let prompt = "The first ten numbers are: 1, 2, 3, 4, 5, ";
    let params = GenerationParams::default()
        .with_max_tokens(20)
        .with_sampling(
            SamplingParams::default()
                .with_temperature(0.8)
                .with_top_p(0.95)
                .with_repetition_penalty(1.2)
                .with_seed(Some(3_132_833_641)),
        );

    println!("=== Testing Prompt ===");
    println!("Prompt: {prompt}");
    let out = generator.generate(prompt, &params)?;
    println!("Response: {}", out.text);
    Ok(())
}
