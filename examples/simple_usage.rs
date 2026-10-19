//! Minimal text completion with the high-level `Generator` API.
//!
//! XORNET_MODEL=models/bitnet-2b cargo run --release --example simple_usage

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;

use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use xor_net::{GenerationParams, Generator, LlamaCausalLM, QuantizationConfig, SamplingParams};

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("XORNET_MODEL").unwrap_or_else(|_| "models/bitnet-2b".into());
    let model = LlamaCausalLM::from_local(
        Path::new(&dir),
        QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true),
    )?;
    let mut generator = Generator::from_dir(Arc::new(model), &dir)?;

    let prompt = "Quantum physics, also known as quantum mechanics, is a branch of physics";
    let params = GenerationParams::default()
        .with_max_tokens(100)
        .with_sampling(SamplingParams::greedy());

    print!("{prompt}");
    let out = generator.generate_stream(prompt, &params, |delta| {
        print!("{delta}");
        ControlFlow::Continue(())
    })?;
    println!(
        "\n[{:?}] {} tokens, prefill {:.0} ms, decode {:.1} tok/s",
        out.finish_reason,
        out.usage.completion_tokens,
        out.timings.prefill_ms(),
        out.timings.decode_tok_per_s()
    );
    Ok(())
}
