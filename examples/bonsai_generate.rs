//! Greedy generation smoke test for Bonsai GGUF models (qwen3 dense or qwen35 hybrid).
//!
//! Usage: bonsai_generate <model.gguf> <tokenizer.json> [prompt] [n_tokens]
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use xor_net::models::bonsai_qwen35::BonsaiQwen35Config;
use xor_net::*;

fn argmax_last(logits: &Tensor) -> anyhow::Result<u32> {
    let v: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let vocab = *logits.dims().last().unwrap();
    let last = &v[v.len() - vocab..];
    let (idx, _) = last
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc });
    Ok(idx as u32)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: bonsai_generate <model.gguf> <tokenizer.json> [prompt] [n]");
    let tok_path = args.next().expect("missing tokenizer.json path");
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".to_string());
    let n: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(32);

    let device = Device::Cpu;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
    let gguf = Arc::new(GgufFile::open(&path)?);
    let arch = gguf.architecture()?;
    let mut tokens = tokenizer.encode(prompt.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let n_prompt = tokens.len();

    let t0 = Instant::now();
    let mut step: Box<dyn FnMut(&[u32], usize) -> anyhow::Result<u32>> = match arch.as_str() {
        "qwen3" => {
            let cfg = BonsaiConfig::from_gguf_metadata(&gguf.metadata.metadata)?;
            let model = BonsaiModel::from_gguf(gguf.clone(), &device)?;
            let mut cache = BonsaiCache::new(&cfg, &device)?;
            Box::new(move |t, pos| argmax_last(&model.forward(t, pos, &mut cache)?))
        }
        "qwen35" => {
            let cfg = BonsaiQwen35Config::from_gguf(&gguf)?;
            let model = BonsaiQwen35Model::from_gguf(gguf.clone(), &device)?;
            let mut cache = BonsaiQwen35Cache::new(&cfg, &device)?;
            Box::new(move |t, pos| argmax_last(&model.forward(t, pos, &mut cache)?))
        }
        other => anyhow::bail!("unsupported architecture: {other}"),
    };
    println!("arch={arch} load={:.2}s prompt_tokens={n_prompt}", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let mut next = step(&tokens, 0)?;
    let prefill = t1.elapsed();
    let t2 = Instant::now();
    for _ in 0..n {
        tokens.push(next);
        next = step(&tokens[tokens.len() - 1..], tokens.len() - 1)?;
    }
    let decode = t2.elapsed();
    let text = tokenizer.decode(&tokens[n_prompt..], true).map_err(anyhow::Error::msg)?;
    println!("--- output ---\n{prompt}{text}\n--------------");
    println!(
        "prefill {:.1} tok/s, decode {:.2} tok/s",
        n_prompt as f64 / prefill.as_secs_f64(),
        n as f64 / decode.as_secs_f64()
    );
    Ok(())
}
