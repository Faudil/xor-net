
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::time::Instant;
use tokenizers::Tokenizer;
use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> anyhow::Result<()> {
    let num_threads = std::env::var("XORNET_THREADS")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(0);
    xor_net::init_threads(num_threads).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let model_id = "HF1BitLLM/Llama3-8B-1.58-100B-tokens";
    println!("Loading 1.58-bit model {}...", model_id);

    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, xor_net::nn::LmHeadConfig::Int4, true);

    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_pretrained(model_id, quantization)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    let mut cache = xor_net::Cache::new(true, &config)?;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let prompt = "The capital of France is";
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();

    println!("Prompt: '{}'", prompt);
    let mut sampler = xor_net::sampler::Sampler::new(299792458, Some(0.7), None, 1.0);

    let mut index_pos = 0;
    let max_len = 50;
    let start_gen = Instant::now();
    let mut generated_tokens = 0usize;

    let mut total_forward_time = std::time::Duration::ZERO;
    let mut total_sample_time = std::time::Duration::ZERO;

    for _loop_idx in 0..max_len {
        let context_size = if index_pos == 0 { tokens.len() } else { 1 };
        let start_pos = tokens.len().saturating_sub(context_size);

        let start_forward = Instant::now();
        let mut logits = model.forward(&tokens[start_pos..], index_pos, &mut cache)?;
        total_forward_time += start_forward.elapsed();

        let start_sample = Instant::now();
        let next_token = sampler.sample(&mut logits.data, &tokens[..])?;
        total_sample_time += start_sample.elapsed();

        tokens.push(next_token);
        index_pos += context_size;
        generated_tokens += 1;

        if Some(next_token) == config.bos_token_id || next_token == 128009 || next_token == 128001 {
            break;
        }
    }

    let generated_text = tokenizer.decode(&tokens, true).unwrap_or(String::new());
    println!("Output: {}", generated_text);

    let elapsed = start_gen.elapsed();
    let tps = generated_tokens as f64 / elapsed.as_secs_f64();
    println!("Generated {} tokens in {:.2?} ({:.2} tokens/sec)", generated_tokens, elapsed, tps);
    println!("  - Model Forward Time: {:.2?}", total_forward_time);
    println!("  - Native Sampling Time: {:.2?}", total_sample_time);

    let (blocks_us, lm_head_us, other_us) = xor_net::models::llama::get_profiling_stats();
    println!("    * [Profile] Transformer Blocks Total: {:.2}ms ({:.2}ms/token)", blocks_us as f64 / 1000.0, (blocks_us as f64 / 1000.0) / generated_tokens as f64);
    println!("    * [Profile] LM Head Total: {:.2}ms ({:.2}ms/token)", lm_head_us as f64 / 1000.0, (lm_head_us as f64 / 1000.0) / generated_tokens as f64);
    println!("    * [Profile] Other Ops Total: {:.2}ms ({:.2}ms/token)", other_us as f64 / 1000.0, (other_us as f64 / 1000.0) / generated_tokens as f64);

    Ok(())
}