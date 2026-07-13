use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;
use xor_net::Sampler;
use xor_net::{AutoModelForCausalLM, Cache, Config, Llama, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::models::llama::LlamaEosToks;
use xor_net::nn::LmHeadConfig;

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const SYSTEM_PROMPT: &str = "You are a helpful, knowledgeable, and friendly AI assistant. You provide clear, concise, and accurate responses.";
const MAX_NEW_TOKENS: usize = 1024;

fn load_hf_model(
    repo_id: &str,
    revision: &str,
    quantization: QuantizationConfig,
) -> anyhow::Result<(Llama, Config, Tokenizer)> {
    eprintln!("Loading model: {repo_id}...");
    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_pretrained(repo_id, quantization)?;
    eprintln!("Model loaded in {:.2?}", load_start.elapsed());

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    Ok((model, config, tokenizer))
}

fn load_local_model(model_dir: &Path) -> anyhow::Result<(Llama, Config, Tokenizer)> {
    eprintln!("Loading model from: {}...", model_dir.display());
    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_local(model_dir, QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4))?;
    eprintln!("Model loaded in {:.2?}", load_start.elapsed());

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?
    } else {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::new(
            "microsoft/bitnet-b1.58-2B-4T".to_string(),
            hf_hub::RepoType::Model,
        ));
        let path = repo.get("tokenizer.json")?;
        Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?
    };

    Ok((model, config, tokenizer))
}

fn is_eos_token(token: u32, config: &Config) -> bool {
    // <|end_of_text|> and <|eot_id|> fallback for Llama3 models
    if token == 128001 || token == 128009 {
        return true;
    }
    // bos_token_id indicates model restarting
    if Some(token) == config.bos_token_id {
        return true;
    }
    match config.eos_token_id {
        Some(LlamaEosToks::Single(id)) => token == id,
        Some(LlamaEosToks::Multiple(ref ids)) => ids.contains(&token),
        None => token == 2,
    }
}

fn uses_llama3_template(config: &Config) -> bool {
    // Llama3 chat models have eos_token_id = [128009] (single), meaning <|eot_id|>.
    // Base models using the Llama3 tokenizer have eos_token_id = [128001, 128009],
    // with both <|end_of_text|> and <|eot_id|>. Only use the chat template for
    // actual chat models, not base models.
    if config.bos_token_id != Some(128000) {
        return false;
    }
    match config.eos_token_id {
        Some(LlamaEosToks::Single(id)) => id == 128009,
        Some(LlamaEosToks::Multiple(_)) => false,
        None => false,
    }
}

fn is_llama3_tokenizer(config: &Config) -> bool {
    config.bos_token_id == Some(128000)
}

fn main() -> anyhow::Result<()> {
    let num_threads = std::env::var("XORNET_THREADS")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(0);
    xor_net::init_threads(num_threads).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let args: Vec<String> = std::env::args().collect();
    let model_arg = args.get(1).map(|s| s.as_str()).unwrap_or("2b");

    let (model, config, tokenizer) = match model_arg {
        "2b" => load_hf_model(
            "microsoft/bitnet-b1.58-2B-4T",
            "main",
            QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int8),
        )?,
        "3b" => load_hf_model(
            "1bitLLM/bitnet_b1_58-3B",
            "main",
            QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4),
        )?,
        "8b" => load_hf_model(
            "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
            "main",
            QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int8),
        )?,
        _ => load_local_model(Path::new(model_arg))?,
    };

    let mut cache = Cache::new(true, &config)?;
    let mut sampler = Sampler::new(42, Some(0.8), Some(0.95), 1.1);

    println!("\n╔══════════════════════════════════════╗");
    println!("║        XorNet Chat v0.1              ║");
    println!("╚══════════════════════════════════════╝");
    println!("Model: {model_arg}");
    println!("Commands: /quit /reset");
    println!("{}", "─".repeat(46));

    let mut context_tokens: Vec<u32> = Vec::new();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let input = line?.trim().to_string();
        if input == "/quit" || input == "/exit" {
            break;
        }
        if input == "/reset" {
            cache = Cache::new(true, &config)?;
            context_tokens.clear();
            println!("Context reset.");
            continue;
        }
        if input.is_empty() {
            continue;
        }

        let use_llama3 = uses_llama3_template(&config);
        let is_llama3_base = !use_llama3 && is_llama3_tokenizer(&config);

        if use_llama3 {
            let user_prompt = if context_tokens.is_empty() {
                format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            } else {
                format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            };

            let add_special = context_tokens.is_empty();
            let user_ids = tokenizer
                .encode(user_prompt.as_str(), add_special)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();

            context_tokens.extend(&user_ids);
        } else if is_llama3_base {
            let user_prompt = if context_tokens.is_empty() {
                format!("System: {}\nUser: {}<|eot_id|>\nAssistant: ", SYSTEM_PROMPT, input)
            } else {
                format!("User: {}<|eot_id|>\nAssistant: ", input)
            };

            let add_special = context_tokens.is_empty();
            let user_ids = tokenizer
                .encode(user_prompt.as_str(), add_special)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();

            context_tokens.extend(&user_ids);
        } else {
            let user_prompt = if context_tokens.is_empty() {
                format!("[System]\n{SYSTEM_PROMPT}\n\n[User]\n{input}\n\n[Assistant]\n")
            } else {
                format!("[User]\n{input}\n\n[Assistant]\n")
            };

            let add_special = context_tokens.is_empty();
            let user_ids = tokenizer
                .encode(user_prompt.as_str(), add_special)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();

            context_tokens.extend(&user_ids);
        }

        let mut tokens = context_tokens.clone();
        let mut index_pos = 0;
        let start_gen = Instant::now();
        let mut generated = 0usize;
        let mut total_forward = std::time::Duration::ZERO;
        let mut total_sample = std::time::Duration::ZERO;
        
        let mut response_tokens: Vec<u32> = Vec::new();
        let mut prev_text = String::new();

        print!("Assistant: ");
        io::stdout().flush()?;

        let (blocks_start, lm_head_start, other_start) =
            xor_net::models::llama::get_profiling_stats();

        const PREFILL_CHUNK_SIZE: usize = 512;

        for _step in 0..MAX_NEW_TOKENS {
            let context_size = if index_pos < context_tokens.len() {
                let remaining = context_tokens.len() - index_pos;
                remaining.min(PREFILL_CHUNK_SIZE)
            } else {
                1
            };
            let start_pos = index_pos;

            let t0 = Instant::now();
            let logits = model.forward(&tokens[start_pos..start_pos + context_size], index_pos, &mut cache)?;
            total_forward += t0.elapsed();

            let t0 = Instant::now();
            let mut logits_data = logits.into_data();

            let next_token = sampler.sample(&mut logits_data, &response_tokens)?;
            total_sample += t0.elapsed();

            tokens.push(next_token);
            response_tokens.push(next_token);
            index_pos += context_size;
            generated += 1;

            if is_eos_token(next_token, &config) {
                response_tokens.pop();
                break;
            }

            if let Ok(text) = tokenizer.decode(&response_tokens, true) {
                if text.len() > prev_text.len() {
                    let new_part = &text[prev_text.len()..];
                    // Only print complete UTF-8 characters
                    if new_part.is_char_boundary(new_part.len()) {
                        print!("{new_part}");
                        io::stdout().flush()?;
                        prev_text = text.clone();
                    }
                }

                if text.ends_with("[User]") || text.ends_with("\nUser:") {
                    break;
                }
            }
        }

        println!();

        context_tokens.extend(&response_tokens);

        let elapsed = start_gen.elapsed();
        let tps = generated as f64 / elapsed.as_secs_f64();

        let (blocks_end, lm_head_end, other_end) =
            xor_net::models::llama::get_profiling_stats();
        let blocks_ms = (blocks_end.saturating_sub(blocks_start)) as f64 / 1000.0;
        let lm_head_ms = (lm_head_end.saturating_sub(lm_head_start)) as f64 / 1000.0;
        let other_ms = (other_end.saturating_sub(other_start)) as f64 / 1000.0;

        println!("{}", "─".repeat(46));
        println!(
            " {generated} tokens in {:.2?} ({:.2} tok/s)",
            elapsed, tps
        );
        println!(" Forward: {total_forward:.2?} | Sample: {total_sample:.2?}");
        if generated > 0 {
            println!(
                " Blocks: {:.2}ms/tok | LM Head: {:.2}ms/tok | Other: {:.2}ms/tok",
                blocks_ms / generated as f64,
                lm_head_ms / generated as f64,
                other_ms / generated as f64,
            );
        }
        println!("{}", "─".repeat(46));
    }

    Ok(())
}
