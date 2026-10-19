use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use xor_net::{BonsaiModel, BonsaiCache, BonsaiConfig, GgufFile};

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Read model path from env or use default
    let model_path = std::env::var("BONSAI_MODEL").ok();
    let model_id = "prism-ml/Bonsai-27B-gguf";

    let gguf_filename = "Bonsai-27B-Q1_0.gguf";

    let (gguf_path, config) = if let Some(path) = model_path {
        let gguf = GgufFile::open(&path)?;
        let cfg = BonsaiConfig::from_gguf_metadata(&gguf.metadata.metadata)?;
        println!("Architecture: {}", cfg.architecture);
        println!("Hidden size: {}", cfg.hidden_size);
        println!("Layers: {}", cfg.num_hidden_layers);
        println!("Heads: {} / {} KV", cfg.num_attention_heads, cfg.num_key_value_heads);
        println!("Vocab: {}", cfg.vocab_size);
        println!("Context: {}", cfg.max_position_embeddings);
        println!("File: {path}");
        (Arc::new(gguf), cfg)
    } else {
        println!("Downloading model {model_id}...");
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        let path = repo.get(gguf_filename)?;
        let gguf = GgufFile::open(&path)?;
        let cfg = BonsaiConfig::from_gguf_metadata(&gguf.metadata.metadata)?;
        println!("Architecture: {}", cfg.architecture);
        println!("Hidden size: {}", cfg.hidden_size);
        println!("Layers: {}", cfg.num_hidden_layers);
        println!("Heads: {} / {} KV", cfg.num_attention_heads, cfg.num_key_value_heads);
        println!("Vocab: {}", cfg.vocab_size);
        println!("Context: {}", cfg.max_position_embeddings);
        println!("File: {}", path.display());
        (Arc::new(gguf), cfg)
    };

    // Build model
    println!("Building model...");
    let load_start = Instant::now();
    let model = BonsaiModel::from_gguf(gguf_path, &device)?;
    println!("Model built in {:.2?}", load_start.elapsed());

    // Load tokenizer
    let tokenizer_path = std::env::var("BONSAI_TOKENIZER").ok().unwrap_or_else(|| {
        let api = Api::new().expect("HF API to initialize");
        let repo = api.repo(Repo::with_revision(
            "prism-ml/Bonsai-27B-gguf".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        repo.get("tokenizer.json")
            .expect("tokenizer.json to exist")
            .display()
            .to_string()
    });

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let prompt = std::env::var("BONSAI_PROMPT").unwrap_or_else(|_| "Q: What is 2+2?\nA:".to_string());

    let tokens = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();

    println!("Prompt: '{prompt}' ({tokens} tokens)", tokens = tokens.len());

    let mut cache = BonsaiCache::new(&config, &device)?;
    let mut logits_processor = LogitsProcessor::new(299792458, None, None);

    let max_len = 20;
    let mut generated = tokens.clone();
    let mut index_pos = 0;
    let start_gen = Instant::now();

    for _ in 0..max_len {
        let ctx_size = if index_pos == 0 { generated.len() } else { 1 };
        let start_pos = generated.len().saturating_sub(ctx_size);

        let logits = model.forward(&generated[start_pos..], index_pos, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let next_token = logits_processor.sample(&logits)?;

        generated.push(next_token);
        index_pos += ctx_size;

        if Some(next_token) == config.eos_token_id || next_token == 2 {
            break;
        }
    }

    let output = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("\nOutput:\n{output}");

    let elapsed = start_gen.elapsed();
    let total_gen = generated.len().saturating_sub(tokens.len());
    if total_gen > 0 {
        let tps = total_gen as f64 / elapsed.as_secs_f64();
        println!("\nGenerated {total_gen} tokens in {elapsed:.2?} ({tps:.2} tok/s)");
    }

    Ok(())
}
