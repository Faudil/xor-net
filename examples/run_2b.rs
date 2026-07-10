use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
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
    // Initialize rayon threads to utilize all logical cores
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let device = Device::Cpu;

    let model_id = "microsoft/bitnet-b1.58-2B-4T";
    println!("Loading 1.58-bit model {}...", model_id);
    
    // Set 1.58-bit ternary quantization mode
    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4);

    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_pretrained(model_id, quantization)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());
    
    // Initialize KV cache
    let mut cache = xor_net::Cache::new(true, &config)?;

    // Fetch the tokenizer from Hugging Face
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
    let mut tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e.to_string()))?.get_ids().to_vec();
    
    println!("Prompt: '{}'", prompt);
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);
    
    let mut index_pos = 0;
    let max_len = 100;
    let start_gen = Instant::now();

    for _loop_idx in 0..max_len {
        let context_size = if index_pos == 0 { tokens.len() } else { 1 };
        let start_pos = tokens.len().saturating_sub(context_size);
        
        let start_forward = Instant::now();
        let logits = model.forward(&tokens[start_pos..], index_pos, &mut cache)?;
        let _elapsed_forward = start_forward.elapsed();
        
        let logits_candle = Tensor::from_vec(logits.data, (logits.shape[2],), &device)?;
        let next_token = logits_processor.sample(&logits_candle)?;
        
        tokens.push(next_token);
        index_pos += context_size;
        
        // Break on standard Eos/Bos tokens or model configuration's bos/eos
        if Some(next_token) == config.bos_token_id || next_token == 128009 || next_token == 128001 || next_token == 2 {
            break;
        }
    }
    
    let generated_text = tokenizer.decode(&tokens, true).unwrap_or(String::new());
    println!("Output: {}", generated_text);
    
    let elapsed = start_gen.elapsed();
    let tps = max_len as f64 / elapsed.as_secs_f64();
    println!("Generated {} tokens in {:.2?} ({:.2} tokens/sec)", max_len, elapsed, tps);
    
    Ok(())
}
