mod baseline_llama;
mod ternary_llama;

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::env;
use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let use_ternary = args.iter().any(|arg| arg == "--mode=ternary");
    let use_baseline = args.iter().any(|arg| arg == "--mode=baseline");

    if !use_ternary && !use_baseline {
        println!("Please specify --mode=ternary or --mode=baseline");
        return Ok(());
    }

    xor_net::init_threads(4)?;
    let device = Device::Cpu;

    println!("Downloading 1bitLLM/bitnet_b1_58-3B...");
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "1bitLLM/bitnet_b1_58-3B".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let file1 = repo.get("model-00001-of-00003.safetensors")?;
    let file2 = repo.get("model-00002-of-00003.safetensors")?;
    let file3 = repo.get("model-00003-of-00003.safetensors")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
    let paths = vec![file1, file2, file3];
    
    let load_start = Instant::now();
    println!("Loading and packing weights from safetensors...");
    
    // Create configs
    let bitnet_cfg_ternary = ternary_llama::Config {
        hidden_size: 3200,
        intermediate_size: 8640,
        vocab_size: 32002,
        num_hidden_layers: 26,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: Some(ternary_llama::LlamaEosToks::Single(2)),
        tie_word_embeddings: true,
        use_flash_attn: false,
        max_position_embeddings: 2048,
        rope_scaling: None,
    };

    let bitnet_cfg_baseline = baseline_llama::Config {
        hidden_size: 3200,
        intermediate_size: 8640,
        vocab_size: 32002,
        num_hidden_layers: 26,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: Some(baseline_llama::LlamaEosToks::Single(2)),
        tie_word_embeddings: true,
        use_flash_attn: false,
        max_position_embeddings: 2048,
        rope_scaling: None,
    };
    
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)?
    };

    let prompt = "The capital of France is";
    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    
    println!("Prompt: '{}'", prompt);
    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);
    
    let mut index_pos = 0;
    let max_len = 50;

    let start_gen;
    let mut generated_text = String::new();

    if use_ternary {
        let model = ternary_llama::Llama::load(vb, &bitnet_cfg_ternary)?;
        let mut cache = ternary_llama::Cache::new(true, candle_core::DType::F32, &bitnet_cfg_ternary, &device)?;
        println!("Loaded Ternary model in {:.2?}", load_start.elapsed());
        start_gen = Instant::now();
        
        for _ in 0..max_len {
            let context_size = if index_pos == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &device)?.unsqueeze(0)?;
            
            let logits = model.forward(&input, index_pos, &mut cache)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let next_token = logits_processor.sample(&logits)?;
            
            tokens.push(next_token);
            index_pos += context_size;
            
            if Some(next_token) == bitnet_cfg_ternary.bos_token_id || next_token == 2 {
                break;
            }
        }
    } else {
        let model = baseline_llama::Llama::load(vb, &bitnet_cfg_baseline)?;
        let mut cache = baseline_llama::Cache::new(true, candle_core::DType::F32, &bitnet_cfg_baseline, &device)?;
        println!("Loaded Baseline model in {:.2?}", load_start.elapsed());
        start_gen = Instant::now();
        
        for _ in 0..max_len {
            let context_size = if index_pos == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &device)?.unsqueeze(0)?;
            
            let logits = model.forward(&input, index_pos, &mut cache)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let next_token = logits_processor.sample(&logits)?;
            
            tokens.push(next_token);
            index_pos += context_size;
            
            if Some(next_token) == bitnet_cfg_baseline.bos_token_id || next_token == 2 {
                break;
            }
        }
    }
    
    generated_text = tokenizer.decode(&tokens, true).unwrap_or(String::new());
    println!("Output: {}", generated_text);
    
    let elapsed = start_gen.elapsed();
    let tokens_generated = index_pos as f64 - tokens.len() as f64 + max_len as f64; // Approx
    let tps = max_len as f64 / elapsed.as_secs_f64();
    println!("Generated {} tokens in {:.2?} ({:.2} tokens/sec)", max_len, elapsed, tps);
    
    Ok(())
}
