mod ternary_llama;

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use ternary_llama::{Cache, Config, Llama};
use tokenizers::Tokenizer;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    
    println!("Loading and packing weights from safetensors...");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&paths, candle_core::DType::F32, &device)?
    };

    let bitnet_cfg = Config {
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
    
    let model = Llama::load(vb, &bitnet_cfg)?;
    let mut cache = Cache::new(true, candle_core::DType::F32, &bitnet_cfg, &device)?;

    let prompt = "The capital of France is";
    let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    
    println!("Prompt: '{}'", prompt);
    print!("{}", prompt);
    std::io::stdout().flush()?;

    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);
    
    let mut index_pos = 0;
    let max_len = 50;

    let start_gen = Instant::now();

    for _ in 0..max_len {
        let context_size = if index_pos == 0 { tokens.len() } else { 1 };
        let start_pos = tokens.len().saturating_sub(context_size);
        let input = Tensor::new(&tokens[start_pos..], &device)?.unsqueeze(0)?;
        
        let logits = model.forward(&input, index_pos, &mut cache)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let next_token = logits_processor.sample(&logits)?;
        
        tokens.push(next_token);
        index_pos += context_size;
        
        if let Some(text) = tokenizer.decode(&[next_token], true).unwrap_or(String::new()).into() {
            print!("{}", text);
            std::io::stdout().flush()?;
        }
        
        // Stop if EOS token
        if Some(next_token) == bitnet_cfg.bos_token_id || next_token == 2 {
            break;
        }
    }
    
    println!();
    
    let elapsed = start_gen.elapsed();
    let tokens_generated = max_len as f64;
    let tps = tokens_generated / elapsed.as_secs_f64();
    println!("Generated {} tokens in {:.2?} ({:.2} tokens/sec)", max_len, elapsed, tps);
    Ok(())
}
