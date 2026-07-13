use std::path::Path;
use tokenizers::Tokenizer;
use xor_net::{AutoModelForCausalLM, Cache, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use xor_net::Sampler;

fn main() -> anyhow::Result<()> {
    let model_path = "/home/faudil/RustroverProjects/XorNet/models/bitnet-2b";
    let (model, config) = AutoModelForCausalLM::from_local(
        Path::new(model_path), 
        QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4)
    )?;
    
    let tokenizer_path = Path::new(model_path).join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() { Tokenizer::from_file(tokenizer_path).unwrap() } else { let api = hf_hub::api::sync::Api::new().unwrap(); let repo = api.repo(hf_hub::Repo::new("microsoft/bitnet-b1.58-2B-4T".to_string(), hf_hub::RepoType::Model)); let path = repo.get("tokenizer.json").unwrap(); Tokenizer::from_file(path).unwrap() };

    let mut sampler = Sampler::new(3132833641, Some(0.8), Some(0.95), 1.2); // use higher temp and repetition penalty 1.2

    let prompt2 = "The first ten numbers are: 1, 2, 3, 4, 5, ";

    let mut tokens = vec![config.bos_token_id.unwrap_or(1)];
    tokens.extend(tokenizer.encode(prompt2, false).unwrap().get_ids());

    println!("=== Testing Prompt ===");
    println!("Prompt: {}", prompt2);
    
    let mut cache = Cache::new(true, &config)?;
    let mut response_tokens = Vec::new();
    
    for i in 0..tokens.len() {
        let logits = model.forward(&tokens[i..i+1], i, &mut cache)?;
        if i == tokens.len() - 1 {
            let mut logits_data = logits.into_data();
            let next_token = sampler.sample(&mut logits_data, &response_tokens)?;
            tokens.push(next_token);
            response_tokens.push(next_token);
        }
    }
    
    let mut current_pos = tokens.len() - 1;
    for _ in 0..20 {
        let logits = model.forward(&tokens[current_pos..current_pos+1], current_pos, &mut cache)?;
        let mut logits_data = logits.into_data();
        let next_token = sampler.sample(&mut logits_data, &response_tokens)?;
        tokens.push(next_token);
        response_tokens.push(next_token);
        current_pos += 1;
        if next_token == 128009 || next_token == 128001 {
            break;
        }
    }
    println!("Response: {}", tokenizer.decode(&response_tokens, true).unwrap());

    Ok(())
}
