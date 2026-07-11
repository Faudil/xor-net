use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_dir = Path::new("/home/faudil/RustroverProjects/XorNet/models/bitnet-2b");

    println!("Loading from local: {:?}", model_dir);
    let (model, config) = AutoModelForCausalLM::from_local(
        model_dir,
        QuantizationConfig::None,
    )?;

    println!("Model loaded! hidden_size={} layers={}", config.hidden_size, config.num_hidden_layers);

    let mut cache = xor_net::Cache::new(true, &config)?;

    let tokens = vec![128000u32, 15043, 29892, 590, 1128]; // "<|begin_of_text|>The capital of France is"
    println!("Tokens: {:?}", tokens);

    // Prefill
    let logits = model.forward(&tokens, 0, &mut cache)?;
    println!("Prefill logits shape: {:?}", logits.shape);
    println!("Logits[0][0..5]: {:?}", &logits.data[..5]);
    println!("Logits sum: {}", logits.data.iter().sum::<f32>());

    // Single token generation
    let next_tokens = vec![13u32]; // just one token
    let logits2 = model.forward(&next_tokens, tokens.len(), &mut cache)?;
    println!("Generate logits shape: {:?}", logits2.shape);

    let stats = xor_net::models::llama::get_profiling_stats();
    println!("Profiling: blocks={}us lm_head={}us other={}us", stats.0, stats.1, stats.2);

    Ok(())
}
