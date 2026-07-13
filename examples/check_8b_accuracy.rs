use hf_hub::{api::sync::Api, Repo, RepoType};
use std::time::Instant;
use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;

fn main() -> anyhow::Result<()> {
    xor_net::init_threads(1).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let model_id = "HF1BitLLM/Llama3-8B-1.58-100B-tokens";
    println!("Loading {}...", model_id);

    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, xor_net::nn::LmHeadConfig::F32, true);

    let load_start = Instant::now();
    let (model, config) = AutoModelForCausalLM::from_pretrained(model_id, quantization)?;
    println!("Loaded in {:.2?}", load_start.elapsed());

    let mut cache = xor_net::Cache::new(true, &config)?;

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let prompt = "The capital of France is";
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    println!("Prompt tokens: {:?}", tokens);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward(&tokens, 0, &mut cache)
    }));
    match result {
        Ok(Ok(logits)) => {
            println!("Logits shape: {:?}", logits.shape);
            let logit_min = logits.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let logit_max = logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            println!("Logits range: [{:.4}, {:.4}]", logit_min, logit_max);
            let mut indices: Vec<usize> = (0..logits.data.len()).collect();
            indices.sort_unstable_by(|&a, &b| logits.data[b].partial_cmp(&logits.data[a]).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top 5 tokens:");
            for i in 0..5.min(indices.len()) {
                let idx = indices[i];
                let tok_str = tokenizer.id_to_token(idx as u32).unwrap_or_default();
                println!("  [{}] token_id={}, logit={:.4}, tok={:?}", i, idx, logits.data[idx], tok_str);
            }
        }
        Ok(Err(e)) => eprintln!("Forward error: {:?}", e),
        Err(e) => eprintln!("Panic: {:?}", e),
    }

    Ok(())
}
