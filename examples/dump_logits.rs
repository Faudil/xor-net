use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;

fn main() -> anyhow::Result<()> {
    let model_id = "HF1BitLLM/Llama3-8B-1.58-100B-tokens";
    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4);
    let (model, _config) = AutoModelForCausalLM::from_pretrained(model_id, quantization)?;
    let mut cache = xor_net::Cache::new(true, &_config)?;

    // "The capital of France is" -> [128000, 791, 6864, 315, 9822, 374]
    let tokens = vec![128000u32, 791, 6864, 315, 9822, 374];
    let logits = model.forward(&tokens, 0, &mut cache)?;
    
    // Logits shape should be [1, 1, 128256]
    let logit_data = &logits.data;
    println!("logits len: {}", logit_data.len());
    println!("logits shape: {:?}", logits.shape);
    println!("First 10 logits: {:?}", &logit_data[..10]);
    
    // Compute stats
    let mean: f32 = logit_data.iter().sum::<f32>() / logit_data.len() as f32;
    let std: f32 = (logit_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logit_data.len() as f32).sqrt();
    let max_val = logit_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = logit_data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Logit stats: mean={:.4}, std={:.4}, max={:.4}, min={:.4}", mean, std, max_val, min_val);
    
    // Top 20 tokens by logit value
    let mut indices: Vec<usize> = (0..logit_data.len()).collect();
    indices.sort_by(|&a, &b| logit_data[b].partial_cmp(&logit_data[a]).unwrap());
    println!("Top 20 token indices and logits:");
    for &idx in indices.iter().take(20) {
        println!("  {:6}: {:.4}", idx, logit_data[idx]);
    }
    
    Ok(())
}
