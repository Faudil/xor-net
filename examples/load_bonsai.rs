//! Load and validate Bonsai GGUF models (both 1.7B qwen3 and 27B qwen35).
use std::sync::Arc;
use xor_net::models::bonsai_qwen35::*;
use xor_net::*;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/bonsai-1.7b-q1_0.gguf".to_string());
    let device = candle_core::Device::Cpu;

    let gguf = GgufFile::open(&path)?;
    let arch = gguf.architecture()?;

    match arch.as_str() {
        "qwen3" => run_qwen3(Arc::new(gguf), &device),
        "qwen35" => run_qwen35(Arc::new(gguf), &device),
        _ => anyhow::bail!("Unknown architecture: {arch}"),
    }
}

fn run_qwen3(gguf: Arc<GgufFile>, device: &candle_core::Device) -> anyhow::Result<()> {
    let cfg = BonsaiConfig::from_gguf_metadata(&gguf.metadata.metadata)?;
    println!("Architecture: qwen3 (full attention, pure)");
    println!(
        "Layers: {}, Hidden: {}, Heads: {} / {} KV, Head dim: {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim
    );
    println!("Building model...");
    let model = BonsaiModel::from_gguf(gguf.clone(), device)?;
    println!("Model built! Running forward pass...");
    let mut cache = BonsaiCache::new(&cfg, device)?;
    let tokens = vec![0u32];
    let _logits = model.forward(&tokens, 0, &mut cache)?;
    println!("✓ Forward pass completed successfully!");
    Ok(())
}

fn run_qwen35(gguf: Arc<GgufFile>, device: &candle_core::Device) -> anyhow::Result<()> {
    let cfg = BonsaiQwen35Config::from_gguf(&gguf)?;
    println!("Architecture: qwen35 (hybrid: SSM + full attention)");
    println!(
        "Layers: {}, Hidden: {}, Heads: {} / {} KV",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
    );
    println!(
        "  Full attention interval: {} (blocks 3,7,11,...)",
        cfg.full_attention_interval
    );
    println!(
        "  SSM: {} groups, {} states, {} v_heads, conv_kernel={}",
        cfg.ssm_group_count, cfg.ssm_state_size, cfg.ssm_num_v_heads, cfg.ssm_conv_kernel
    );
    println!("Building model...");
    let model = BonsaiQwen35Model::from_gguf(gguf.clone(), device)?;
    println!("Model built! Running forward pass...");
    let mut cache = BonsaiQwen35Cache::new(&cfg, device)?;
    let tokens = vec![0u32];
    let logits = model.forward(&tokens, 0, &mut cache)?;
    let lv: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
    let nan = lv.iter().filter(|x| x.is_nan()).count();
    let min = lv.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = lv.iter().sum::<f32>() / lv.len() as f32;
    println!(
        "✓ Forward pass completed! logits: nan={nan} min={min:.3} max={max:.3} mean={mean:.3}"
    );
    Ok(())
}
