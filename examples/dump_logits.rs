use xor_net::{AutoModelForCausalLM, QuantizationConfig, Cache, Config};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use std::time::Instant;

fn main() {
    let (model, config) = AutoModelForCausalLM::from_pretrained(
        "1bitLLM/bitnet_b1_58-3B", 
        QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int8)
    ).unwrap();

    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let mut cache1 = Cache::new(true, &config).unwrap();
    let logits1 = model.forward(&tokens, 0, &mut cache1).unwrap();

    let mut cache2 = Cache::new(true, &config).unwrap();
    let mut logits2 = None;
    for (i, &t) in tokens.iter().enumerate() {
        logits2 = Some(model.forward(&[t], i, &mut cache2).unwrap());
    }

    println!("Batched logits: {:?}", &logits1.data[..10]);
    println!("Unbatched logits: {:?}", &logits2.unwrap().data[..10]);
}
