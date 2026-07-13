use xor_net::{AutoModelForCausalLM, QuantizationConfig};
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;

fn main() {
    println!("Loading 3B model...");
    let (model, _) = AutoModelForCausalLM::from_pretrained(
        "1bitLLM/bitnet_b1_58-3B", 
        QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int8, true)
    ).unwrap();
    println!("Model loaded!");
}
