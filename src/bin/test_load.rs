use xor_net::models::llama::Llama;
use xor_net::AutoModelForCausalLM;
use std::path::Path;

fn main() {
    let _ = AutoModelForCausalLM::from_local(Path::new("/home/faudil/RustroverProjects/XorNet/models/bitnet-2b"), xor_net::QuantizationConfig::Bit1_58(xor_net::bit1_58::quantization::TernaryPackType::Pack4, xor_net::nn::LmHeadConfig::Int4));
}
