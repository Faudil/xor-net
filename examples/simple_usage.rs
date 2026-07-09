use xor_net::bit1::layers::BitLinear;
use xor_net::bit1_58::layers::TernaryLinear;
use xor_net::bit1_58::quantization::TernaryPackType;
use candle_core::{Device, Tensor, Module};



fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    
    let in_dim = 4;
    let out_dim = 2;
    let bit_weights = vec![
        1.0, -1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0, -1.0,
    ];
    
    let bit_linear = BitLinear::new(in_dim, out_dim, &bit_weights)?;
    
    let input_data = vec![
        0.5f32, 1.0, -0.5, 2.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    let input = Tensor::from_vec(input_data.clone(), (2, in_dim), &device)?;
    
    println!("--- BitLinear (1-bit) ---");
    let out_bit = bit_linear.forward(&input)?;
    println!("Input:\n{}", input);
    println!("Output:\n{}", out_bit);

    let ternary_weights = vec![
        1.0, 0.0, -1.0, 1.0,
        0.0, -1.0, 1.0, 0.0,
    ];
    
    let ternary_linear_pack4 = TernaryLinear::new(in_dim, out_dim, &ternary_weights, TernaryPackType::Pack4)?;
    let ternary_linear_pack5 = TernaryLinear::new(in_dim, out_dim, &ternary_weights, TernaryPackType::Pack5)?;
    
    println!("\n--- TernaryLinear (1.58-bit) Pack4 ---");
    let out_ternary4 = ternary_linear_pack4.forward(&input)?;
    println!("Output (Pack4):\n{}", out_ternary4);

    println!("\n--- TernaryLinear (1.58-bit) Pack5 ---");
    let out_ternary5 = ternary_linear_pack5.forward(&input)?;
    println!("Output (Pack5):\n{}", out_ternary5);

    Ok(())
}
