use xor_net::bit1_58::quantization::*;
use xor_net::bit1_58::simd::*;

fn main() {
    let w_f32 = vec![-1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0];
    let w_packed = pack_1_58bit_4pack(&w_f32, 1.0);
    
    let a_f32 = vec![10.0, -5.0, 3.0, 7.0, -2.0, 4.0, -6.0, 8.0];
    let (a_i8, inv_scale) = quantize_f32_to_i8(&a_f32);
    
    let mut dot_ref = 0.0;
    for i in 0..8 {
        dot_ref += a_i8[i] as f32 * w_f32[i];
    }
    
    let dot_simd = ternary_dot_product_pack4(&a_i8, &w_packed, 8);
    println!("Ref: {}, SIMD: {}, inv_scale: {}", dot_ref, dot_simd, inv_scale);
}
