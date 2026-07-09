use xor_net::bit1_58::quantization::{
    pack_1_58bit_4pack, unpack_1_58bit_4pack,
    pack_1_58bit_5pack, unpack_1_58bit_5pack,
    quantize_f32_to_i8, TernaryPackType
};
use xor_net::bit1_58::simd::{ternary_dot_product_pack4, ternary_dot_product_pack4_avx2, ternary_dot_product_pack5_scalar};
use xor_net::bit1_58::layers::TernaryLinear;
use candle_core::{Device, Tensor, Module};

#[test]
fn test_ternary_pack_unpack_roundtrip() {
    let weights = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0];
    
    let packed4 = pack_1_58bit_4pack(&weights, 1.0);
    let unpacked4 = unpack_1_58bit_4pack(&packed4, weights.len());
    assert_eq!(weights, unpacked4);

    let packed5 = pack_1_58bit_5pack(&weights, 1.0);
    let unpacked5 = unpack_1_58bit_5pack(&packed5, weights.len());
    assert_eq!(weights, unpacked5);
}

#[test]
fn test_absmax_quantization() {
    let acts = vec![0.0, -2.5, 5.0, 1.25];
    let (quantized, scale) = quantize_f32_to_i8(&acts);
    
    assert!((scale - (5.0 / 127.0)).abs() < 1e-5);
    assert_eq!(quantized, vec![0, -64, 127, 32]);
    
    let zero_acts = vec![0.0, 0.0, 0.0];
    let (quantized_zeros, scale_zeros) = quantize_f32_to_i8(&zero_acts);
    assert_eq!(quantized_zeros, vec![0, 0, 0]);
    assert_eq!(scale_zeros, 1.0);
}

#[test]
fn test_ternary_simd_parity() {
    let lengths = [4, 8, 12, 16, 20, 32, 36, 64, 68, 128, 132];
    for &len in &lengths {
        let acts: Vec<i8> = (0..len).map(|i| ((i % 255) as i32 - 127) as i8).collect();
        let w_floats: Vec<f32> = (0..len).map(|i| {
            match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            }
        }).collect();
        
        let w_packed = pack_1_58bit_4pack(&w_floats, 1.0);
        
        let scalar_res = ternary_dot_product_pack4(&acts, &w_packed, len);
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                let avx2_res = unsafe { ternary_dot_product_pack4_avx2(&acts, &w_packed, len) };
                assert_eq!(scalar_res, avx2_res, "Mismatch at len {}", len);
            }
        }
        
        let w_packed5 = pack_1_58bit_5pack(&w_floats, 1.0);
        let scalar_pack5_res = ternary_dot_product_pack5_scalar(&acts, &w_packed5, len);
        assert_eq!(scalar_res, scalar_pack5_res, "Pack5 scalar mismatch at len {}", len);
    }
}

#[test]
fn test_ternary_linear_layer() {
    let device = Device::Cpu;
    let in_dim = 4;
    let out_dim = 2;
    let weights = vec![
         1.0,  0.0, -1.0,  1.0,
         0.0, -1.0,  1.0,  0.0,
    ];
    
    let layer_pack4 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack4).unwrap();
    let layer_pack5 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack5).unwrap();
    
    let input = Tensor::from_vec(vec![0.5f32, 1.0, -0.5, 2.0], (1, in_dim), &device).unwrap();
    
    let out4 = layer_pack4.forward(&input).unwrap();
    let out4_vec = out4.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let out5 = layer_pack5.forward(&input).unwrap();
    let out5_vec = out5.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(out4_vec, out5_vec);
    
    assert!((out4_vec[0] - 1.879921).abs() < 1e-4);
    assert!((out4_vec[1] - (-0.944881)).abs() < 1e-4);
}
