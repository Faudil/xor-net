use xor_net::bit1::quantization::pack_1bit;
use xor_net::bit1::simd::{xnor_dot_product, xnor_dot_product_avx2};
use xor_net::bit1::layers::BitLinear;
use candle_core::{Device, Tensor, Module};

#[test]
fn test_bit1_packing() {
    let weights = vec![1.5, -0.3, 0.0, -10.0, 5.0, 0.0, -1.0, 1.2];
    let packed = pack_1bit(&weights);
    assert_eq!(packed.len(), 1);
    assert_eq!(packed[0], 181);
}

#[test]
fn test_bit1_simd_parity() {
    let lengths = [8, 16, 32, 40, 64, 96, 100, 128, 256, 260];
    for &len in &lengths {
        let a_floats: Vec<f32> = (0..len).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let b_floats: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        
        let a_packed = pack_1bit(&a_floats);
        let b_packed = pack_1bit(&b_floats);
        
        let scalar_res = xnor_dot_product(&a_packed, &b_packed, len);
        
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                let avx2_res = unsafe { xnor_dot_product_avx2(&a_packed, &b_packed, len) };
                assert_eq!(scalar_res, avx2_res, "Mismatch at len {}", len);
            }
        }
    }
}

#[test]
fn test_bit1_linear_layer() {
    let device = Device::Cpu;
    let in_dim = 4;
    let out_dim = 2;
    let weights = vec![
         1.0, -1.0,  1.0,  1.0,
        -1.0, -1.0,  1.0, -1.0,
    ];
    let layer = BitLinear::new(in_dim, out_dim, &weights).unwrap();
    
    let input = Tensor::from_vec(vec![0.5f32, 1.0, -0.5, 2.0], (1, in_dim), &device).unwrap();
    let out = layer.forward(&input).unwrap();
    let out_vec = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    
    assert_eq!(out_vec, vec![0.0, -4.0]);
}
