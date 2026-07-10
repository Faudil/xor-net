//! Non-regression tests for linear layers: F32Linear, TernaryLinear, BitLinear.
//! Each test verifies forward-pass correctness against a hand-computed
//! reference matmul or known mathematical properties.

use xor_net::tensor::FastTensor;
use xor_net::nn::dynamic_linear::F32Linear;
use xor_net::bit1_58::layers::TernaryLinear;
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::bit1::layers::BitLinear;

const TOL: f32 = 1e-4;

fn ref_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // weight layout: [out_dim, in_dim] (row-major)
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        for d in 0..in_dim {
            out[o] += input[d] * weight[o * in_dim + d];
        }
    }
    out
}

// ── F32Linear ───────────────────────────────────────────────────────────

#[test]
fn f32_linear_identity_weight() {
    // 4×4 identity matrix → output == input
    let mut w_data = vec![0.0f32; 16];
    for i in 0..4 { w_data[i * 4 + i] = 1.0; }
    let w = FastTensor::new(w_data, vec![4, 4]);
    let layer = F32Linear::new(w);
    let input = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f32_linear_known_matmul() {
    // W = [[1,2],[3,4],[5,6]], shape [3,2], in=2, out=3
    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_data = vec![0.5, -1.0];
    let expected = ref_matmul(&input_data, &w_data, 2, 3);
    let w = FastTensor::new(w_data, vec![3, 2]);
    let layer = F32Linear::new(w);
    let input = FastTensor::new(input_data, vec![1, 2]);
    let out = layer.forward(&input).unwrap();
    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < TOL, "F32Linear idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn f32_linear_batched() {
    let w_data = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
    let w = FastTensor::new(w_data, vec![2, 2]);
    let layer = F32Linear::new(w);
    let input = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out.shape, vec![2, 2]);
}

// ── TernaryLinear ───────────────────────────────────────────────────────

#[test]
fn ternary_linear_pack4_vs_pack5_parity() {
    let in_dim = 20;
    let out_dim = 4;
    let weights: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| match i % 3 { 0 => -1.0, 1 => 0.0, _ => 1.0 })
        .collect();
    let input_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 10.0) * 0.1).collect();
    let input = FastTensor::new(input_data, vec![1, in_dim]);

    let l4 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack4).unwrap();
    let l5 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack5).unwrap();
    let o4 = l4.forward(&input).unwrap();
    let o5 = l5.forward(&input).unwrap();

    for (i, (&a, &b)) in o4.data.iter().zip(o5.data.iter()).enumerate() {
        assert!((a - b).abs() < TOL, "Pack4 vs Pack5 mismatch idx {}: {} vs {}", i, a, b);
    }
}

#[test]
fn ternary_linear_zero_input() {
    let weights = vec![1.0, -1.0, 0.0, 1.0];
    let layer = TernaryLinear::new(2, 2, &weights, TernaryPackType::Pack4).unwrap();
    let input = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let out = layer.forward(&input).unwrap();
    for &v in &out.data {
        assert!(v.abs() < TOL, "Zero input should produce ~zero output, got {}", v);
    }
}

// ── BitLinear ───────────────────────────────────────────────────────────

#[test]
fn bit_linear_known_result() {
    // W = [[+1,-1,+1,+1], [-1,-1,+1,-1]]
    // input sign-packed: [0.5, 1.0, -0.5, 2.0] → bits [1,1,0,1]
    // XNOR(input_bits, row0_bits) where row0=[1,0,1,1]:
    //   xnor = [1,0,0,1] → matches=2, result=2*2-4=0
    // XNOR(input_bits, row1_bits) where row1=[0,0,1,0]:
    //   xnor = [0,0,0,0] → matches=0, result=2*0-4=-4
    let weights = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
    let layer = BitLinear::new(4, 2, &weights).unwrap();
    let input = FastTensor::new(vec![0.5, 1.0, -0.5, 2.0], vec![1, 4]);
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.data, vec![0.0, -4.0]);
}

#[test]
fn bit_linear_identical_weights_rows() {
    // All rows identical → all outputs equal
    let row = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let weights: Vec<f32> = row.iter().cycle().take(row.len() * 3).cloned().collect();
    let layer = BitLinear::new(8, 3, &weights).unwrap();
    let input = FastTensor::new(vec![1.0; 8], vec![1, 8]);
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.data[0], out.data[1]);
    assert_eq!(out.data[1], out.data[2]);
}
