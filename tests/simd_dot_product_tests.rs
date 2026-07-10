//! Non-regression tests for SIMD dot-product kernels.
//!
//! Tests cover: ternary_dot_product_pack4 (scalar + AVX2 parity),
//! ternary_dot_product_pack5_scalar, xnor_dot_product, and known
//! hand-computed reference values for small inputs.

use xor_net::bit1_58::quantization::{pack_1_58bit_4pack, pack_1_58bit_5pack};
use xor_net::bit1_58::simd::{ternary_dot_product_pack4, ternary_dot_product_pack5_scalar};
use xor_net::bit1::quantization::pack_1bit;
use xor_net::bit1::simd::xnor_dot_product;

#[test]
fn ternary_pack4_dot_product_hand_computed() {
    // activations:    [2, -3, 1, 4]
    // weights (f32):  [-1, 0, 1, -1]  →  ternary: -1, 0, +1, -1
    // dot product = 2*(-1) + (-3)*(0) + 1*(1) + 4*(-1) = -2 + 0 + 1 - 4 = -5
    let acts: Vec<i8> = vec![2, -3, 1, 4];
    let w_f32 = vec![-1.0, 0.0, 1.0, -1.0];
    let w_packed = pack_1_58bit_4pack(&w_f32, 1.0);
    let result = ternary_dot_product_pack4(&acts, &w_packed, 4);
    assert_eq!(result, -5, "Hand-computed ternary pack4 dot product mismatch");
}

#[test]
fn ternary_pack5_dot_product_hand_computed() {
    // Same activations and weights as above but through pack5
    let acts: Vec<i8> = vec![2, -3, 1, 4, 5];
    let w_f32 = vec![-1.0, 0.0, 1.0, -1.0, 1.0];
    let w_packed = pack_1_58bit_5pack(&w_f32, 1.0);
    // dot = 2*(-1) + (-3)*0 + 1*1 + 4*(-1) + 5*1 = -2 + 0 + 1 - 4 + 5 = 0
    let result = ternary_dot_product_pack5_scalar(&acts, &w_packed, 5);
    assert_eq!(result, 0, "Hand-computed ternary pack5 dot product mismatch");
}

#[test]
fn ternary_pack4_all_zeros_weights() {
    // All-zero weights → dot product must be 0 regardless of activations
    let acts: Vec<i8> = vec![127, -128, 50, -50, 1, -1, 0, 100];
    let w_f32 = vec![0.0f32; 8];
    let w_packed = pack_1_58bit_4pack(&w_f32, 1.0);
    let result = ternary_dot_product_pack4(&acts, &w_packed, 8);
    assert_eq!(result, 0, "All-zero weights must produce zero dot product");
}

#[test]
fn ternary_pack4_identity_weights() {
    // All weights = +1 → dot product equals sum of activations
    let acts: Vec<i8> = vec![10, -20, 30, -40];
    let w_f32 = vec![1.0f32; 4];
    let w_packed = pack_1_58bit_4pack(&w_f32, 1.0);
    let result = ternary_dot_product_pack4(&acts, &w_packed, 4);
    let expected: i32 = acts.iter().map(|&x| x as i32).sum();
    assert_eq!(result, expected, "All +1 weights: dot product must equal sum of activations");
}

#[test]
fn ternary_pack4_negation_weights() {
    // All weights = -1 → dot product equals negated sum of activations
    let acts: Vec<i8> = vec![10, -20, 30, -40];
    let w_f32 = vec![-1.0f32; 4];
    let w_packed = pack_1_58bit_4pack(&w_f32, 1.0);
    let result = ternary_dot_product_pack4(&acts, &w_packed, 4);
    let expected: i32 = -(acts.iter().map(|&x| x as i32).sum::<i32>());
    assert_eq!(result, expected, "All -1 weights: dot product must negate");
}

#[test]
fn ternary_pack4_vs_pack5_parity_large() {
    // For lengths that are multiples of both 4 and 5 (i.e. 20, 40, 60), verify parity
    for len in [20, 40, 60, 100, 200] {
        let acts: Vec<i8> = (0..len).map(|i| ((i * 7 + 13) % 255) as i8).collect();
        let w_f32: Vec<f32> = (0..len).map(|i| match i % 3 { 0 => -1.0, 1 => 0.0, _ => 1.0 }).collect();

        let w4 = pack_1_58bit_4pack(&w_f32, 1.0);
        let w5 = pack_1_58bit_5pack(&w_f32, 1.0);

        let r4 = ternary_dot_product_pack4(&acts, &w4, len);
        let r5 = ternary_dot_product_pack5_scalar(&acts, &w5, len);

        assert_eq!(r4, r5, "Pack4 vs Pack5 mismatch at len={}", len);
    }
}

#[test]
fn xnor_dot_product_hand_computed() {
    // a = [+1, -1, +1, +1]  →  bits: 1,0,1,1 → packed byte = 0b00001101 = 13
    // b = [+1, +1, -1, +1]  →  bits: 1,1,0,1 → packed byte = 0b00001011 = 11
    // XNOR on bits: 1,0,0,1 → 2 matches out of 4
    // result = 2 * matches - total = 2 * 2 - 4 = 0
    let a_f32 = vec![1.0, -1.0, 1.0, 1.0];
    let b_f32 = vec![1.0, 1.0, -1.0, 1.0];
    let a = pack_1bit(&a_f32);
    let b = pack_1bit(&b_f32);
    let result = xnor_dot_product(&a, &b, 4);
    assert_eq!(result, 0.0, "XNOR dot product hand-computed mismatch");
}

#[test]
fn xnor_dot_product_identical_vectors() {
    // Identical vectors → all bits match → matches = total_bits
    // result = 2*total - total = total
    for len in [8, 16, 32, 64, 128, 256] {
        let f: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let packed = pack_1bit(&f);
        let result = xnor_dot_product(&packed, &packed, len);
        assert_eq!(result, len as f32, "Identical vectors must produce dot product = len ({})", len);
    }
}

#[test]
fn xnor_dot_product_opposite_vectors() {
    // Flipped vectors → 0 matches → result = 2*0 - total = -total
    for len in [8, 16, 32, 64] {
        let a_f32: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let b_f32: Vec<f32> = (0..len).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let a = pack_1bit(&a_f32);
        let b = pack_1bit(&b_f32);
        let result = xnor_dot_product(&a, &b, len);
        assert_eq!(result, -(len as f32), "Opposite vectors must produce dot product = -len ({})", len);
    }
}
