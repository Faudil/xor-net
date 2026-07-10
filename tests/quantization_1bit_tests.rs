//! Non-regression tests for 1-bit quantization (pack_1bit).
//!
//! These tests verify the bit-packing sign convention (>= 0 → 1, < 0 → 0),
//! correct handling of non-8-aligned lengths, and edge cases.

use xor_net::bit1::quantization::pack_1bit;

#[test]
fn pack_1bit_known_byte() {
    // Manual verification: [+, -, +, -, +, +, -, +]
    // Bits (LSB first): 1,0,1,0,1,1,0,1 = 0b10110101 = 181
    let weights = vec![1.5, -0.3, 0.0, -10.0, 5.0, 0.0, -1.0, 1.2];
    let packed = pack_1bit(&weights);
    assert_eq!(packed.len(), 1);
    assert_eq!(packed[0], 0b10110101, "Known bit pattern mismatch");
}

#[test]
fn pack_1bit_zero_is_positive() {
    // The convention is: >= 0 maps to bit 1 (+1)
    let weights = vec![0.0f32; 8];
    let packed = pack_1bit(&weights);
    assert_eq!(packed[0], 0xFF, "All zeros should pack as all-ones (0xFF)");
}

#[test]
fn pack_1bit_all_negative() {
    let weights = vec![-1.0f32; 8];
    let packed = pack_1bit(&weights);
    assert_eq!(packed[0], 0x00, "All negatives should pack as 0x00");
}

#[test]
fn pack_1bit_all_positive() {
    let weights = vec![1.0f32; 8];
    let packed = pack_1bit(&weights);
    assert_eq!(packed[0], 0xFF, "All positives should pack as 0xFF");
}

#[test]
fn pack_1bit_non_aligned_length() {
    // Length 3: only 3 bits should be set in the first byte, rest should be 0
    let weights = vec![1.0, -1.0, 1.0];
    let packed = pack_1bit(&weights);
    assert_eq!(packed.len(), 1);
    // Bits: 1, 0, 1, 0, 0, 0, 0, 0 = 0b00000101 = 5
    assert_eq!(packed[0], 0b00000101, "Non-aligned packing must zero-pad high bits");
}

#[test]
fn pack_1bit_multi_byte() {
    // 16 elements → 2 bytes
    let weights: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let packed = pack_1bit(&weights);
    assert_eq!(packed.len(), 2);
    // Alternating pattern: bits 1,0,1,0,1,0,1,0 = 0x55
    assert_eq!(packed[0], 0x55);
    assert_eq!(packed[1], 0x55);
}

#[test]
fn pack_1bit_large_values() {
    // Very large magnitudes should still follow the sign convention
    let weights = vec![1e30, -1e30, f32::MAX, f32::MIN, 1e-38, -1e-38, 0.0, f32::EPSILON];
    let packed = pack_1bit(&weights);
    // Expected bits (LSB first): 1,0,1,0,1,0,1,1 = 0b11010101 = 213
    assert_eq!(packed[0], 0b11010101);
}
