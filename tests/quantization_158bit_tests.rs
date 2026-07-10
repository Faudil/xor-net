//! Non-regression tests for 1.58-bit quantization (pack4, pack5, f32→i8).
//!
//! These tests verify that pack/unpack are exact inverses, that quantization
//! maps boundary values correctly, and that edge-case inputs (all-zero,
//! all-negative, non-aligned lengths) produce correct results.

use xor_net::bit1_58::quantization::{
    pack_1_58bit_4pack, unpack_1_58bit_4pack,
    pack_1_58bit_5pack, unpack_1_58bit_5pack,
    quantize_f32_to_i8,
};

// ── Pack4 roundtrip ─────────────────────────────────────────────────────

#[test]
fn pack4_roundtrip_exact_ternary_values() {
    // All three ternary values in every possible position within a 4-element group
    let weights: Vec<f32> = vec![
        -1.0, 0.0, 1.0, -1.0,
         1.0, 1.0, 0.0,  0.0,
        -1.0, -1.0,
    ];
    let packed = pack_1_58bit_4pack(&weights, 1.0);
    let unpacked = unpack_1_58bit_4pack(&packed, weights.len());
    assert_eq!(weights, unpacked, "Pack4 roundtrip must be lossless for exact ternary values");
}

#[test]
fn pack4_roundtrip_non_aligned_length() {
    // Length not a multiple of 4: ensure padding doesn't corrupt values
    for len in [1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 17] {
        let weights: Vec<f32> = (0..len).map(|i| match i % 3 { 0 => -1.0, 1 => 0.0, _ => 1.0 }).collect();
        let packed = pack_1_58bit_4pack(&weights, 1.0);
        let unpacked = unpack_1_58bit_4pack(&packed, weights.len());
        assert_eq!(weights, unpacked, "Pack4 roundtrip failed for len={}", len);
    }
}

#[test]
fn pack4_all_zeros() {
    let weights = vec![0.0f32; 20];
    let packed = pack_1_58bit_4pack(&weights, 1.0);
    let unpacked = unpack_1_58bit_4pack(&packed, weights.len());
    assert_eq!(weights, unpacked, "All-zero weights must roundtrip as all-zero");
}

#[test]
fn pack4_all_negative() {
    let weights = vec![-1.0f32; 12];
    let packed = pack_1_58bit_4pack(&weights, 1.0);
    let unpacked = unpack_1_58bit_4pack(&packed, weights.len());
    assert_eq!(weights, unpacked, "All-negative weights must roundtrip as all-negative");
}

// ── Pack5 roundtrip ─────────────────────────────────────────────────────

#[test]
fn pack5_roundtrip_exact_ternary_values() {
    let weights: Vec<f32> = vec![
        -1.0, 0.0, 1.0, -1.0, 1.0,
         0.0, 0.0, -1.0, 1.0, -1.0,
         1.0,
    ];
    let packed = pack_1_58bit_5pack(&weights, 1.0);
    let unpacked = unpack_1_58bit_5pack(&packed, weights.len());
    assert_eq!(weights, unpacked, "Pack5 roundtrip must be lossless for exact ternary values");
}

#[test]
fn pack5_roundtrip_non_aligned_length() {
    for len in [1, 2, 3, 4, 6, 7, 8, 9, 11, 14, 16, 21] {
        let weights: Vec<f32> = (0..len).map(|i| match i % 3 { 0 => -1.0, 1 => 0.0, _ => 1.0 }).collect();
        let packed = pack_1_58bit_5pack(&weights, 1.0);
        let unpacked = unpack_1_58bit_5pack(&packed, weights.len());
        assert_eq!(weights, unpacked, "Pack5 roundtrip failed for len={}", len);
    }
}

// ── quantize_f32_to_i8 ─────────────────────────────────────────────────

#[test]
fn quantize_i8_all_zeros_returns_scale_one() {
    let acts = vec![0.0f32; 16];
    let (quantized, inv_scale) = quantize_f32_to_i8(&acts);
    assert_eq!(inv_scale, 1.0, "inv_scale must be 1.0 when all activations are zero");
    assert!(quantized.iter().all(|&x| x == 0), "All-zero input must quantize to all-zero i8");
}

#[test]
fn quantize_i8_max_abs_maps_to_127() {
    let acts = vec![0.0, -3.0, 3.0, 1.5];
    let (quantized, inv_scale) = quantize_f32_to_i8(&acts);

    // max_abs = 3.0, so scale = 127/3, inv_scale = 3/127
    let expected_inv_scale = 3.0 / 127.0;
    assert!(
        (inv_scale - expected_inv_scale).abs() < 1e-6,
        "inv_scale mismatch: got {}, expected {}", inv_scale, expected_inv_scale
    );
    // The max absolute value should map to ±127
    assert_eq!(quantized[2], 127, "max positive value must map to 127");
    assert_eq!(quantized[1], -127, "max negative value must map to -127");
}

#[test]
fn quantize_i8_reconstruction_error_bounded() {
    // Verify that round-tripping through quantize then dequantize keeps error small.
    let acts: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
    let (quantized, inv_scale) = quantize_f32_to_i8(&acts);

    for (i, (&orig, &q)) in acts.iter().zip(quantized.iter()).enumerate() {
        let reconstructed = q as f32 * inv_scale;
        let abs_err = (orig - reconstructed).abs();
        // Error should be bounded by the quantization step size (inv_scale)
        assert!(
            abs_err <= inv_scale + 1e-6,
            "Reconstruction error too large at index {}: orig={}, recon={}, err={}, step={}",
            i, orig, reconstructed, abs_err, inv_scale
        );
    }
}
