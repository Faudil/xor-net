//! Non-regression tests for linear layers: F32Linear, TernaryLinear, BitLinear.
//! Each test verifies forward-pass correctness against a hand-computed
//! reference matmul or known mathematical properties.

use xor_net::bit1::layers::BitLinear;
use xor_net::bit1_58::layers::TernaryLinear;
use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::dynamic_linear::F32Linear;
use xor_net::tensor::FastTensor;

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
    for i in 0..4 {
        w_data[i * 4 + i] = 1.0;
    }
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
        .map(|i| match i % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    let input_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 - 10.0) * 0.1).collect();
    let input = FastTensor::new(input_data, vec![1, in_dim]);

    let l4 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack4, None).unwrap();
    let l5 = TernaryLinear::new(in_dim, out_dim, &weights, TernaryPackType::Pack5, None).unwrap();
    let o4 = l4.forward(&input).unwrap();
    let o5 = l5.forward(&input).unwrap();

    for (i, (&a, &b)) in o4.data.iter().zip(o5.data.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL,
            "Pack4 vs Pack5 mismatch idx {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn ternary_linear_zero_input() {
    let weights = vec![1.0, -1.0, 0.0, 1.0];
    let layer = TernaryLinear::new(2, 2, &weights, TernaryPackType::Pack4, None).unwrap();
    let input = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let out = layer.forward(&input).unwrap();
    for &v in &out.data {
        assert!(
            v.abs() < TOL,
            "Zero input should produce ~zero output, got {}",
            v
        );
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
    let row = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let weights: Vec<f32> = row.iter().cycle().take(row.len() * 3).cloned().collect();
    let layer = BitLinear::new(8, 3, &weights).unwrap();
    let input = FastTensor::new(vec![1.0; 8], vec![1, 8]);
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.data[0], out.data[1]);
    assert_eq!(out.data[1], out.data[2]);
}

// ───────────── Int4 / Int8 / forward_with_quantized regressions ─────────────

use xor_net::bit1_58::quantization::quantize_f32_to_i8;
use xor_net::nn::dynamic_linear::{Int4Linear, Int8Linear};

/// Exact integer reference for Int4Linear: decode the packed nibbles and
/// accumulate against the quantised activation.
fn int4_reference(lin: &Int4Linear, q: &[i8], inv_scale: f32) -> Vec<f32> {
    let bpr = lin.in_dim.div_ceil(2);
    (0..lin.out_dim)
        .map(|o| {
            let row = &lin.weight_i4[o * bpr..(o + 1) * bpr];
            let mut dot = 0i32;
            for i in 0..lin.in_dim {
                let b = row[i / 2];
                let nib = if i % 2 == 0 { b & 0x0F } else { b >> 4 };
                let w = ((nib << 4) as i8) >> 4; // sign-extend 4 bits
                dot += q[i] as i32 * w as i32;
            }
            dot as f32 * inv_scale * lin.scales[o]
        })
        .collect()
}

fn lcg_weights(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn int4_linear_odd_in_dim_includes_last_weight() {
    // Regression: `chunks_exact(2)` dropped the final weight when in_dim is odd.
    for &in_dim in &[1usize, 3, 5, 7, 129, 257] {
        let out_dim = 4;
        let w = lcg_weights(in_dim * out_dim, in_dim as u32);
        let lin = Int4Linear::new(&w, out_dim, in_dim);
        let x = lcg_weights(in_dim, 99);
        let mut q = vec![0i8; in_dim];
        let inv = quantize_f32_to_i8(&x, &mut q);
        let want = int4_reference(&lin, &q, inv);

        let xs = FastTensor::new(x.clone(), vec![1, in_dim]);
        let got = lin.forward(&xs).unwrap();
        assert_eq!(got.data, want, "forward in_dim={in_dim}");

        let got_q = lin.forward_with_quantized(&xs, &q, inv).unwrap();
        assert_eq!(got_q.data, want, "forward_with_quantized in_dim={in_dim}");

        // Batched path (b_size > 1) uses the same row kernel.
        let xs2 = FastTensor::new([x.clone(), x.clone()].concat(), vec![2, in_dim]);
        let got2 = lin.forward(&xs2).unwrap();
        assert_eq!(
            &got2.data[..out_dim],
            &want[..],
            "batched row 0 in_dim={in_dim}"
        );
        assert_eq!(
            &got2.data[out_dim..],
            &want[..],
            "batched row 1 in_dim={in_dim}"
        );
    }
}

#[test]
fn int4_last_weight_contributes() {
    // in_dim = 3, only the last weight is non-zero: the output must be non-zero.
    let w = vec![0.0, 0.0, 1.0];
    let lin = Int4Linear::new(&w, 1, 3);
    let xs = FastTensor::new(vec![0.0, 0.0, 1.0], vec![1, 3]);
    let out = lin.forward(&xs).unwrap();
    assert!(
        (out.data[0] - 1.0).abs() < 1e-2,
        "last weight ignored: {}",
        out.data[0]
    );
}

#[test]
fn int8_linear_forward_with_quantized_matches_forward() {
    for &in_dim in &[1usize, 63, 64, 65, 200, 1031] {
        let out_dim = 5;
        let w = lcg_weights(in_dim * out_dim, 7 + in_dim as u32);
        let lin = Int8Linear::new(&w, out_dim, in_dim);
        let x = lcg_weights(in_dim, 3);
        let mut q = vec![0i8; in_dim];
        let inv = quantize_f32_to_i8(&x, &mut q);
        let xs = FastTensor::new(x.clone(), vec![1, in_dim]);
        let a = lin.forward(&xs).unwrap();
        let b = lin.forward_with_quantized(&xs, &q, inv).unwrap();
        assert_eq!(a.data, b.data, "in_dim={in_dim}");
        assert_eq!(b.shape, vec![1, out_dim]);
        // Integer reference (the VNNI zero-point path must be exact).
        for o in 0..out_dim {
            let dot: i32 = (0..in_dim)
                .map(|i| q[i] as i32 * lin.weight_i8[o * in_dim + i] as i32)
                .sum();
            assert_eq!(b.data[o], dot as f32 * inv * lin.scales[o], "row {o}");
        }
    }
}

#[test]
fn forward_with_quantized_rejects_batched_input() {
    // Regression: the output buffer was sized for one row while the shape kept
    // the batch dims, which panicked (or mis-shaped) downstream.
    let (in_dim, out_dim) = (8, 4);
    let w = lcg_weights(in_dim * out_dim, 1);
    let xs = FastTensor::new(vec![0.5; 2 * in_dim], vec![2, in_dim]);
    let q = vec![1i8; 2 * in_dim];

    let i8l = Int8Linear::new(&w, out_dim, in_dim);
    assert!(i8l.forward_with_quantized(&xs, &q, 1.0).is_err());
    let i4l = Int4Linear::new(&w, out_dim, in_dim);
    assert!(i4l.forward_with_quantized(&xs, &q, 1.0).is_err());
    let tl = TernaryLinear::new(in_dim, out_dim, &w, TernaryPackType::Pack4, None).unwrap();
    assert!(tl.forward_with_quantized(&xs, &q, 1.0).is_err());

    // A single row with extra leading unit dims is fine.
    let xs1 = FastTensor::new(vec![0.5; in_dim], vec![1, 1, in_dim]);
    let out = tl.forward_with_quantized(&xs1, &q[..in_dim], 1.0).unwrap();
    assert_eq!(out.shape, vec![1, 1, out_dim]);
    assert_eq!(out.data.len(), out_dim);
}

#[test]
fn forward_with_quantized_rejects_short_quantized_input() {
    let (in_dim, out_dim) = (8, 4);
    let w = lcg_weights(in_dim * out_dim, 2);
    let xs = FastTensor::new(vec![0.5; in_dim], vec![1, in_dim]);
    let q = vec![1i8; in_dim - 1];
    assert!(
        Int8Linear::new(&w, out_dim, in_dim)
            .forward_with_quantized(&xs, &q, 1.0)
            .is_err()
    );
    assert!(
        Int4Linear::new(&w, out_dim, in_dim)
            .forward_with_quantized(&xs, &q, 1.0)
            .is_err()
    );
}
