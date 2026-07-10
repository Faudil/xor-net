//! Non-regression tests for RMSNorm: edge cases, numerical stability,
//! and correctness against hand-computed references.

use xor_net::tensor::FastTensor;

const EPS: f32 = 1e-5;

#[test]
fn rmsnorm_uniform_input() {
    // All elements equal → RMS = |x|, so normalized = sign(x) * weight
    let x = FastTensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![1, 4]);
    let w = FastTensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let out = x.rmsnorm(&w, 1e-6).unwrap();
    // RMS = sqrt(mean(4)) = 2, inv_rms = 0.5, output = 2 * 0.5 * 1 = 1
    for &v in &out.data {
        assert!((v - 1.0).abs() < EPS, "Uniform input rmsnorm: got {}", v);
    }
}

#[test]
fn rmsnorm_unit_weight() {
    // weight = 1 → rmsnorm is purely normalization
    let x = FastTensor::new(vec![3.0, 4.0], vec![1, 2]);
    let w = FastTensor::new(vec![1.0, 1.0], vec![2]);
    let out = x.rmsnorm(&w, 1e-6).unwrap();
    // RMS = sqrt((9+16)/2) = sqrt(12.5), inv = 1/sqrt(12.5)
    let rms = (12.5f32).sqrt();
    let expected = [3.0 / rms, 4.0 / rms];
    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "rmsnorm unit weight idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn rmsnorm_zero_input() {
    // All-zero input → sum_sq = 0, inv_std = 1/sqrt(eps), output ≈ 0
    let x = FastTensor::new(vec![0.0; 4], vec![1, 4]);
    let w = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let out = x.rmsnorm(&w, 1e-6).unwrap();
    for &v in &out.data {
        assert!(v.abs() < EPS, "Zero input rmsnorm should be ~0, got {}", v);
    }
}

#[test]
fn rmsnorm_batched() {
    // Two rows should be normalized independently
    let x = FastTensor::new(vec![1.0, 0.0, 3.0, 0.0, 0.0, 4.0], vec![2, 3]);
    let w = FastTensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let out = x.rmsnorm(&w, 1e-6).unwrap();
    // Row 0: rms = sqrt((1+0+9)/3) = sqrt(10/3)
    let rms0 = ((1.0 + 0.0 + 9.0) / 3.0f32).sqrt();
    assert!((out.data[0] - 1.0 / rms0).abs() < EPS);
    assert!((out.data[1] - 0.0).abs() < EPS);
    assert!((out.data[2] - 3.0 / rms0).abs() < EPS);
    // Row 1: rms = sqrt((0+0+16)/3) = sqrt(16/3)
    let rms1 = ((0.0 + 0.0 + 16.0) / 3.0f32).sqrt();
    assert!((out.data[3] - 0.0).abs() < EPS);
    assert!((out.data[4] - 0.0).abs() < EPS);
    assert!((out.data[5] - 4.0 / rms1).abs() < EPS);
}

#[test]
fn rmsnorm_large_values_no_nan() {
    // Large values should not produce NaN or Inf
    let x = FastTensor::new(vec![1e6, -1e6, 1e6, -1e6], vec![1, 4]);
    let w = FastTensor::new(vec![1.0; 4], vec![4]);
    let out = x.rmsnorm(&w, 1e-6).unwrap();
    for &v in &out.data {
        assert!(v.is_finite(), "rmsnorm must not produce NaN/Inf for large values");
    }
}

#[test]
fn rmsnorm_weight_scaling() {
    // weight = 2 → output should be 2× the unit-weight case
    let x = FastTensor::new(vec![3.0, 4.0], vec![1, 2]);
    let w1 = FastTensor::new(vec![1.0, 1.0], vec![2]);
    let w2 = FastTensor::new(vec![2.0, 2.0], vec![2]);
    let out1 = x.rmsnorm(&w1, 1e-6).unwrap();
    let out2 = x.rmsnorm(&w2, 1e-6).unwrap();
    for (i, (&a, &b)) in out2.data.iter().zip(out1.data.iter()).enumerate() {
        assert!((a - 2.0 * b).abs() < EPS, "2× weight idx {}: {} vs 2*{}", i, a, b);
    }
}
