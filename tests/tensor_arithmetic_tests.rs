//! Non-regression tests for FastTensor arithmetic: add, add_inplace,
//! silu_mul, silu_mul_inplace, gelu.

use xor_net::tensor::FastTensor;

const EPS: f32 = 1e-6;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", msg);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{}: idx {}  got {}, expected {}", msg, i, a, e);
    }
}

fn ref_silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn ref_gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x3)).tanh())
}

#[test]
fn add_known_values() {
    let a = FastTensor::new(vec![1.0, -2.0, 3.0, 0.0], vec![2, 2]);
    let b = FastTensor::new(vec![4.0, 5.0, -6.0, 7.0], vec![2, 2]);
    let c = a.add(&b).unwrap();
    assert_close(&c.data, &[5.0, 3.0, -3.0, 7.0], EPS, "add");
}

#[test]
fn add_inplace_matches_add() {
    let a = FastTensor::new(vec![0.5, -1.5, 2.5, -3.5, 4.5, -5.5], vec![2, 3]);
    let b = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out_add = a.add(&b).unwrap();
    let out_ip = a.clone().add_inplace(&b).unwrap();
    assert_close(&out_ip.data, &out_add.data, EPS, "add_inplace parity");
}

#[test]
fn add_zero_is_identity() {
    let a = FastTensor::new(vec![3.14, -2.71, 1.41, 0.0], vec![4]);
    let z = FastTensor::zeros(vec![4]);
    let c = a.add(&z).unwrap();
    assert_close(&c.data, &a.data, EPS, "add zero identity");
}

#[test]
fn silu_mul_known_values() {
    let a_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let b_data = vec![1.0; 6];
    let a = FastTensor::new(a_data.clone(), vec![6]);
    let b = FastTensor::new(b_data, vec![6]);
    let c = a.silu_mul(&b).unwrap();
    let expected: Vec<f32> = a_data.iter().map(|&x| ref_silu(x)).collect();
    assert_close(&c.data, &expected, EPS, "silu_mul b=1");
}

#[test]
fn silu_mul_zero_input() {
    let a = FastTensor::new(vec![0.0; 3], vec![3]);
    let b = FastTensor::new(vec![100.0, -50.0, 0.0], vec![3]);
    let c = a.silu_mul(&b).unwrap();
    assert_close(&c.data, &[0.0, 0.0, 0.0], EPS, "silu(0)*b=0");
}

#[test]
fn silu_mul_inplace_matches() {
    let a = FastTensor::new(vec![0.5, -1.5, 2.5, -3.5], vec![4]);
    let b = FastTensor::new(vec![2.0, 3.0, -1.0, 0.5], vec![4]);
    let out_fn = a.silu_mul(&b).unwrap();
    let out_ip = a.clone().silu_mul_inplace(&b).unwrap();
    assert_close(&out_ip.data, &out_fn.data, EPS, "silu_mul_inplace parity");
}

#[test]
fn gelu_known_values() {
    let x = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let t = FastTensor::new(x.clone(), vec![5]);
    let g = t.gelu().unwrap();
    let expected: Vec<f32> = x.iter().map(|&v| ref_gelu(v)).collect();
    assert_close(&g.data, &expected, EPS, "gelu ref");
}

#[test]
fn gelu_zero_is_zero() {
    let t = FastTensor::new(vec![0.0], vec![1]);
    assert!((t.gelu().unwrap().data[0]).abs() < EPS);
}

#[test]
fn relu2_mul_known_values() {
    let a = FastTensor::new(vec![-2.0, 0.0, 1.0, 3.0], vec![4]);
    let b = FastTensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let c = a.relu2_mul_inplace(&b).unwrap();
    assert_close(&c.data, &[0.0, 0.0, 2.0, 18.0], EPS, "relu2_mul");
}

#[test]
fn relu2_mul_negative_input() {
    let a = FastTensor::new(vec![-5.0, -0.5, -100.0], vec![3]);
    let b = FastTensor::new(vec![1.0, 1.0, 1.0], vec![3]);
    let c = a.relu2_mul_inplace(&b).unwrap();
    assert_close(&c.data, &[0.0, 0.0, 0.0], EPS, "relu2_mul negative");
}

#[test]
fn gelu_positive_stays_positive() {
    let x: Vec<f32> = (1..=10).map(|i| i as f32 * 0.5).collect();
    let t = FastTensor::new(x, vec![10]);
    for (i, &v) in t.gelu().unwrap().data.iter().enumerate() {
        assert!(v > 0.0, "GELU(x>0) must be >0, idx {}", i);
    }
}

// The dispatched ternary GEMV (VNNI / AVX-512 / AVX2 / scalar) must match the
// reference decode exactly.
#[test]
fn ternary_dot_product_pack4_matches_scalar() {
    use xor_net::bit1_58::quantization::pack_1_58bit_4pack;
    use xor_net::bit1_58::simd::ternary_dot_product_pack4;

    let in_dim = 256;
    let mut rng = 12345u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as f32
    };

    let weights: Vec<f32> = (0..in_dim)
        .map(|_| {
            let r = next();
            if r < 0.33 { -1.0 } else if r > 0.66 { 1.0 } else { 0.0 }
        })
        .collect();
    let acts_f32: Vec<f32> = (0..in_dim).map(|_| (next() - 0.5) * 40.0).collect();
    let acts_i8: Vec<i8> = acts_f32.iter().map(|&x| x.round().max(-127.0).min(127.0) as i8).collect();

    let packed = pack_1_58bit_4pack(&weights, 1.0);
    let got = ternary_dot_product_pack4(&acts_i8, &packed, in_dim);

    let expected: i32 = acts_i8
        .iter()
        .zip(weights.iter())
        .map(|(&a, &w)| a as i32 * w as i32)
        .sum();

    assert_eq!(got, expected, "ternary GEMV mismatch");
}

// Vectorized SiLU (AVX-512) must match the scalar libm reference, including
// values that exercise the exp() range clamp and the tail handling.
#[test]
fn silu_inplace_vectorized_matches_scalar() {
    use xor_net::bit1_58::quantization::silu_inplace;

    let gate: Vec<f32> = (-40..40)
        .map(|i| i as f32 * 0.7)
        .chain(std::iter::once(200.0))
        .chain(std::iter::once(-200.0))
        .collect();
    let up: Vec<f32> = (0..gate.len()).map(|i| (i as f32 * 0.13 - 5.0)).collect();

    // use_silu = true
    let mut g1 = gate.clone();
    silu_inplace(&mut g1, &up, true);
    let expected_silu: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
        .collect();
    assert_close(&g1, &expected_silu, 1e-2, "silu_inplace vectorized (SiLU)");

    // use_silu = false (ReLU²)
    let mut g2 = gate.clone();
    silu_inplace(&mut g2, &up, false);
    let expected_relu2: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let r = if g > 0.0 { g } else { 0.0 };
            r * r * u
        })
        .collect();
    assert_close(&g2, &expected_relu2, 1e-2, "silu_inplace vectorized (ReLU²)");
}
