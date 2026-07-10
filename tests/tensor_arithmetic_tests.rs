//! Non-regression tests for FastTensor arithmetic: add, add_inplace,
//! silu_mul, silu_mul_inplace, gelu.

use xor_net::tensor::FastTensor;

const EPS: f32 = 1e-6;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", msg);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{}: idx {} — got {}, expected {}", msg, i, a, e);
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
fn gelu_positive_stays_positive() {
    let x: Vec<f32> = (1..=10).map(|i| i as f32 * 0.5).collect();
    let t = FastTensor::new(x, vec![10]);
    for (i, &v) in t.gelu().unwrap().data.iter().enumerate() {
        assert!(v > 0.0, "GELU(x>0) must be >0, idx {}", i);
    }
}
