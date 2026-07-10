//! Non-regression tests for RoPE (rotary positional embeddings)
//! applied on FastTensor.

use xor_net::tensor::FastTensor;
use std::f32::consts::PI;

const EPS: f32 = 1e-5;

#[test]
fn rope_identity_at_zero_angle() {
    // cos=1, sin=0 → rotation is identity
    let x = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]);
    let cos = FastTensor::new(vec![1.0, 1.0], vec![1, 2]);
    let sin = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let out = x.rope_inplace(&cos, &sin, 0).unwrap();
    for (i, (&a, &e)) in out.data.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
        assert!((a - e).abs() < EPS, "Identity rope idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn rope_90_degree_rotation() {
    // cos=0, sin=1 → 90° rotation
    // x = [r0, r1, i0, i1], half_dim=2
    // out_real[d] = r*0 - i*1 = -i
    // out_imag[d] = r*1 + i*0 = r
    let x = FastTensor::new(vec![3.0, 5.0, 7.0, 11.0], vec![1, 1, 1, 4]);
    let cos = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let sin = FastTensor::new(vec![1.0, 1.0], vec![1, 2]);
    let out = x.rope_inplace(&cos, &sin, 0).unwrap();
    let expected = [-7.0, -11.0, 3.0, 5.0];
    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "90° rope idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn rope_180_degree_rotation() {
    // cos=-1, sin=0 → negation
    let x = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]);
    let cos = FastTensor::new(vec![-1.0, -1.0], vec![1, 2]);
    let sin = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let out = x.rope_inplace(&cos, &sin, 0).unwrap();
    let expected = [-1.0, -2.0, -3.0, -4.0];
    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "180° rope idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn rope_preserves_norm() {
    // Rotation should preserve the L2 norm of each (real, imag) pair
    let x_data = vec![3.0, 4.0, 1.0, 2.0];
    let angle = PI / 6.0; // 30 degrees
    let cos_t = FastTensor::new(vec![angle.cos(), angle.cos()], vec![1, 2]);
    let sin_t = FastTensor::new(vec![angle.sin(), angle.sin()], vec![1, 2]);
    let x = FastTensor::new(x_data.clone(), vec![1, 1, 1, 4]);
    let out = x.rope_inplace(&cos_t, &sin_t, 0).unwrap();

    let orig_norm_0 = (x_data[0] * x_data[0] + x_data[2] * x_data[2]).sqrt();
    let out_norm_0 = (out.data[0] * out.data[0] + out.data[2] * out.data[2]).sqrt();
    assert!((orig_norm_0 - out_norm_0).abs() < EPS, "RoPE must preserve norm pair 0");

    let orig_norm_1 = (x_data[1] * x_data[1] + x_data[3] * x_data[3]).sqrt();
    let out_norm_1 = (out.data[1] * out.data[1] + out.data[3] * out.data[3]).sqrt();
    assert!((orig_norm_1 - out_norm_1).abs() < EPS, "RoPE must preserve norm pair 1");
}

#[test]
fn rope_multi_head() {
    // 2 heads, 1 token, head_dim=4
    let x = FastTensor::new(
        vec![1.0, 0.0, 0.0, 1.0,   // head 0
             0.0, 1.0, 1.0, 0.0],  // head 1
        vec![1, 2, 1, 4],
    );
    let cos = FastTensor::new(vec![0.0, 0.0], vec![1, 2]);
    let sin = FastTensor::new(vec![1.0, 1.0], vec![1, 2]);
    let out = x.rope_inplace(&cos, &sin, 0).unwrap();
    // head 0: real=[1,0], imag=[0,1] → out_real=[1*0-0*1, 0*0-1*1]=[ 0,-1]
    //                                   out_imag=[1*1+0*0, 0*1+1*0]=[ 1, 0]
    // head 1: real=[0,1], imag=[1,0] → out_real=[0*0-1*1, 1*0-0*1]=[-1, 0]
    //                                   out_imag=[0*1+1*0, 1*1+0*0]=[ 0, 1]
    let expected = [0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0];
    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "multi-head rope idx {}: {} vs {}", i, a, e);
    }
}
