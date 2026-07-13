//! Non-regression tests for attention numerics: softmax row sums,
//! causal mask correctness, and cache write behavior.

use xor_net::nn::{fast_attention, CpuRingCache};
use xor_net::tensor::FastTensor;

const EPS: f32 = 1e-4;

#[test]
fn attention_single_token_is_identity() {
    // With a single KV pair (seq_len=1, pos=0), softmax = 1.0
    // So output = V exactly
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;
    let mut cache = CpuRingCache::new(1, num_kv_heads, 32, head_dim, false);

    let q = FastTensor::new(vec![1.0; num_heads * head_dim], vec![1, num_heads, 1, head_dim]);
    let k = FastTensor::new(vec![1.0; num_kv_heads * head_dim], vec![1, num_kv_heads, 1, head_dim]);
    let v_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| i as f32).collect();
    let v = FastTensor::new(v_data.clone(), vec![1, num_kv_heads, 1, head_dim]);

    let out = fast_attention(&q, &k, &v, 0, 0, &mut cache, num_heads, num_kv_heads, head_dim).unwrap();
    // With GQA ratio 1:1, head h uses kv head h
    // Output should equal V for each head
    assert_eq!(out.shape, vec![1, num_heads, 1, head_dim]);
    for (i, (&a, &e)) in out.data.iter().zip(v_data.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "Single-token attention idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn attention_causal_mask_second_token_no_future() {
    // First token at pos=0, second at pos=1.
    // The second token should attend to both pos 0 and pos 1, but not beyond.
    let num_heads = 1;
    let num_kv_heads = 1;
    let head_dim = 2;
    let mut cache = CpuRingCache::new(1, num_kv_heads, 32, head_dim, false);

    // Step 0: insert K=[1,0], V=[10,0]
    let q0 = FastTensor::new(vec![1.0, 0.0], vec![1, 1, 1, 2]);
    let k0 = FastTensor::new(vec![1.0, 0.0], vec![1, 1, 1, 2]);
    let v0 = FastTensor::new(vec![10.0, 0.0], vec![1, 1, 1, 2]);
    let _ = fast_attention(&q0, &k0, &v0, 0, 0, &mut cache, num_heads, num_kv_heads, head_dim).unwrap();

    // Step 1: insert K=[0,1], V=[0,20], query=[1,1]
    let q1 = FastTensor::new(vec![1.0, 1.0], vec![1, 1, 1, 2]);
    let k1 = FastTensor::new(vec![0.0, 1.0], vec![1, 1, 1, 2]);
    let v1 = FastTensor::new(vec![0.0, 20.0], vec![1, 1, 1, 2]);
    let out = fast_attention(&q1, &k1, &v1, 0, 1, &mut cache, num_heads, num_kv_heads, head_dim).unwrap();

    // Q=[1,1] · K0=[1,0] = 1, Q=[1,1] · K1=[0,1] = 1 (equal scores)
    // After softmax with equal scores → [0.5, 0.5]
    // Output = 0.5 * [10,0] + 0.5 * [0,20] = [5, 10]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let s0 = 1.0 * scale;
    let s1 = 1.0 * scale;
    let max_s = s0.max(s1);
    let e0 = (s0 - max_s).exp();
    let e1 = (s1 - max_s).exp();
    let sum = e0 + e1;
    let w0 = e0 / sum;
    let w1 = e1 / sum;
    let expected = [w0 * 10.0 + w1 * 0.0, w0 * 0.0 + w1 * 20.0];

    for (i, (&a, &e)) in out.data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < EPS, "Causal attn idx {}: {} vs {}", i, a, e);
    }
}

#[test]
fn attention_output_finite() {
    // Stress test: random-ish values should not produce NaN/Inf
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let seq_len = 6;
    let mut cache = CpuRingCache::new(1, num_kv_heads, 64, head_dim, false);

    let q_data: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let k_data: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();
    let v_data: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
        .map(|i| i as f32 * 0.03 - 1.0)
        .collect();

    let q = FastTensor::new(q_data, vec![1, num_heads, seq_len, head_dim]);
    let k = FastTensor::new(k_data, vec![1, num_kv_heads, seq_len, head_dim]);
    let v = FastTensor::new(v_data, vec![1, num_kv_heads, seq_len, head_dim]);

    let out = fast_attention(&q, &k, &v, 0, 0, &mut cache, num_heads, num_kv_heads, head_dim).unwrap();
    for (i, &v) in out.data.iter().enumerate() {
        assert!(v.is_finite(), "Attention output NaN/Inf at idx {}", i);
    }
}

#[test]
fn cache_write_correctness() {
    let num_kv_heads = 1;
    let head_dim = 2;
    let mut cache = CpuRingCache::new(1, num_kv_heads, 8, head_dim, false);

    // Write token at pos 0 with K=[5,6]
    let q = FastTensor::new(vec![1.0, 1.0], vec![1, 1, 1, 2]);
    let k = FastTensor::new(vec![5.0, 6.0], vec![1, 1, 1, 2]);
    let v = FastTensor::new(vec![0.0, 0.0], vec![1, 1, 1, 2]);
    let _ = fast_attention(&q, &k, &v, 0, 0, &mut cache, 1, 1, 2).unwrap();

    // Verify cache slot 0 has [5, 6]
    assert_eq!(cache.k_buffers[0][0], 5.0);
    assert_eq!(cache.k_buffers[0][1], 6.0);
}
