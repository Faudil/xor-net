use xor_net::sampler::Sampler;

#[test]
fn test_greedy_argmax() {
    let mut s = Sampler::new(42, Some(0.0), None, 1.0);
    let mut logits = vec![-5.0, 1.0, 3.0, -2.0, 0.5];
    let token = s.sample(&mut logits, &[]).unwrap();
    assert_eq!(token, 2);
}

#[test]
fn test_greedy_negative_only() {
    let mut s = Sampler::new(42, Some(0.0), None, 1.0);
    let mut logits = vec![-10.0, -3.0, -8.0, -1.0];
    let token = s.sample(&mut logits, &[]).unwrap();
    assert_eq!(token, 3);
}

#[test]
fn test_seed_zero_uses_time() {
    let s = Sampler::new(0, None, None, 1.0);
    assert!(s.seed > 0);
}

#[test]
fn test_seed_preserved() {
    let s = Sampler::new(12345, None, None, 1.0);
    assert_eq!(s.seed, 12345);
}

#[test]
fn test_temperature_sample_produces_valid_token() {
    let mut s = Sampler::new(42, Some(1.0), None, 1.0);
    let mut logits: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0).abs()).collect();
    for _ in 0..50 {
        let token = s.sample(&mut logits, &[]).unwrap();
        assert!(token < 50);
    }
}

#[test]
fn test_deterministic_seed() {
    let mut logits: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
    let mut a = Sampler::new(999, Some(1.0), Some(0.9), 1.0);
    let mut b = Sampler::new(999, Some(1.0), Some(0.9), 1.0);
    let ta = a.sample(&mut logits.clone(), &[]).unwrap();
    let tb = b.sample(&mut logits, &[]).unwrap();
    assert_eq!(ta, tb);
}

#[test]
fn test_top_p_filters_low_probability() {
    let mut s = Sampler::new(42, Some(1.0), Some(0.5), 1.0);
    let mut logits = vec![100.0, -100.0, -100.0, -100.0, -100.0];
    let token = s.sample(&mut logits, &[]).unwrap();
    assert_eq!(token, 0);
}

#[test]
fn test_temperature_scaling_changes_distribution() {
    let logits_base: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let mut cold = Sampler::new(42, Some(0.1), None, 1.0);
    let mut cold_logits = logits_base.clone();
    assert_eq!(cold.sample(&mut cold_logits, &[]).unwrap(), 9);
    let hot_tokens: Vec<u32> = (0..20).map(|_| {
        let mut s = Sampler::new(42, Some(10.0), None, 1.0);
        s.sample(&mut logits_base.clone(), &[]).unwrap()
    }).collect();
    assert!(hot_tokens.iter().any(|&t| t != 9));
}

#[test]
fn test_sample_with_temperature_only() {
    let mut s = Sampler::new(123, Some(0.5), None, 1.0);
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    let token = s.sample(&mut logits, &[]).unwrap();
    assert!(token < 4);
}

#[test]
fn test_empty_logits_edge_case() {
    let mut s = Sampler::new(42, Some(1.0), None, 1.0);
    let mut logits = vec![0.0];
    assert_eq!(s.sample(&mut logits, &[]).unwrap(), 0);
}
