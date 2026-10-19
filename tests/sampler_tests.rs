use xor_net::sampler::{Sampler, SamplingParams};

fn sampler(p: SamplingParams) -> Sampler {
    Sampler::from_params(p.with_seed(Some(1234)))
}

/// Empirical token frequencies over `n` draws.
fn histogram(s: &mut Sampler, logits: &[f32], ctx: &[u32], n: usize) -> Vec<usize> {
    let mut h = vec![0usize; logits.len()];
    let mut buf = logits.to_vec();
    for _ in 0..n {
        buf.copy_from_slice(logits);
        h[s.sample(&mut buf, ctx).unwrap() as usize] += 1;
    }
    h
}

fn softmax(l: &[f32], t: f32) -> Vec<f64> {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = l.iter().map(|&x| (((x - m) / t) as f64).exp()).collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

// --- greedy -----------------------------------------------------------------

#[test]
fn greedy_argmax() {
    let mut s = Sampler::new(42, Some(0.0), None, 1.0);
    let mut logits = vec![-5.0, 1.0, 3.0, -2.0, 0.5];
    assert_eq!(s.sample(&mut logits, &[]).unwrap(), 2);
}

#[test]
fn greedy_negative_only() {
    let mut s = Sampler::new(42, None, None, 1.0);
    let mut logits = vec![-10.0, -3.0, -8.0, -1.0];
    assert_eq!(s.sample(&mut logits, &[]).unwrap(), 3);
}

#[test]
fn greedy_first_of_ties_and_large_vocab() {
    let mut s = sampler(SamplingParams::greedy());
    let mut logits = vec![0.0f32; 100_003];
    logits[70_001] = 5.0;
    logits[90_000] = 5.0;
    assert_eq!(s.sample(&mut logits, &[]).unwrap(), 70_001);
}

#[test]
fn single_logit() {
    let mut s = Sampler::new(42, Some(1.0), None, 1.0);
    assert_eq!(s.sample(&mut [0.0], &[]).unwrap(), 0);
    assert!(s.sample(&mut [], &[]).is_err());
}

// --- seeding ----------------------------------------------------------------

#[test]
fn seed_zero_compat_uses_entropy() {
    let a = Sampler::new(0, None, None, 1.0);
    let b = Sampler::new(0, None, None, 1.0);
    assert_ne!(a.seed(), b.seed());
}

#[test]
fn seed_preserved() {
    assert_eq!(Sampler::new(12345, None, None, 1.0).seed(), 12345);
    let s = Sampler::from_params(SamplingParams::default().with_seed(Some(0)));
    assert_eq!(s.seed(), 0, "explicit Some(0) is a valid seed");
    let e = Sampler::from_params(SamplingParams::default());
    let replay = Sampler::from_params(SamplingParams::default().with_seed(Some(e.seed())));
    assert_eq!(replay.seed(), e.seed());
}

#[test]
fn deterministic_with_same_seed() {
    let logits: Vec<f32> = (0..5000)
        .map(|i| ((i * 7919) % 1000) as f32 / 200.0)
        .collect();
    for p in [
        SamplingParams::default(),
        SamplingParams::default()
            .with_temperature(0.8)
            .with_top_p(0.9),
        SamplingParams::default().with_top_k(50),
        SamplingParams::default().with_min_p(0.1),
    ] {
        let mut a = Sampler::from_params(p.clone().with_seed(Some(999)));
        let mut b = Sampler::from_params(p.clone().with_seed(Some(999)));
        let mut c = Sampler::from_params(p.clone().with_seed(Some(1000)));
        let sa: Vec<u32> = (0..200)
            .map(|_| a.sample(&mut logits.clone(), &[]).unwrap())
            .collect();
        let sb: Vec<u32> = (0..200)
            .map(|_| b.sample(&mut logits.clone(), &[]).unwrap())
            .collect();
        let sc: Vec<u32> = (0..200)
            .map(|_| c.sample(&mut logits.clone(), &[]).unwrap())
            .collect();
        assert_eq!(sa, sb, "{p:?}");
        assert_ne!(sa, sc, "{p:?}");
    }
}

// --- distribution -------------------------------------------------------------

#[test]
fn temperature_distribution_matches_softmax() {
    let logits = vec![0.0f32, 1.0, 2.0, 0.5, -1.0];
    for t in [0.5f32, 1.0, 2.0] {
        let mut s = sampler(SamplingParams::default().with_temperature(t));
        let n = 200_000;
        let h = histogram(&mut s, &logits, &[], n);
        let p = softmax(&logits, t);
        for i in 0..logits.len() {
            let emp = h[i] as f64 / n as f64;
            let sd = (p[i] * (1.0 - p[i]) / n as f64).sqrt();
            assert!(
                (emp - p[i]).abs() < 5.0 * sd + 1e-4,
                "t={t} token {i}: emp {emp} vs {}",
                p[i]
            );
        }
    }
}

#[test]
fn hot_temperature_spreads() {
    let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let mut cold = sampler(SamplingParams::default().with_temperature(0.01));
    assert_eq!(cold.sample(&mut logits.clone(), &[]).unwrap(), 9);
    let mut hot = sampler(SamplingParams::default().with_temperature(10.0));
    let h = histogram(&mut hot, &logits, &[], 2000);
    assert!(h.iter().filter(|&&c| c > 0).count() >= 8);
}

// --- truncation -------------------------------------------------------------

#[test]
fn top_k_truncates() {
    let logits: Vec<f32> = (0..1000).map(|i| (i % 97) as f32 / 10.0).collect();
    // top 3 values: 9.6 at tokens with i % 97 == 96 -> 96, 193, 290, ... (10 of them).
    let mut s = sampler(
        SamplingParams::default()
            .with_temperature(5.0)
            .with_top_k(3),
    );
    let h = histogram(&mut s, &logits, &[], 3000);
    let used: Vec<usize> = (0..1000).filter(|&i| h[i] > 0).collect();
    assert!(used.len() <= 3, "{used:?}");
    assert!(used.iter().all(|&i| i % 97 == 96));

    // Large k goes through the select_nth path.
    let logits: Vec<f32> = (0..5000).map(|i| i as f32 / 1000.0).collect();
    let mut s = sampler(
        SamplingParams::default()
            .with_temperature(100.0)
            .with_top_k(400),
    );
    let h = histogram(&mut s, &logits, &[], 20_000);
    assert!((0..4600).all(|i| h[i] == 0));
    assert!((4600..5000).filter(|&i| h[i] > 0).count() > 350);
}

#[test]
fn top_k_distribution_is_renormalised() {
    let logits = vec![2.0f32, 1.0, 0.0, -1.0, 3.0];
    let mut s = sampler(SamplingParams::default().with_top_k(2));
    let n = 100_000;
    let h = histogram(&mut s, &logits, &[], n);
    let p = softmax(&[2.0, 3.0], 1.0);
    assert_eq!(h[1] + h[2] + h[3], 0);
    assert!((h[0] as f64 / n as f64 - p[0]).abs() < 0.01);
    assert!((h[4] as f64 / n as f64 - p[1]).abs() < 0.01);
}

/// Regression test for the old bug: after finding the nucleus the sampler
/// drew from the whole top-1024 tail instead of the nucleus only.
#[test]
fn top_p_samples_only_inside_nucleus() {
    // probs ~ [0.5, 0.3, 0.2 spread over 2000 tail tokens]
    let mut logits = vec![0.0f32; 2002];
    let tail_each = 0.2f32 / 2000.0;
    logits[0] = (0.5f32 / tail_each).ln();
    logits[1] = (0.3f32 / tail_each).ln();
    // nucleus for top_p = 0.75 is {0, 1} (0.5 < 0.75 <= 0.8)
    let mut s = sampler(SamplingParams::default().with_top_p(0.75));
    let h = histogram(&mut s, &logits, &[], 20_000);
    let outside: usize = h[2..].iter().sum();
    assert_eq!(outside, 0, "sampled outside the nucleus");
    let r = h[0] as f64 / (h[0] + h[1]) as f64;
    assert!((r - 0.625).abs() < 0.02, "renormalised ratio {r}");

    // top_p = 0.45 -> only the first token.
    let mut s = sampler(SamplingParams::default().with_top_p(0.45));
    let h = histogram(&mut s, &logits, &[], 2000);
    assert_eq!(h[0], 2000);
}

#[test]
fn top_p_on_flat_large_vocab() {
    // Uniform: the whole tie class is kept. (Kept small: tests run in debug.)
    let n = 20_000;
    let logits = vec![0.0f32; n];
    let mut s = sampler(SamplingParams::default().with_top_p(0.25));
    let h = histogram(&mut s, &logits, &[], 500);
    assert!(h.iter().sum::<usize>() == 500);

    // Linear logits: nucleus is a contiguous top range.
    let logits: Vec<f32> = (0..n).map(|i| i as f32 * 5e-4).collect();
    let probs = softmax(&logits, 1.0);
    let mut acc = 0.0;
    let mut cut = n;
    for i in (0..n).rev() {
        acc += probs[i];
        if acc >= 0.3 {
            cut = i;
            break;
        }
    }
    let mut s = sampler(SamplingParams::default().with_top_p(0.3));
    let h = histogram(&mut s, &logits, &[], 500);
    let below: usize = h[..cut.saturating_sub(2)].iter().sum();
    assert_eq!(below, 0, "sampled below the nucleus cut {cut}");
}

#[test]
fn top_p_zero_is_argmax() {
    let logits = vec![0.1f32, 0.3, 0.2];
    let mut s = sampler(SamplingParams::default().with_top_p(0.0));
    let h = histogram(&mut s, &logits, &[], 500);
    assert_eq!(h[1], 500);
}

#[test]
fn top_k_then_top_p() {
    // After top_k=3 the probs are softmax([3,2,1]) = [0.665, 0.245, 0.090];
    // top_p=0.7 keeps {0,1}.
    let logits = vec![3.0f32, 2.0, 1.0, 0.0, -1.0];
    let mut s = sampler(SamplingParams::default().with_top_k(3).with_top_p(0.7));
    let h = histogram(&mut s, &logits, &[], 5000);
    assert_eq!(h[2] + h[3] + h[4], 0);
    assert!(h[1] > 0);
}

#[test]
fn min_p_truncates() {
    // p relative to max: exp(0)=1, exp(-1)=0.37, exp(-3)=0.05, exp(-5)=0.0067
    let logits = vec![5.0f32, 4.0, 2.0, 0.0];
    for (min_p, allowed) in [(0.5f32, 1usize), (0.3, 2), (0.04, 3), (0.001, 4)] {
        let mut s = sampler(SamplingParams::default().with_min_p(min_p));
        let h = histogram(&mut s, &logits, &[], 20_000);
        let used = h.iter().filter(|&&c| c > 0).count();
        assert_eq!(used, allowed, "min_p={min_p}: {h:?}");
        // and through the top-k code path
        let mut s = sampler(SamplingParams::default().with_min_p(min_p).with_top_k(3));
        let h = histogram(&mut s, &logits, &[], 20_000);
        assert_eq!(h.iter().filter(|&&c| c > 0).count(), allowed.min(3));
    }
}

// --- penalties ----------------------------------------------------------------

#[test]
fn repetition_penalty_hf_semantics() {
    let mut s = sampler(SamplingParams::greedy().with_repetition_penalty(2.0));
    // token 0 positive -> /2 = 1.5 ; token 1 negative -> *2 = -2 ; 2 untouched = 1.6
    let mut logits = vec![3.0f32, -1.0, 1.6];
    assert_eq!(s.sample(&mut logits, &[0, 1, 0, 0]).unwrap(), 2);
    assert_eq!(
        logits,
        vec![1.5, -2.0, 1.6],
        "applied once per distinct token"
    );
}

#[test]
fn frequency_and_presence_penalties_openai_semantics() {
    let mut s = sampler(
        SamplingParams::greedy()
            .with_frequency_penalty(0.5)
            .with_presence_penalty(0.25),
    );
    let mut logits = vec![3.0f32, 2.0, 1.0];
    s.sample(&mut logits, &[0, 0, 0, 1]).unwrap();
    assert_eq!(logits, vec![3.0 - 1.5 - 0.25, 2.0 - 0.5 - 0.25, 1.0]);
}

#[test]
fn sample_split_penalises_output_only_for_frequency() {
    let mut s = sampler(
        SamplingParams::greedy()
            .with_frequency_penalty(1.0)
            .with_repetition_penalty(2.0),
    );
    let mut logits = vec![4.0f32, 4.0, 4.0];
    s.sample_split(&mut logits, &[0], &[1]).unwrap();
    // token 0: prompt only -> rep only = 2 ; token 1: rep + freq = 2 - 1 = 1
    assert_eq!(logits, vec![2.0, 1.0, 4.0]);
}

#[test]
fn penalty_last_n_window() {
    let mut s = sampler(
        SamplingParams::greedy()
            .with_presence_penalty(10.0)
            .with_penalty_last_n(Some(2)),
    );
    let mut logits = vec![5.0f32, 4.0, 3.0];
    // window = [1, 2]: token 0 is outside it
    s.sample(&mut logits, &[0, 0, 1, 2]).unwrap();
    assert_eq!(logits, vec![5.0, -6.0, -7.0]);
    // repeated calls reuse buffers correctly (no stale counts)
    let mut logits = vec![5.0f32, 4.0, 3.0];
    s.sample(&mut logits, &[2, 2]).unwrap();
    assert_eq!(logits, vec![5.0, 4.0, -7.0]);
    // out-of-vocab ids are ignored
    let mut logits = vec![5.0f32, 4.0, 3.0];
    s.sample(&mut logits, &[99, 100]).unwrap();
    assert_eq!(logits, vec![5.0, 4.0, 3.0]);
}

// --- robustness -------------------------------------------------------------

#[test]
fn nan_and_inf_logits_do_not_panic() {
    let cases: Vec<Vec<f32>> = vec![
        vec![f32::NAN, 1.0, 2.0],
        vec![f32::NAN, f32::NAN],
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
        vec![f32::INFINITY, 1.0, f32::NAN],
        vec![f32::INFINITY, f32::INFINITY],
        vec![f32::MAX, -f32::MAX, 0.0],
    ];
    let params = [
        SamplingParams::greedy(),
        SamplingParams::default(),
        SamplingParams::default()
            .with_temperature(0.3)
            .with_top_p(0.9),
        SamplingParams::default().with_top_k(2).with_min_p(0.1),
        SamplingParams::default().with_min_p(0.5),
        SamplingParams::default()
            .with_repetition_penalty(1.3)
            .with_frequency_penalty(0.7),
    ];
    for p in &params {
        for c in &cases {
            let mut s = sampler(p.clone());
            let r = s.sample(&mut c.clone(), &[0, 1]);
            if let Ok(t) = r {
                assert!((t as usize) < c.len());
                assert!(!c[t as usize].is_nan(), "{p:?} {c:?} -> {t}");
            }
        }
    }
    // +inf dominates
    let mut s = sampler(SamplingParams::default().with_top_p(0.9));
    assert_eq!(s.sample(&mut [1.0, f32::INFINITY, 2.0], &[]).unwrap(), 1);
    // all-NaN is an error, not a panic or a silent token
    let mut s = sampler(SamplingParams::default());
    assert!(s.sample(&mut [f32::NAN, f32::NAN], &[]).is_err());
}

/// Rough timing on a 128k vocabulary. Run with
/// `cargo test --release --test sampler_tests bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_sampling_128k() {
    let n = 128_256;
    let mut x = 0x1234_5678u64;
    let base: Vec<f32> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0
        })
        .collect();
    let ctx: Vec<u32> = (0..64).map(|i| i * 977).collect();
    for (name, p) in [
        ("greedy", SamplingParams::greedy()),
        (
            "temp0.8 top_p0.9",
            SamplingParams::default()
                .with_temperature(0.8)
                .with_top_p(0.9),
        ),
        ("top_k40", SamplingParams::default().with_top_k(40)),
        ("min_p0.05", SamplingParams::default().with_min_p(0.05)),
    ] {
        let mut s = sampler(p);
        let mut buf = base.clone();
        let it = 200;
        let t = std::time::Instant::now();
        for _ in 0..it {
            buf.copy_from_slice(&base);
            std::hint::black_box(s.sample(&mut buf, &ctx).unwrap());
        }
        println!(
            "{name}: {:.1} us/token",
            t.elapsed().as_secs_f64() * 1e6 / it as f64
        );
    }
}
