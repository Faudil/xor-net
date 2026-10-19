//! Validates the fast 1-bit Qwen3 (`xor_net::models::q1::qwen3`) against the
//! slow candle oracle in `support/bonsai_reference.rs` (f32 dequantized
//! weights, f32 activations).
//!
//! Needs real weights, so it only runs when `BONSAI_GGUF` points at a dense
//! Bonsai GGUF (it is skipped otherwise). The oracle is slow; use release:
//!
//! ```text
//! BONSAI_GGUF=models/bonsai/Bonsai-1.7B-Q1_0.gguf \
//!     cargo test --release --test bonsai_reference -- --nocapture
//! ```
//!
//! The fast path quantizes activations to int8 (per 32 elements), so logits
//! are not bit-exact; the test reports the error and checks top-1 agreement
//! and that the greedy continuations agree.

#[path = "support/bonsai_reference.rs"]
#[allow(clippy::all)]
mod bonsai_reference;

use std::sync::Arc;

use bonsai_reference::{BonsaiCache, BonsaiModel};
use xor_net::loader::gguf::{GgufFile, build_tokenizer};
use xor_net::models::q1::qwen3::Qwen3Model;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn rank_of(v: &[f32], id: u32) -> usize {
    let x = v[id as usize];
    v.iter().filter(|&&y| y > x).count()
}

struct Stats {
    max_abs: f32,
    rel: f32,
    /// KL(reference || fast) of the next-token distributions, in nats.
    kl: f64,
    top1: bool,
    rank: usize,
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
    v.iter().map(|&x| x as f64 - lse).collect()
}

fn compare(reference: &[f32], fast: &[f32]) -> Stats {
    assert_eq!(reference.len(), fast.len());
    let max_abs = reference
        .iter()
        .zip(fast)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let r = argmax(reference);
    let (lp, lq) = (log_softmax(reference), log_softmax(fast));
    let kl = lp
        .iter()
        .zip(&lq)
        .map(|(a, b)| a.exp() * (a - b))
        .sum::<f64>();
    Stats {
        max_abs,
        rel: max_abs / scale,
        kl,
        top1: r == argmax(fast),
        rank: rank_of(fast, r),
    }
}

#[test]
fn fast_qwen3_matches_reference() -> anyhow::Result<()> {
    let Ok(path) = std::env::var("BONSAI_GGUF") else {
        eprintln!("BONSAI_GGUF not set: skipping the oracle comparison");
        return Ok(());
    };
    let steps: usize = std::env::var("BONSAI_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let gguf = Arc::new(GgufFile::open(&path)?);
    let tok = build_tokenizer(&gguf)?;
    let prompt = "The capital of France is";
    let ids: Vec<u32> = tok
        .encode(prompt, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let device = candle_core::Device::Cpu;
    let max_len = ids.len() + steps + 1;

    let reference = BonsaiModel::from_gguf(gguf.clone(), max_len, &device)?;
    let fast = Qwen3Model::from_gguf(&gguf)?;
    let mut ref_cache = BonsaiCache::new(&reference.config, &device)?;
    let mut st = fast.new_qwen3_state();

    // Teacher-forced on the reference's greedy tokens: logits error per step.
    let mut ref_tokens = Vec::new();
    let mut stats = Vec::new();
    let mut r: Vec<f32> = reference
        .forward(&ids, 0, &mut ref_cache)?
        .flatten_all()?
        .to_vec1()?;
    let mut f = fast.forward_tokens(&ids, 0, &mut st)?;
    let mut pos = ids.len();
    for step in 0..=steps {
        let s = compare(&r, &f);
        eprintln!(
            "step {step:2}: max|Δlogit| {:.4} (rel {:.4}) KL {:.2e} top1 {} ref-top1 rank in fast {}",
            s.max_abs, s.rel, s.kl, s.top1, s.rank
        );
        stats.push(s);
        if step == steps {
            break;
        }
        let next = argmax(&r);
        ref_tokens.push(next);
        r = reference
            .forward(&[next], pos, &mut ref_cache)?
            .flatten_all()?
            .to_vec1()?;
        f = fast.forward_tokens(&[next], pos, &mut st)?;
        pos += 1;
    }

    // Free-running greedy decode of the fast model from the same prompt.
    let mut st = fast.new_qwen3_state();
    let mut f = fast.forward_tokens(&ids, 0, &mut st)?;
    let mut fast_tokens = Vec::new();
    for i in 0..steps {
        let next = argmax(&f);
        fast_tokens.push(next);
        f = fast.forward_tokens(&[next], ids.len() + i, &mut st)?;
    }
    let agree = ref_tokens
        .iter()
        .zip(&fast_tokens)
        .take_while(|(a, b)| a == b)
        .count();
    let dec = |t: &[u32]| tok.decode(t, false).unwrap_or_default();
    eprintln!("reference: {prompt}{}", dec(&ref_tokens));
    eprintln!("fast     : {prompt}{}", dec(&fast_tokens));

    let n = stats.len() as f32;
    let top1 = stats.iter().filter(|s| s.top1).count() as f32 / n;
    let worst_rel = stats.iter().map(|s| s.rel).fold(0.0f32, f32::max);
    let mean_abs = stats.iter().map(|s| s.max_abs).sum::<f32>() / n;
    let mean_kl = stats.iter().map(|s| s.kl).sum::<f64>() / n as f64;
    eprintln!(
        "summary: top-1 agreement {:.1}% over {} steps, mean max|Δlogit| {mean_abs:.4}, worst rel {worst_rel:.4}, mean KL {mean_kl:.2e} nats, greedy prefix agreement {agree}/{steps} tokens",
        top1 * 100.0,
        stats.len()
    );
    assert!(top1 >= 0.9, "top-1 agreement {top1} < 90%");
    assert!(mean_kl < 0.01, "mean KL {mean_kl} too large");
    assert!(
        stats.iter().all(|s| s.rank <= 2),
        "a reference top-1 token fell below rank 2 in the fast model"
    );
    assert!(
        agree >= steps.min(20),
        "greedy continuations diverge after {agree} tokens"
    );
    Ok(())
}
