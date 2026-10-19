//! Prefill / decode throughput of a Bonsai model (`.gguf` or `.xq1`).
//!
//! ```text
//! cargo run --release --example bonsai_bench -- <model> [--threads N] [--prompt-len P] [--gen G] [--reps R]
//! ```
//!
//! Prefill feeds `P` synthetic tokens in one call; decode then generates `G`
//! tokens one at a time (greedy argmax, no sampling overhead). Reports the
//! best of `R` repetitions, load time and peak RSS.
use std::time::Instant;

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let l = s.lines().find(|l| l.starts_with("VmHWM:"))?.to_string();
            l.split_whitespace().nth(1)?.parse::<f64>().ok()
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let (mut threads, mut plen, mut gen_n, mut reps) = (0usize, 128usize, 64usize, 2usize);
    while let Some(a) = args.next() {
        let mut v = || -> anyhow::Result<usize> {
            Ok(args
                .next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))?
                .parse()?)
        };
        match a.as_str() {
            "--threads" | "-t" => threads = v()?,
            "--prompt-len" | "-p" => plen = v()?,
            "--gen" | "-g" => gen_n = v()?,
            "--reps" | "-r" => reps = v()?,
            _ => path = Some(a),
        }
    }
    let path = path.ok_or_else(|| {
        anyhow::anyhow!(
            "usage: bonsai_bench <model> [--threads N] [--prompt-len P] [--gen G] [--reps R]"
        )
    })?;
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t0 = Instant::now();
    let model = xor_net::models::q1::load_causal_lm(&path)?;
    let load = t0.elapsed().as_secs_f64();
    let vocab = model.vocab_size() as u32;
    let prompt: Vec<u32> = (0..plen as u32)
        .map(|i| (i * 7919 + 13) % vocab.min(100_000))
        .collect();
    let (mut best_pf, mut best_dec) = (0.0f64, 0.0f64);
    for _ in 0..reps {
        let mut st = model.new_state()?;
        let t = Instant::now();
        let mut logits = model.forward(&prompt, 0, st.as_mut())?;
        best_pf = best_pf.max(plen as f64 / t.elapsed().as_secs_f64());
        let t = Instant::now();
        for i in 0..gen_n {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            logits = model.forward(&[next], plen + i, st.as_mut())?;
        }
        best_dec = best_dec.max(gen_n as f64 / t.elapsed().as_secs_f64());
    }
    println!(
        "{path}: threads {} | load {load:.2}s | prefill {plen} tok: {best_pf:.1} tok/s | decode {gen_n} tok: {best_dec:.1} tok/s | peak RSS {:.0} MB",
        rayon::current_num_threads(),
        peak_rss_mb()
    );
    Ok(())
}
