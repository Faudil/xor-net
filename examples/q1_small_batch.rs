//! Q1_0 GEMM cost at small token counts (continuous-batching decode).
//!
//! ```text
//! cargo run --release --example q1_small_batch -- [--rows R] [--cols C] [--mats M] [--max-n N] [--reps K]
//! ```
//!
//! Streams `M` random `[R, C]` 1-bit matrices (default: 8 x Bonsai-27B
//! `gate_up`, ~190 MB, far beyond L3) through every supported kernel for
//! `n = 1 ..= N` tokens and reports the time per sweep, the effective weight
//! bandwidth and the cost per token relative to `n = 1` (GEMV). A
//! memory-bound kernel costs the same for every `n`; a compute-bound one
//! grows linearly.
use std::time::Instant;

use half::f16;
use xor_net::bit1::q1_0::{ByteBuf, GemmKernel, Q1Weight, Q8Acts, gemm, gemm_with};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut rows, mut cols, mut mats, mut max_n, mut reps) =
        (34816usize, 5120usize, 8usize, 16usize, 5usize);
    while let Some(a) = args.next() {
        let mut v = || -> anyhow::Result<usize> {
            Ok(args
                .next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))?
                .parse()?)
        };
        match a.as_str() {
            "--rows" => rows = v()?,
            "--cols" => cols = v()?,
            "--mats" => mats = v()?,
            "--max-n" => max_n = v()?,
            "--reps" => reps = v()?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let block = 128;
    let ws: Vec<Q1Weight> = (0..mats)
        .map(|_| {
            let bits: Vec<u8> = (0..rows * cols / 8).map(|_| rnd() as u8).collect();
            let scales: Vec<u8> = (0..rows * cols / block)
                .flat_map(|_| {
                    f16::from_f32(0.01 + (rnd() % 100) as f32 * 1e-4)
                        .to_le_bytes()
                        .into_iter()
                })
                .collect();
            Q1Weight::new(
                rows,
                cols,
                block,
                ByteBuf::from_vec(bits),
                ByteBuf::from_vec(scales),
            )
        })
        .collect::<anyhow::Result<_>>()?;
    let bytes: usize = ws.iter().map(Q1Weight::nbytes).sum();
    println!(
        "{mats} x [{rows} x {cols}] block {block}: {:.0} MB of weights, {} threads",
        bytes as f64 / 1e6,
        rayon::current_num_threads()
    );
    let kernels = GemmKernel::all_supported();
    let mut out = vec![0.0f32; max_n * rows];
    let mut acts = Q8Acts::new();
    let mut base = 0.0f64;
    print!("{:>3} {:>9} {:>7}", "n", "gemm ms", "x/tok");
    for k in &kernels {
        print!(" {:>22}", format!("{k:?}"));
    }
    println!();
    for n in 1..=max_n {
        let x: Vec<f32> = (0..n * cols)
            .map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        acts.quantize(&x, n, cols, 32);
        let mut time = |f: &mut dyn FnMut(&Q1Weight, &mut [f32])| {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = Instant::now();
                for w in &ws {
                    f(w, &mut out[..n * rows]);
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let t_default = time(&mut |w, o| gemm(w, &acts, o));
        if n == 1 {
            base = t_default;
        }
        print!(
            "{n:>3} {:>9.2} {:>7.2}",
            t_default * 1e3,
            t_default / (base * n as f64)
        );
        for &k in &kernels {
            let t = time(&mut |w, o| gemm_with(k, w, &acts, o));
            print!(" {:>9.2}ms {:>5.1}GB/s", t * 1e3, bytes as f64 / t / 1e9);
        }
        println!();
    }
    Ok(())
}
