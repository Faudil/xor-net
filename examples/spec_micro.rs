//! Micro-timings of the pieces of a speculative round on a Bonsai model:
//! plain 1-token decode, an n-token forward, an n-token provisional verify
//! (+ taps, all logits) and its commit, and (optionally) a DSpark round.
//!
//! ```text
//! cargo run --release --example spec_micro -- <model.gguf> [--dspark d.gguf] [--head own|shared] [-n 5] [--reps 10]
//! ```
use std::sync::Arc;
use std::time::Instant;

use xor_net::models::q1::dspark::{DsparkHead, DsparkModel};
use xor_net::models::traits::{CausalLM, ForwardOpts, HiddenTaps};
use xor_net::speculative::{DraftModel, DraftSampler};

struct Greedy;
impl DraftSampler for Greedy {
    fn pick(&mut self, l: &mut [f32], _d: &[u32]) -> anyhow::Result<(u32, Option<Vec<f32>>)> {
        let mut best = 0;
        for (i, &v) in l.iter().enumerate() {
            if v > l[best] {
                best = i;
            }
        }
        Ok((best as u32, None))
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut dspark = None;
    let mut head = DsparkHead::Own;
    let mut n = 5usize;
    let mut reps = 10usize;
    let mut threads = 6usize;
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        match a.as_str() {
            "--dspark" => dspark = Some(val()?),
            "--head" => head = val()?.parse()?,
            "-n" => n = val()?.parse()?,
            "--reps" => reps = val()?.parse()?,
            "--threads" => threads = val()?.parse()?,
            "--q41-bench" => {
                xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;
                return q41_bench(reps);
            }
            _ if model.is_none() => model = Some(a),
            _ => anyhow::bail!("unknown argument {a}"),
        }
    }
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;
    let m: Arc<dyn CausalLM> = Arc::from(xor_net::models::q1::load_causal_lm(
        model.as_deref().unwrap(),
    )?);
    let prompt: Vec<u32> = (0..200u32).map(|i| 1000 + (i * 7919) % 5000).collect();
    let mut st = m.new_state()?;
    m.forward(&prompt, 0, st.as_mut())?;
    let pos = prompt.len();
    let toks: Vec<u32> = (0..n as u32).map(|i| 2000 + i).collect();
    let time = |label: &str, f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<()> {
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            f()?;
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("{label:<40} {best:8.2} ms");
        Ok(())
    };
    // Each closure leaves the state at `pos` (commit / checkpoint).
    st.checkpoint();
    let base = pos;
    time("decode 1 token", &mut || {
        m.forward(&toks[..1], base, st.as_mut())?;
        anyhow::ensure!(st.truncate(base));
        Ok(())
    })?;
    time(&format!("forward {n} tokens (plain)"), &mut || {
        m.forward(&toks, base, st.as_mut())?;
        anyhow::ensure!(st.truncate(base));
        Ok(())
    })?;
    let layers = vec![1usize, 16, 31, 46, 61];
    let mut taps = HiddenTaps {
        layers: layers.clone(),
        data: Vec::new(),
    };
    time(
        &format!("verify {n} (provisional, all logits)"),
        &mut || {
            let mut o = ForwardOpts {
                all_logits: true,
                provisional: true,
                taps: None,
            };
            m.forward_with(&toks, base, st.as_mut(), &mut o)?;
            st.commit(base)?;
            Ok(())
        },
    )?;
    time(&format!("verify {n} + taps + commit(all)"), &mut || {
        let mut o = ForwardOpts {
            all_logits: true,
            provisional: true,
            taps: Some(&mut taps),
        };
        m.forward_with(&toks, base, st.as_mut(), &mut o)?;
        st.commit(base + n)?;
        anyhow::ensure!(st.truncate(base));
        Ok(())
    })?;
    let mut commit_ms = 0.0;
    for _ in 0..reps {
        let mut o = ForwardOpts {
            all_logits: true,
            provisional: true,
            taps: None,
        };
        m.forward_with(&toks, base, st.as_mut(), &mut o)?;
        let t = Instant::now();
        st.commit(base + n / 2)?;
        commit_ms += t.elapsed().as_secs_f64() * 1e3;
        anyhow::ensure!(st.truncate(base));
    }
    println!(
        "{:<40} {:8.2} ms",
        format!("commit({}) alone (mean)", n / 2),
        commit_ms / reps as f64
    );

    if let Some(d) = dspark {
        let dm = DsparkModel::load_for(&d, &*m, head)?;
        let mut dr = dm.new_drafter()?;
        // Context: taps of the prompt.
        let mut taps = HiddenTaps {
            layers,
            data: Vec::new(),
        };
        let mut st2 = m.new_state()?;
        m.forward_with(
            &prompt,
            0,
            st2.as_mut(),
            &mut ForwardOpts {
                taps: Some(&mut taps),
                ..Default::default()
            },
        )?;
        let t = Instant::now();
        dr.observe(&prompt, 0, Some(&taps.data))?;
        println!(
            "{:<40} {:8.2} ms",
            "dspark observe 200 prompt rows",
            t.elapsed().as_secs_f64() * 1e3
        );
        let w = taps.data.len() / prompt.len();
        let three = taps.data[..3 * w].to_vec();
        let mut ctx = prompt.clone();
        ctx.push(2000);
        time("dspark observe 3 rows", &mut || {
            dr.observe(&prompt[..3], pos, Some(&three))?;
            dr.rewind(&prompt);
            Ok(())
        })?;
        time("dspark draft (block of 4)", &mut || {
            dr.draft(&ctx, 4, &mut Greedy)?;
            Ok(())
        })?;
    }
    let _ = pos;
    Ok(())
}

/// Q4_1 GEMM throughput on random weights.
fn q41_bench(reps: usize) -> anyhow::Result<()> {
    use xor_net::models::q1::dspark::q4_1::{self, Isa, Q8Blocks, Q41Weight};
    let mut seed = 0x9E37_79B9u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for &(rows, cols) in &[
        (5120usize, 5120usize),
        (10240, 5120),
        (248320, 5120),
        (248320, 256),
    ] {
        let nb = cols / 32;
        let mut data = vec![0u8; rows * nb * 20];
        for (i, b) in data.iter_mut().enumerate() {
            *b = if i % 20 < 4 {
                [0x00, 0x20, 0x00, 0xa0][i % 20 % 4]
            } else {
                next() as u8
            };
        }
        let w = Q41Weight::from_gguf(&data, rows, cols)?;
        drop(data);
        for n in [1usize, 4, 5] {
            let x: Vec<f32> = (0..n * cols)
                .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
                .collect();
            let mut a = Q8Blocks::new();
            a.quantize(&x, n, cols);
            let mut out = vec![0.0; n * rows];
            for isa in Isa::all_supported()
                .into_iter()
                .filter(|i| *i != Isa::Scalar)
            {
                let mut best = f64::MAX;
                for _ in 0..reps {
                    let t = Instant::now();
                    q4_1::gemm_with(isa, &w, &a, &mut out);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                println!(
                    "q4_1 {rows}x{cols} n={n} {isa:?}: {:.2} ms, {:.1} GB/s",
                    best * 1e3,
                    w.nbytes() as f64 / best / 1e9
                );
            }
            if n == 1 {
                let mut best = f64::MAX;
                for _ in 0..reps {
                    let t = Instant::now();
                    q4_1::gemv_f32(&w, &x, &mut out);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                println!("q4_1 {rows}x{cols} gemv_f32: {:.2} ms", best * 1e3);
            }
        }
    }
    Ok(())
}
