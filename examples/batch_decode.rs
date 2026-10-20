//! Continuous-batching check and benchmark for Bonsai models.
//!
//! ```text
//! cargo run --release --example batch_decode -- <model> [--check N] [--bench] [--max-b B] [--steps S] [--ctx P] [--reps R] [--threads T]
//! ```
//!
//! `--check N`: four different prompts, greedy, `N` tokens each, decoded
//! alone (`CausalLM::forward`) and batched (`BatchedCausalLM::forward_batch`,
//! the fourth prompt joining while the others decode); asserts identical
//! tokens and bit-identical logits at every step.
//!
//! `--bench`: decode-step throughput of `forward_batch` for batch sizes
//! 1, 2, 4, 6, 8 (up to `--max-b`) after a `P`-token prefill per sequence:
//! step time, aggregate and per-stream tok/s (best of `R`).
use std::time::Instant;

use xor_net::models::batched::{BatchItem, BatchedCausalLM};
use xor_net::models::traits::ModelState;

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn hash(l: &[f32]) -> u64 {
    l.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn items<'a>(
    states: &'a mut [Box<dyn ModelState>],
    feeds: &'a [(usize, Vec<u32>, usize)],
) -> Vec<BatchItem<'a>> {
    // feeds: (sequence index, tokens, pos), ascending sequence index.
    let mut out = Vec::new();
    let mut rest = states;
    let mut base = 0;
    for (i, toks, pos) in feeds {
        let (_, tail) = rest.split_at_mut(i - base);
        let (one, tail) = tail.split_at_mut(1);
        rest = tail;
        base = i + 1;
        out.push(BatchItem {
            tokens: toks,
            pos: *pos,
            state: one[0].as_mut(),
        });
    }
    out
}

fn check(
    model: &dyn BatchedCausalLM,
    generator: &xor_net::Generator,
    n: usize,
) -> anyhow::Result<()> {
    let prompts = [
        "The capital of France is",
        "Write a haiku about the sea:\n",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "Explain in one paragraph why the sky is blue. Mention Rayleigh scattering and the wavelength dependence.",
    ];
    let toks: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| generator.encode(p, true))
        .collect::<anyhow::Result<_>>()?;
    // Alone.
    let mut solo_tokens = Vec::new();
    let mut solo_hashes = Vec::new();
    let t = Instant::now();
    for p in &toks {
        let mut st = model.new_state()?;
        let mut logits = model.forward(p, 0, st.as_mut())?;
        let (mut out, mut hs) = (Vec::new(), vec![hash(&logits)]);
        for _ in 0..n {
            let tok = argmax(&logits);
            out.push(tok);
            logits = model.forward(&[tok], p.len() + out.len() - 1, st.as_mut())?;
            hs.push(hash(&logits));
        }
        solo_tokens.push(out);
        solo_hashes.push(hs);
    }
    let solo_s = t.elapsed().as_secs_f64();
    // Batched: sequences 0..3 prefill together at step 0; sequence 3 joins
    // (prefills) at step 5 while the others decode.
    let join = |i: usize| if i == 3 { 5 } else { 0 };
    let mut states: Vec<Box<dyn ModelState>> = (0..toks.len())
        .map(|_| model.new_state())
        .collect::<anyhow::Result<_>>()?;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); toks.len()];
    let mut step_of = vec![0usize; toks.len()];
    let t = Instant::now();
    for step in 0..n + 6 {
        let mut feeds = Vec::new();
        for (i, p) in toks.iter().enumerate() {
            if step < join(i) || step_of[i] > n {
                continue;
            }
            if step_of[i] == 0 {
                feeds.push((i, p.clone(), 0));
            } else {
                let last = *out[i].last().unwrap();
                feeds.push((i, vec![last], p.len() + out[i].len() - 1));
            }
        }
        if feeds.is_empty() {
            break;
        }
        let idx: Vec<usize> = feeds.iter().map(|f| f.0).collect();
        let logits = model.forward_batch(&mut items(&mut states, &feeds))?;
        for (i, l) in idx.into_iter().zip(logits) {
            let k = step_of[i];
            anyhow::ensure!(
                hash(&l) == solo_hashes[i][k],
                "sequence {i} step {k}: batched logits differ from solo"
            );
            step_of[i] += 1;
            if out[i].len() < n {
                out[i].push(argmax(&l));
            }
        }
    }
    let batch_s = t.elapsed().as_secs_f64();
    for (i, (a, b)) in solo_tokens.iter().zip(&out).enumerate() {
        anyhow::ensure!(a == b, "sequence {i}: tokens differ");
        println!(
            "seq {i}: {} prompt tokens, {} generated, identical: {:?}",
            toks[i].len(),
            a.len(),
            generator
                .decode(a, true)?
                .chars()
                .take(60)
                .collect::<String>()
        );
    }
    println!(
        "check OK: 4 sequences x {n} tokens bit-identical (solo {solo_s:.2}s, batched {batch_s:.2}s)"
    );
    Ok(())
}

/// `DecodeSession`s stepped together (continuous batching, prefill chunks
/// of 16 mixed with decode steps) against `Generator` alone: same tokens,
/// text and finish reasons, for greedy and seeded sampling, stop strings
/// and `max_tokens`.
fn sessions(
    model: &dyn BatchedCausalLM,
    generator: &xor_net::Generator,
    n: usize,
) -> anyhow::Result<()> {
    use xor_net::batch_session::{DecodeSession, SessionShared, step};
    use xor_net::{ChatMessage, GenerationParams, SamplingParams};
    let chats = [
        "What is the capital of Italy?",
        "List three prime numbers.",
        "Write one sentence about cats.",
        "Count from 1 to 20 separated by commas.",
    ];
    let params: Vec<GenerationParams> = (0..chats.len())
        .map(|i| {
            let p = GenerationParams::default().with_max_tokens(n);
            match i {
                1 => p.with_sampling(
                    SamplingParams::default()
                        .with_temperature(0.8)
                        .with_seed(Some(7)),
                ),
                2 => p.with_sampling(SamplingParams::greedy()).with_stop(["."]),
                3 => p
                    .with_sampling(SamplingParams::greedy().with_repetition_penalty(1.1))
                    .with_max_tokens(n / 2),
                _ => p.with_sampling(SamplingParams::greedy()),
            }
        })
        .collect();
    let prompts: Vec<Vec<u32>> = chats
        .iter()
        .map(|c| {
            let text = generator.apply_chat_template(&[ChatMessage::user(*c)], true)?;
            generator.encode(&text, false)
        })
        .collect::<anyhow::Result<_>>()?;
    let mut want = Vec::new();
    for (p, prm) in prompts.iter().zip(&params) {
        let mut g = generator.fork();
        want.push(g.generate_from_tokens_with_boundaries(p, prm, &[], |_| {
            std::ops::ControlFlow::Continue(())
        })?);
    }
    let shared = SessionShared::from_generator(generator);
    let mut ss: Vec<DecodeSession> = prompts
        .iter()
        .zip(&params)
        .map(|(p, prm)| DecodeSession::start(&shared, p, prm, &[], None))
        .collect::<anyhow::Result<_>>()?;
    let mut texts = vec![String::new(); ss.len()];
    let fwd = |items: &mut [BatchItem<'_>]| model.forward_batch(items);
    loop {
        let mut refs: Vec<&mut DecodeSession> = ss.iter_mut().collect();
        if step(&fwd, &mut refs, 16, 24) == 0 {
            break;
        }
        for (s, t) in ss.iter_mut().zip(&mut texts) {
            for (_, piece) in s.take_events() {
                t.push_str(&piece);
            }
        }
    }
    for (i, (s, w)) in ss.iter().zip(&want).enumerate() {
        anyhow::ensure!(s.error().is_none(), "session {i}: {:?}", s.error());
        let o = s.outcome().expect("finished");
        anyhow::ensure!(
            o.tokens == w.tokens,
            "session {i}: tokens differ: {:?} vs {:?} ({:?} / {:?}) {:?} vs {:?}",
            o.tokens,
            w.tokens,
            o.finish,
            w.finish_reason,
            texts[i],
            w.text
        );
        anyhow::ensure!(texts[i] == w.text, "session {i}: text differs");
        anyhow::ensure!(o.finish == w.finish_reason, "session {i}: finish differs");
        anyhow::ensure!(
            o.usage.completion_tokens == w.usage.completion_tokens,
            "session {i}: usage"
        );
        println!(
            "session {i}: {:?}, {} tokens: {:?}",
            o.finish,
            o.tokens.len(),
            texts[i]
        );
    }
    println!("sessions OK: batched DecodeSessions match Generator");
    Ok(())
}

fn bench(
    model: &dyn BatchedCausalLM,
    max_b: usize,
    steps: usize,
    ctx: usize,
    reps: usize,
) -> anyhow::Result<()> {
    let vocab = model.vocab_size() as u32;
    println!(
        "{:>3} {:>10} {:>12} {:>14}",
        "B", "step ms", "aggregate", "per stream"
    );
    for b in [1usize, 2, 3, 4, 6, 8, 12, 16]
        .into_iter()
        .filter(|&b| b <= max_b)
    {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut states: Vec<Box<dyn ModelState>> = (0..b)
                .map(|_| model.new_state())
                .collect::<anyhow::Result<_>>()?;
            let mut last = Vec::new();
            for (i, st) in states.iter_mut().enumerate() {
                let p: Vec<u32> = (0..ctx as u32)
                    .map(|j| (j * 7919 + 13 + i as u32 * 101) % vocab.min(100_000))
                    .collect();
                last.push(argmax(&model.forward(&p, 0, st.as_mut())?));
            }
            let t = Instant::now();
            for s in 0..steps {
                let feeds: Vec<(usize, Vec<u32>, usize)> =
                    (0..b).map(|i| (i, vec![last[i]], ctx + s)).collect();
                let logits = model.forward_batch(&mut items(&mut states, &feeds))?;
                last = logits.iter().map(|l| argmax(l)).collect();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let step = best / steps as f64;
        println!(
            "{b:>3} {:>10.2} {:>8.1} tok/s {:>8.1} tok/s",
            step * 1e3,
            b as f64 / step,
            1.0 / step
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut sessions_n = 0usize;
    let (mut check_n, mut do_bench, mut max_b, mut steps, mut ctx, mut reps, mut threads) =
        (0usize, false, 8usize, 32usize, 128usize, 3usize, 0usize);
    while let Some(a) = args.next() {
        let mut v = || -> anyhow::Result<usize> {
            Ok(args
                .next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))?
                .parse()?)
        };
        match a.as_str() {
            "--check" => check_n = v()?,
            "--sessions" => sessions_n = v()?,
            "--bench" => do_bench = true,
            "--max-b" => max_b = v()?,
            "--steps" => steps = v()?,
            "--ctx" => ctx = v()?,
            "--reps" => reps = v()?,
            "--threads" | "-t" => threads = v()?,
            _ => path = Some(a),
        }
    }
    let path = path.ok_or_else(|| anyhow::anyhow!("usage: batch_decode <model> [options]"))?;
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (model, generator) = xor_net::models::q1::batch::load_batched(&path, None)?;
    println!("{path}: {} threads", rayon::current_num_threads());
    if check_n > 0 {
        check(model.as_ref(), &generator, check_n)?;
    }
    if sessions_n > 0 {
        sessions(model.as_ref(), &generator, sessions_n)?;
    }
    if do_bench {
        bench(model.as_ref(), max_b, steps, ctx, reps)?;
    }
    Ok(())
}
