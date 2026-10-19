//! Generate text with a Bonsai 1-bit model (GGUF `qwen3` / `qwen35`, or a
//! converted `.xq1` file) through the `Generator` API.
//!
//! ```text
//! cargo run --release --example bonsai_generate -- <model.gguf|model.xq1> \
//!     [--tokenizer tokenizer.json] [--prompt TEXT] [--chat] [-n N] \
//!     [--temp T] [--threads N] [--bench]
//! ```
//!
//! Without `--tokenizer` the tokenizer embedded in the model file is used.
//! `--chat` wraps the prompt in the model's chat template. `--bench` uses
//! greedy decoding with `ignore_eos` and prints prefill/decode tok/s and the
//! peak RSS.
use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Instant;

use xor_net::{ChatMessage, GenerationParams, SamplingParams};

fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut chat = false;
    let mut n = 64usize;
    let mut temp = 0.0f32;
    let mut threads = 0usize;
    let mut bench = false;
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        match a.as_str() {
            "--tokenizer" => tokenizer = Some(val()?.into()),
            "--prompt" | "-p" => prompt = val()?,
            "--chat" => chat = true,
            "-n" => n = val()?.parse()?,
            "--temp" => temp = val()?.parse()?,
            "--threads" | "-t" => threads = val()?.parse()?,
            "--bench" => bench = true,
            _ if model.is_none() && !a.starts_with('-') => model = Some(a.into()),
            _ => anyhow::bail!("unknown argument {a}"),
        }
    }
    let model = model.ok_or_else(|| {
        anyhow::anyhow!("usage: bonsai_generate <model.gguf|model.xq1> [--tokenizer t.json] [--prompt P] [--chat] [-n N] [--temp T] [--threads N] [--bench]")
    })?;
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let mut generator = xor_net::models::q1::load_generator(&model, tokenizer.as_deref())?;
    eprintln!(
        "loaded {} in {:.2}s ({} threads, rss {:.0} MB)",
        model.display(),
        t0.elapsed().as_secs_f64(),
        rayon::current_num_threads(),
        peak_rss_mb().unwrap_or(0.0)
    );

    let sampling = if temp > 0.0 {
        SamplingParams::default().with_temperature(temp)
    } else {
        SamplingParams::greedy()
    };
    let params = GenerationParams::default()
        .with_max_tokens(n)
        .with_sampling(sampling)
        .with_ignore_eos(bench)
        .with_add_special_tokens(false);

    let mut stdout = std::io::stdout();
    let on_text = |t: &str| {
        print!("{t}");
        let _ = stdout.flush();
        ControlFlow::Continue(())
    };
    let out = if chat {
        generator.chat_stream(&[ChatMessage::user(prompt.clone())], &params, on_text)?
    } else {
        print!("{prompt}");
        generator.generate_stream(&prompt, &params, on_text)?
    };
    println!();
    eprintln!(
        "prompt {} tok, completion {} tok, finish {:?} | prefill {:.1} tok/s, decode {:.1} tok/s | peak rss {:.0} MB",
        out.usage.prompt_tokens,
        out.usage.completion_tokens,
        out.finish_reason,
        out.timings.prefill_tok_per_s(),
        out.timings.decode_tok_per_s(),
        peak_rss_mb().unwrap_or(0.0)
    );
    Ok(())
}
