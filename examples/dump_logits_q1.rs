//! Dump xor-net logits for cross-checking against another implementation
//! (e.g. llama.cpp) on identical token ids.
//!
//! ```text
//! dump_logits_q1 tokenize <model.gguf|.xq1> <text.txt> <out_tokens.u32>
//! dump_logits_q1 logits   <model.gguf|.xq1> <tokens.u32> <out_logits.f32> [--prefill]
//! ```
//!
//! `logits` writes row-major `[n_tokens, vocab]` f32: by default the tokens
//! are fed one per call (decode path); with `--prefill` each position's logits
//! come from one prefill call over the whole prefix (prefill path).
use std::io::Write;

use xor_net::AutoModelForCausalLM;

fn read_u32s(p: &str) -> anyhow::Result<Vec<u32>> {
    let b = std::fs::read(p)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        a.len() >= 4,
        "usage: dump_logits_q1 tokenize|logits <model> <in> <out> [--prefill]"
    );
    xor_net::init_threads(6).map_err(|e| anyhow::anyhow!("{e}"))?;
    match a[0].as_str() {
        "tokenize" => {
            let g = AutoModelForCausalLM::generator_from_file(&a[1], None)?;
            let text = std::fs::read_to_string(&a[2])?;
            let enc = g
                .tokenizer()
                .encode(text.as_str(), false)
                .map_err(anyhow::Error::msg)?;
            let mut f = std::fs::File::create(&a[3])?;
            for t in enc.get_ids() {
                f.write_all(&t.to_le_bytes())?;
            }
            eprintln!("tokens: {}", enc.get_ids().len());
        }
        "logits" => {
            let model = AutoModelForCausalLM::from_file(&a[1])?;
            let toks = read_u32s(&a[2])?;
            let prefill = a.iter().any(|s| s == "--prefill");
            let mut f = std::io::BufWriter::new(std::fs::File::create(&a[3])?);
            let mut state = model.new_state()?;
            for i in 0..toks.len() {
                let logits = if prefill {
                    state.reset()?;
                    model.forward(&toks[..=i], 0, &mut *state)?
                } else {
                    model.forward(&toks[i..=i], i, &mut *state)?
                };
                for v in logits.iter() {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            eprintln!("wrote {} x {} logits", toks.len(), model.vocab_size());
        }
        m => anyhow::bail!("unknown mode {m}"),
    }
    Ok(())
}
