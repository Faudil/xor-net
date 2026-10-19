//! Numerics regression check for a Bonsai `.xq1` model: prefill a prompt in
//! one call, then greedy-decode `G` tokens one at a time. Writes the prefill
//! logits and every generated token id to `--out`, and with `--compare`
//! reports the max |Δlogit|, the KL divergence of the prefill distribution
//! and whether the greedy tokens are identical to a previous run's dump.
//!
//! ```text
//! cargo run --release --example bonsai_logits_check -- <model.xq1> --prompt-file P \
//!     [--gen 64] [--out run.bin] [--compare base.bin] [--one-by-one]
//! ```
//!
//! `--one-by-one` feeds the prompt one token per call (the decode path)
//! instead of one prefill call.
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut model, mut prompt_file, mut out, mut cmp) = (None, None, None, None);
    let mut gen_n = 64usize;
    let mut one_by_one = false;
    while let Some(a) = args.next() {
        let mut v = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        match a.as_str() {
            "--prompt-file" => prompt_file = Some(PathBuf::from(v()?)),
            "--gen" => gen_n = v()?.parse()?,
            "--out" => out = Some(PathBuf::from(v()?)),
            "--compare" => cmp = Some(PathBuf::from(v()?)),
            "--one-by-one" => one_by_one = true,
            _ => model = Some(PathBuf::from(a)),
        }
    }
    let model = model.ok_or_else(|| anyhow::anyhow!("usage: bonsai_logits_check <model.xq1> --prompt-file P [--gen G] [--out F] [--compare F]"))?;
    let text =
        std::fs::read_to_string(prompt_file.ok_or_else(|| anyhow::anyhow!("--prompt-file"))?)?;
    let (lm, assets) = xor_net::models::q1::format::load_with_assets(&model)?;
    let prompt: Vec<u32> = assets
        .tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .to_vec();
    let mut st = lm.new_state()?;
    let t = std::time::Instant::now();
    let logits = if one_by_one {
        let mut l = Vec::new();
        for (i, &t) in prompt.iter().enumerate() {
            l = lm.forward(&[t], i, st.as_mut())?;
        }
        l
    } else {
        lm.forward(&prompt, 0, st.as_mut())?
    };
    let pf = t.elapsed().as_secs_f64();
    let argmax = |l: &[f32]| {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    let mut toks = Vec::with_capacity(gen_n);
    // Top-1 minus top-2 logit at every greedy step (how close to a tie).
    let mut margins = Vec::with_capacity(gen_n);
    let mut cur = logits.clone();
    for i in 0..gen_n {
        let next = argmax(&cur);
        let top1 = cur[next as usize];
        let top2 = cur
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != next as usize)
            .map(|(_, &v)| v)
            .fold(f32::NEG_INFINITY, f32::max);
        margins.push(top1 - top2);
        toks.push(next);
        cur = lm.forward(&[next], prompt.len() + i, st.as_mut())?;
    }
    println!(
        "prompt {} tok, prefill {:.1} tok/s, generated: {:?}",
        prompt.len(),
        prompt.len() as f64 / pf,
        assets.tokenizer.decode(&toks, false).unwrap_or_default()
    );
    let mut bytes = Vec::new();
    bytes.extend((logits.len() as u32).to_le_bytes());
    bytes.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
    bytes.extend((toks.len() as u32).to_le_bytes());
    bytes.extend(toks.iter().flat_map(|v| v.to_le_bytes()));
    if let Some(o) = out {
        std::fs::write(o, &bytes)?;
    }
    if let Some(c) = cmp {
        let b = std::fs::read(c)?;
        let rd = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let nl = rd(0) as usize;
        let base: Vec<f32> = (0..nl).map(|i| f32::from_bits(rd(4 + 4 * i))).collect();
        let nt = rd(4 + 4 * nl) as usize;
        let btoks: Vec<u32> = (0..nt).map(|i| rd(8 + 4 * nl + 4 * i)).collect();
        anyhow::ensure!(nl == logits.len(), "vocab mismatch");
        let maxd = logits
            .iter()
            .zip(&base)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let lse = |l: &[f32]| {
            let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
        };
        let (za, zb) = (lse(&logits), lse(&base));
        // KL(base || new)
        let kl: f64 = base
            .iter()
            .zip(&logits)
            .map(|(&b, &a)| {
                let lb = b as f64 - zb;
                lb.exp() * (lb - (a as f64 - za))
            })
            .sum();
        let same = btoks == toks;
        let first_diff = btoks.iter().zip(&toks).position(|(a, b)| a != b);
        println!(
            "compare: prefill max|Δlogit| {maxd:.3e}, KL {kl:.3e}, greedy tokens identical: {same} (first diff {first_diff:?}{}), prefill argmax same: {}",
            first_diff.map_or(String::new(), |i| format!(
                ", top-2 margin there {:.3}",
                margins[i]
            )),
            argmax(&base) == argmax(&logits)
        );
    }
    Ok(())
}
