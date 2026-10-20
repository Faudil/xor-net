//! Speculative decoding benchmark and exactness check.
//!
//! ```text
//! cargo run --release --example spec_bench -- <model.gguf|.xq1> \
//!     [--draft none,lookup,<dspark.gguf>] [--k 1,2,3,4] [--head own|shared] \
//!     [--adaptive] [--ngram MIN:MAX] [--max-draft N] [-n 128] [--threads 6] \
//!     [--prompts prose,code,summary,edit] [--temp T --seed S] [--reps R]
//! ```
//!
//! For every prompt a plain baseline runs first; every speculative config
//! is then checked to produce token-identical output (greedy) and reported
//! with decode tok/s, mean accepted length τ (tokens committed per verify
//! round) and draft acceptance rate. With `--reps R` each run repeats and
//! the best (min-time) run is reported.
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use xor_net::models::q1::dspark::{DsparkHead, DsparkModel};
use xor_net::speculative::{DraftModel, PromptLookup, SpecConfig};
use xor_net::{ChatMessage, GenerationParams, SamplingParams};

const PASSAGE: &str = "The Great Fire of London broke out in the early hours of Sunday, \
2 September 1666, in a bakery on Pudding Lane belonging to Thomas Farriner. Fanned by a strong \
easterly wind and fed by timber houses dried out by a long, hot summer, the fire spread rapidly \
through the narrow streets of the medieval city. The Lord Mayor, Sir Thomas Bloodworth, at first \
dismissed the danger, and the usual technique of demolishing houses to create firebreaks was \
applied too late. By Tuesday the fire had reached St Paul's Cathedral, whose lead roof melted and \
poured into the streets. The fire was finally brought under control on Thursday, 6 September, \
after the wind dropped and the garrison of the Tower of London used gunpowder to blow up houses \
in its path. It destroyed some 13,200 houses, 87 parish churches and most of the buildings of the \
City authorities, leaving perhaps 70,000 of the city's 80,000 inhabitants homeless. The recorded \
death toll was small, but historians believe many deaths of poor and middle-class people went \
unrecorded. In the aftermath, King Charles II appointed commissioners to plan the rebuilding, \
and Christopher Wren was given the task of designing a new St Paul's Cathedral and fifty-one city \
churches. The Rebuilding Act of 1667 required new houses to be built of brick or stone, and the \
Monument to the Great Fire, designed by Wren and Robert Hooke, was completed in 1677 near the \
place where the fire started.";

const CODE: &str = r#"class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, name, qty, price):
        if name in self.items:
            self.items[name]["qty"] += qty
        else:
            self.items[name] = {"qty": qty, "price": price}

    def remove(self, name, qty):
        if name not in self.items:
            raise KeyError(name)
        self.items[name]["qty"] -= qty
        if self.items[name]["qty"] <= 0:
            del self.items[name]

    def total_value(self):
        total = 0
        for name, item in self.items.items():
            total += item["qty"] * item["price"]
        return total

    def report(self):
        lines = []
        for name, item in sorted(self.items.items()):
            lines.append(f"{name}: {item['qty']} x {item['price']:.2f}")
        return "\n".join(lines)
"#;

fn prompt_text(kind: &str) -> anyhow::Result<String> {
    Ok(match kind {
        "prose" => "Write a short story about a lighthouse keeper who finds a message in a \
                    bottle that seems to be addressed to her."
            .into(),
        "code" => "Write a Python function that reads a CSV file and returns, for every \
                   numeric column, its mean, minimum and maximum. Include type hints, a \
                   docstring and a small usage example."
            .into(),
        "summary" => format!(
            "Summarize the following text in about 150 words, keeping the key names, \
             dates and numbers.\n\n{PASSAGE}"
        ),
        "edit" => format!(
            "Add type hints and a docstring to every method of this class. Return the \
             complete updated code only.\n\n```python\n{CODE}```"
        ),
        other => anyhow::bail!("unknown prompt kind {other:?} (prose|code|summary|edit)"),
    })
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let mut drafts = vec!["none".to_string(), "lookup".to_string()];
    let mut ks = vec![(4usize, false)];
    let mut head = DsparkHead::SharedTarget;
    let mut adaptive = false;
    let mut ngram = (1usize, 3usize);
    let mut max_draft = 8usize;
    let mut n = 128usize;
    let mut threads = 0usize;
    let mut prompts = vec![
        "prose".to_string(),
        "code".into(),
        "summary".into(),
        "edit".into(),
    ];
    let mut temp = 0.0f32;
    let mut seed = 1u64;
    let mut reps = 1usize;
    let mut think = false;
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        let list = |s: String| s.split(',').map(str::to_owned).collect::<Vec<_>>();
        match a.as_str() {
            "--draft" => drafts = list(val()?),
            "--k" => {
                // `4` = fixed k, `4a` = adaptive with at most 4.
                ks = list(val()?)
                    .iter()
                    .map(|k| match k.strip_suffix('a') {
                        Some(k) => k.parse().map(|k| (k, true)),
                        None => k.parse().map(|k| (k, adaptive)),
                    })
                    .collect::<Result<_, _>>()?
            }
            "--head" => head = val()?.parse()?,
            "--adaptive" => adaptive = true,
            "--no-adaptive" => adaptive = false,
            "--ngram" => {
                let v = val()?;
                let (a, b) = v
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("--ngram MIN:MAX"))?;
                ngram = (a.parse()?, b.parse()?);
            }
            "--max-draft" => max_draft = val()?.parse()?,
            "-n" => n = val()?.parse()?,
            "--threads" | "-t" => threads = val()?.parse()?,
            "--prompts" => prompts = list(val()?),
            "--temp" => temp = val()?.parse()?,
            "--seed" => seed = val()?.parse()?,
            "--reps" => reps = val()?.parse()?,
            "--think" => think = true,
            _ if model.is_none() && !a.starts_with('-') => model = Some(a.into()),
            _ => anyhow::bail!("unknown argument {a}"),
        }
    }
    let model = model.ok_or_else(|| anyhow::anyhow!("usage: spec_bench <model> [options]"))?;
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t0 = Instant::now();
    let mut g = xor_net::models::q1::load_generator(&model, None)?;
    g.set_prefix_caching(false);
    eprintln!(
        "loaded {} in {:.1}s",
        model.display(),
        t0.elapsed().as_secs_f64()
    );

    let mut drafters: Vec<(String, Option<Arc<dyn DraftModel>>)> = Vec::new();
    for d in &drafts {
        match d.as_str() {
            "none" => drafters.push(("plain".into(), None)),
            "lookup" => drafters.push((
                format!("lookup{}:{}", ngram.0, ngram.1),
                Some(Arc::new(PromptLookup::new(ngram.0, ngram.1, max_draft))),
            )),
            path => {
                let t = Instant::now();
                let m = DsparkModel::load_for(path, &**g.model(), head)?;
                eprintln!(
                    "loaded DSpark {path} ({head:?} head, {:.0} MB/round) in {:.1}s",
                    m.bytes_per_round() as f64 / 1e6,
                    t.elapsed().as_secs_f64()
                );
                drafters.push((format!("dspark-{head:?}").to_lowercase(), Some(Arc::new(m))));
            }
        }
    }

    let sampling = if temp > 0.0 {
        SamplingParams::default()
            .with_temperature(temp)
            .with_seed(Some(seed))
    } else {
        SamplingParams::greedy()
    };
    let base_params = GenerationParams::default()
        .with_max_tokens(n)
        .with_sampling(sampling)
        .with_ignore_eos(true)
        .with_enable_thinking(think);

    println!(
        "| prompt | drafter | k | prompt tok | tokens | decode tok/s | speedup | τ | accept | identical | draft ms | verify ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for kind in &prompts {
        let msgs = [ChatMessage::user(prompt_text(kind)?)];
        let mut baseline: Option<(Vec<u32>, f64)> = None;
        for (name, d) in &drafters {
            let kset: Vec<(usize, bool)> = if d.is_some() {
                ks.clone()
            } else {
                vec![(0, false)]
            };
            for &(k, adaptive) in &kset {
                let params = match d {
                    Some(d) => base_params
                        .clone()
                        .with_speculative(SpecConfig::new(d.clone(), k).with_adaptive(adaptive)),
                    None => base_params.clone(),
                };
                let mut best: Option<xor_net::Generation> = None;
                for _ in 0..reps.max(1) {
                    g.release_state();
                    let out = g.chat_stream(&msgs, &params, |_| ControlFlow::Continue(()))?;
                    if best
                        .as_ref()
                        .is_none_or(|b| out.timings.decode < b.timings.decode)
                    {
                        best = Some(out);
                    }
                }
                let out = best.unwrap();
                let tps = out.timings.decode_tok_per_s();
                if baseline.is_none() {
                    baseline = Some((out.tokens.clone(), tps));
                    if d.is_some() {
                        eprintln!(
                            "note: first config of {kind} is speculative; it is the reference"
                        );
                    }
                }
                let (bt, btps) = baseline.as_ref().unwrap();
                let ident = if temp > 0.0 {
                    "n/a".to_string()
                } else if out.tokens == *bt {
                    "yes".into()
                } else {
                    let at = out
                        .tokens
                        .iter()
                        .zip(bt)
                        .take_while(|(a, b)| a == b)
                        .count();
                    format!("NO (diverges at {at})")
                };
                let s = out.speculative.unwrap_or_default();
                println!(
                    "| {kind} | {name} | {k}{} | {} | {} | {tps:.2} | {:.2}x | {:.2} | {:.0}% | {ident} | {:.0} | {:.0} |",
                    if adaptive { "a" } else { "" },
                    out.usage.prompt_tokens,
                    out.tokens.len(),
                    tps / btps,
                    s.mean_accepted_len(),
                    100.0 * s.acceptance_rate(),
                    s.draft_time.as_secs_f64() * 1e3,
                    s.verify_time.as_secs_f64() * 1e3,
                );
                if std::env::var_os("SPEC_BENCH_TEXT").is_some() {
                    eprintln!("--- {kind} / {name} k={k}:\n{}\n---", out.text);
                }
            }
        }
    }
    Ok(())
}
