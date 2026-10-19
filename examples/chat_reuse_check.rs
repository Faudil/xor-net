//! Multi-turn chat prefix-reuse check: each turn runs with the reused state
//! and again from scratch (fresh generator, no caching); the greedy token
//! ids must be identical.
//!
//! ```text
//! cargo run --release --example chat_reuse_check -- <model.xq1> [turns] [max_tokens]
//! ```
use std::time::Instant;

use xor_net::models::q1::qwen35::Qwen35State;
use xor_net::{AutoModelForCausalLM, ChatMessage, GenerationParams, SamplingParams};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("model path");
    let turns: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(40);
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut g = AutoModelForCausalLM::generator_from_file(path, None)?;

    // Checkpoint cost (qwen35 only).
    let mut st = g.model().new_state()?;
    if st.as_any_mut().downcast_mut::<Qwen35State>().is_some() {
        st.checkpoint();
        let t = Instant::now();
        st.checkpoint();
        let ck = t.elapsed();
        let t = Instant::now();
        let ok = st.truncate(0);
        let bytes = st
            .as_any_mut()
            .downcast_mut::<Qwen35State>()
            .unwrap()
            .checkpoint_bytes();
        println!(
            "checkpoint: {:.1} MiB, copy {:.2} ms (truncate(0) {ok} {:.2} ms)",
            bytes as f64 / (1 << 20) as f64,
            ck.as_secs_f64() * 1e3,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    drop(st);

    let params = GenerationParams::default()
        .with_max_tokens(n)
        .with_sampling(SamplingParams::greedy())
        .with_enable_thinking(false);
    let users = [
        "Name three primary colors.",
        "Which of them is used in the French flag?",
        "Write one sentence about that flag.",
        "Thanks. Now count from one to five.",
    ];
    let mut msgs = vec![ChatMessage::system("You are a concise assistant.")];
    let mut all_same = true;
    for (turn, u) in users.iter().cycle().take(turns).enumerate() {
        msgs.push(ChatMessage::user(*u));
        let out = g.chat(&msgs, &params)?;
        let mut fresh_g = g.fork();
        fresh_g.set_prefix_caching(false);
        let fresh = fresh_g.chat(&msgs, &params)?;
        drop(fresh_g);
        let same = out.tokens == fresh.tokens;
        all_same &= same;
        println!(
            "turn {}: prompt {} cached {} prefill {:.0} ms (fresh {:.0} ms) | {} tokens | identical={same}",
            turn + 1,
            out.usage.prompt_tokens,
            out.usage.cached_prompt_tokens,
            out.timings.prefill_ms(),
            fresh.timings.prefill_ms(),
            out.tokens.len(),
        );
        if !same {
            println!("  reused: {:?}\n  fresh:  {:?}", out.tokens, fresh.tokens);
        }
        println!("  {:?}", out.text);
        msgs.push(ChatMessage::assistant(out.text));
    }
    println!("ALL IDENTICAL: {all_same}");
    Ok(())
}
