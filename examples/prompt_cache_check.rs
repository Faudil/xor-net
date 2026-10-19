//! Cross-request prompt cache check on a real model.
//!
//! 1. Snapshot cost: prefill `N` tokens (default 512), then time
//!    `snapshot(N)` and `restore` and report the snapshot size.
//! 2. Two generators (as two server slots) share one `PromptCache`.
//!    Request A = long system prompt + question 1, then request B (on the
//!    *other* generator) = same system prompt + question 2. B must reuse at
//!    least the system prompt and produce the same greedy tokens as B run
//!    fresh (no cache, no prefix reuse). Prints TTFT (prefill wall time,
//!    restore included) for B fresh vs. cached.
//!
//! ```text
//! cargo run --release --example prompt_cache_check -- <model.xq1> [max_tokens] [snapshot_tokens (0 = skip)]
//! ```
use std::sync::Arc;
use std::time::Instant;

use xor_net::prompt_cache::PromptCache;
use xor_net::{AutoModelForCausalLM, ChatMessage, GenerationParams, SamplingParams};

const SYSTEM: &str = "You are Atlas, the support assistant of the Northwind \
public library network. Follow these rules in every answer. \
1. Be concise: answer in at most three sentences unless the user asks for detail. \
2. Opening hours: the central branch opens from nine in the morning to eight in \
the evening on weekdays, and from ten to five on Saturdays; it is closed on Sundays \
and public holidays. Neighbourhood branches open from noon to six on weekdays only. \
3. Membership is free for residents. Visitors pay twelve euros per year. A member \
may borrow up to fifteen items at a time: at most five films and at most three \
board games. \
4. Loans last three weeks and can be renewed twice online, unless another member \
has reserved the item. Late returns cost twenty cents per day per item, capped at \
five euros per item. Children's cards never pay late fees. \
5. Lost or damaged items are charged at their replacement price plus a three euro \
processing fee. Never promise a waiver; direct such cases to the branch manager. \
6. The digital catalogue offers e-books, audiobooks, newspapers and language \
courses. Members sign in with their card number and a four digit PIN. A forgotten \
PIN can only be reset in person, with an identity document. \
7. Meeting rooms can be booked up to thirty days in advance, for at most four hours \
per day, by members aged eighteen or older. Cancellations later than twenty four \
hours before the booking count as a no-show; three no-shows suspend booking rights \
for two months. \
8. Printing costs ten cents per black and white page and fifty cents per colour \
page. Scanning to e-mail or USB is free. \
9. Accessibility: every branch has step-free access, hearing loops at the front \
desk, and large-print collections. Assistance dogs are welcome everywhere. \
10. Privacy: never ask for or repeat a member's full card number, address or \
borrowing history. If the user shares personal data, do not repeat it back. \
11. If you do not know an answer, say so and suggest calling the central desk. \
Do not invent services, prices or events. \
12. Tone: friendly, plain language, no emojis, no marketing phrases.";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("model path");
    let n: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let snap_n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(512);
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!("{e}"))?;
    let g = AutoModelForCausalLM::generator_from_file(path, None)?;
    let model = g.model().clone();

    // --- 1. snapshot size / cost ------------------------------------------
    if snap_n > 0 {
        let ids: Vec<u32> = (0..snap_n as u32)
            .map(|i| 1000 + (i * 7919) % 20000)
            .collect();
        let mut st = model.new_state()?;
        let t = Instant::now();
        for (i, c) in ids.chunks(512).enumerate() {
            model.forward(c, i * 512, &mut *st)?;
        }
        let pf = t.elapsed();
        let t = Instant::now();
        let snap = st.snapshot(snap_n).expect("snapshot");
        let ts = t.elapsed();
        let mut st2 = model.new_state()?;
        let t = Instant::now();
        assert!(st2.restore(&snap));
        let tr_cold = t.elapsed();
        let t = Instant::now();
        assert!(st2.restore(&snap));
        let tr_warm = t.elapsed();
        // Restored state continues exactly like the original.
        let next = [42u32];
        let a = model.forward(&next, snap_n, &mut *st)?;
        let b = model.forward(&next, snap_n, &mut *st2)?;
        println!(
            "snapshot @{snap_n}: {:.1} MiB ({:.1} KiB/token incl. fixed state), snapshot {:.2} ms, \
             restore {:.2} ms (fresh state) / {:.2} ms (warm); prefill {:.0} ms; continue identical: {}",
            snap.bytes() as f64 / (1 << 20) as f64,
            snap.bytes() as f64 / 1024.0 / snap_n as f64,
            ts.as_secs_f64() * 1e3,
            tr_cold.as_secs_f64() * 1e3,
            tr_warm.as_secs_f64() * 1e3,
            pf.as_secs_f64() * 1e3,
            a == b
        );
    }

    // --- 2. two generators sharing a cache --------------------------------
    let params = GenerationParams::default()
        .with_max_tokens(n)
        .with_sampling(SamplingParams::greedy())
        .with_enable_thinking(false);
    let cache = Arc::new(PromptCache::new(1 << 30, 64));
    let mut ga = g.fork().with_prompt_cache(cache.clone());
    let mut gb = ga.fork();
    let sys = ChatMessage::system(SYSTEM);
    let ma = [
        sys.clone(),
        ChatMessage::user("Can I renew a film I borrowed two weeks ago?"),
    ];
    let mb = [
        sys.clone(),
        ChatMessage::user("How much does colour printing cost?"),
    ];
    // The system-prompt boundary the generator snapshots at.
    let kw = &params.chat_template_kwargs;
    let ids_b = g.encode(&g.apply_chat_template_with_kwargs(&mb, true, kw)?, false)?;
    let sys_len = g.chat_prefix_boundaries(&mb, kw, &ids_b)[0];

    let mut fresh_g = g.fork();
    fresh_g.set_prefix_caching(false);
    let fresh = fresh_g.chat(&mb, &params)?;
    drop(fresh_g);

    let a = ga.chat(&ma, &params)?;
    let b = gb.chat(&mb, &params)?;
    let st = cache.stats();
    println!(
        "system prompt: {sys_len} tokens; cache after A: {:?} ({:.1} MiB)",
        cache.entry_lens(),
        st.bytes as f64 / (1 << 20) as f64
    );
    println!(
        "A: prompt {} cached {} TTFT {:.0} ms",
        a.usage.prompt_tokens,
        a.usage.cached_prompt_tokens,
        a.timings.prefill_ms()
    );
    println!(
        "B fresh : prompt {} cached {} TTFT {:.0} ms",
        fresh.usage.prompt_tokens,
        fresh.usage.cached_prompt_tokens,
        fresh.timings.prefill_ms()
    );
    println!(
        "B cached: prompt {} cached {} TTFT {:.0} ms ({:.1}x)",
        b.usage.prompt_tokens,
        b.usage.cached_prompt_tokens,
        b.timings.prefill_ms(),
        fresh.timings.prefill_ms() / b.timings.prefill_ms().max(1e-3)
    );
    let same = b.tokens == fresh.tokens;
    println!("B text: {:?}", b.text);
    if !same {
        println!("  cached: {:?}\n  fresh:  {:?}", b.tokens, fresh.tokens);
    }
    let ok = same && b.usage.cached_prompt_tokens >= sys_len;
    println!(
        "B identical to fresh: {same}; cached >= system prompt: {}; stats {:?}",
        b.usage.cached_prompt_tokens >= sys_len,
        cache.stats()
    );
    println!("PASS: {ok}");
    Ok(())
}
