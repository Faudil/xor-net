//! Interactive multi-turn chat on top of `Generator`.
//!
//! cargo run --release --example chat -- [2b|3b|8b|<local model dir>]
//!
//! Env: XORNET_THREADS (pool size), XORNET_LMHEAD (int8|int4|ternary).
//! Turns reuse the model state of the previous turn (prefix caching), so
//! only the new user message is prefilled.

use std::io::{self, BufRead, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::nn::LmHeadConfig;
use xor_net::{
    ChatMessage, ChatTemplate, GenerationParams, Generator, LlamaCausalLM, QuantizationConfig,
    SamplingParams,
};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

const SYSTEM_PROMPT: &str = "You are a helpful, knowledgeable, and friendly AI assistant. You provide clear, concise, and accurate responses.";

fn lmhead_cfg(default: LmHeadConfig) -> LmHeadConfig {
    match std::env::var("XORNET_LMHEAD").as_deref() {
        Ok("int8") => LmHeadConfig::Int8,
        Ok("int4") => LmHeadConfig::Int4,
        Ok("ternary") => LmHeadConfig::Ternary,
        _ => default,
    }
}

/// Download tokenizer files of an HF repo; returns their directory.
fn hub_tokenizer_dir(repo_id: &str) -> anyhow::Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(repo_id.to_string());
    let tok = repo.get("tokenizer.json")?;
    for optional in ["tokenizer_config.json", "generation_config.json"] {
        let _ = repo.get(optional);
    }
    Ok(tok.parent().unwrap().to_path_buf())
}

fn main() -> anyhow::Result<()> {
    let threads = std::env::var("XORNET_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    xor_net::init_threads(threads).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let arg = std::env::args().nth(1).unwrap_or_else(|| "2b".into());
    let quant =
        |head, mixed| QuantizationConfig::Bit1_58(TernaryPackType::Pack4, lmhead_cfg(head), mixed);
    // (repo, quantization, is a base model without a usable chat template)
    let hub = match arg.as_str() {
        "2b" => Some((
            "microsoft/bitnet-b1.58-2B-4T",
            quant(LmHeadConfig::Int4, false),
            false,
        )),
        "3b" => Some((
            "1bitLLM/bitnet_b1_58-3B",
            quant(LmHeadConfig::Int4, false),
            true,
        )),
        "8b" => Some((
            "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
            quant(LmHeadConfig::Int8, true),
            true,
        )),
        _ => None,
    };

    eprintln!("Loading {arg}...");
    let t = std::time::Instant::now();
    let (model, tok_dir, base) = match hub {
        Some((repo, q, base)) => (
            LlamaCausalLM::from_pretrained(repo, q)?,
            hub_tokenizer_dir(repo)?,
            base,
        ),
        None => {
            let dir = Path::new(&arg);
            (
                LlamaCausalLM::from_local(dir, quant(LmHeadConfig::Int4, false))?,
                dir.to_path_buf(),
                false,
            )
        }
    };
    eprintln!("Model loaded in {:.2?}", t.elapsed());

    let mut generator = Generator::from_dir(Arc::new(model), &tok_dir)?;
    if base || generator.chat_template().is_none() {
        // Plain "System:/User:/Assistant:" transcript for base models.
        let (bos, eos) = generator
            .chat_template()
            .map(|t| (t.bos_token().to_owned(), t.eos_token().to_owned()))
            .unwrap_or_default();
        generator = generator.with_chat_template(ChatTemplate::fallback(bos, eos));
    }

    let params = GenerationParams::default()
        .with_max_tokens(2048)
        .with_stop(["\nUser:", "[User]"])
        .with_sampling(
            SamplingParams::default()
                .with_temperature(0.8)
                .with_top_p(0.95)
                .with_repetition_penalty(1.2)
                .with_penalty_last_n(Some(256)),
        );

    println!("XorNet chat | model: {arg} | commands: /quit /reset");
    let mut history = vec![ChatMessage::system(SYSTEM_PROMPT)];
    for line in io::stdin().lock().lines() {
        let input = line?.trim().to_string();
        match input.as_str() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                history.truncate(1);
                println!("Context reset.");
                continue;
            }
            _ => {}
        }
        history.push(ChatMessage::user(input));

        print!("Assistant: ");
        io::stdout().flush()?;
        // Leave room in the context window for the reply.
        let prompt_len = generator
            .encode(&generator.apply_chat_template(&history, true)?, false)?
            .len();
        let max_tokens = generator.model().max_seq_len().saturating_sub(prompt_len);
        let p = params
            .clone()
            .with_max_tokens(params.max_tokens.min(max_tokens));
        let out = generator.chat_stream(&history, &p, |delta| {
            print!("{delta}");
            let _ = io::stdout().flush();
            ControlFlow::Continue(())
        })?;
        println!();
        history.push(ChatMessage::assistant(out.text.trim()));

        let tm = out.timings;
        println!(
            "── {} tokens | prefill {} tok in {:.0} ms ({} cached) | decode {:.2} tok/s | {:?}",
            out.usage.completion_tokens,
            tm.prefill_tokens,
            tm.prefill_ms(),
            out.usage.cached_prompt_tokens,
            tm.decode_tok_per_s(),
            out.finish_reason,
        );
    }
    Ok(())
}
