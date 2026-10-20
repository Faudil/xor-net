//! `xor-net-server`: a small OpenAI / llama.cpp-compatible HTTP inference
//! server for the xor-net engine.
//!
//! Design:
//!   * OpenAI surface: `GET /v1/models`, `POST /v1/completions`,
//!     `POST /v1/chat/completions` (streaming + non-streaming), llama.cpp
//!     `POST /completion` and `GET /health`.
//!   * Two schedulers (`--scheduler`). `batch` (default for Bonsai models):
//!     continuous batching, one engine thread steps every active request
//!     together (`--max-batch` of them) through the model's batched forward
//!     on all physical cores, so each weight block is read once per step for
//!     all requests (`batcher`). `slots`: `--slots` worker threads each run
//!     one decode inside their own rayon pool of `--threads-per-slot`
//!     threads, sharing one `Arc` model; each slot owns a `Generator` fork
//!     (state / KV cache, sampler); defaults split the physical cores over 2
//!     slots (see `config::default_parallelism`).
//!   * One `PromptCache` shared by all slots (`--prompt-cache-mb`): state
//!     snapshots at the end of the system prompt and of the chat history, so
//!     a shared prefix is prefilled once across requests and slots. The job
//!     queue matches jobs to the slot whose state holds the longest prefix
//!     of their prompt (small lookahead window, no starvation).
//!   * Hand-rolled, defensive HTTP/1.1 (no async runtime): thread per
//!     connection with a connection cap, bounded job queue, size limits and
//!     timeouts everywhere. The release profile uses `panic = "abort"`, so
//!     request paths must never panic.
//!   * All model interaction goes through the `engine::Engine` trait,
//!     implemented by `GeneratorEngine` on top of `xor_net::Generator`
//!     (real chat templates, `chat_template_kwargs`, reasoning separation).
//!   * Models: Bonsai `.gguf` / `.xq1` files (`AutoModelForCausalLM::
//!     from_file`), Llama / BitNet safetensors dirs, or an HF repo id.
//!
//! Run `xor-net-server --help` for options.

mod api;
mod batcher;
mod config;
mod engine;
mod http;
mod serve;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
use xor_net::generation::{GenerationParams, Generator};
use xor_net::models::batched::BatchedCausalLM;
use xor_net::models::traits::CausalLM;
use xor_net::nn::LmHeadConfig;
use xor_net::prompt_cache::PromptCache;
use xor_net::{LlamaCausalLM, QuantizationConfig, TernaryPackType};

use crate::batcher::BatchOptions;
use crate::config::{Parsed, Scheduler, ServerConfig};
use crate::engine::{Engine, GeneratorEngine};
use crate::serve::App;

fn hub_file(repo_id: &str, file: &str) -> anyhow::Result<PathBuf> {
    let api = Api::new().map_err(|e| anyhow!(e.to_string()))?;
    api.repo(Repo::new(repo_id.to_string(), RepoType::Model))
        .get(file)
        .map_err(|e| anyhow!("{repo_id}/{file}: {e}"))
}

fn read_tokenizer(path: &Path) -> anyhow::Result<Tokenizer> {
    println!("Tokenizer: {}", path.display());
    Tokenizer::from_file(path).map_err(|e| anyhow!("tokenizer {}: {e}", path.display()))
}

/// What `--model` points at.
enum Source<'a> {
    /// Bonsai `.gguf` / `.xq1`.
    File(&'a Path),
    /// HF-style safetensors directory (Llama / BitNet).
    Dir(&'a Path),
    /// HF hub repo id.
    Hub(&'a str),
}

fn source(model: &str) -> Source<'_> {
    let p = Path::new(model);
    if p.is_dir() {
        Source::Dir(p)
    } else if p.is_file() {
        Source::File(p)
    } else {
        Source::Hub(model)
    }
}

/// A loaded model: its `Generator` (tokenizer, chat template, EOS ids), the
/// default model id and, for Bonsai models, the batched forward of the same
/// model (continuous batching).
type Loaded = (Generator, String, Option<Arc<dyn BatchedCausalLM>>);

/// Load the model and wrap it in a `Generator`.
fn load(cfg: &ServerConfig) -> anyhow::Result<Loaded> {
    let explicit_tok = cfg.tokenizer.as_deref().map(Path::new);
    let quant = || QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true);
    match source(&cfg.model) {
        Source::File(path) => {
            println!("Loading Bonsai model file: {}", path.display());
            let (batched, g) = xor_net::models::q1::batch::load_batched(path, explicit_tok)?;
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&cfg.model)
                .to_string();
            Ok((g, id, Some(batched)))
        }
        Source::Dir(dir) => {
            println!("Loading from local directory: {}", dir.display());
            let model: Arc<dyn CausalLM> = Arc::new(LlamaCausalLM::from_local(dir, quant())?);
            let g = if let Some(t) = explicit_tok {
                Generator::from_dir_with_tokenizer(model, dir, read_tokenizer(t)?)?
            } else if dir.join("tokenizer.json").is_file() {
                println!("Tokenizer: {}", dir.join("tokenizer.json").display());
                Generator::from_dir(model, dir)?
            } else {
                eprintln!(
                    "warning: no tokenizer.json with the model; downloading from {}",
                    cfg.tokenizer_repo
                );
                let tok = read_tokenizer(&hub_file(&cfg.tokenizer_repo, "tokenizer.json")?)?;
                Generator::from_dir_with_tokenizer(model, dir, tok)?
            };
            let canon = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
            let id = canon
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&cfg.model)
                .to_string();
            Ok((g, id, None))
        }
        Source::Hub(repo) => {
            println!("Loading from HuggingFace: {repo}");
            let model: Arc<dyn CausalLM> = Arc::new(LlamaCausalLM::from_pretrained(repo, quant())?);
            // hf-hub puts every file of a revision in one snapshot dir, so
            // fetching the configs next to tokenizer.json lets from_dir find
            // the chat template and EOS ids.
            let tok_path = match hub_file(repo, "tokenizer.json") {
                Ok(p) => p,
                Err(_) => {
                    eprintln!(
                        "warning: {repo} has no tokenizer.json; using {}",
                        cfg.tokenizer_repo
                    );
                    hub_file(&cfg.tokenizer_repo, "tokenizer.json")?
                }
            };
            for f in ["tokenizer_config.json", "generation_config.json"] {
                let _ = hub_file(repo, f);
            }
            let tok = read_tokenizer(explicit_tok.unwrap_or(&tok_path))?;
            let snapshot = tok_path.parent().unwrap_or(Path::new("."));
            let g = Generator::from_dir_with_tokenizer(model, snapshot, tok)?;
            Ok((g, repo.to_string(), None))
        }
    }
}

/// Physical cores, assuming SMT-2 as `config::default_parallelism` does.
fn physical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .div_ceil(2)
        .max(1)
}

/// `--draft`: prompt lookup or a DSpark drafter for speculative decoding.
fn speculative(
    cfg: &ServerConfig,
    g: &xor_net::Generator,
) -> anyhow::Result<Option<xor_net::speculative::SpecConfig>> {
    use xor_net::models::q1::dspark::{DsparkHead, DsparkModel};
    use xor_net::speculative::{DraftModel, PromptLookup, SpecConfig};
    let Some(d) = &cfg.draft else {
        return Ok(None);
    };
    let (m, k): (Arc<dyn DraftModel>, usize) = if d == "lookup" {
        (
            Arc::new(PromptLookup::new(3, 5, cfg.draft_k.unwrap_or(4))),
            4,
        )
    } else {
        let head = match cfg.draft_head.as_deref() {
            Some("own") => DsparkHead::Own,
            _ => DsparkHead::SharedTarget,
        };
        let m = DsparkModel::load_for(d, &**g.model(), head)?;
        let k = m.config().block_size;
        (Arc::new(m), k)
    };
    println!(
        "Speculative decoding: {} drafter, k = {}",
        m.name(),
        cfg.draft_k.unwrap_or(k)
    );
    Ok(Some(SpecConfig::new(m, cfg.draft_k.unwrap_or(k))))
}

fn main() -> anyhow::Result<()> {
    let cfg = match config::parse(std::env::args().skip(1), |k| std::env::var(k).ok()) {
        Ok(Parsed::Run(c)) => *c,
        Ok(Parsed::Help) => {
            print!("{}", config::USAGE);
            return Ok(());
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{}", config::USAGE);
            std::process::exit(2);
        }
    };

    let mut cfg = cfg;
    // Auto: continuous batching for Bonsai models (they have a batched
    // forward), independent slots otherwise.
    let scheduler = cfg.scheduler.unwrap_or(match source(&cfg.model) {
        // Speculative decoding runs in the slot scheduler.
        _ if cfg.draft.is_some() => Scheduler::Slots,
        Source::File(_) => Scheduler::Batch,
        _ => Scheduler::Slots,
    });
    cfg.scheduler = Some(scheduler);
    let threads = match scheduler {
        // Slots: the global pool only serves loading; decoding runs in
        // per-slot pools.
        Scheduler::Slots => cfg.slots * cfg.threads_per_slot,
        // Batch: every batched forward uses the global pool, one thread per
        // physical core (SMT siblings only add contention).
        Scheduler::Batch => physical_cores(),
    };
    xor_net::init_threads(threads).map_err(|e| anyhow!(e.to_string()))?;

    let (mut generator, default_id, batched) = load(&cfg)?;
    println!("Model loaded.");
    let cache_bytes = match cfg.prompt_cache_mb {
        Some(mb) => mb.saturating_mul(1 << 20),
        None => PromptCache::default_budget(),
    };
    if cache_bytes > 0 {
        // Set before the slots fork the generator: they all share it.
        generator.set_prompt_cache(Some(Arc::new(PromptCache::new(
            cache_bytes,
            cfg.prompt_cache_min_tokens,
        ))));
        println!(
            "Prompt cache: {} MiB, prefixes >= {} tokens",
            cache_bytes >> 20,
            cfg.prompt_cache_min_tokens
        );
    } else {
        println!("Prompt cache: disabled");
    }
    let id = cfg.model_id.clone().unwrap_or(default_id);
    let spec = speculative(&cfg, &generator)?;
    if spec.is_some() && scheduler == Scheduler::Batch {
        eprintln!("warning: --draft is ignored by the batch scheduler (use --scheduler slots)");
    }
    let engine = match scheduler {
        Scheduler::Slots => GeneratorEngine::new(generator, cfg.slots, id).with_speculative(spec),
        Scheduler::Batch => {
            if batched.is_none() {
                eprintln!(
                    "warning: this model has no batched forward; the batch scheduler \
                     forwards requests one by one (--scheduler slots is faster)"
                );
            }
            // One slot per concurrent sequence.
            cfg.slots = cfg.max_batch;
            GeneratorEngine::new_batched(
                generator,
                batched,
                BatchOptions {
                    max_batch: cfg.max_batch,
                    prefill_chunk: GenerationParams::default().prefill_chunk_size,
                    prefill_budget: cfg.batch_prefill,
                },
                id,
            )?
        }
    };
    println!(
        "Model id: {}, context: {}, EOS ids: {:?}, chat template: {}",
        engine.model_id(),
        engine.context_length(),
        engine.eos_ids(),
        if engine.has_chat_template() {
            "model"
        } else {
            "fallback (System:/User:/Assistant:)"
        }
    );
    if engine.eos_ids().is_empty() {
        eprintln!(
            "warning: no EOS token found; generation only stops on max_tokens / stop strings"
        );
    }

    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))
        .with_context(|| format!("bind {}:{}", cfg.host, cfg.port))?;
    let app = Arc::new(App::new(Arc::new(engine), cfg));
    let workers = serve::spawn_workers(&app)?;
    match scheduler {
        Scheduler::Slots => println!(
            "Server: slot scheduler, {} worker slots, {} thread(s)/slot, reasoning separation {}, listening on {}:{}",
            app.cfg.slots,
            app.cfg.threads_per_slot,
            if app.reasoning { "on" } else { "off" },
            app.cfg.host,
            app.cfg.port
        ),
        Scheduler::Batch => println!(
            "Server: batch scheduler, up to {} concurrent requests, {} threads, reasoning separation {}, listening on {}:{}",
            app.cfg.max_batch,
            rayon::current_num_threads(),
            if app.reasoning { "on" } else { "off" },
            app.cfg.host,
            app.cfg.port
        ),
    }

    serve::serve(listener, Arc::clone(&app));

    app.queue.shutdown();
    for w in workers {
        let _ = w.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
