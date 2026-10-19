//! `xor-net-server`: a small OpenAI / llama.cpp-compatible HTTP inference
//! server for the xor-net engine.
//!
//! Design:
//!   * OpenAI surface: `GET /v1/models`, `POST /v1/completions`,
//!     `POST /v1/chat/completions` (streaming + non-streaming), llama.cpp
//!     `POST /completion` and `GET /health`.
//!   * Aggregate throughput comes from physical cores: `--slots` worker
//!     threads each run one decode inside their own (default 1-thread) rayon
//!     pool. See `OPTIMISATION.md`.
//!   * Hand-rolled, defensive HTTP/1.1 (no async runtime): thread per
//!     connection with a connection cap, bounded job queue, size limits and
//!     timeouts everywhere. The release profile uses `panic = "abort"`, so
//!     request paths must never panic.
//!   * All model interaction goes through the `engine::Engine` trait.
//!
//! Run `xor-net-server --help` for options.

mod api;
mod config;
mod engine;
mod http;
mod serve;
mod text;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
use xor_net::nn::LmHeadConfig;
use xor_net::{AutoModelForCausalLM, QuantizationConfig, TernaryPackType};

use crate::config::{Parsed, ServerConfig};
use crate::engine::{Engine, LlamaEngine};
use crate::serve::App;

fn hub_file(repo_id: &str, file: &str) -> anyhow::Result<PathBuf> {
    let api = Api::new().map_err(|e| anyhow!(e.to_string()))?;
    api.repo(Repo::new(repo_id.to_string(), RepoType::Model))
        .get(file)
        .map_err(|e| anyhow!("{repo_id}/{file}: {e}"))
}

/// tokenizer.json from the model dir (or hub repo), falling back to
/// `fallback_repo` only when it is missing.
fn load_tokenizer(
    model_dir: Option<&Path>,
    hub_id: Option<&str>,
    fallback_repo: &str,
) -> anyhow::Result<Tokenizer> {
    let path = if let Some(p) = model_dir
        .map(|d| d.join("tokenizer.json"))
        .filter(|p| p.is_file())
    {
        p
    } else if let Some(p) = hub_id.and_then(|id| hub_file(id, "tokenizer.json").ok()) {
        p
    } else {
        eprintln!("warning: no tokenizer.json with the model; downloading from {fallback_repo}");
        hub_file(fallback_repo, "tokenizer.json")?
    };
    println!("Tokenizer: {}", path.display());
    Tokenizer::from_file(&path).map_err(|e| anyhow!("tokenizer {}: {e}", path.display()))
}

fn model_id(cfg: &ServerConfig, model_dir: Option<&Path>) -> String {
    if let Some(id) = &cfg.model_id {
        return id.clone();
    }
    if let Some(dir) = model_dir {
        let canon = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        if let Some(name) = canon.file_name().and_then(|s| s.to_str()) {
            return name.to_string();
        }
    }
    cfg.model.clone()
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

    xor_net::init_threads(cfg.slots).map_err(|e| anyhow!(e.to_string()))?;

    let quant = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true);
    let local = Path::new(&cfg.model).is_dir();
    let model_dir = local.then(|| Path::new(&cfg.model));
    let (model, mconfig) = if let Some(dir) = model_dir {
        println!("Loading from local directory: {}", dir.display());
        AutoModelForCausalLM::from_local(dir, quant)?
    } else {
        println!("Loading from HuggingFace: {}", cfg.model);
        AutoModelForCausalLM::from_pretrained(&cfg.model, quant)?
    };
    println!("Model loaded.");

    let tokenizer = load_tokenizer(
        model_dir,
        (!local).then_some(cfg.model.as_str()),
        &cfg.tokenizer_repo,
    )?;
    let id = model_id(&cfg, model_dir);
    let engine = LlamaEngine::new(model, mconfig, tokenizer, model_dir, id);
    println!(
        "Model id: {}, context: {}, EOS ids: {:?}",
        engine.model_id(),
        engine.context_length(),
        engine.eos_ids()
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
    println!(
        "Server: {} worker slots, {} thread(s)/slot, listening on {}:{}",
        app.cfg.slots, app.cfg.threads_per_slot, app.cfg.host, app.cfg.port
    );

    serve::serve(listener, Arc::clone(&app));

    app.queue.shutdown();
    for w in workers {
        let _ = w.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
