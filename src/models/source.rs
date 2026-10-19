//! Resolve a user-supplied model reference and load a ready [`Generator`].
//!
//! One entry point for the CLI, the Python bindings and applications that
//! want "give me a path or a Hub id, I want to generate":
//!
//! | reference                              | resolves to                          |
//! |----------------------------------------|--------------------------------------|
//! | `model.gguf` / `model.xq1`             | Bonsai 1-bit (Q1_0) model file       |
//! | directory with `config.json`           | Llama / BitNet b1.58 safetensors dir |
//! | directory with a `.xq1` / `.gguf`      | that Bonsai model file               |
//! | `owner/repo` (Hub, e.g. `prism-ml/Bonsai-1.7B-gguf`) | its Q1_0 `.gguf`, else its safetensors |
//! | `owner/repo:file.gguf`                 | that file of the repo                |
//!
//! Hub downloads go through `hf-hub` into the standard Hugging Face cache
//! (`HF_HOME`, default `~/.cache/huggingface`), so they are shared with
//! Python tooling and only happen once.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use xor_net::models::source::{load_generator, LoadOptions};
//! let mut g = load_generator("models/bonsai/Bonsai-1.7B.xq1", &LoadOptions::default())?;
//! let out = g.chat(&[xor_net::ChatMessage::user("Hi!")], &Default::default())?;
//! println!("{}", out.text);
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};

use crate::bit1_58::quantization::TernaryPackType;
use crate::generation::Generator;
use crate::models::llama::causal_lm::LlamaCausalLM;
use crate::nn::{LmHeadConfig, QuantizationConfig};

/// Which runtime a resolved model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// 1-bit Bonsai (`.gguf` with Q1_0 weights, or converted `.xq1`).
    Bonsai,
    /// HF Llama-architecture directory (BitNet b1.58, Llama3-8B-1.58, ...).
    Llama,
}

/// A model reference resolved to local files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub kind: ModelKind,
    /// The model file (Bonsai) or directory (Llama).
    pub path: PathBuf,
}

/// Options for [`load_generator`].
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Override the tokenizer (`tokenizer.json`).
    pub tokenizer: Option<PathBuf>,
    /// Weight quantization for Llama/BitNet safetensors models (ignored for
    /// Bonsai, which is natively 1-bit).
    pub quantization: QuantizationConfig,
    /// Hub repo to fetch `tokenizer.json` (+ config) from when a Llama
    /// directory ships without one.
    pub fallback_tokenizer_repo: String,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            tokenizer: None,
            quantization: QuantizationConfig::Bit1_58(
                TernaryPackType::Pack4,
                LmHeadConfig::Int4,
                true,
            ),
            fallback_tokenizer_repo: "microsoft/bitnet-b1.58-2B-4T".into(),
        }
    }
}

/// Whether `spec` looks like a Hub repo id (`owner/name[:file]`) rather than
/// a local path. Only meaningful when no file of that name exists.
pub fn looks_like_hub_id(spec: &str) -> bool {
    let repo = spec.split_once(':').map_or(spec, |(r, _)| r);
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    let ok = |s: &str| {
        !s.is_empty()
            && s != "."
            && s != ".."
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    ok(owner) && ok(name) && !spec.starts_with('.')
}

/// Pick the Bonsai file inside a local directory, preferring `.xq1`.
fn bonsai_file_in(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    let ext = |p: &Path, e: &str| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(e));
    files
        .iter()
        .find(|p| ext(p, "xq1"))
        .or_else(|| pick_gguf(files.iter().filter(|p| ext(p, "gguf"))))
        .cloned()
}

/// Among GGUF names, prefer a Q1_0 one.
fn pick_gguf<'a, P: AsRef<Path> + 'a>(names: impl Iterator<Item = &'a P>) -> Option<&'a P> {
    let names: Vec<&P> = names.collect();
    let is_q1 = |p: &P| {
        p.as_ref()
            .to_string_lossy()
            .to_ascii_lowercase()
            .contains("q1_0")
    };
    names
        .iter()
        .find(|p| is_q1(p))
        .or_else(|| names.first())
        .copied()
}

fn resolve_local(path: &Path) -> anyhow::Result<ResolvedModel> {
    if path.is_file() {
        return Ok(ResolvedModel {
            kind: ModelKind::Bonsai,
            path: path.to_path_buf(),
        });
    }
    anyhow::ensure!(path.is_dir(), "{} does not exist", path.display());
    if path.join("config.json").is_file() {
        return Ok(ResolvedModel {
            kind: ModelKind::Llama,
            path: path.to_path_buf(),
        });
    }
    match bonsai_file_in(path) {
        Some(f) => Ok(ResolvedModel {
            kind: ModelKind::Bonsai,
            path: f,
        }),
        None => anyhow::bail!(
            "{}: no config.json (Llama/BitNet) and no .xq1/.gguf (Bonsai) inside",
            path.display()
        ),
    }
}

fn hub_err(repo: &str, e: impl std::fmt::Display) -> anyhow::Error {
    let msg = e.to_string();
    if msg.contains("status: 401") || msg.contains("status: 404") {
        anyhow::anyhow!(
            "Hugging Face Hub: {repo} not found (or gated/private: set HF_TOKEN). \
             If you meant a local path, it does not exist."
        )
    } else {
        anyhow::anyhow!("Hugging Face Hub ({repo}): {msg}")
    }
}

fn hub_api(progress: bool) -> Result<hf_hub::api::sync::Api, hf_hub::api::sync::ApiError> {
    let mut b = ApiBuilder::from_env().with_progress(progress);
    if let Ok(t) = std::env::var("HF_TOKEN")
        && !t.is_empty()
    {
        b = b.with_token(Some(t));
    }
    b.build()
}

/// Look for a `.gguf` already in the local HF cache (offline fallback).
fn cached_gguf(repo_id: &str) -> Option<PathBuf> {
    let dir = Cache::from_env()
        .path()
        .join(Repo::model(repo_id.to_string()).folder_name())
        .join("snapshots");
    let mut found = Vec::new();
    for snap in std::fs::read_dir(dir).ok()?.flatten() {
        for f in std::fs::read_dir(snap.path()).ok()?.flatten() {
            if f.path().extension().is_some_and(|e| e == "gguf") {
                found.push(f.path());
            }
        }
    }
    found.sort();
    pick_gguf(found.iter()).cloned()
}

fn resolve_hub(spec: &str) -> anyhow::Result<ResolvedModel> {
    let (repo_id, file) = match spec.split_once(':') {
        Some((r, f)) => (r, Some(f)),
        None => (spec, None),
    };
    let api = hub_api(true).map_err(|e| hub_err(repo_id, e))?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    if let Some(f) = file {
        let path = repo.get(f).map_err(|e| hub_err(repo_id, e))?;
        return resolve_local(&path);
    }
    let info = match repo.info() {
        Ok(i) => i,
        Err(e) => {
            // Offline: reuse a cached GGUF if there is one.
            if let Some(p) = cached_gguf(repo_id) {
                return Ok(ResolvedModel {
                    kind: ModelKind::Bonsai,
                    path: p,
                });
            }
            return Err(hub_err(repo_id, e));
        }
    };
    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    if let Some(g) = pick_gguf(names.iter().filter(|n| n.ends_with(".gguf"))) {
        let path = repo.get(g).map_err(|e| hub_err(repo_id, e))?;
        return Ok(ResolvedModel {
            kind: ModelKind::Bonsai,
            path,
        });
    }
    if let Some(x) = names.iter().find(|n| n.ends_with(".xq1")) {
        let path = repo.get(x).map_err(|e| hub_err(repo_id, e))?;
        return Ok(ResolvedModel {
            kind: ModelKind::Bonsai,
            path,
        });
    }
    anyhow::ensure!(
        names.iter().any(|n| n == "config.json"),
        "{repo_id}: no .gguf, .xq1 or config.json in the repo"
    );
    // Llama / BitNet safetensors: fetch everything a local dir would have.
    let wanted = |n: &str| {
        !n.contains('/')
            && (n.ends_with(".safetensors")
                || matches!(
                    n,
                    "config.json"
                        | "model.safetensors.index.json"
                        | "tokenizer.json"
                        | "tokenizer_config.json"
                        | "generation_config.json"
                        | "chat_template.jinja"
                ))
    };
    let mut dir = None;
    for n in names.iter().filter(|n| wanted(n)) {
        let p = repo.get(n).map_err(|e| hub_err(repo_id, e))?;
        dir = p.parent().map(Path::to_path_buf);
    }
    let dir = dir.ok_or_else(|| anyhow::anyhow!("{repo_id}: nothing to download"))?;
    Ok(ResolvedModel {
        kind: ModelKind::Llama,
        path: dir,
    })
}

/// Resolve a model reference (see the module docs), downloading from the
/// Hub if needed.
pub fn resolve(spec: &str) -> anyhow::Result<ResolvedModel> {
    let path = Path::new(spec);
    if path.exists() || !looks_like_hub_id(spec) {
        return resolve_local(path);
    }
    resolve_hub(spec)
}

/// Directory holding `tokenizer.json` for a Llama model: the model dir if it
/// has one, else a Hub snapshot of `fallback_repo`.
fn llama_tokenizer_dir(dir: &Path, fallback_repo: &str) -> anyhow::Result<PathBuf> {
    if dir.join("tokenizer.json").is_file() {
        return Ok(dir.to_path_buf());
    }
    let api = hub_api(false).map_err(|e| hub_err(fallback_repo, e))?;
    let repo = api.repo(Repo::new(fallback_repo.to_string(), RepoType::Model));
    let tok = repo
        .get("tokenizer.json")
        .map_err(|e| hub_err(fallback_repo, e))?;
    for optional in ["tokenizer_config.json", "generation_config.json"] {
        let _ = repo.get(optional);
    }
    Ok(tok
        .parent()
        .ok_or_else(|| anyhow::anyhow!("bad tokenizer path"))?
        .to_path_buf())
}

/// Load an already resolved model into a [`Generator`].
pub fn load_resolved(m: &ResolvedModel, opts: &LoadOptions) -> anyhow::Result<Generator> {
    match m.kind {
        ModelKind::Bonsai => crate::models::q1::load_generator(&m.path, opts.tokenizer.as_deref()),
        ModelKind::Llama => {
            let model = LlamaCausalLM::from_local(&m.path, opts.quantization)?;
            let model: Arc<dyn crate::CausalLM> = Arc::new(model);
            match &opts.tokenizer {
                Some(t) => {
                    let tok = tokenizers::Tokenizer::from_file(t)
                        .map_err(|e| anyhow::anyhow!("loading {}: {e}", t.display()))?;
                    Ok(Generator::new(model, tok))
                }
                None => Generator::from_dir(
                    model,
                    llama_tokenizer_dir(&m.path, &opts.fallback_tokenizer_repo)?,
                ),
            }
        }
    }
}

/// Resolve `spec` (path or Hub id) and load it into a [`Generator`].
pub fn load_generator(spec: &str, opts: &LoadOptions) -> anyhow::Result<Generator> {
    load_resolved(&resolve(spec)?, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_id_detection() {
        assert!(looks_like_hub_id("prism-ml/Bonsai-1.7B-gguf"));
        assert!(looks_like_hub_id("microsoft/bitnet-b1.58-2B-4T"));
        assert!(looks_like_hub_id(
            "prism-ml/Bonsai-8B-gguf:Bonsai-8B-Q1_0.gguf"
        ));
        assert!(!looks_like_hub_id("models/bonsai/Bonsai-1.7B.xq1/x"));
        assert!(!looks_like_hub_id("./a/b"));
        assert!(!looks_like_hub_id("../a"));
        assert!(!looks_like_hub_id("/abs/path"));
        assert!(!looks_like_hub_id("model.gguf"));
        assert!(!looks_like_hub_id("a/"));
    }

    #[test]
    fn gguf_preference() {
        let names = ["Bonsai-8B-F16.gguf", "Bonsai-8B-Q1_0.gguf"];
        assert_eq!(pick_gguf(names.iter()), Some(&"Bonsai-8B-Q1_0.gguf"));
        let names = ["only.gguf"];
        assert_eq!(pick_gguf(names.iter()), Some(&"only.gguf"));
    }

    #[test]
    fn missing_local_path_errors() {
        let e = resolve("definitely/not/here.gguf").unwrap_err();
        assert!(e.to_string().contains("does not exist"), "{e}");
    }
}
