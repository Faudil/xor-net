//! `xor-net convert | bench | info | classify | pull | serve`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use serde_json::Value;
use xor_net::loader::gguf::GgufFile;
use xor_net::models::q1::{ModelAssets, format};
use xor_net::models::source::{self, LoadOptions, ModelKind};

use crate::args::{BenchArgs, ClassifyArgs, ConvertArgs, InfoArgs, PullArgs};
use crate::generate::{fmt_rss, init_threads};

// --- convert -----------------------------------------------------------------

pub fn default_xq1_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    // Bonsai-8B-Q1_0.gguf -> Bonsai-8B.xq1
    let stem = stem
        .strip_suffix("-Q1_0")
        .or_else(|| stem.strip_suffix("-q1_0"))
        .unwrap_or(&stem)
        .to_string();
    input.with_file_name(format!("{stem}.xq1"))
}

pub fn convert(a: ConvertArgs) -> anyhow::Result<()> {
    let resolved = source::resolve(&a.input)?;
    anyhow::ensure!(
        resolved.kind == ModelKind::Bonsai,
        "{}: convert takes a Bonsai Q1_0 .gguf",
        resolved.path.display()
    );
    let input = resolved.path;
    anyhow::ensure!(
        !format::is_xq1(&input)?,
        "{} is already an .xq1 file",
        input.display()
    );
    let output = a.output.unwrap_or_else(|| {
        // Hub downloads live in the read-only-ish HF cache: write to cwd.
        let p = default_xq1_path(&input);
        if Path::new(&a.input).exists() {
            p
        } else {
            PathBuf::from(p.file_name().unwrap_or_default())
        }
    });
    anyhow::ensure!(
        input.canonicalize().ok() != output.canonicalize().ok() || !output.exists(),
        "output would overwrite the input"
    );
    let t0 = Instant::now();
    eprintln!("converting {} -> {} ...", input.display(), output.display());
    let g = GgufFile::open(&input)?;
    let mut assets = ModelAssets::from_gguf(&g)?;
    if let Some(p) = &a.tokenizer {
        assets.tokenizer = tokenizers::Tokenizer::from_file(p)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", p.display()))?;
    }
    let arch = format::convert_gguf(&g, &output, &assets)?;
    let size = std::fs::metadata(&output)?.len();
    println!(
        "{} ({arch}) -> {} ({:.1} MB) in {:.1}s, verified bit-identical",
        input.display(),
        output.display(),
        size as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// --- bench -------------------------------------------------------------------

pub fn bench(a: BenchArgs) -> anyhow::Result<()> {
    init_threads(a.m.threads)?;
    let resolved = source::resolve(&a.m.model)?;
    let t0 = Instant::now();
    let model = match resolved.kind {
        ModelKind::Bonsai => {
            std::sync::Arc::from(xor_net::models::q1::load_causal_lm(&resolved.path)?)
        }
        ModelKind::Llama => {
            let opts = LoadOptions::default();
            let m = xor_net::LlamaCausalLM::from_local(&resolved.path, opts.quantization)?;
            std::sync::Arc::new(m) as std::sync::Arc<dyn xor_net::CausalLM>
        }
    };
    let load = t0.elapsed().as_secs_f64();
    anyhow::ensure!(
        a.prompt_len + a.gen_tokens <= model.max_seq_len(),
        "prompt-len + gen exceeds the context ({})",
        model.max_seq_len()
    );
    let vocab = model.vocab_size() as u32;
    let prompt: Vec<u32> = (0..a.prompt_len as u32)
        .map(|i| (i.wrapping_mul(7919) + 13) % vocab.clamp(1, 100_000))
        .collect();
    let (mut best_pf, mut best_dec) = (0.0f64, 0.0f64);
    for _ in 0..a.reps {
        let mut st = model.new_state()?;
        let t = Instant::now();
        let mut logits = model.forward(&prompt, 0, st.as_mut())?;
        best_pf = best_pf.max(a.prompt_len as f64 / t.elapsed().as_secs_f64());
        let t = Instant::now();
        for i in 0..a.gen_tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map_or(0, |(i, _)| i as u32);
            logits = model.forward(&[next], a.prompt_len + i, st.as_mut())?;
        }
        if a.gen_tokens > 0 {
            best_dec = best_dec.max(a.gen_tokens as f64 / t.elapsed().as_secs_f64());
        }
    }
    println!(
        "{}: threads {} | load {load:.2}s | prefill {} tok: {best_pf:.1} tok/s | decode {} tok: {best_dec:.1} tok/s | peak RSS {}",
        resolved.path.display(),
        rayon::current_num_threads(),
        a.prompt_len,
        a.gen_tokens,
        fmt_rss()
    );
    Ok(())
}

// --- info --------------------------------------------------------------------

fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.1} MB", b / 1e6)
    }
}

fn human_params(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.2} B", n / 1e9)
    } else {
        format!("{:.0} M", n / 1e6)
    }
}

/// Facts about a model, gathered without loading its weights.
#[derive(Default)]
struct Info {
    rows: Vec<(&'static str, String)>,
    template: Option<String>,
}

impl Info {
    fn row(&mut self, k: &'static str, v: impl Into<String>) {
        self.rows.push((k, v.into()));
    }
}

fn template_row(t: Option<&str>) -> String {
    match t {
        None => "none (a generic fallback is used for chat)".into(),
        Some(t) => {
            let mut s = format!("yes ({} chars", t.len());
            if t.contains("enable_thinking") {
                s += ", reads enable_thinking";
            } else if t.contains("<think>") {
                s += ", <think> blocks, no enable_thinking switch";
            }
            s + ")"
        }
    }
}

fn arch_rows(info: &mut Info, arch: &str, cfg: &Value) {
    let u = |k: &str| cfg.get(k).and_then(Value::as_u64);
    let desc = match arch {
        "qwen3" => "qwen3 (dense transformer)",
        "qwen35" => "qwen35 (hybrid: gated attention + Gated DeltaNet)",
        other => other,
    };
    info.row("architecture", desc);
    if let Some(l) = u("layers") {
        let extra = cfg
            .get("recurrent")
            .and_then(Value::as_array)
            .map(|r| {
                let n = r.iter().filter(|v| v.as_bool() == Some(true)).count();
                format!(" ({} attention, {n} DeltaNet)", r.len() - n)
            })
            .unwrap_or_default();
        info.row("layers", format!("{l}{extra}"));
    }
    if let Some(h) = u("hidden") {
        info.row("hidden size", h.to_string());
    }
    if let (Some(h), Some(kv)) = (u("heads"), u("kv_heads")) {
        info.row(
            "attention",
            format!(
                "{h} heads, {kv} kv heads, head_dim {}",
                u("head_dim").unwrap_or(0)
            ),
        );
    }
    if let Some(c) = u("context_length") {
        info.row("context", format!("{c} tokens"));
    }
    if let Some(v) = u("vocab") {
        info.row("vocab", v.to_string());
    }
}

fn info_xq1(path: &Path, info: &mut Info) -> anyhow::Result<()> {
    let r = format::XqReader::open(path)?;
    info.row("format", "xq1 v1 (lossless Q1_0, zero-copy mmap)");
    arch_rows(info, r.arch(), &r.config::<Value>()?);
    info.row("parameters", human_params(r.param_count()));
    info.row(
        "quantization",
        match r.q1_block() {
            Some(b) => format!("1-bit Q1_0: sign bits + one f16 scale per {b} weights"),
            None => "unquantized".into(),
        },
    );
    info.row(
        "tokenizer",
        if r.has_tokenizer() {
            "embedded"
        } else {
            "none"
        },
    );
    info.row("chat template", template_row(r.chat_template()));
    info.template = r.chat_template().map(str::to_owned);
    Ok(())
}

fn gguf_dtype_name(d: u32) -> String {
    match d {
        0 => "F32".into(),
        1 => "F16".into(),
        30 => "BF16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        8 => "Q8_0".into(),
        41 => "Q1_0 (g128)".into(),
        42 => "Q1_0 (g32)".into(),
        43 => "Q1_0_g128".into(),
        other => format!("type {other}"),
    }
}

fn info_gguf(path: &Path, info: &mut Info) -> anyhow::Result<()> {
    let g = GgufFile::open(path)?;
    let arch = g.architecture()?;
    let meta_s = |k: &str| {
        g.get_metadata(k)
            .and_then(|v| v.to_string())
            .map(str::to_owned)
    };
    info.row("format", "GGUF");
    if let Some(n) = meta_s("general.name") {
        info.row("name", n);
    }
    let cfg = match arch.as_str() {
        "qwen3" => serde_json::to_value(xor_net::Qwen3Config::from_gguf(&g)?)?,
        "qwen35" => serde_json::to_value(xor_net::Qwen35Config::from_gguf(&g)?)?,
        _ => Value::Null,
    };
    arch_rows(info, &arch, &cfg);
    if cfg.is_null() {
        info.row("supported", "no (xor-net runs GGUF archs qwen3 / qwen35)");
    }
    let tensors = &g.metadata.tensor_infos;
    let params: u64 = tensors.values().map(|t| t.elem_count as u64).sum();
    info.row("parameters", human_params(params));
    let mut by_type: std::collections::BTreeMap<String, u64> = Default::default();
    for t in tensors.values() {
        *by_type.entry(gguf_dtype_name(t.dtype_u32)).or_default() += t.elem_count as u64;
    }
    let mut q: Vec<_> = by_type.into_iter().collect();
    q.sort_by(|a, b| b.1.cmp(&a.1));
    info.row(
        "quantization",
        q.iter()
            .map(|(k, n)| format!("{k} {:.1}%", 100.0 * *n as f64 / params.max(1) as f64))
            .collect::<Vec<_>>()
            .join(", "),
    );
    info.row(
        "tokenizer",
        if g.get_metadata("tokenizer.ggml.tokens").is_some() {
            "embedded"
        } else {
            "none"
        },
    );
    let t = meta_s("tokenizer.chat_template");
    info.row("chat template", template_row(t.as_deref()));
    info.template = t;
    Ok(())
}

fn info_llama(dir: &Path, info: &mut Info) -> anyhow::Result<()> {
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?,
    )?;
    let u = |k: &str| cfg.get(k).and_then(Value::as_u64);
    let s = |k: &str| {
        cfg.get(k)
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string()
    };
    info.row(
        "format",
        "HF safetensors directory (ternary-packed at load)",
    );
    let arch_list = cfg
        .get("architectures")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    info.row("architecture", format!("{} ({arch_list})", s("model_type")));
    if let Some(l) = u("num_hidden_layers") {
        info.row("layers", l.to_string());
    }
    if let Some(h) = u("hidden_size") {
        info.row("hidden size", h.to_string());
    }
    if let Some(h) = u("num_attention_heads") {
        info.row(
            "attention",
            format!(
                "{h} heads, {} kv heads",
                u("num_key_value_heads").unwrap_or(h)
            ),
        );
    }
    if let Some(c) = u("max_position_embeddings") {
        info.row("context", format!("{c} tokens"));
    }
    if let Some(v) = u("vocab_size") {
        info.row("vocab", v.to_string());
    }
    info.row(
        "quantization",
        "1.58-bit ternary (BitNet b1.58), packed to Pack4 + Int4 LM head at load",
    );
    let has_tok = dir.join("tokenizer.json").is_file();
    info.row(
        "tokenizer",
        if has_tok {
            "tokenizer.json"
        } else {
            "missing (fetched from microsoft/bitnet-b1.58-2B-4T at load)"
        },
    );
    let tmpl = std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| {
            v.get("chat_template")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());
    info.row("chat template", template_row(tmpl.as_deref()));
    info.template = tmpl;
    Ok(())
}

fn dir_size(p: &Path) -> u64 {
    if p.is_file() {
        return std::fs::metadata(p).map_or(0, |m| m.len());
    }
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

pub fn info(a: InfoArgs) -> anyhow::Result<()> {
    let resolved = source::resolve(&a.model)?;
    let mut info = Info::default();
    info.row("model", resolved.path.display().to_string());
    match resolved.kind {
        ModelKind::Bonsai if format::is_xq1(&resolved.path)? => {
            info_xq1(&resolved.path, &mut info)?
        }
        ModelKind::Bonsai => info_gguf(&resolved.path, &mut info)?,
        ModelKind::Llama => info_llama(&resolved.path, &mut info)?,
    }
    info.row("size on disk", human_bytes(dir_size(&resolved.path)));
    for (k, v) in &info.rows {
        println!("{k:<14} {v}");
    }
    if a.template {
        match &info.template {
            Some(t) => println!("\n--- chat template ---\n{t}"),
            None => println!("\n(no chat template)"),
        }
    }
    Ok(())
}

// --- classify ----------------------------------------------------------------

pub fn classify(a: ClassifyArgs) -> anyhow::Result<()> {
    if a.threads > 0 {
        init_threads(a.threads)?;
    }
    anyhow::ensure!(
        a.model.is_file(),
        "vision model {} not found (pass -m, or set XORNET_VISION_MODEL; QuickNet is \
         produced by scripts/vision/convert_larq_zoo.py)",
        a.model.display()
    );
    let t = Instant::now();
    let model = xor_net::vision::ImageClassifier::load(&a.model)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "{} ({} classes) loaded in {load_ms:.1} ms",
        model.name(),
        model.num_classes()
    );
    for path in &a.images {
        let img = image::open(path)
            .with_context(|| format!("decoding {}", path.display()))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.into_raw();
        let t = Instant::now();
        let preds = model.predict(&rgb, w, h, a.k)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if a.images.len() > 1 {
            println!("{}  ({ms:.1} ms)", path.display());
        } else {
            eprintln!("{} ({w}x{h}) in {ms:.1} ms", path.display());
        }
        for p in preds {
            println!("{:>6.2}%  {}", 100.0 * p.probability, p.label);
        }
    }
    Ok(())
}

// --- pull / serve ------------------------------------------------------------

pub fn pull(a: PullArgs) -> anyhow::Result<()> {
    let r = source::resolve(&a.model)?;
    println!("{}", r.path.display());
    Ok(())
}

pub fn serve(args: Vec<String>) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let name = format!("xor-net-server{}", std::env::consts::EXE_SUFFIX);
    let sibling = exe.with_file_name(&name);
    let prog = if sibling.is_file() {
        sibling
    } else {
        PathBuf::from(&name)
    };
    let status = std::process::Command::new(&prog).args(&args).status();
    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => anyhow::bail!(
            "could not start {} ({e}). Build it with `cargo build --release --bin xor-net-server` \
             or install it with `cargo install xor-net --features cli`.",
            prog.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xq1_default_name() {
        assert_eq!(
            default_xq1_path(Path::new("m/Bonsai-8B-Q1_0.gguf")),
            PathBuf::from("m/Bonsai-8B.xq1")
        );
        assert_eq!(
            default_xq1_path(Path::new("x.gguf")),
            PathBuf::from("x.xq1")
        );
    }

    #[test]
    fn humanize() {
        assert_eq!(human_params(1_720_000_000), "1.72 B");
        assert_eq!(human_params(400_000_000), "400 M");
        assert_eq!(human_bytes(248_664_203), "248.7 MB");
    }
}
