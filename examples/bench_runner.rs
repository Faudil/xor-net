//! `bench_runner`  single-config benchmark worker for the xor-net matrix driver.
//!
//! One invocation benchmarks exactly ONE configuration (quant format, LM-head
//! precision, weight format, thread count, single vs concurrent mode) and emits
//! a single JSON object recording everything: system metadata, the full config,
//! per-phase profiler deltas, memory, and the result. The driver
//! (`scripts/bench_all.sh`) loops over the matrix and re-invokes this binary
//! once per combo, because `init_threads` installs a GLOBAL rayon pool that can
//! only be set once per process.
//!
//! All knobs are environment variables (see README of the script). Output:
//!   * human summary  -> stderr
//!   * one JSON line  -> stdout (and appended to $XORNET_LOG if set)

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use rayon::ThreadPoolBuilder;
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use xor_net::bit1_58::quantization::TernaryPackType;
use xor_net::models::llama::{Cache, Config, Llama, LlamaEosToks};
use xor_net::nn::LmHeadConfig;
use xor_net::sampler::Sampler;
use xor_net::{AutoModelForCausalLM, QuantizationConfig};

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

fn env_usize(key: &str, default: usize) -> usize {
    env_str(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn build_quant(quant: &str, lmhead: LmHeadConfig) -> QuantizationConfig {
    match quant {
        "none" => QuantizationConfig::None,
        "bit1" => QuantizationConfig::Bit1(lmhead),
        "bit1_58_pack4" => QuantizationConfig::Bit1_58(TernaryPackType::Pack4, lmhead, false),
        "bit1_58_pack5" => QuantizationConfig::Bit1_58(TernaryPackType::Pack5, lmhead, false),
        _ => QuantizationConfig::Bit1_58(TernaryPackType::Pack4, lmhead, false),
    }
}

fn effective_lmhead(quant: &str, requested: &str) -> String {
    if quant == "none" {
        "f32".to_string()
    } else {
        requested.to_string()
    }
}

fn parse_lmhead(s: &str) -> LmHeadConfig {
    match s {
        "f16" => LmHeadConfig::F16,
        "int8" => LmHeadConfig::Int8,
        "int4" => LmHeadConfig::Int4,
        "ternary" => LmHeadConfig::Ternary,
        _ => LmHeadConfig::F32,
    }
}

fn load_tokenizer(model_path: &str, tok_repo: &str) -> anyhow::Result<Tokenizer> {
    let local = Path::new(model_path).join("tokenizer.json");
    if local.exists() {
        return Tokenizer::from_file(local).map_err(|e| anyhow::anyhow!(e.to_string()));
    }
    let api = Api::new()?;
    let repo = api.repo(Repo::new(tok_repo.to_string(), RepoType::Model));
    let p = repo.get("tokenizer.json")?;
    Tokenizer::from_file(p).map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn eos_list(cfg: &Config) -> Vec<u32> {
    match &cfg.eos_token_id {
        Some(LlamaEosToks::Single(t)) => vec![*t],
        Some(LlamaEosToks::Multiple(v)) => v.clone(),
        None => vec![2],
    }
}

struct Prof {
    blocks: u64,
    lm_head: u64,
    other: u64,
    attn: u64,
    mlp: u64,
    norm: u64,
    silu: u64,
}

fn snap() -> Prof {
    use xor_net::models::llama::*;
    Prof {
        blocks: TIME_BLOCKS.load(Ordering::Relaxed),
        lm_head: TIME_LM_HEAD.load(Ordering::Relaxed),
        other: TIME_OTHER.load(Ordering::Relaxed),
        attn: TIME_ATTN_GEMV.load(Ordering::Relaxed),
        mlp: TIME_MLP_GEMV.load(Ordering::Relaxed),
        norm: TIME_NORM.load(Ordering::Relaxed),
        silu: get_mlp_silu_time(),
    }
}

fn delta(a: &Prof, b: &Prof) -> Prof {
    Prof {
        blocks: b.blocks.saturating_sub(a.blocks),
        lm_head: b.lm_head.saturating_sub(a.lm_head),
        other: b.other.saturating_sub(a.other),
        attn: b.attn.saturating_sub(a.attn),
        mlp: b.mlp.saturating_sub(a.mlp),
        norm: b.norm.saturating_sub(a.norm),
        silu: b.silu.saturating_sub(a.silu),
    }
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1).map(|v| v.trim().to_string()))
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn target_features() -> Vec<String> {
    let mut v = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            v.push("avx2".into());
        }
        if std::is_x86_feature_detected!("avx512f") {
            v.push("avx512f".into());
        }
        if std::is_x86_feature_detected!("avx512bw") {
            v.push("avx512bw".into());
        }
        if std::is_x86_feature_detected!("avx512vnni") {
            v.push("avx512vnni".into());
        }
        if std::is_x86_feature_detected!("avx512vbmi") {
            v.push("avx512vbmi".into());
        }
    }
    v
}

fn git_sha() -> String {
    if let Some(s) = env_str("XORNET_GIT_SHA") {
        return s;
    }
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn mem_kb() -> (u64, u64) {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        let mut rss = 0u64;
        let mut hwm = 0u64;
        for l in s.lines() {
            if let Some(rest) = l.strip_prefix("VmRSS:") {
                rss = rest
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
            }
            if let Some(rest) = l.strip_prefix("VmHWM:") {
                hwm = rest
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
            }
        }
        (rss, hwm)
    } else {
        (0, 0)
    }
}

struct SingleResult {
    tokens_generated: usize,
    prefill_ms: f64,
    decode_ms: f64,
    prompt_len: usize,
    eos_hit: bool,
    prof: Prof,
    tail: String,
}

fn run_single(
    model: &Llama,
    cfg: &Config,
    tokenizer: &Tokenizer,
    prompt: &str,
    tokens: usize,
    seed: u64,
) -> anyhow::Result<SingleResult> {
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    let eos = eos_list(cfg);

    let warmup = env_usize("XORNET_WARMUP", 0);
    if warmup > 0 {
        let mut cache = Cache::new(true, cfg)?;
        let mut toks = prompt_ids.clone();
        let mut sampler = Sampler::new(seed, None, None, 1.0);
        let mut idx = 0usize;
        for _ in 0..warmup {
            let ctx = if idx == 0 { toks.len() } else { 1 };
            let start = toks.len().saturating_sub(ctx);
            let logits = model.forward(&toks[start..], idx, &mut cache)?;
            let mut ld = logits.into_data();
            let nt = sampler.sample(&mut ld, &toks)?;
            toks.push(nt);
            idx += ctx;
            if eos.contains(&nt) {
                break;
            }
        }
    }

    let mut cache = Cache::new(true, cfg)?;
    let mut toks = prompt_ids.clone();
    let mut sampler = Sampler::new(seed, None, None, 1.0);
    let init_len = toks.len();
    let p0 = snap();

    let t_prefill = Instant::now();
    let logits = model.forward(&toks[..], 0, &mut cache)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let prompt_len = toks.len();

    let mut ld = logits.into_data();
    let mut nt = sampler.sample(&mut ld, &toks)?;
    toks.push(nt);

    let mut idx = prompt_len;
    let t_decode = Instant::now();
    let mut gen_n = 1usize;
    let mut eos_hit = eos.contains(&nt);
    for _ in 1..tokens {
        let logits = model.forward(&[nt], idx, &mut cache)?;
        let mut ld = logits.into_data();
        nt = sampler.sample(&mut ld, &toks)?;
        toks.push(nt);
        idx += 1;
        gen_n += 1;
        if eos.contains(&nt) {
            eos_hit = true;
            break;
        }
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let p1 = snap();

    let tail = tokenizer
        .decode(&toks[init_len..], true)
        .unwrap_or_default();

    Ok(SingleResult {
        tokens_generated: gen_n,
        prefill_ms,
        decode_ms,
        prompt_len,
        eos_hit,
        prof: delta(&p0, &p1),
        tail,
    })
}

struct ConcurrentResult {
    total_gen: u64,
    wall_ms: f64,
    per_stream: Vec<f64>,
    prof: Prof,
}

fn run_stream(
    model: &Llama,
    cfg: &Config,
    prompt_ids: &[u32],
    tokens: usize,
    eos: Vec<u32>,
    seed: u64,
) -> (usize, f64) {
    let mut cache = Cache::new(true, cfg).expect("cache");
    let mut sampler = Sampler::new(seed, None, None, 1.0);
    let mut toks = prompt_ids.to_vec();
    let mut idx = 0usize;
    let init = toks.len();
    let t = Instant::now();
    for _ in 0..tokens {
        let ctx = if idx == 0 { toks.len() } else { 1 };
        let start = toks.len().saturating_sub(ctx);
        let logits = model
            .forward(&toks[start..], idx, &mut cache)
            .expect("forward");
        let mut ld = logits.into_data();
        let nt = sampler.sample(&mut ld, &toks).expect("sample");
        toks.push(nt);
        idx += ctx;
        if eos.contains(&nt) {
            break;
        }
    }
    let ng = toks.len() - init;
    let tps = ng as f64 / t.elapsed().as_secs_f64();
    (ng, tps)
}

#[allow(clippy::too_many_arguments)] // benchmark driver: one knob per argument
fn run_concurrent(
    model: Arc<Llama>,
    cfg: Arc<Config>,
    tokenizer: Arc<Tokenizer>,
    prompt: &str,
    tokens: usize,
    streams: usize,
    pool_threads: usize,
    seed_base: u64,
) -> ConcurrentResult {
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let prompt_ids = Arc::new(prompt_ids);
    let eos = eos_list(&cfg);

    let pools: Vec<Arc<rayon::ThreadPool>> = if pool_threads > 0 {
        (0..streams)
            .map(|_| {
                Arc::new(
                    ThreadPoolBuilder::new()
                        .num_threads(pool_threads)
                        .build()
                        .expect("pool"),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let p0 = snap();
    let wall = Instant::now();
    let mut handles = Vec::with_capacity(streams);
    for s in 0..streams {
        let m = Arc::clone(&model);
        let c = Arc::clone(&cfg);
        let p = Arc::clone(&prompt_ids);
        let pool = pools.get(s).cloned();
        let eos = eos.clone();
        handles.push(std::thread::spawn(move || {
            let run = || run_stream(&m, &c, &p, tokens, eos, seed_base + s as u64);
            match pool {
                Some(p) => p.install(run),
                None => run(),
            }
        }));
    }

    let mut total_gen = 0u64;
    let mut per_stream = Vec::with_capacity(streams);
    for h in handles {
        let (ng, tps) = h.join().expect("join");
        total_gen += ng as u64;
        per_stream.push(tps);
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let p1 = snap();

    ConcurrentResult {
        total_gen,
        wall_ms,
        per_stream,
        prof: delta(&p0, &p1),
    }
}

fn main() -> anyhow::Result<()> {
    let quant = env_str("XORNET_QUANT").unwrap_or_else(|| "bit1_58_pack4".into());
    let lmhead_req = env_str("XORNET_LMHEAD").unwrap_or_else(|| "int4".into());
    let lmhead = parse_lmhead(&lmhead_req);
    let weight_fmt = env_str("XORNET_WEIGHT_FMT").unwrap_or_else(|| "dense".into());
    if weight_fmt == "sparse" {
        unsafe {
            std::env::set_var("XORNET_WEIGHT_FMT", "sparse");
        }
    }
    let threads = env_usize("XORNET_THREADS", 0);
    let mode = env_str("XORNET_MODE").unwrap_or_else(|| "single".into());
    let streams = env_usize("XORNET_STREAMS", 4);
    let pool_threads = env_usize("XORNET_POOL_THREADS", 0);
    let tokens = env_usize("XORNET_TOKENS", 128);
    let prompt =
        env_str("XORNET_PROMPT").unwrap_or_else(|| "The history of the Roman Empire is".into());
    let model_path = env_str("XORNET_MODEL").unwrap_or_else(|| "models/bitnet-2b".into());
    let tok_repo =
        env_str("XORNET_TOKENIZER_REPO").unwrap_or_else(|| "microsoft/bitnet-b1.58-2B-4T".into());
    let seed_base = env_usize("XORNET_SEED", 299792458) as u64;

    let _ = xor_net::init_threads(threads);
    let qcfg = build_quant(&quant, lmhead);

    eprintln!(
        "[bench_runner] model={} quant={} lmhead={} fmt={} threads={} mode={} tokens={}",
        model_path, quant, lmhead_req, weight_fmt, threads, mode, tokens
    );

    let load_start = Instant::now();
    let (model, config) = if Path::new(&model_path).is_dir() {
        AutoModelForCausalLM::from_local(Path::new(&model_path), qcfg)?
    } else {
        AutoModelForCausalLM::from_pretrained(&model_path, qcfg)?
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[bench_runner] loaded in {:.1} ms", load_ms);

    let tokenizer = Arc::new(load_tokenizer(&model_path, &tok_repo)?);
    let cfg = Arc::new(config);
    let model = Arc::new(model);

    let mut obj = Map::new();
    let run_id = env_str("XORNET_RUN_ID").unwrap_or_else(|| {
        format!(
            "{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    });
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| {
            std::process::Command::new("date")
                .arg("-u")
                .arg("+%Y-%m-%dT%H:%M:%S.%3NZ")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| format!("{}.{:03}Z", d.as_secs(), d.subsec_millis()))
        })
        .unwrap_or_else(|| "unknown".to_string());

    obj.insert("run_id".into(), json!(run_id));
    obj.insert("timestamp".into(), json!(timestamp));
    obj.insert("git_sha".into(), json!(git_sha()));
    obj.insert(
        "rustc".into(),
        json!(env_str("XORNET_RUSTC").unwrap_or_default()),
    );
    obj.insert(
        "features".into(),
        json!(env_str("XORNET_FEATURES").unwrap_or_default()),
    );
    obj.insert("cpu_model".into(), json!(cpu_model()));
    obj.insert("target_features".into(), json!(target_features()));
    obj.insert("model".into(), json!(model_path));
    obj.insert("quant".into(), json!(quant));
    obj.insert(
        "lm_head".into(),
        json!(effective_lmhead(&quant, &lmhead_req)),
    );
    obj.insert("lm_head_requested".into(), json!(lmhead_req));
    obj.insert("weight_fmt".into(), json!(weight_fmt));
    obj.insert("threads".into(), json!(threads));
    obj.insert("mode".into(), json!(mode));
    obj.insert("streams".into(), json!(streams));
    obj.insert("pool_threads".into(), json!(pool_threads));
    obj.insert("prompt".into(), json!(prompt));
    obj.insert("tokens_requested".into(), json!(tokens));
    obj.insert("load_ms".into(), json!(load_ms));
    obj.insert(
        "weight_bytes_per_token".into(),
        json!(model.weight_bytes_per_token()),
    );
    let (rss, hwm) = mem_kb();
    obj.insert("rss_kb".into(), json!(rss));
    obj.insert("vmhwm_kb".into(), json!(hwm));

    if mode == "concurrent" {
        let r = run_concurrent(
            Arc::clone(&model),
            Arc::clone(&cfg),
            Arc::clone(&tokenizer),
            &prompt,
            tokens,
            streams,
            pool_threads,
            seed_base,
        );
        let agg = r.total_gen as f64 / (r.wall_ms / 1000.0);
        let avg = if r.per_stream.is_empty() {
            0.0
        } else {
            r.per_stream.iter().sum::<f64>() / r.per_stream.len() as f64
        };
        obj.insert("aggregate_tps".into(), json!(agg));
        obj.insert("wall_ms".into(), json!(r.wall_ms));
        obj.insert("total_gen".into(), json!(r.total_gen));
        obj.insert("per_stream_tps".into(), json!(r.per_stream));
        obj.insert("avg_per_stream_tps".into(), json!(avg));
        obj.insert("prefill_ms".into(), Value::Null);
        obj.insert("decode_ms".into(), Value::Null);
        obj.insert("prefill_tps".into(), Value::Null);
        obj.insert("decode_tps".into(), Value::Null);
        obj.insert("overall_tps".into(), Value::Null);
        obj.insert("tokens_generated".into(), Value::Null);
        obj.insert("eos".into(), Value::Null);
        obj.insert("output_tail".into(), Value::Null);
        obj.insert("blocks_ms".into(), json!(r.prof.blocks as f64 / 1e6));
        obj.insert("lm_head_ms".into(), json!(r.prof.lm_head as f64 / 1e6));
        obj.insert("other_ms".into(), json!(r.prof.other as f64 / 1e6));
        obj.insert("attn_ms".into(), json!(r.prof.attn as f64 / 1e6));
        obj.insert("mlp_ms".into(), json!(r.prof.mlp as f64 / 1e6));
        obj.insert("norm_ms".into(), json!(r.prof.norm as f64 / 1e6));
        obj.insert("silu_ms".into(), json!(r.prof.silu as f64 / 1e6));
        eprintln!(
            "[bench_runner] aggregate={:.2} tok/s | streams={} total_gen={} wall={:.1?} avg/stream={:.2}",
            agg,
            streams,
            r.total_gen,
            std::time::Duration::from_secs_f64(r.wall_ms / 1000.0),
            avg
        );
    } else {
        let r = run_single(&model, &cfg, &tokenizer, &prompt, tokens, seed_base)?;
        let prefill_tps = r.prompt_len as f64 / (r.prefill_ms / 1000.0);
        let decode_tps = (r.tokens_generated as f64 - 1.0) / (r.decode_ms / 1000.0);
        let overall_tps = r.tokens_generated as f64 / ((r.prefill_ms + r.decode_ms) / 1000.0);
        obj.insert("prefill_ms".into(), json!(r.prefill_ms));
        obj.insert("decode_ms".into(), json!(r.decode_ms));
        obj.insert("prefill_tps".into(), json!(prefill_tps));
        obj.insert("decode_tps".into(), json!(decode_tps));
        obj.insert("overall_tps".into(), json!(overall_tps));
        obj.insert("tokens_generated".into(), json!(r.tokens_generated));
        obj.insert("eos".into(), json!(r.eos_hit));
        obj.insert("output_tail".into(), json!(r.tail));
        obj.insert("aggregate_tps".into(), Value::Null);
        obj.insert("wall_ms".into(), Value::Null);
        obj.insert("total_gen".into(), Value::Null);
        obj.insert("per_stream_tps".into(), Value::Null);
        obj.insert("avg_per_stream_tps".into(), Value::Null);
        obj.insert("blocks_ms".into(), json!(r.prof.blocks as f64 / 1e6));
        obj.insert("lm_head_ms".into(), json!(r.prof.lm_head as f64 / 1e6));
        obj.insert("other_ms".into(), json!(r.prof.other as f64 / 1e6));
        obj.insert("attn_ms".into(), json!(r.prof.attn as f64 / 1e6));
        obj.insert("mlp_ms".into(), json!(r.prof.mlp as f64 / 1e6));
        obj.insert("norm_ms".into(), json!(r.prof.norm as f64 / 1e6));
        obj.insert("silu_ms".into(), json!(r.prof.silu as f64 / 1e6));
        eprintln!(
            "[bench_runner] prefill={:.2} tok/s decode={:.2} tok/s overall={:.2} tok/s ({} tok){}",
            prefill_tps,
            decode_tps,
            overall_tps,
            r.tokens_generated,
            if r.eos_hit { " [eos]" } else { "" }
        );
    }

    let json_line = Value::Object(obj).to_string();
    if let Some(log) = env_str("XORNET_LOG")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
    {
        let _ = f.write_all(json_line.as_bytes());
        let _ = f.write_all(b"\n");
    }
    println!("{}", json_line);
    Ok(())
}
