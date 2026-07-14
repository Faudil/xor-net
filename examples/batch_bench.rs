//! Concurrent multi-stream serving benchmark.
//!
//! Loads the model once and serves `N` independent generation streams in
//! parallel, each with its own KV cache. This measures *aggregate* throughput
//! (tokens/sec across all streams) versus the single-stream figure, and tells
//! us whether the engine is compute-bound (aggregate flat ~= single-stream) or
//! has headroom from batching / concurrency.
//!
//! Two concurrency modes (see `XORNET_POOL_THREADS`):
//!   * shared global pool (default): all streams share one rayon pool of
//!     `XORNET_GLOBAL_THREADS` threads. Simple, but the matmul is
//!     cache/memory-bound so >~8 threads contend and aggregate plateaus.
//!   * per-stream pools: each stream gets its own rayon pool of
//!     `XORNET_POOL_THREADS` threads pinned (by OS scheduling) to a core
//!     subset. With `pool = cores / streams` each stream keeps the optimal
//!     thread count and aggregate scales ~linearly with streams.
//!
//! Env knobs:
//!   XORNET_STREAMS        number of concurrent streams   (default 4)
//!   XORNET_TOKENS         tokens generated per stream    (default 64)
//!   XORNET_PROMPT         prompt text                   (default a short sentence)
//!   XORNET_GLOBAL_THREADS rayon global pool size        (default 0 = auto ~8)
//!   XORNET_POOL_THREADS  if >0, per-stream pool size  (default 0 = shared)
//!   XORNET_MODEL          model dir / hub id            (default models/bitnet-2b)

use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use xor_net::models::llama::Config;
use xor_net::sampler::Sampler;
use xor_net::{
    AutoModelForCausalLM, Cache, Llama, QuantizationConfig, TernaryPackType,
};
use xor_net::nn::LmHeadConfig;

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const EOS: &[u32] = &[2, 128001, 128009];

fn main() -> anyhow::Result<()> {
    let streams: usize = std::env::var("XORNET_STREAMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let tokens: usize = std::env::var("XORNET_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let prompt = std::env::var("XORNET_PROMPT").unwrap_or_else(|_| {
        "The history of the Roman Empire is".to_string()
    });
    let gthreads: usize = std::env::var("XORNET_GLOBAL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let pool_threads: usize = std::env::var("XORNET_POOL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bitnet-2b".to_string());

    let _ = xor_net::init_threads(gthreads);

    let quantization = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true);

    let load_start = Instant::now();
    let (model, config): (Llama, Config) = if Path::new(&model_path).is_dir() {
        println!("Loading from local directory: {}", model_path);
        AutoModelForCausalLM::from_local(Path::new(&model_path), quantization)?
    } else {
        println!("Loading from HuggingFace: {}", model_path);
        AutoModelForCausalLM::from_pretrained(&model_path, quantization)?
    };
    println!("Model loaded in {:.2?}", load_start.elapsed());

    let api = Api::new().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let repo = api.repo(Repo::new(
        "microsoft/bitnet-b1.58-2B-4T".to_string(),
        RepoType::Model,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Arc::new(
        Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e.to_string()))?,
    );

    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();

    let model = Arc::new(model);
    let config = Arc::new(config);
    let prompt_ids = Arc::new(prompt_ids);

    // Per-stream rayon pools (when XORNET_POOL_THREADS > 0). Each pool is
    // built once and `instal`led around the stream's forward so its matmuls run
    // on a fixed core subset, keeping the optimal thread count per stream.
    let pools: Vec<Arc<rayon::ThreadPool>> = if pool_threads > 0 {
        (0..streams)
            .map(|_| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(pool_threads)
                    .build()
                    .ok()
                    .map(Arc::new)
            })
            .flatten()
            .collect()
    } else {
        Vec::new()
    };

    println!(
        "Serving {} streams x {} tokens | prompt='{}' ({} tok) | mode={}",
        streams,
        tokens,
        prompt,
        prompt_ids.len(),
        if pool_threads > 0 {
            format!("per-stream pools of {pool_threads}")
        } else {
            "shared global pool".to_string()
        }
    );

    let wall_start = Instant::now();
    let mut handles = Vec::with_capacity(streams);
    for s in 0..streams {
        let model = Arc::clone(&model);
        let cfg = Arc::clone(&config);
        let pids = Arc::clone(&prompt_ids);
        let tok = Arc::clone(&tokenizer);
        let pool: Option<Arc<rayon::ThreadPool>> = pools.get(s).cloned();
        handles.push(std::thread::spawn(move || {
            let run = || run_stream(&*model, &*cfg, &pids[..], &*tok, tokens, s as u64);
            match pool {
                Some(p) => p.install(run),
                None => run(),
            }
        }));
    }

    let mut total_gen = 0u64;
    let mut per_streams = Vec::with_capacity(streams);
    for h in handles {
        let res: (usize, f64, String) = h.join().expect("join");
        let (ng, per, text) = res;
        total_gen += ng as u64;
        per_streams.push(per);
        println!("  stream ng={} ({:.2} tok/s): {}", ng, per, text.replace('\n', " "));
    }
    let wall = wall_start.elapsed();

    let agg = total_gen as f64 / wall.as_secs_f64();
    let avg_per = per_streams.iter().sum::<f64>() / per_streams.len() as f64;
    println!("---");
    println!(
        "streams={} total_gen={} wall={:.2?} aggregate_tps={:.2} avg_per_stream_tps={:.2}",
        streams, total_gen, wall, agg, avg_per
    );
    Ok(())
}

// keep Duration referenced for potential tuning logs
#[allow(dead_code)]
fn _d(_: Duration) {}

/// Generate `tokens` tokens for one stream and return (generated, per-stream
/// tok/s, decoded tail). `s` only seeds the sampler so streams diverge.
fn run_stream(
    model: &Llama,
    cfg: &Config,
    pids: &[u32],
    tok: &Tokenizer,
    tokens: usize,
    s: u64,
) -> (usize, f64, String) {
    let mut cache = Cache::new(true, cfg).expect("cache");
    let mut sampler = Sampler::new(299792458u64 + s, None, None, 1.0);
    let mut tokens_vec = pids.to_vec();
    let mut index_pos = 0usize;
    let init_len = tokens_vec.len();
    let gen_start = Instant::now();
    for _ in 0..tokens {
        let ctx = if index_pos == 0 { tokens_vec.len() } else { 1 };
        let start_pos = tokens_vec.len().saturating_sub(ctx);
        let logits = model
            .forward(&tokens_vec[start_pos..], index_pos, &mut cache)
            .expect("forward");
        let mut ld = logits.into_data();
        let nt = sampler.sample(&mut ld, &tokens_vec).expect("sample");
        tokens_vec.push(nt);
        index_pos += ctx;
        if EOS.contains(&nt) {
            break;
        }
    }
    let ng = tokens_vec.len() - init_len;
    let per = ng as f64 / gen_start.elapsed().as_secs_f64();
    let text = tok.decode(&tokens_vec[init_len..], true).unwrap_or_default();
    (ng, per, text)
}
