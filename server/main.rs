//! `xor-net-server`  a small OpenAI / llama.cpp-compatible HTTP
//! inference server for the BitNet engine.
//!
//! Design goals:
//!   * OpenAI-compatible surface: `GET /v1/models`, `POST /v1/completions`,
//!     `POST /v1/chat/completions` (streaming + non-streaming), plus the
//!     llama.cpp legacy `POST /completion`.
//!   * Maximise *aggregate* throughput on this 6C/12T Zen5 box. The engine
//!     decode GEMV is memory/cache-bound, so the lever is *physical cores*:
//!     we run `XORNET_SLOTS` worker threads (default = logical/2 = 6), each
//!     pinned to a dedicated core via its own 1-thread rayon pool. Up to
//!     `SLOTS` decodes run concurrently -> ~110 tok/s aggregate (vs ~59
//!     single-stream). See `OPTIMISATION.md`.
//!   * Hand-rolled HTTP/1.1 (no async framework) because the engine is
//!     synchronous/blocking; a thread-per-connection front-end feeds a shared
//!     job queue that the slot workers drain.
//!
//! Env knobs:
//!   XORNET_MODEL            model dir / hub id (default models/bitnet-2b)
//!   XORNET_SERVER_HOST     bind host       (default 127.0.0.1)
//!   XORNET_SERVER_PORT     bind port       (default 8080)
//!   XORNET_SLOTS          worker slots   (default = logical/2)
//!   XORNET_THREADS_PER_SLOT  rayon threads per slot (default 1 -> 110 tok/s)

use std::io::{BufRead, Write};
use std::net::{TcpListener, TcpStream};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use hf_hub::{api::sync::Api, Repo, RepoType};
use xor_net::models::llama::{Cache, Config, Llama};
use xor_net::nn::LmHeadConfig;
use xor_net::sampler::Sampler;
use xor_net::{
    AutoModelForCausalLM, QuantizationConfig, TernaryPackType,
};

const EOS: &[u32] = &[2, 128001, 128009];
const DEFAULT_MODEL_ID: &str = "bitnet-b1.58-2B-4T";

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GenParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    stream: bool,
    seed: u64,
}

enum GenEvent {
    Token { id: u32, text: String },
    Done { stopped: bool, completion_tokens: usize },
    Error(String),
}

struct Job {
    prompt_ids: Vec<u32>,
    params: GenParams,
    tx: mpsc::Sender<GenEvent>,
}

/// Job queue: `None` is the stop sentinel.
type Queue = Arc<(Mutex<VecDeque<Option<Job>>>, Condvar)>;

static REQ_ID: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Generation (one decode, runs inside a slot's 1-thread rayon pool)
// ---------------------------------------------------------------------------

fn generate(
    model: &Llama,
    config: &Config,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    params: &GenParams,
    tx: &mpsc::Sender<GenEvent>,
) {
    let mut cache = match Cache::new(true, config) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(GenEvent::Error(format!("cache: {e}")));
            return;
        }
    };

    let temperature = if params.temperature <= 0.0 {
        None
    } else {
        Some(params.temperature)
    };
    let top_p = if params.top_p <= 0.0 {
        None
    } else {
        Some(params.top_p)
    };
    let mut sampler = Sampler::new(
        params.seed,
        temperature,
        top_p,
        params.repetition_penalty,
    );

    let mut toks = prompt_ids.to_vec();
    let mut index_pos = 0usize;
    let init_len = toks.len();
    let mut ng = 0usize;
    let mut stopped = false;

    for _ in 0..params.max_tokens {
        let ctx = if index_pos == 0 { toks.len() } else { 1 };
        let start = toks.len().saturating_sub(ctx);
        let logits = match model.forward(&toks[start..], index_pos, &mut cache) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.send(GenEvent::Error(format!("forward: {e}")));
                return;
            }
        };
        let mut ld = logits.into_data();
        let nt = match sampler.sample(&mut ld, &toks) {
            Ok(n) => n,
            Err(e) => {
                let _ = tx.send(GenEvent::Error(format!("sample: {e}")));
                return;
            }
        };
        toks.push(nt);
        index_pos += ctx;
        ng += 1;

        let text = tokenizer
            .decode(&[nt], false)
            .unwrap_or_default();
        if tx.send(GenEvent::Token { id: nt, text }).is_err() {
            return; // client disconnected
        }
        if EOS.contains(&nt) {
            stopped = true;
            break;
        }
    }

    let _ = tx.send(GenEvent::Done {
        stopped,
        completion_tokens: ng,
    });
    let _ = init_len;
}

// ---------------------------------------------------------------------------
// Worker slots
// ---------------------------------------------------------------------------

fn worker(
    slot: usize,
    model: Arc<Llama>,
    config: Arc<Config>,
    tokenizer: Arc<Tokenizer>,
    queue: Queue,
    threads_per_slot: usize,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads_per_slot.max(1))
        .build()
        .expect("slot pool");

    loop {
        let job = {
            let (q, cv) = &*queue;
            let mut g = q.lock().unwrap();
            while g.is_empty() {
                g = cv.wait(g).unwrap();
            }
            g.pop_front().unwrap()
        };

        match job {
            None => break, // stop sentinel
            Some(Job {
                prompt_ids,
                params,
                tx,
            }) => {
                let _ = slot;
                pool.install(|| {
                    generate(
                        &model,
                        &config,
                        &tokenizer,
                        &prompt_ids,
                        &params,
                        &tx,
                    )
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP front-end
// ---------------------------------------------------------------------------

fn read_request(r: &mut impl BufRead) -> anyhow::Result<(String, Vec<String>, Vec<u8>)> {
    let mut request_line = String::new();
    r.read_line(&mut request_line)?;
    if request_line.is_empty() {
        return Err(anyhow!("empty request"));
    }

    let mut headers = Vec::new();
    let mut content_len: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 || line == "\r\n" {
            break;
        }
        if let Some(idx) = line.to_ascii_lowercase().find("content-length:") {
            if let Ok(v) = line[idx + 15..].trim().parse::<usize>() {
                content_len = Some(v);
            }
        }
        headers.push(line);
    }

    let body = match content_len {
        Some(cl) => {
            let mut b = vec![0u8; cl];
            r.read_exact(&mut b)?;
            b
        }
        None => Vec::new(),
    };

    Ok((request_line, headers, body))
}

fn build_prompt_ids(
    route: &str,
    v: &Value,
    tokenizer: &Tokenizer,
) -> anyhow::Result<Vec<u32>> {
    let prompt: String = if route == "chat" {
        // Minimal chat template: role: content, then an assistant prompt.
        let mut p = String::new();
        if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
            for m in messages {
                let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("user");
                let content = m.get("content").and_then(|x| x.as_str()).unwrap_or("");
                p.push_str(&format!("{role}: {content}\n"));
            }
        }
        p.push_str("assistant: ");
        p
    } else {
        // completions / llama.cpp: `prompt` is a string (or array of strings).
        match v.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => return Err(anyhow!("missing 'prompt'")),
        }
    };

    let ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    Ok(ids)
}

fn gen_params(v: &Value) -> GenParams {
    let max_tokens = v
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(256) as usize;
    let max_tokens = max_tokens.max(1).min(4096);
    let temperature = v
        .get("temperature")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0) as f32;
    let top_p = v
        .get("top_p")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0) as f32;
    let repetition_penalty = v
        .get("repetition_penalty")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0) as f32;
    let stream = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
        ^ REQ_ID.fetch_add(1, Ordering::Relaxed);
    GenParams {
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        stream,
        seed,
    }
}

fn sse(w: &mut impl Write, obj: &Value) {
    if let Ok(s) = serde_json::to_string(obj) {
        let _ = write!(w, "data: {s}\n\n");
        let _ = w.flush();
    }
}

fn handle(
    stream: TcpStream,
    model: Arc<Llama>,
    config: Arc<Config>,
    tokenizer: Arc<Tokenizer>,
    queue: Queue,
    model_id: &str,
) {
    let mut reader = std::io::BufReader::new(&stream);
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };

    let (request_line, _headers, body) = match read_request(&mut reader) {
        Ok(r) => r,
        Err(_) => return,
    };

    let mut parts = request_line.trim_end().split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("").split('?').next().unwrap_or("");
    let path = path.trim_start_matches('/');

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let req_id = format!("cmpl-{}", REQ_ID.fetch_add(1, Ordering::Relaxed));

    if method == "GET" && (path == "health" || path == "v1/health") {
        let _ = write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            json!({"status":"ok"}).to_string().len(),
            json!({"status":"ok"}).to_string()
        );
        return;
    }

    if method == "GET" && path == "v1/models" {
        let body = json!({
            "object": "list",
            "data": [{
                "id": model_id,
                "object": "model",
                "owned_by": "xor-net",
                "created": created,
            }],
        })
        .to_string();
        let _ = write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        return;
    }

    if method != "POST" || body.is_empty() {
        let _ = writer.write_all(
            b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 22\r\nConnection: close\r\n\r\n{\"error\":\"bad request\"}",
        );
        return;
    }

    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let b = json!({"error": format!("json: {e}")}).to_string();
            let _ = write!(
                writer,
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                b.len(),
                b
            );
            return;
        }
    };

    let route = if path == "v1/chat/completions" {
        "chat"
    } else {
        "completion"
    };

    let prompt_ids = match build_prompt_ids(route, &v, &tokenizer) {
        Ok(ids) => ids,
        Err(e) => {
            let b = json!({"error": e.to_string()}).to_string();
            let _ = write!(
                writer,
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                b.len(),
                b
            );
            return;
        }
    };

    let params = gen_params(&v);
    let prompt_tokens = prompt_ids.len();
    let (tx, rx) = mpsc::channel::<GenEvent>();

    {
        let (q, cv) = &*queue;
        let mut g = q.lock().unwrap();
        g.push_back(Some(Job {
            prompt_ids,
            params: params.clone(),
            tx,
        }));
        cv.notify_one();
    }

    if params.stream {
        let _ = write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );

        let mut full = String::new();
        let mut completion_tokens = 0usize;
        let mut stopped = false;
        for ev in rx {
            match ev {
                GenEvent::Token { text, .. } => {
                    full.push_str(&text);
                    if route == "chat" {
                        sse(
                            &mut writer,
                            &json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": text},
                                    "finish_reason": Value::Null,
                                }],
                            }),
                        );
                    } else {
                        sse(
                            &mut writer,
                            &json!({
                                "id": req_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_id,
                                "choices": [{
                                    "text": text,
                                    "index": 0,
                                    "finish_reason": Value::Null,
                                }],
                                "usage": Value::Null,
                            }),
                        );
                    }
                }
                GenEvent::Done {
                    stopped,
                    completion_tokens,
                } => {
                    let _ = write!(writer, "data: [DONE]\n\n");
                    break;
                }
                GenEvent::Error(e) => {
                    let _ = write!(writer, "data: {}\n\n", json!({"error": e}).to_string());
                    break;
                }
            }
        }
        let _ = stopped;
        let _ = completion_tokens;
        let _ = full;
    } else {
        let mut full = String::new();
        let mut completion_tokens = 0usize;
        let mut stopped = true;
        for ev in rx {
            match ev {
                GenEvent::Token { text, .. } => full.push_str(&text),
                GenEvent::Done {
                    stopped: s,
                    completion_tokens: c,
                } => {
                    stopped = s;
                    completion_tokens = c;
                    break;
                }
                GenEvent::Error(e) => {
                    let b = json!({"error": e}).to_string();
                    let _ = write!(
                        writer,
                        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        b.len(),
                        b
                    );
                    return;
                }
            }
        }

        let finish = if stopped { "stop" } else { "length" };
        let body = if route == "chat" {
            json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        } else {
            json!({
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": full,
                    "index": 0,
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        }
        .to_string();

        let _ = write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("XORNET_MODEL")
        .unwrap_or_else(|_| "models/bitnet-2b".to_string());
    let host = std::env::var("XORNET_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("XORNET_SERVER_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let slots: usize = std::env::var("XORNET_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(logical / 2);
    let slots = slots.max(1).min(logical);
    let threads_per_slot: usize = std::env::var("XORNET_THREADS_PER_SLOT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    xor_net::init_threads(slots).map_err(|e| anyhow!(e.to_string()))?;

    let quant = QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, true);
    let (model, config) = if std::path::Path::new(&model_path).is_dir() {
        println!("Loading from local directory: {}", model_path);
        AutoModelForCausalLM::from_local(std::path::Path::new(&model_path), quant)?
    } else {
        println!("Loading from HuggingFace: {}", model_path);
        AutoModelForCausalLM::from_pretrained(&model_path, quant)?
    };
    println!("Model loaded.");

    let api = Api::new().map_err(|e| anyhow!(e.to_string()))?;
    let repo = api.repo(Repo::new(
        "microsoft/bitnet-b1.58-2B-4T".to_string(),
        RepoType::Model,
    ));
    let tok_path = repo
        .get("tokenizer.json")
        .map_err(|e| anyhow!(e.to_string()))?;
    let tokenizer = Arc::new(
        Tokenizer::from_file(tok_path).map_err(|e| anyhow!(e.to_string()))?,
    );

    let model = Arc::new(model);
    let config = Arc::new(config);
    let queue: Queue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));

    for s in 0..slots {
        let m = Arc::clone(&model);
        let c = Arc::clone(&config);
        let tk = Arc::clone(&tokenizer);
        let q = Arc::clone(&queue);
        thread::spawn(move || worker(s, m, c, tk, q, threads_per_slot));
    }
    println!(
        "Server: {} worker slots, {} thread(s)/slot, listening on {}:{}",
        slots, threads_per_slot, host, port
    );

    let listener = TcpListener::bind((host.as_str(), port))?;
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let m = Arc::clone(&model);
                let c = Arc::clone(&config);
                let tk = Arc::clone(&tokenizer);
                let q = Arc::clone(&queue);
                let mid = model_id_for(&model_path);
                thread::spawn(move || handle(s, m, c, tk, q, &mid));
            }
            Err(_) => continue,
        }
    }

    // Stop sentinels (only reached on listener drop).
    let (q, cv) = &*queue;
    let mut g = q.lock().unwrap();
    for _ in 0..slots {
        g.push_back(None);
    }
    cv.notify_all();
    Ok(())
}

fn model_id_for(model_path: &str) -> String {
    if model_path.contains("2B") || model_path.contains("2b") {
        DEFAULT_MODEL_ID.to_string()
    } else {
        std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(DEFAULT_MODEL_ID)
            .to_string()
    }
}
