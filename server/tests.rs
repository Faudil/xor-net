//! End-to-end tests of the HTTP server over loopback with a mock engine.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::config::ServerConfig;
use crate::engine::mock::MockEngine;
use crate::serve::{self, App};

fn test_cfg() -> ServerConfig {
    let mut c = ServerConfig::defaults();
    c.slots = 2;
    c.read_timeout = Duration::from_secs(5);
    c.idle_timeout = Duration::from_secs(2);
    c.write_timeout = Duration::from_secs(5);
    c.max_tokens_cap = 1000;
    c
}

fn start(engine: MockEngine, cfg: ServerConfig) -> (SocketAddr, Arc<MockEngine>, Arc<App>) {
    let engine = Arc::new(engine);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Arc::new(App::new(engine.clone(), cfg));
    serve::spawn_workers(&app).unwrap();
    let app2 = Arc::clone(&app);
    std::thread::spawn(move || serve::serve(listener, app2));
    (addr, engine, app)
}

fn connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Resp {
    fn header(&self, n: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(n))
            .map(|(_, v)| v.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.body))
    }
}

/// Read one response: Content-Length framed, or until EOF.
fn read_resp(r: &mut BufReader<TcpStream>) -> Resp {
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    let status: u16 = line.split(' ').nth(1).unwrap_or("0").parse().unwrap();
    let mut headers = Vec::new();
    loop {
        let mut l = String::new();
        r.read_line(&mut l).unwrap();
        let l = l.trim_end();
        if l.is_empty() {
            break;
        }
        let (k, v) = l.split_once(':').unwrap();
        headers.push((k.to_string(), v.trim().to_string()));
    }
    let cl = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.parse::<usize>().unwrap());
    let mut body = Vec::new();
    match cl {
        Some(n) => {
            body.resize(n, 0);
            r.read_exact(&mut body).unwrap();
        }
        None => {
            r.read_to_end(&mut body).unwrap();
        }
    }
    Resp {
        status,
        headers,
        body: String::from_utf8(body).unwrap(),
    }
}

fn raw(addr: SocketAddr, req: &[u8]) -> Resp {
    let mut s = connect(addr);
    s.write_all(req).unwrap();
    read_resp(&mut BufReader::new(s))
}

fn post(addr: SocketAddr, path: &str, body: &str) -> Resp {
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    raw(addr, req.as_bytes())
}

fn get(addr: SocketAddr, path: &str) -> Resp {
    raw(
        addr,
        format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
    )
}

/// Split an SSE body into `data:` payloads, asserting the framing.
fn sse_events(body: &str) -> Vec<String> {
    assert!(body.ends_with("\n\n"), "unterminated SSE body: {body:?}");
    body.split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| {
            e.strip_prefix("data: ")
                .unwrap_or_else(|| panic!("bad event {e:?}"))
                .to_string()
        })
        .collect()
}

#[test]
fn routing_404_405_options_health_models() {
    let (addr, _, _) = start(MockEngine::new("hi"), test_cfg());
    let r = get(addr, "/nope");
    assert_eq!(r.status, 404);
    assert_eq!(r.json()["error"]["code"], "not_found");
    let r = get(addr, "/v1/completions");
    assert_eq!(r.status, 405);
    assert_eq!(r.header("Allow"), Some("POST, OPTIONS"));
    let r = post(addr, "/health", "{}");
    assert_eq!(r.status, 405);
    let r = post(addr, "/v1/embeddings", "{}");
    assert_eq!(r.status, 404);
    let r = raw(
        addr,
        b"OPTIONS /v1/chat/completions HTTP/1.1\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(r.status, 204);
    assert!(r.header("Access-Control-Allow-Origin").is_none());
    let r = get(addr, "/health");
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["status"], "ok");
    let r = get(addr, "/v1/models");
    assert_eq!(r.json()["data"][0]["id"], "mock-model");
}

#[test]
fn cors_when_enabled() {
    let mut cfg = test_cfg();
    cfg.cors_origin = Some("*".into());
    let (addr, _, _) = start(MockEngine::new("hi"), cfg);
    let r = raw(
        addr,
        b"OPTIONS /v1/completions HTTP/1.1\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(r.status, 204);
    assert_eq!(r.header("Access-Control-Allow-Origin"), Some("*"));
    assert!(
        r.header("Access-Control-Allow-Methods")
            .unwrap()
            .contains("POST")
    );
    assert_eq!(
        get(addr, "/health").header("Access-Control-Allow-Origin"),
        Some("*")
    );
}

#[test]
fn completion_non_stream_usage_and_finish() {
    let (addr, _, _) = start(MockEngine::new("Hello wörld"), test_cfg());
    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":50}"#,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "Hello wörld");
    assert_eq!(v["choices"][0]["finish_reason"], "length"); // script exhausted
    assert_eq!(v["usage"]["prompt_tokens"], 4); // BOS + 3 bytes
    assert_eq!(v["usage"]["completion_tokens"], "Hello wörld".len());
    assert_eq!(v["usage"]["total_tokens"], 4 + "Hello wörld".len());

    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":3}"#,
    );
    let v = r.json();
    assert_eq!(v["choices"][0]["text"], "Hel");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 3);
}

#[test]
fn chat_eos_and_stop_strings() {
    let (addr, _, _) = start(MockEngine::new("abc STOP def\0ignored"), test_cfg());
    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}],"stop":["STOP"]}"#,
    );
    let v = r.json();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "abc ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"], 8);

    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}]}"#,
    );
    let v = r.json();
    assert_eq!(v["choices"][0]["message"]["content"], "abc STOP def");
    assert_eq!(v["choices"][0]["finish_reason"], "stop"); // EOS
    // usage: prompt = BOS + "user: x\nassistant: "
    assert_eq!(
        v["usage"]["prompt_tokens"],
        1 + "user: x\nassistant: ".len()
    );
    assert_eq!(v["usage"]["completion_tokens"], 13); // includes EOS
}

#[test]
fn chat_stream_framing_with_usage_and_multibyte() {
    let text = "€uro 🦀!";
    let (addr, _, _) = start(MockEngine::new(text), test_cfg());
    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.header("Content-Type"), Some("text/event-stream"));
    let ev = sse_events(&r.body);
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let mut content = String::new();
    for c in &chunks[..chunks.len() - 2] {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert!(c["usage"].is_null());
        let d = c["choices"][0]["delta"]["content"].as_str().unwrap();
        assert!(!d.contains('\u{FFFD}'));
        content.push_str(d);
    }
    assert_eq!(content, text);
    let fin = &chunks[chunks.len() - 2];
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
    let usage = &chunks[chunks.len() - 1];
    assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
    assert_eq!(usage["usage"]["completion_tokens"], text.len());
}

#[test]
fn completions_stream_without_usage() {
    let (addr, _, _) = start(MockEngine::new("ab"), test_cfg());
    let r = post(addr, "/v1/completions", r#"{"prompt":"x","stream":true}"#);
    let ev = sse_events(&r.body);
    assert_eq!(ev.len(), 4); // a, b, final, [DONE]
    let fin: Value = serde_json::from_str(&ev[2]).unwrap();
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
    assert!(fin.get("usage").is_none());
    assert_eq!(ev[3], "[DONE]");
}

#[test]
fn llama_cpp_completion() {
    let (addr, _, _) = start(MockEngine::new("hey you"), test_cfg());
    let r = post(addr, "/completion", r#"{"prompt":"x","n_predict":3}"#);
    let v = r.json();
    assert_eq!(v["content"], "hey");
    assert_eq!(v["stop"], true);
    assert_eq!(v["stop_type"], "limit");
    assert_eq!(v["tokens_predicted"], 3);
    let r = post(
        addr,
        "/completion",
        r#"{"prompt":"x","stream":true,"stop":[" y"]}"#,
    );
    let ev = sse_events(&r.body);
    assert_ne!(ev.last().unwrap(), "[DONE]");
    let last: Value = serde_json::from_str(ev.last().unwrap()).unwrap();
    assert_eq!(last["stop"], true);
    assert_eq!(last["stop_type"], "word");
    assert_eq!(last["stopping_word"], " y");
    let text: String = ev[..ev.len() - 1]
        .iter()
        .map(|e| {
            serde_json::from_str::<Value>(e).unwrap()["content"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(text, "hey");
}

#[test]
fn validation_errors_over_http() {
    let (addr, _, _) = start(MockEngine::new("x"), test_cfg());
    let r = post(addr, "/v1/completions", r#"{"prompt":"a","temperature":3}"#);
    assert_eq!(r.status, 400);
    let e = &r.json()["error"];
    assert_eq!(e["type"], "invalid_request_error");
    assert_eq!(e["param"], "temperature");
    assert!(e.get("code").is_some() && e.get("message").is_some());

    let r = post(addr, "/v1/completions", "{bad json");
    assert_eq!(r.status, 400);
    let r = post(addr, "/v1/completions", "");
    assert_eq!(r.status, 400);
    let r = post(addr, "/v1/completions", r#"{"prompt":"a","n":3}"#);
    assert_eq!(r.json()["error"]["param"], "n");
    let r = post(addr, "/v1/completions", r#"{"prompt":[999]}"#);
    assert_eq!(r.status, 400);

    // Context length (mock ctx = 64).
    let long = "a".repeat(100);
    let r = post(
        addr,
        "/v1/completions",
        &format!(r#"{{"prompt":"{long}"}}"#),
    );
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":61}"#,
    );
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"abc","max_tokens":60}"#,
    );
    assert_eq!(r.status, 200);
}

#[test]
fn oversized_and_malformed_requests_dont_kill_server() {
    let mut cfg = test_cfg();
    cfg.max_body_bytes = 1024;
    let (addr, _, _) = start(MockEngine::new("ok"), cfg);
    let r = raw(
        addr,
        b"POST /v1/completions HTTP/1.1\r\nContent-Length: 99999999999999\r\n\r\n",
    );
    assert_eq!(r.status, 413);
    let r = raw(
        addr,
        b"POST /v1/completions HTTP/1.1\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n",
    );
    assert_eq!(r.status, 400);
    let mut big = b"GET /health HTTP/1.1\r\nX: ".to_vec();
    big.extend(std::iter::repeat_n(b'a', 20_000));
    big.extend_from_slice(b"\r\n\r\n");
    assert_eq!(raw(addr, &big).status, 431);
    assert_eq!(raw(addr, b"GARBAGE\r\n\r\n").status, 400);
    // Chunked JSON body works.
    let body = r#"{"prompt":"x"}"#;
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{body}\r\n0\r\n\r\n",
        body.len()
    );
    let r = raw(addr, req.as_bytes());
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(get(addr, "/health").status, 200);
}

#[test]
fn expect_continue() {
    let (addr, _, _) = start(MockEngine::new("ok"), test_cfg());
    let body = r#"{"prompt":"x"}"#;
    let mut s = connect(addr);
    write!(
        s,
        "POST /v1/completions HTTP/1.1\r\nExpect: 100-continue\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    assert_eq!(line, "HTTP/1.1 100 Continue\r\n");
    r.read_line(&mut line).unwrap();
    s.write_all(body.as_bytes()).unwrap();
    assert_eq!(read_resp(&mut r).status, 200);
}

#[test]
fn slowloris_times_out() {
    let mut cfg = test_cfg();
    cfg.read_timeout = Duration::from_millis(400);
    let (addr, _, _) = start(MockEngine::new("ok"), cfg);
    let mut s = connect(addr);
    let t0 = Instant::now();
    // Trickle one byte every 100 ms: each read is "in time" but the request
    // as a whole is not.
    let req = b"GET /health HTTP/1.1\r\nX-Slow: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n\r\n";
    let mut got_408 = false;
    for &b in req.iter() {
        if s.write_all(&[b]).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
        if t0.elapsed() > Duration::from_secs(3) {
            break;
        }
    }
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    if buf.starts_with("HTTP/1.1 408") {
        got_408 = true;
    }
    assert!(got_408, "got {buf:?}");
    assert!(t0.elapsed() < Duration::from_secs(4));

    // Idle connection that never sends anything is closed too.
    let mut s = connect(addr);
    let t0 = Instant::now();
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    assert!(t0.elapsed() < Duration::from_secs(3));
}

#[test]
fn keep_alive_reuses_connection() {
    let (addr, _, _) = start(MockEngine::new("ok"), test_cfg());
    let s = connect(addr);
    let mut w = s.try_clone().unwrap();
    let mut r = BufReader::new(s);
    let body = r#"{"prompt":"x"}"#;
    for _ in 0..3 {
        write!(
            w,
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let resp = read_resp(&mut r);
        assert_eq!(resp.status, 200);
        assert_eq!(resp.header("Connection"), Some("keep-alive"));
        assert_eq!(resp.json()["choices"][0]["text"], "ok");
    }
    w.write_all(b"GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        .unwrap();
    let resp = read_resp(&mut r);
    assert_eq!(resp.header("Connection"), Some("close"));
    // Server closed the connection.
    let mut rest = Vec::new();
    assert_eq!(r.read_to_end(&mut rest).unwrap(), 0);
}

fn wait_idle(addr: SocketAddr) {
    let t0 = Instant::now();
    loop {
        let v = get(addr, "/health").json();
        if v["slots_processing"] == 0 && v["queued"] == 0 {
            return;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "slot never freed: {v}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn client_disconnect_aborts_generation() {
    let mut eng = MockEngine::new(&"x".repeat(60));
    eng.token_delay = Duration::from_millis(20);
    let (addr, engine, _) = start(eng, test_cfg());

    // Streaming: read the head, then hang up.
    let mut s = connect(addr);
    let body = r#"{"prompt":"a","stream":true}"#;
    write!(
        s,
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut buf = [0u8; 64];
    let _ = s.read(&mut buf).unwrap();
    drop(s);
    wait_idle(addr);
    let produced = engine.produced.load(Ordering::SeqCst);
    assert!(
        produced < 40,
        "kept generating after disconnect: {produced}"
    );

    // Non-streaming: nothing is written until the end, so detection relies on
    // polling the socket.
    engine.produced.store(0, Ordering::SeqCst);
    let mut s = connect(addr);
    let body = r#"{"prompt":"a"}"#;
    write!(
        s,
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    drop(s);
    wait_idle(addr);
    let produced = engine.produced.load(Ordering::SeqCst);
    assert!(
        produced < 40,
        "kept generating after disconnect: {produced}"
    );
}

#[test]
fn queue_full_returns_503() {
    let mut eng = MockEngine::new(&"x".repeat(30));
    eng.token_delay = Duration::from_millis(20);
    let mut cfg = test_cfg();
    cfg.slots = 1;
    cfg.max_queue = 1;
    let (addr, _, _) = start(eng, cfg);
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(move || post(addr, "/v1/completions", r#"{"prompt":"a"}"#)))
        .collect();
    let resps: Vec<Resp> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let busy: Vec<&Resp> = resps.iter().filter(|r| r.status == 503).collect();
    assert!(!busy.is_empty(), "expected at least one 503");
    assert_eq!(busy[0].header("Retry-After"), Some("1"));
    assert_eq!(busy[0].json()["error"]["code"], "server_busy");
    assert!(resps.iter().any(|r| r.status == 200));
}

#[test]
fn connection_cap_returns_503() {
    let mut cfg = test_cfg();
    cfg.max_connections = 1;
    let (addr, _, app) = start(MockEngine::new("x"), cfg);
    let _held = connect(addr);
    let t0 = Instant::now();
    while app.conns.load(Ordering::SeqCst) == 0 {
        assert!(t0.elapsed() < Duration::from_secs(2));
        std::thread::sleep(Duration::from_millis(5));
    }
    // Don't send anything: the server answers 503 right after accept.
    let r = read_resp(&mut BufReader::new(connect(addr)));
    assert_eq!(r.status, 503);
}

fn thinking_engine(script: &str) -> MockEngine {
    let mut e = MockEngine::new(script);
    e.has_thinking = true;
    e
}

const THINK_SCRIPT: &str = "<think>\nadd 2+2\n</think>\n\nIt is 4.";

#[test]
fn chat_reasoning_content_non_stream() {
    let (addr, engine, app) = start(thinking_engine(THINK_SCRIPT), test_cfg());
    assert!(app.reasoning, "auto: template has <think>");
    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":true,
            "chat_template_kwargs":{"custom":1}}"#,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "It is 4.");
    assert_eq!(msg["reasoning_content"], "add 2+2");
    assert_eq!(v["usage"]["completion_tokens"], THINK_SCRIPT.len());
    let kw = engine.last_kwargs.lock().unwrap().clone();
    assert_eq!(
        Value::Object(kw),
        serde_json::json!({"custom":1,"enable_thinking":true})
    );

    // No reasoning produced: reasoning_content is null.
    let (addr, _, _) = start(thinking_engine("plain"), test_cfg());
    let v = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}]}"#,
    )
    .json();
    assert_eq!(v["choices"][0]["message"]["content"], "plain");
    assert!(v["choices"][0]["message"]["reasoning_content"].is_null());

    // Completions never split.
    let (addr, _, _) = start(thinking_engine(THINK_SCRIPT), test_cfg());
    let v = post(addr, "/v1/completions", r#"{"prompt":"x","max_tokens":60}"#).json();
    assert_eq!(v["choices"][0]["text"], THINK_SCRIPT);
}

#[test]
fn chat_reasoning_content_stream() {
    let (addr, _, _) = start(thinking_engine(THINK_SCRIPT), test_cfg());
    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}],"stream":true}"#,
    );
    let ev = sse_events(&r.body);
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let (mut content, mut reasoning) = (String::new(), String::new());
    for e in &ev[1..ev.len() - 2] {
        let c: Value = serde_json::from_str(e).unwrap();
        let d = &c["choices"][0]["delta"];
        match (d.get("content"), d.get("reasoning_content")) {
            (Some(t), None) => content.push_str(t.as_str().unwrap()),
            (None, Some(t)) => reasoning.push_str(t.as_str().unwrap()),
            _ => panic!("bad delta {d}"),
        }
    }
    assert_eq!(
        (content.as_str(), reasoning.as_str()),
        ("It is 4.", "add 2+2")
    );
}

#[test]
fn reasoning_format_none_keeps_raw_text() {
    let mut cfg = test_cfg();
    cfg.reasoning_format = Some(xor_net::generation::ReasoningFormat::None);
    let (addr, _, app) = start(thinking_engine(THINK_SCRIPT), cfg);
    assert!(!app.reasoning);
    let v = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"x"}]}"#,
    )
    .json();
    assert_eq!(v["choices"][0]["message"]["content"], THINK_SCRIPT);
    assert!(
        v["choices"][0]["message"]
            .get("reasoning_content")
            .is_none()
    );

    // Forced on for a template without <think>.
    let mut cfg = test_cfg();
    cfg.reasoning_format = Some(xor_net::generation::ReasoningFormat::Separate);
    let (_, _, app) = start(MockEngine::new("x"), cfg);
    assert!(app.reasoning);
    let (_, _, app) = start(MockEngine::new("x"), test_cfg());
    assert!(!app.reasoning);
}

/// End-to-end over HTTP with a real Bonsai model. Run with
/// `BONSAI_XQ1=models/bonsai/Bonsai-1.7B.xq1 cargo test --release --bin xor-net-server -- --ignored bonsai`
/// (inside a memory cap: `systemd-run --user --scope -p MemoryMax=4G ...`).
#[test]
#[ignore]
fn bonsai_e2e() {
    use crate::engine::GeneratorEngine;
    use xor_net::generation::{ChatTemplate, GenerationParams, ReasoningFormat};

    let Ok(path) = std::env::var("BONSAI_XQ1") else {
        eprintln!("BONSAI_XQ1 not set; skipping");
        return;
    };
    let g = xor_net::AutoModelForCausalLM::generator_from_file(&path, None).unwrap();

    // Reasoning separation on real tokens: the Bonsai 1.7B template always
    // disables thinking, so use a Qwen3.5-style template that opens <think>
    // in the prompt unless enable_thinking is false.
    let qwen3 = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n\
{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}";
    let mut think = g
        .fork()
        .with_chat_template(ChatTemplate::new(qwen3, "", "<|im_end|>").unwrap());
    let p = GenerationParams::default()
        .with_max_tokens(400)
        .with_sampling(xor_net::SamplingParams::greedy())
        .with_reasoning_format(ReasoningFormat::Separate);
    let msgs = [xor_net::ChatMessage::user("What is 17 + 25?")];
    let raw = think
        .chat(
            &msgs,
            &p.clone().with_reasoning_format(ReasoningFormat::None),
        )
        .unwrap();
    let out = think.chat(&msgs, &p).unwrap();
    eprintln!(
        "raw: {:?}\nsplit: {:?} / {:?}",
        raw.text, out.reasoning, out.text
    );
    // Bonsai-1.7B is a no-think model: it usually closes the block at once,
    // but the tag must never reach the visible text.
    let (r, t) = raw
        .text
        .split_once("</think>")
        .expect("model closes <think>");
    assert_eq!(out.reasoning.as_deref().unwrap_or(""), r.trim_matches('\n'));
    assert_eq!(out.text, t.trim_start_matches('\n'));
    let out = think
        .chat(&msgs, &p.clone().with_enable_thinking(false))
        .unwrap();
    assert_eq!(out.reasoning, None, "{out:?}");
    assert!(out.text.contains("42"), "{:?}", out.text);

    let engine = GeneratorEngine::new(g, 2, "bonsai".into());
    let mut cfg = test_cfg();
    cfg.slots = 2;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Arc::new(App::new(Arc::new(engine), cfg));
    assert!(app.reasoning, "the Bonsai template has <think>");
    serve::spawn_workers(&app).unwrap();
    let app2 = Arc::clone(&app);
    std::thread::spawn(move || serve::serve(listener, app2));

    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":20,"temperature":0}"#,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("Paris"), "{content:?}");
    assert!(!content.contains("<think>") && !content.contains("<|im_end|>"));
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let r = post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"stream":true,"enable_thinking":false}"#,
    );
    let ev = sse_events(&r.body);
    assert_eq!(ev.last().unwrap(), "[DONE]");
    assert!(ev.len() > 3);

    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"The capital of France is","max_tokens":4,"temperature":0}"#,
    );
    let v = r.json();
    assert!(
        v["choices"][0]["text"].as_str().unwrap().contains("Paris"),
        "{v}"
    );
    assert_eq!(v["usage"]["completion_tokens"], 4);

    // Token prompts are range-checked against the vocabulary.
    let r = post(
        addr,
        "/v1/completions",
        r#"{"prompt":[9707,11],"max_tokens":2}"#,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let r = post(addr, "/v1/completions", r#"{"prompt":[999999999]}"#);
    assert_eq!(r.status, 400);
}
