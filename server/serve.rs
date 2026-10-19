//! Connection handling, routing, the bounded job queue and the slot workers.

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::api::{self, Api, ApiError, RespMeta};
use crate::config::ServerConfig;
use crate::engine::{Engine, GenRequest, GenResult};
use crate::http::{self, Limits, Request, Response, TimedReader};
use xor_net::generation::{ReasoningFormat, StreamEvent};

/// How often a waiting connection checks whether its client went away.
const POLL: Duration = Duration::from_millis(200);

// ---------------------------------------------------------------------------
// Job queue
// ---------------------------------------------------------------------------

pub enum Event {
    Delta(String),
    /// Reasoning (`<think>`) delta; only with reasoning separation.
    Reasoning(String),
    Done(GenResult),
    Error(String),
}

pub struct Job {
    pub req: GenRequest,
    pub tx: mpsc::Sender<Event>,
    pub cancel: Arc<AtomicBool>,
}

/// Slot affinity: how many leading prompt tokens slot `slot`'s state already
/// holds (see `Engine::slot_affinity`). Must not block.
pub type AffinityFn = Box<dyn Fn(usize, &[u32]) -> usize + Send + Sync>;

/// A freed slot may pick any of the first `AFFINITY_WINDOW` queued jobs
/// (the one whose prompt its state covers best); beyond that, FIFO.
pub const AFFINITY_WINDOW: usize = 4;
/// A job at the head of the queue that was passed over this many times is
/// taken next regardless of affinity (no starvation).
pub const AFFINITY_MAX_SKIPS: u32 = 4;

struct Queued {
    job: Job,
    skipped: u32,
}

struct QState {
    jobs: VecDeque<Queued>,
    /// Jobs handed straight to an idle slot (its state matches the prompt
    /// best), not yet picked up.
    mailbox: Vec<Option<Job>>,
    /// Slots blocked in `pop` with nothing to do.
    idle: Vec<bool>,
    busy: usize,
    shutdown: bool,
}

/// Bounded job queue shared by connection threads (producers) and slot
/// workers. FIFO, except that with an affinity function a job is matched to
/// the slot whose state already holds the longest prefix of its prompt:
/// an arriving job goes to the best idle slot (if any slot has a non-zero
/// match), and a slot that frees up picks the best match among the first
/// [`AFFINITY_WINDOW`] queued jobs (ties and zero matches: oldest first; a
/// head job skipped [`AFFINITY_MAX_SKIPS`] times is taken next).
/// Poisoning is ignored: the state is always consistent between operations.
pub struct JobQueue {
    state: Mutex<QState>,
    cv: Condvar,
    cap: usize,
    affinity: Option<AffinityFn>,
}

impl JobQueue {
    pub fn new(cap: usize, slots: usize, affinity: Option<AffinityFn>) -> Self {
        let slots = slots.max(1);
        Self {
            state: Mutex::new(QState {
                jobs: VecDeque::new(),
                mailbox: (0..slots).map(|_| None).collect(),
                idle: vec![false; slots],
                busy: 0,
                shutdown: false,
            }),
            cv: Condvar::new(),
            cap,
            affinity,
        }
    }

    fn lock(&self) -> MutexGuard<'_, QState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Enqueue, or give the job back if the queue is full / shut down.
    pub fn try_push(&self, job: Job) -> Result<(), Box<Job>> {
        let mut g = self.lock();
        let pending = g.jobs.len() + g.mailbox.iter().flatten().count();
        if g.shutdown || pending >= self.cap {
            return Err(Box::new(job));
        }
        if let Some(aff) = self.affinity.as_ref().filter(|_| g.jobs.is_empty()) {
            let best = (0..g.idle.len())
                .filter(|&s| g.idle[s] && g.mailbox[s].is_none())
                .map(|s| (aff(s, &job.req.prompt_ids), s))
                .filter(|&(score, _)| score > 0)
                .max_by_key(|&(score, s)| (score, std::cmp::Reverse(s)));
            if let Some((_, slot)) = best {
                g.idle[slot] = false;
                g.mailbox[slot] = Some(job);
                drop(g);
                self.cv.notify_all();
                return Ok(());
            }
        }
        g.jobs.push_back(Queued { job, skipped: 0 });
        drop(g);
        self.cv.notify_one();
        Ok(())
    }

    /// Index of the queued job slot `slot` should run next.
    fn pick(&self, g: &mut QState, slot: usize) -> usize {
        let Some(aff) = &self.affinity else {
            return 0;
        };
        if g.jobs.len() < 2 || g.jobs[0].skipped >= AFFINITY_MAX_SKIPS {
            return 0;
        }
        let window = g.jobs.len().min(AFFINITY_WINDOW);
        let (best, _) = (0..window)
            .map(|i| (i, aff(slot, &g.jobs[i].job.req.prompt_ids)))
            .fold(
                (0, 0),
                |acc, (i, score)| {
                    if score > acc.1 { (i, score) } else { acc }
                },
            );
        for q in g.jobs.iter_mut().take(best) {
            q.skipped += 1;
        }
        best
    }

    /// Block until a job is available for worker `slot` (`None` on
    /// shutdown). Marks the caller busy until [`JobQueue::finish`].
    pub fn pop(&self, slot: usize) -> Option<Job> {
        let mut g = self.lock();
        let slot = slot.min(g.idle.len() - 1);
        loop {
            if g.shutdown {
                return None;
            }
            let job = match g.mailbox[slot].take() {
                Some(j) => Some(j),
                None if !g.jobs.is_empty() => {
                    let i = self.pick(&mut g, slot);
                    g.jobs.remove(i).map(|q| q.job)
                }
                None => None,
            };
            if let Some(j) = job {
                g.idle[slot] = false;
                g.busy += 1;
                return Some(j);
            }
            g.idle[slot] = true;
            g = self.cv.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub fn finish(&self) {
        let mut g = self.lock();
        g.busy = g.busy.saturating_sub(1);
    }

    pub fn shutdown(&self) {
        self.lock().shutdown = true;
        self.cv.notify_all();
    }

    /// (queued, busy)
    pub fn stats(&self) -> (usize, usize) {
        let g = self.lock();
        (g.jobs.len() + g.mailbox.iter().flatten().count(), g.busy)
    }
}

// ---------------------------------------------------------------------------
// App state and workers
// ---------------------------------------------------------------------------

pub struct App {
    pub engine: Arc<dyn Engine>,
    /// Chat responses carry `reasoning_content` (resolved from
    /// `--reasoning-format` and the model's template).
    pub reasoning: bool,
    pub queue: JobQueue,
    pub cfg: ServerConfig,
    pub limits: Limits,
    pub conns: AtomicUsize,
    req_counter: AtomicU64,
}

impl App {
    pub fn new(engine: Arc<dyn Engine>, cfg: ServerConfig) -> Self {
        let reasoning = match cfg.reasoning_format {
            Some(f) => f == ReasoningFormat::Separate,
            None => engine.chat_template_has_thinking(),
        };
        let aff_engine = Arc::clone(&engine);
        let affinity: AffinityFn = Box::new(move |slot, ids| aff_engine.slot_affinity(slot, ids));
        Self {
            engine,
            reasoning,
            queue: JobQueue::new(cfg.max_queue, cfg.slots, Some(affinity)),
            limits: Limits {
                max_line: cfg.max_header_line,
                max_headers: cfg.max_headers,
                max_body: cfg.max_body_bytes,
            },
            cfg,
            conns: AtomicUsize::new(0),
            req_counter: AtomicU64::new(0),
        }
    }

    fn next_id(&self) -> u64 {
        self.req_counter.fetch_add(1, Ordering::Relaxed)
    }

    fn cors_headers(&self) -> Vec<(&'static str, String)> {
        match &self.cfg.cors_origin {
            Some(o) => vec![
                ("Access-Control-Allow-Origin", o.clone()),
                ("Vary", "Origin".into()),
            ],
            None => Vec::new(),
        }
    }
}

/// Spawn `cfg.slots` workers, each with its own rayon pool.
pub fn spawn_workers(app: &Arc<App>) -> anyhow::Result<Vec<thread::JoinHandle<()>>> {
    let mut handles = Vec::new();
    for slot in 0..app.cfg.slots {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(app.cfg.threads_per_slot.max(1))
            .thread_name(move |i| format!("slot{slot}-rayon{i}"))
            .build()?;
        let app = Arc::clone(app);
        let h = thread::Builder::new()
            .name(format!("slot{slot}"))
            .spawn(move || worker(&app, slot, &pool))?;
        handles.push(h);
    }
    Ok(handles)
}

fn worker(app: &App, slot: usize, pool: &rayon::ThreadPool) {
    while let Some(job) = app.queue.pop(slot) {
        if !job.cancel.load(Ordering::Relaxed) {
            run_job(app.engine.as_ref(), slot, &job, pool);
        }
        app.queue.finish();
    }
}

fn run_job(engine: &dyn Engine, slot: usize, job: &Job, pool: &rayon::ThreadPool) {
    let tx = &job.tx;
    let cancel = &job.cancel;
    let res = pool.install(|| {
        engine.generate(slot, &job.req, &mut |ev: StreamEvent<'_>| {
            if cancel.load(Ordering::Relaxed) {
                return ControlFlow::Break(());
            }
            let msg = match ev {
                StreamEvent::Text(d) if !d.is_empty() => Event::Delta(d.to_string()),
                StreamEvent::Reasoning(d) if !d.is_empty() => Event::Reasoning(d.to_string()),
                _ => return ControlFlow::Continue(()),
            };
            if tx.send(msg).is_err() {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        })
    });
    let _ = match res {
        Ok(r) => tx.send(Event::Done(r)),
        Err(e) => tx.send(Event::Error(e.to_string())),
    };
}

// ---------------------------------------------------------------------------
// Accept loop / connections
// ---------------------------------------------------------------------------

struct ConnGuard<'a>(&'a AtomicUsize);
impl Drop for ConnGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Sets the job's cancel flag when the connection handler exits for any
/// reason (error, disconnect, early return).
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub fn serve(listener: TcpListener, app: Arc<App>) {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let prev = app.conns.fetch_add(1, Ordering::AcqRel);
        if prev >= app.cfg.max_connections {
            app.conns.fetch_sub(1, Ordering::AcqRel);
            reject_busy(stream);
            continue;
        }
        let app2 = Arc::clone(&app);
        let spawned = thread::Builder::new().name("conn".into()).spawn(move || {
            let _g = ConnGuard(&app2.conns);
            handle_connection(stream, &app2);
        });
        if spawned.is_err() {
            // The closure (and its guard) was dropped without running.
            app.conns.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn reject_busy(stream: TcpStream) {
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    let mut s = stream;
    let _ = ApiError::new(
        503,
        "server_error",
        "server_busy",
        "too many open connections",
    )
    .response()
    .with_header("Retry-After", "1")
    .write_to(&mut s, false, &[]);
    // Drain whatever already arrived (without waiting) so closing doesn't
    // turn into an RST that could discard the 503 on the client side.
    if s.set_nonblocking(true).is_ok() {
        let mut buf = [0u8; 4096];
        for _ in 0..16 {
            match io::Read::read(&mut s, &mut buf) {
                Ok(n) if n > 0 => continue,
                _ => break,
            }
        }
    }
    let _ = s.shutdown(std::net::Shutdown::Write);
}

/// Returns true if the peer has closed its side of the connection.
fn peer_closed(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut b = [0u8; 1];
    let closed = match stream.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => false,
        Err(e) if e.kind() == io::ErrorKind::Interrupted => false,
        Err(_) => true,
    };
    let _ = stream.set_nonblocking(false);
    closed
}

/// Rate-limited [`peer_closed`] check.
struct PeerWatch<'a> {
    stream: &'a TcpStream,
    last: Instant,
}

impl<'a> PeerWatch<'a> {
    fn new(stream: &'a TcpStream) -> Self {
        Self {
            stream,
            last: Instant::now(),
        }
    }

    fn gone(&mut self) -> bool {
        if self.last.elapsed() < POLL {
            return false;
        }
        self.last = Instant::now();
        peer_closed(self.stream)
    }
}

pub fn handle_connection(stream: TcpStream, app: &App) {
    let cfg = &app.cfg;
    let _ = stream.set_nodelay(true);
    if stream.set_write_timeout(Some(cfg.write_timeout)).is_err() {
        return;
    }
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let mut reader = BufReader::new(TimedReader::new(&stream, cfg.read_timeout));
    let mut served = 0usize;

    loop {
        // Wait for the first byte of the next request.
        let wait = if served == 0 {
            cfg.read_timeout
        } else {
            cfg.idle_timeout
        };
        reader.get_mut().set_per_read(wait);
        reader.get_mut().set_deadline(Some(Instant::now() + wait));
        match reader.fill_buf() {
            Ok([]) => return,
            Ok(_) => {}
            Err(e) => {
                // First request never arrived in time: tell the client.
                if served == 0 && http::is_timeout(&e) {
                    let _ = error_response(408, "request timed out").write_to(
                        &mut writer,
                        false,
                        &app.cors_headers(),
                    );
                }
                return;
            }
        }

        // The whole request must arrive within read_timeout (slowloris).
        reader.get_mut().set_per_read(cfg.read_timeout);
        reader
            .get_mut()
            .set_deadline(Some(Instant::now() + cfg.read_timeout));
        let mut on_continue = || -> io::Result<()> {
            writer.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
            writer.flush()
        };
        let req = match http::read_request_with(&mut reader, &app.limits, &mut on_continue) {
            Ok(r) => r,
            Err(e) => {
                if let Some(status) = e.status() {
                    let _ = error_response(status, &e.message()).write_to(
                        &mut writer,
                        false,
                        &app.cors_headers(),
                    );
                }
                return;
            }
        };
        reader.get_mut().set_deadline(None);
        served += 1;
        let keep = req.wants_keep_alive() && served < cfg.keep_alive_max;
        let reusable = dispatch(&req, app, &mut writer, &stream, keep);
        if !(keep && reusable) {
            return;
        }
    }
}

fn error_response(status: u16, msg: &str) -> Response {
    let (kind, code) = match status {
        404 => ("invalid_request_error", "not_found"),
        405 => ("invalid_request_error", "method_not_allowed"),
        408 => ("invalid_request_error", "request_timeout"),
        413 => ("invalid_request_error", "payload_too_large"),
        414 | 431 => ("invalid_request_error", "headers_too_large"),
        501 => ("invalid_request_error", "not_implemented"),
        503 => ("server_error", "server_busy"),
        s if s >= 500 => ("server_error", "internal_error"),
        _ => ("invalid_request_error", "bad_request"),
    };
    ApiError::new(status, kind, code, msg).response()
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum Route {
    Health,
    Models,
    Generate(Api),
}

/// (route, allowed method) for a path.
pub fn lookup(path: &str) -> Option<(Route, &'static str)> {
    let path = if path.len() > 1 {
        path.trim_end_matches('/')
    } else {
        path
    };
    Some(match path {
        "/health" | "/v1/health" => (Route::Health, "GET"),
        "/v1/models" | "/models" => (Route::Models, "GET"),
        "/v1/chat/completions" | "/chat/completions" => (Route::Generate(Api::Chat), "POST"),
        "/v1/completions" | "/completions" => (Route::Generate(Api::Completions), "POST"),
        "/completion" => (Route::Generate(Api::LlamaCpp), "POST"),
        _ => return None,
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Handle one parsed request. Returns whether the connection may be reused.
fn dispatch(req: &Request, app: &App, w: &mut TcpStream, stream: &TcpStream, keep: bool) -> bool {
    let cors = app.cors_headers();
    let send = |w: &mut TcpStream, r: Response| r.write_to(w, keep, &cors).is_ok();

    let Some((route, allowed)) = lookup(&req.path) else {
        return send(
            w,
            error_response(404, &format!("no route for {}", req.path)),
        );
    };

    if req.method == "OPTIONS" {
        let mut r = Response::empty(204).with_header("Allow", format!("{allowed}, OPTIONS"));
        if app.cfg.cors_origin.is_some() {
            r = r
                .with_header(
                    "Access-Control-Allow-Methods",
                    format!("{allowed}, OPTIONS"),
                )
                .with_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization",
                )
                .with_header("Access-Control-Max-Age", "600");
        }
        return send(w, r);
    }
    if req.method != allowed {
        let r = error_response(
            405,
            &format!("method {} not allowed on {}", req.method, req.path),
        )
        .with_header("Allow", format!("{allowed}, OPTIONS"));
        return send(w, r);
    }

    match route {
        Route::Health => {
            let (queued, busy) = app.queue.stats();
            let mut body = json!({
                "status": "ok",
                "slots_idle": app.cfg.slots.saturating_sub(busy),
                "slots_processing": busy,
                "queued": queued,
            });
            if let Some(st) = app.engine.prompt_cache_stats() {
                body["prompt_cache"] = json!(st);
            }
            send(w, Response::json(200, &body))
        }
        Route::Models => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": app.engine.model_id(),
                    "object": "model",
                    "created": now_secs(),
                    "owned_by": "xor-net",
                    "meta": {"n_ctx_train": app.engine.context_length()},
                }],
            });
            send(w, Response::json(200, &body))
        }
        Route::Generate(api) => match generate(req, api, app, w, stream, keep) {
            Ok(reusable) => reusable,
            Err(e) => {
                let mut r = e.response();
                if e.status == 503 {
                    r = r.with_header("Retry-After", "1");
                }
                send(w, r)
            }
        },
    }
}

/// Validate, enqueue and answer a generation request. `Err` is returned
/// only before anything was written to the client.
fn generate(
    req: &Request,
    api: Api,
    app: &App,
    w: &mut TcpStream,
    stream: &TcpStream,
    keep: bool,
) -> Result<bool, ApiError> {
    let engine = app.engine.as_ref();
    let p = api::parse_request(api, &req.body, app.cfg.max_tokens_cap)?;
    let prompt_ids = engine
        .encode(&p.prompt, &p.chat_template_kwargs)
        .map_err(|m| {
            ApiError::invalid(
                if api == Api::Chat {
                    "messages"
                } else {
                    "prompt"
                },
                m,
            )
        })?;
    let boundaries = engine.prefix_boundaries(&p.prompt, &p.chat_template_kwargs, &prompt_ids);
    let max_tokens = api::resolve_max_tokens(
        p.max_tokens,
        prompt_ids.len(),
        engine.context_length(),
        app.cfg.default_max_tokens,
        api,
    )?;
    let mut sampling = p.sampling.clone();
    if !p.seeded {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        sampling.seed = (nanos ^ app.next_id().wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    }

    let prompt_tokens = prompt_ids.len();
    let cancel = Arc::new(AtomicBool::new(false));
    let _cancel_guard = CancelOnDrop(Arc::clone(&cancel));
    let (tx, rx) = mpsc::channel();
    let job = Job {
        req: GenRequest {
            prompt_ids,
            max_tokens,
            sampling,
            stop: p.stop.clone(),
            reasoning: api == Api::Chat && app.reasoning,
            boundaries,
        },
        tx,
        cancel,
    };
    if app.queue.try_push(job).is_err() {
        return Err(ApiError::new(
            503,
            "server_error",
            "server_busy",
            "the server is overloaded, retry later",
        ));
    }

    let id = match api {
        Api::Chat => format!("chatcmpl-{}", app.next_id()),
        _ => format!("cmpl-{}", app.next_id()),
    };
    let meta = RespMeta {
        id: &id,
        created: now_secs(),
        model: engine.model_id(),
        prompt_tokens,
    };
    let cors = app.cors_headers();

    if p.stream {
        // The SSE body is close-delimited, so the connection is never reused.
        // A write error drops `_cancel_guard`, which cancels the job.
        let _ = stream_events(api, &meta, &p, rx, w, stream, &cors);
        Ok(false)
    } else {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut watch = PeerWatch::new(stream);
        loop {
            match rx.recv_timeout(POLL) {
                Ok(Event::Delta(d)) => {
                    text.push_str(&d);
                    if watch.gone() {
                        return Ok(false);
                    }
                }
                Ok(Event::Reasoning(d)) => {
                    reasoning.push_str(&d);
                    if watch.gone() {
                        return Ok(false);
                    }
                }
                Ok(Event::Done(res)) => {
                    let r = (api == Api::Chat && app.reasoning).then_some(reasoning.as_str());
                    let body = api::full_body(api, &meta, &text, r, &res);
                    return Ok(Response::json(200, &body).write_to(w, keep, &cors).is_ok());
                }
                Ok(Event::Error(e)) => {
                    let r = error_response(500, &format!("generation failed: {e}"));
                    return Ok(r.write_to(w, keep, &cors).is_ok());
                }
                Err(RecvTimeoutError::Timeout) => {
                    if watch.gone() {
                        return Ok(false);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let r = error_response(500, "generation worker stopped");
                    let _ = r.write_to(w, false, &cors);
                    return Ok(false);
                }
            }
        }
    }
}

fn sse_json(w: &mut TcpStream, v: &serde_json::Value) -> io::Result<()> {
    http::write_sse_data(w, &v.to_string())
}

/// Write the SSE stream for one job.
fn stream_events(
    api: Api,
    meta: &RespMeta,
    p: &api::ParsedRequest,
    rx: mpsc::Receiver<Event>,
    w: &mut TcpStream,
    stream: &TcpStream,
    cors: &[(&'static str, String)],
) -> io::Result<()> {
    http::write_sse_head(w, cors)?;
    if api == Api::Chat {
        sse_json(w, &api::delta_chunk(api, meta, "", true, p.include_usage))?;
    }
    let mut watch = PeerWatch::new(stream);
    loop {
        match rx.recv_timeout(POLL) {
            Ok(Event::Delta(d)) => {
                sse_json(w, &api::delta_chunk(api, meta, &d, false, p.include_usage))?;
                if watch.gone() {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "client closed"));
                }
            }
            Ok(Event::Reasoning(d)) => {
                if api == Api::Chat {
                    sse_json(w, &api::reasoning_chunk(meta, &d, p.include_usage))?;
                }
                if watch.gone() {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "client closed"));
                }
            }
            Ok(Event::Done(res)) => {
                for c in api::final_chunks(api, meta, &res, p.include_usage) {
                    sse_json(w, &c)?;
                }
                if api::sends_done(api) {
                    http::write_sse_data(w, "[DONE]")?;
                }
                return Ok(());
            }
            Ok(Event::Error(e)) => {
                let err = ApiError::new(
                    500,
                    "server_error",
                    "internal_error",
                    format!("generation failed: {e}"),
                );
                return sse_json(w, &err.body());
            }
            Err(RecvTimeoutError::Timeout) => {
                if watch.gone() {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "client closed"));
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let err = ApiError::new(
                    500,
                    "server_error",
                    "internal_error",
                    "generation worker stopped",
                );
                return sse_json(w, &err.body());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::SamplingParams;

    fn job(ids: &[u32]) -> (Job, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel();
        let req = GenRequest {
            prompt_ids: ids.to_vec(),
            max_tokens: 1,
            sampling: SamplingParams::default(),
            stop: Vec::new(),
            reasoning: false,
            boundaries: Vec::new(),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        (Job { req, tx, cancel }, rx)
    }

    /// Slot 0 holds [1, 2, 3], slot 1 holds [9, 9].
    fn queue(cap: usize) -> JobQueue {
        let states: [&[u32]; 2] = [&[1, 2, 3], &[9, 9]];
        let aff: AffinityFn = Box::new(move |slot, ids| {
            states[slot]
                .iter()
                .zip(ids)
                .take_while(|(a, b)| a == b)
                .count()
        });
        JobQueue::new(cap, 2, Some(aff))
    }

    #[test]
    fn freed_slot_prefers_its_prefix_within_window() {
        let q = queue(16);
        let mut rxs = Vec::new();
        for ids in [&[5u32][..], &[9, 9, 1], &[1, 2, 3, 4]] {
            let (j, rx) = job(ids);
            rxs.push(rx);
            assert!(q.try_push(j).is_ok());
        }
        assert_eq!(q.pop(1).unwrap().req.prompt_ids, [9, 9, 1]);
        assert_eq!(q.pop(0).unwrap().req.prompt_ids, [1, 2, 3, 4]);
        assert_eq!(q.pop(1).unwrap().req.prompt_ids, [5]);
        assert_eq!(q.stats(), (0, 3));

        // Beyond the window: FIFO.
        for i in 0..AFFINITY_WINDOW as u32 {
            let (j, rx) = job(&[100 + i]);
            rxs.push(rx);
            assert!(q.try_push(j).is_ok());
        }
        let (j, rx) = job(&[1, 2, 3]);
        rxs.push(rx);
        assert!(q.try_push(j).is_ok());
        assert_eq!(q.pop(0).unwrap().req.prompt_ids, [100]);
    }

    #[test]
    fn head_job_is_not_starved() {
        let q = queue(16);
        let mut rxs = Vec::new();
        let (j, rx) = job(&[7]);
        rxs.push(rx);
        assert!(q.try_push(j).is_ok());
        for _ in 0..AFFINITY_MAX_SKIPS {
            let (j, rx) = job(&[1, 2, 3]);
            rxs.push(rx);
            assert!(q.try_push(j).is_ok());
            assert_eq!(q.pop(0).unwrap().req.prompt_ids, [1, 2, 3]);
        }
        let (j, rx) = job(&[1, 2, 3]);
        rxs.push(rx);
        assert!(q.try_push(j).is_ok());
        assert_eq!(q.pop(0).unwrap().req.prompt_ids, [7], "skipped too often");
    }

    #[test]
    fn arriving_job_goes_to_best_idle_slot() {
        let q = Arc::new(queue(16));
        let (tx, rx) = mpsc::channel();
        let workers: Vec<_> = (0..2)
            .map(|slot| {
                let q = Arc::clone(&q);
                let tx = tx.clone();
                thread::spawn(move || {
                    while let Some(j) = q.pop(slot) {
                        tx.send((slot, j.req.prompt_ids.clone())).unwrap();
                        q.finish();
                    }
                })
            })
            .collect();
        // Wait until both workers are blocked in `pop`.
        let deadline = Instant::now() + Duration::from_secs(5);
        while q.lock().idle != [true, true] {
            assert!(Instant::now() < deadline, "workers never went idle");
            thread::sleep(Duration::from_millis(5));
        }
        let mut rxs = Vec::new();
        for ids in [&[9u32, 9, 4][..], &[1, 2, 7]] {
            let (j, r) = job(ids);
            rxs.push(r);
            assert!(q.try_push(j).is_ok());
            let (slot, got) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(got, ids);
            assert_eq!(slot, if ids[0] == 9 { 1 } else { 0 });
            let deadline = Instant::now() + Duration::from_secs(5);
            while q.lock().idle != [true, true] {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(5));
            }
        }
        q.shutdown();
        for w in workers {
            w.join().unwrap();
        }
    }
}
