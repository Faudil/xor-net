//! Python bindings (`pip install xor-net` / `maturin develop`).
//!
//! ```python
//! import xor_net
//! model = xor_net.load("prism-ml/Bonsai-1.7B-gguf")   # or a .gguf/.xq1 path, or a BitNet dir
//! print(model.chat([{"role": "user", "content": "Hi!"}], max_tokens=64))
//! for piece in model.generate_stream("The capital of France is", max_tokens=16):
//!     print(piece, end="", flush=True)
//!
//! clf = xor_net.ImageClassifier("quicknet.safetensors")
//! clf.predict("cat.jpg", k=5)          # path, encoded bytes, numpy HxWx3 uint8, or PIL image
//! ```
//!
//! Every model call releases the GIL. Streams run generation on a
//! background thread; dropping the iterator (or `cancel()`) stops it at the
//! next token.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use crate::bit1_58::quantization::TernaryPackType;
use crate::generation::{
    ChatMessage, FinishReason, Generation, GenerationParams, Generator, ReasoningFormat,
    StreamEvent,
};
use crate::init_threads;
use crate::models::llama::causal_lm::LlamaCausalLM;
use crate::models::llama::{Config, LlamaConfig};
use crate::models::source::{self, LoadOptions, ModelKind};
use crate::nn::{LmHeadConfig, QuantizationConfig};
use crate::sampler::SamplingParams;
use crate::vision;

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn any_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// Initialise the global thread pool once. A second, different request is
/// reported as a warning (the pool can only be built once per process).
fn ensure_threads(py: Python<'_>, threads: Option<usize>) -> PyResult<()> {
    static INIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let want = threads.unwrap_or(0);
    let mut fresh = false;
    let got = *INIT.get_or_init(|| {
        fresh = true;
        // Fails only if someone else (e.g. rayon's lazy default) already
        // built the global pool; that pool is then used as-is.
        let _ = init_threads(want);
        rayon::current_num_threads()
    });
    if !fresh && threads.is_some() && want != got && want != 0 {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            &std::ffi::CString::new(format!(
                "xor_net thread pool already initialised with {got} threads; threads={want} ignored"
            ))
            .unwrap_or_default(),
            1,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Legacy configuration classes (Llama / BitNet)
// ---------------------------------------------------------------------------

#[pyclass(name = "QuantizationConfigPy")]
#[derive(Clone)]
pub struct QuantizationConfigPy {
    inner: QuantizationConfig,
}

#[pymethods]
impl QuantizationConfigPy {
    #[new]
    #[pyo3(signature = (pack_type = "pack4", lm_head = "int4"))]
    fn new(pack_type: &str, lm_head: &str) -> PyResult<Self> {
        let pack = match pack_type.to_lowercase().as_str() {
            "pack4" => TernaryPackType::Pack4,
            "pack5" => TernaryPackType::Pack5,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown pack type: {pack_type}"
                )));
            }
        };
        let lm_head_config = match lm_head.to_lowercase().as_str() {
            "int4" => LmHeadConfig::Int4,
            "int8" => LmHeadConfig::Int8,
            "fp32" | "f32" => LmHeadConfig::F32,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown lm_head config: {lm_head}"
                )));
            }
        };
        Ok(Self {
            inner: QuantizationConfig::Bit1_58(pack, lm_head_config, true),
        })
    }
}

#[pyclass]
pub struct ModelConfig {
    inner: Config,
}

#[pymethods]
impl ModelConfig {
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let llama_config: LlamaConfig = serde_json::from_str(json_str).map_err(py_err)?;
        Ok(Self {
            inner: llama_config.into_config(false),
        })
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }
    fn num_layers(&self) -> usize {
        self.inner.num_hidden_layers
    }
    fn num_heads(&self) -> usize {
        self.inner.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Generation helpers
// ---------------------------------------------------------------------------

fn lock(g: &Mutex<Generator>) -> std::sync::MutexGuard<'_, Generator> {
    g.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sampling / stopping options shared by every entry point. Defaults are the
/// Qwen3 non-thinking recommendation (temperature 0.7, top-p 0.8, top-k 20);
/// `temperature=0` is greedy.
#[derive(Clone, Default)]
struct Opts {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
    enable_thinking: Option<bool>,
    ignore_eos: bool,
}

impl Opts {
    fn params(&self) -> PyResult<GenerationParams> {
        let temperature = self.temperature.unwrap_or(0.7);
        let top_p = self.top_p.unwrap_or(0.8);
        if !(temperature >= 0.0 && temperature.is_finite()) {
            return Err(PyValueError::new_err("temperature must be >= 0"));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(PyValueError::new_err("top_p must be in (0, 1]"));
        }
        let sampling = if temperature == 0.0 {
            SamplingParams::greedy()
        } else {
            SamplingParams {
                temperature,
                top_p,
                top_k: self.top_k.unwrap_or(20),
                min_p: self.min_p.unwrap_or(0.0),
                ..SamplingParams::default()
            }
        };
        let sampling = SamplingParams {
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            seed: self.seed,
            ..sampling
        };
        let mut p = GenerationParams::default()
            .with_max_tokens(self.max_tokens.unwrap_or(256))
            .with_stop(self.stop.clone().unwrap_or_default())
            .with_sampling(sampling)
            .with_ignore_eos(self.ignore_eos)
            .with_reasoning_format(ReasoningFormat::Separate);
        if let Some(on) = self.enable_thinking {
            p = p.with_enable_thinking(on);
        }
        Ok(p)
    }
}

fn to_messages(messages: Vec<HashMap<String, String>>) -> PyResult<Vec<ChatMessage>> {
    messages
        .into_iter()
        .map(|m| match (m.get("role"), m.get("content")) {
            (Some(r), Some(c)) => Ok(ChatMessage::new(r.clone(), c.clone())),
            _ => Err(PyValueError::new_err(
                "each message needs 'role' and 'content' keys",
            )),
        })
        .collect()
}

fn finish_str(f: FinishReason) -> &'static str {
    f.as_openai_str()
}

/// Full result of a generation (`details=True`).
#[pyclass(frozen, get_all)]
#[derive(Clone)]
pub struct Completion {
    /// Visible completion text (reasoning excluded).
    text: String,
    /// Content of `<think>` blocks, if the model produced any.
    reasoning: Option<String>,
    /// "stop" or "length".
    finish_reason: &'static str,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    completion_tokens: usize,
    prefill_tok_per_s: f64,
    decode_tok_per_s: f64,
}

impl From<Generation> for Completion {
    fn from(g: Generation) -> Self {
        Self {
            prefill_tok_per_s: g.timings.prefill_tok_per_s(),
            decode_tok_per_s: g.timings.decode_tok_per_s(),
            text: g.text,
            reasoning: g.reasoning,
            finish_reason: finish_str(g.finish_reason),
            prompt_tokens: g.usage.prompt_tokens,
            cached_prompt_tokens: g.usage.cached_prompt_tokens,
            completion_tokens: g.usage.completion_tokens,
        }
    }
}

#[pymethods]
impl Completion {
    fn __repr__(&self) -> String {
        format!(
            "Completion(text={:?}, reasoning={}, finish_reason={:?}, prompt_tokens={}, completion_tokens={}, decode_tok_per_s={:.1})",
            self.text,
            self.reasoning
                .as_ref()
                .map_or("None".into(), |r| format!("{r:?}")),
            self.finish_reason,
            self.prompt_tokens,
            self.completion_tokens,
            self.decode_tok_per_s
        )
    }
    fn __str__(&self) -> String {
        self.text.clone()
    }
}

enum StreamMsg {
    Delta(bool /* reasoning */, String),
    Done(Box<Completion>),
    Error(String),
}

/// Iterator over the deltas of a streaming generation. Yields `str` pieces
/// of visible text, or `(kind, text)` tuples with `kind` in
/// `{"text", "reasoning"}` when created with `reasoning=True`.
#[pyclass]
pub struct TextStream {
    rx: Mutex<Receiver<StreamMsg>>,
    cancel: Arc<AtomicBool>,
    with_kinds: bool,
    done: bool,
    result: Option<Completion>,
}

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        loop {
            if self.done {
                return Ok(None);
            }
            let rx = &self.rx;
            let msg = py.allow_threads(|| rx.lock().unwrap_or_else(|e| e.into_inner()).recv());
            match msg {
                Ok(StreamMsg::Delta(reasoning, s)) => {
                    if self.with_kinds {
                        let kind = if reasoning { "reasoning" } else { "text" };
                        return Ok(Some(
                            PyTuple::new(py, [kind, s.as_str()])?.into_any().unbind(),
                        ));
                    }
                    if !reasoning {
                        return Ok(Some(s.into_pyobject(py)?.into_any().unbind()));
                    }
                }
                Ok(StreamMsg::Done(c)) => {
                    self.done = true;
                    self.result = Some(*c);
                    return Ok(None);
                }
                Ok(StreamMsg::Error(e)) => {
                    self.done = true;
                    return Err(PyRuntimeError::new_err(e));
                }
                Err(_) => {
                    self.done = true;
                    return Err(PyRuntimeError::new_err(
                        "generation thread ended unexpectedly",
                    ));
                }
            }
        }
    }

    /// "stop" / "length" once the stream is exhausted, else None.
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.result.as_ref().map(|c| c.finish_reason)
    }

    /// The full [`Completion`] (usage, timings, reasoning) once exhausted.
    #[getter]
    fn completion(&self) -> Option<Completion> {
        self.result.clone()
    }

    /// Stop generation at the next token.
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for TextStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A loaded language model: Bonsai 1-bit (`.gguf` / `.xq1`) or a Llama /
/// BitNet b1.58 safetensors directory. Create with `xor_net.load(...)`.
#[pyclass(name = "Model")]
pub struct PyModel {
    generator: Arc<Mutex<Generator>>,
    path: String,
    kind: &'static str,
    llama_config: Option<Config>,
}

impl PyModel {
    fn from_generator(g: Generator, path: String, kind: &'static str, cfg: Option<Config>) -> Self {
        Self {
            generator: Arc::new(Mutex::new(g)),
            path,
            kind,
            llama_config: cfg,
        }
    }

    fn run(&self, py: Python<'_>, prompt: Prompt, opts: Opts, details: bool) -> PyResult<PyObject> {
        let params = opts.params()?;
        let generator = &self.generator;
        let out = py
            .allow_threads(move || {
                let mut g = lock(generator);
                match prompt {
                    Prompt::Text(p) => g.generate(&p, &params),
                    Prompt::Chat(m) => g.chat(&m, &params),
                }
            })
            .map_err(any_err)?;
        if details {
            Ok(Completion::from(out).into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(out.text.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn stream(&self, prompt: Prompt, opts: Opts, with_kinds: bool) -> PyResult<TextStream> {
        let params = opts.params()?;
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let generator = self.generator.clone();
        let stop = cancel.clone();
        std::thread::spawn(move || {
            let mut g = lock(&generator);
            let on_event = |ev: StreamEvent<'_>| {
                let msg = match ev {
                    StreamEvent::Text(t) => StreamMsg::Delta(false, t.to_owned()),
                    StreamEvent::Reasoning(r) => StreamMsg::Delta(true, r.to_owned()),
                };
                if stop.load(Ordering::Relaxed) || tx.send(msg).is_err() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let res = match prompt {
                Prompt::Text(p) => g.generate_stream_events(&p, &params, on_event),
                Prompt::Chat(m) => g.chat_stream_events(&m, &params, on_event),
            };
            let _ = tx.send(match res {
                Ok(out) => StreamMsg::Done(Box::new(out.into())),
                Err(e) => StreamMsg::Error(format!("{e:#}")),
            });
        });
        Ok(TextStream {
            rx: Mutex::new(rx),
            cancel,
            with_kinds,
            done: false,
            result: None,
        })
    }

    fn legacy(model: LlamaCausalLM, tokenizer_dir: &Path, path: String) -> PyResult<Self> {
        let config = model.config().clone();
        let generator = Generator::from_dir(Arc::new(model), tokenizer_dir).map_err(any_err)?;
        Ok(Self::from_generator(generator, path, "llama", Some(config)))
    }
}

fn quant(q: Option<QuantizationConfigPy>) -> QuantizationConfig {
    q.map(|q| q.inner)
        .unwrap_or_else(|| LoadOptions::default().quantization)
}

/// Load a model from a path or Hugging Face repo id (see `xor_net.load`).
fn load_impl(
    py: Python<'_>,
    model: &str,
    threads: Option<usize>,
    tokenizer: Option<PathBuf>,
    quantization: Option<QuantizationConfigPy>,
) -> PyResult<PyModel> {
    ensure_threads(py, threads)?;
    let opts = LoadOptions {
        tokenizer,
        quantization: quant(quantization),
        ..LoadOptions::default()
    };
    let spec = model.to_string();
    let (g, resolved) = py
        .allow_threads(move || -> anyhow::Result<_> {
            let r = source::resolve(&spec)?;
            Ok((source::load_resolved(&r, &opts)?, r))
        })
        .map_err(any_err)?;
    let kind = match resolved.kind {
        ModelKind::Bonsai => "bonsai",
        ModelKind::Llama => "llama",
    };
    Ok(PyModel::from_generator(
        g,
        resolved.path.display().to_string(),
        kind,
        None,
    ))
}

macro_rules! opts {
    ($($f:ident),* ; $ignore_eos:expr) => {
        Opts { $($f,)* ignore_eos: $ignore_eos }
    };
}

#[pymethods]
impl PyModel {
    /// Same as `xor_net.load`.
    #[staticmethod]
    #[pyo3(signature = (model, threads=None, tokenizer=None, quantization=None))]
    fn load(
        py: Python<'_>,
        model: &str,
        threads: Option<usize>,
        tokenizer: Option<PathBuf>,
        quantization: Option<QuantizationConfigPy>,
    ) -> PyResult<Self> {
        load_impl(py, model, threads, tokenizer, quantization)
    }

    /// Legacy: Llama/BitNet from the Hub (prefer `xor_net.load`).
    #[staticmethod]
    #[pyo3(signature = (model_id, quantization=None, num_threads=None))]
    fn from_pretrained(
        py: Python<'_>,
        model_id: &str,
        quantization: Option<QuantizationConfigPy>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        ensure_threads(py, num_threads)?;
        let q = quant(quantization);
        let id = model_id.to_string();
        let (model, dir) = py
            .allow_threads(move || -> anyhow::Result<_> {
                let model = LlamaCausalLM::from_pretrained(&id, q)?;
                let api = hf_hub::api::sync::Api::new()?;
                let repo = api.repo(hf_hub::Repo::new(id.clone(), hf_hub::RepoType::Model));
                let tok = repo.get("tokenizer.json")?;
                for optional in ["tokenizer_config.json", "generation_config.json"] {
                    let _ = repo.get(optional);
                }
                let dir = tok
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("bad tokenizer path"))?
                    .to_path_buf();
                Ok((model, dir))
            })
            .map_err(any_err)?;
        Self::legacy(model, &dir, model_id.to_string())
    }

    /// Legacy: Llama/BitNet from a local directory (prefer `xor_net.load`).
    #[staticmethod]
    #[pyo3(signature = (model_dir, quantization=None, num_threads=None))]
    fn from_local(
        py: Python<'_>,
        model_dir: &str,
        quantization: Option<QuantizationConfigPy>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        ensure_threads(py, num_threads)?;
        let dir = Path::new(model_dir);
        let model = LlamaCausalLM::from_local(dir, quant(quantization)).map_err(any_err)?;
        Self::legacy(model, dir, model_dir.to_string())
    }

    /// Complete a raw text prompt. Returns the completion text (not the
    /// prompt), or a `Completion` with `details=True`.
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None, enable_thinking=None, ignore_eos=false, details=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
        enable_thinking: Option<bool>,
        ignore_eos: bool,
        details: bool,
    ) -> PyResult<PyObject> {
        let o = opts!(max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            frequency_penalty, presence_penalty, stop, seed, enable_thinking; ignore_eos);
        self.run(py, Prompt::Text(prompt), o, details)
    }

    /// Streaming `generate`: an iterator of text deltas (or `(kind, text)`
    /// tuples with `reasoning=True`).
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None, enable_thinking=None, ignore_eos=false, reasoning=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate_stream(
        &self,
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
        enable_thinking: Option<bool>,
        ignore_eos: bool,
        reasoning: bool,
    ) -> PyResult<TextStream> {
        let o = opts!(max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            frequency_penalty, presence_penalty, stop, seed, enable_thinking; ignore_eos);
        self.stream(Prompt::Text(prompt), o, reasoning)
    }

    /// Chat completion over `[{"role": ..., "content": ...}, ...]` with the
    /// model's chat template. Returns the assistant reply (reasoning
    /// excluded), or a `Completion` with `details=True`.
    #[pyo3(signature = (messages, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None, enable_thinking=None, ignore_eos=false, details=false))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        py: Python<'_>,
        messages: Vec<HashMap<String, String>>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
        enable_thinking: Option<bool>,
        ignore_eos: bool,
        details: bool,
    ) -> PyResult<PyObject> {
        let o = opts!(max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            frequency_penalty, presence_penalty, stop, seed, enable_thinking; ignore_eos);
        self.run(py, Prompt::Chat(to_messages(messages)?), o, details)
    }

    /// Streaming `chat`.
    #[pyo3(signature = (messages, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None, enable_thinking=None, ignore_eos=false, reasoning=false))]
    #[allow(clippy::too_many_arguments)]
    fn chat_stream(
        &self,
        messages: Vec<HashMap<String, String>>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        stop: Option<Vec<String>>,
        seed: Option<u64>,
        enable_thinking: Option<bool>,
        ignore_eos: bool,
        reasoning: bool,
    ) -> PyResult<TextStream> {
        let o = opts!(max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            frequency_penalty, presence_penalty, stop, seed, enable_thinking; ignore_eos);
        self.stream(Prompt::Chat(to_messages(messages)?), o, reasoning)
    }

    /// Render messages with the chat template (adds the generation prompt).
    #[pyo3(signature = (messages, enable_thinking=None))]
    fn apply_chat_template(
        &self,
        messages: Vec<HashMap<String, String>>,
        enable_thinking: Option<bool>,
    ) -> PyResult<String> {
        let msgs = to_messages(messages)?;
        let mut kw = serde_json::Map::new();
        if let Some(on) = enable_thinking {
            kw.insert("enable_thinking".into(), on.into());
        }
        lock(&self.generator)
            .apply_chat_template_with_kwargs(&msgs, true, &kw)
            .map_err(any_err)
    }

    #[pyo3(signature = (text, add_special_tokens=true))]
    fn tokenize(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        lock(&self.generator)
            .encode(text, add_special_tokens)
            .map_err(any_err)
    }

    #[pyo3(signature = (tokens, skip_special_tokens=true))]
    fn decode(&self, tokens: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        lock(&self.generator)
            .decode(&tokens, skip_special_tokens)
            .map_err(any_err)
    }

    /// Free the KV cache / recurrent state until the next request.
    fn reset(&self) {
        lock(&self.generator).release_state();
    }

    /// Legacy Llama config (None for Bonsai models).
    fn config(&self) -> Option<ModelConfig> {
        self.llama_config.clone().map(|inner| ModelConfig { inner })
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }
    /// "bonsai" or "llama".
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }
    #[getter]
    fn vocab_size(&self) -> usize {
        lock(&self.generator).model().vocab_size()
    }
    /// Maximum context length in tokens.
    #[getter]
    fn context_length(&self) -> usize {
        lock(&self.generator).model().max_seq_len()
    }
    #[getter]
    fn has_chat_template(&self) -> bool {
        lock(&self.generator).chat_template().is_some()
    }
    /// The chat template reads `enable_thinking`.
    #[getter]
    fn supports_thinking(&self) -> bool {
        lock(&self.generator)
            .chat_template()
            .is_some_and(|t| t.supports_enable_thinking())
    }

    fn __repr__(&self) -> String {
        format!("xor_net.Model(kind={:?}, path={:?})", self.kind, self.path)
    }
}

/// Load a language model.
///
/// `model` is a `.gguf` / `.xq1` Bonsai file, a Llama/BitNet safetensors
/// directory, or a Hugging Face repo id (`"prism-ml/Bonsai-1.7B-gguf"`, or
/// `"owner/repo:file.gguf"`), downloaded into the HF cache. `threads` sizes
/// the global thread pool on first use (default: auto, max 8).
#[pyfunction]
#[pyo3(signature = (model, threads=None, tokenizer=None, quantization=None))]
fn load(
    py: Python<'_>,
    model: &str,
    threads: Option<usize>,
    tokenizer: Option<PathBuf>,
    quantization: Option<QuantizationConfigPy>,
) -> PyResult<PyModel> {
    load_impl(py, model, threads, tokenizer, quantization)
}

// ---------------------------------------------------------------------------
// Vision
// ---------------------------------------------------------------------------

/// Decode `obj` into interleaved RGB8: a numpy `HxWx3`/`HxWx4`/`HxW`
/// uint8 array, a PIL image, raw RGB bytes (with width and height),
/// encoded JPEG/PNG bytes, or a path.
fn image_from_py(
    obj: &Bound<'_, PyAny>,
    width: Option<usize>,
    height: Option<usize>,
) -> PyResult<(Vec<u8>, usize, usize)> {
    let decode = |bytes: &[u8]| -> PyResult<(Vec<u8>, usize, usize)> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| PyValueError::new_err(format!("decoding image: {e}")))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Ok((img.into_raw(), w, h))
    };
    let raw_or_decode = |bytes: Vec<u8>| -> PyResult<(Vec<u8>, usize, usize)> {
        match (width, height) {
            (Some(w), Some(h)) => {
                if bytes.len() != w * h * 3 {
                    return Err(PyValueError::new_err(format!(
                        "raw RGB buffer has {} bytes, expected {w}*{h}*3 = {}",
                        bytes.len(),
                        w * h * 3
                    )));
                }
                Ok((bytes, w, h))
            }
            (None, None) => decode(&bytes),
            _ => Err(PyValueError::new_err(
                "pass both width and height for raw RGB bytes",
            )),
        }
    };
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return raw_or_decode(b.as_bytes().to_vec());
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        return raw_or_decode(b.to_vec());
    }
    // PIL.Image.Image
    if obj.hasattr("mode")? && obj.hasattr("size")? && obj.hasattr("convert")? {
        let rgb = obj.call_method1("convert", ("RGB",))?;
        let (w, h): (usize, usize) = rgb.getattr("size")?.extract()?;
        let data: Vec<u8> = rgb.call_method0("tobytes")?.extract()?;
        return Ok((data, w, h));
    }
    // numpy-like array
    if obj.hasattr("shape")? && obj.hasattr("dtype")? && obj.hasattr("tobytes")? {
        let dtype: String = obj.getattr("dtype")?.getattr("str")?.extract()?;
        if dtype != "|u1" {
            return Err(PyValueError::new_err(format!(
                "array dtype must be uint8 (got {dtype}); use arr.astype('uint8')"
            )));
        }
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        let data: Vec<u8> = obj.call_method0("tobytes")?.extract()?; // C order
        return match shape.as_slice() {
            [h, w, 3] => Ok((data, *w, *h)),
            [h, w, 4] => Ok((
                data.chunks_exact(4)
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect(),
                *w,
                *h,
            )),
            [h, w] => Ok((data.iter().flat_map(|&v| [v, v, v]).collect(), *w, *h)),
            s => Err(PyValueError::new_err(format!(
                "expected an HxWx3 (RGB), HxWx4 or HxW array, got shape {s:?}"
            ))),
        };
    }
    if let Ok(p) = obj.extract::<PathBuf>() {
        let bytes = std::fs::read(&p)
            .map_err(|e| PyValueError::new_err(format!("reading {}: {e}", p.display())))?;
        return decode(&bytes);
    }
    Err(PyValueError::new_err(
        "image must be a path, bytes, a numpy uint8 array or a PIL image",
    ))
}

/// Binary-CNN image classifier (e.g. QuickNet, `xornet-vision-v1` file).
#[pyclass(name = "ImageClassifier", frozen)]
pub struct PyImageClassifier {
    inner: vision::ImageClassifier,
}

#[pymethods]
impl PyImageClassifier {
    #[new]
    #[pyo3(signature = (path, threads=None))]
    fn new(py: Python<'_>, path: PathBuf, threads: Option<usize>) -> PyResult<Self> {
        ensure_threads(py, threads)?;
        let inner = py
            .allow_threads(|| vision::ImageClassifier::load(&path))
            .map_err(any_err)?;
        Ok(Self { inner })
    }

    /// Top-`k` classes for one image, as a list of
    /// `{"label", "probability", "class_index"}` dicts, most likely first.
    /// `image`: path, encoded JPEG/PNG bytes, raw RGB bytes (with `width`
    /// and `height`), numpy `HxWx3` uint8 array, or PIL image.
    #[pyo3(signature = (image, k=5, width=None, height=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        k: usize,
        width: Option<usize>,
        height: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let (rgb, w, h) = image_from_py(image, width, height)?;
        let preds = py
            .allow_threads(|| self.inner.predict(&rgb, w, h, k))
            .map_err(any_err)?;
        let out = PyList::empty(py);
        for p in preds {
            let d = PyDict::new(py);
            d.set_item("label", p.label)?;
            d.set_item("probability", p.probability)?;
            d.set_item("class_index", p.class_index)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Raw logits (one per class).
    #[pyo3(signature = (image, width=None, height=None))]
    fn logits(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        width: Option<usize>,
        height: Option<usize>,
    ) -> PyResult<Vec<f32>> {
        let (rgb, w, h) = image_from_py(image, width, height)?;
        py.allow_threads(|| self.inner.classify(&rgb, w, h))
            .map(|c| c.logits)
            .map_err(any_err)
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }
    #[getter]
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels().to_vec()
    }
    fn __repr__(&self) -> String {
        format!(
            "xor_net.ImageClassifier(name={:?}, num_classes={})",
            self.inner.name(),
            self.inner.num_classes()
        )
    }
}

#[pyfunction]
fn init_threads_py(num_threads: usize) -> PyResult<()> {
    init_threads(num_threads).map_err(py_err)
}

#[pymodule]
fn xor_net(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(init_threads_py, m)?)?;
    m.add_class::<PyModel>()?;
    m.add_class::<Completion>()?;
    m.add_class::<TextStream>()?;
    m.add_class::<PyImageClassifier>()?;
    m.add_class::<QuantizationConfigPy>()?;
    m.add_class::<ModelConfig>()?;
    // Backwards-compatible names.
    m.add("XorNetModel", m.getattr("Model")?)?;
    m.add("QuantizationConfig", m.getattr("QuantizationConfigPy")?)?;
    m.add("init_threads", m.getattr("init_threads_py")?)?;
    Ok(())
}
