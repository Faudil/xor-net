use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use crate::bit1_58::quantization::TernaryPackType;
use crate::generation::{ChatMessage, GenerationParams, Generator};
use crate::init_threads;
use crate::models::llama::causal_lm::LlamaCausalLM;
use crate::models::llama::{Config, LlamaConfig};
use crate::nn::{LmHeadConfig, QuantizationConfig};
use crate::sampler::SamplingParams;

#[pyclass]
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
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown pack type: {pack_type}"
                )));
            }
        };
        let lm_head_config = match lm_head.to_lowercase().as_str() {
            "int4" => LmHeadConfig::Int4,
            "int8" => LmHeadConfig::Int8,
            "fp32" => LmHeadConfig::F32,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown lm_head config: {lm_head}"
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
        let llama_config: LlamaConfig =
            serde_json::from_str(json_str).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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

fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn lock(g: &Mutex<Generator>) -> std::sync::MutexGuard<'_, Generator> {
    g.lock().unwrap_or_else(|e| e.into_inner())
}

/// Python-side sampling / stopping options, shared by every entry point.
#[allow(clippy::too_many_arguments)]
fn build_params(
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
) -> GenerationParams {
    GenerationParams::default()
        .with_max_tokens(max_tokens.unwrap_or(256))
        .with_stop(stop.unwrap_or_default())
        .with_sampling(SamplingParams {
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(0.9),
            top_k: top_k.unwrap_or(0),
            min_p: min_p.unwrap_or(0.0),
            repetition_penalty: repetition_penalty.unwrap_or(1.0),
            frequency_penalty: frequency_penalty.unwrap_or(0.0),
            presence_penalty: presence_penalty.unwrap_or(0.0),
            penalty_last_n: None,
            seed,
        })
}

fn to_messages(messages: Vec<HashMap<String, String>>) -> PyResult<Vec<ChatMessage>> {
    messages
        .into_iter()
        .map(|m| {
            let role = m.get("role").cloned();
            let content = m.get("content").cloned();
            match (role, content) {
                (Some(r), Some(c)) => Ok(ChatMessage::new(r, c)),
                _ => Err(PyRuntimeError::new_err(
                    "each message needs 'role' and 'content' keys",
                )),
            }
        })
        .collect()
}

enum StreamMsg {
    Delta(String),
    Done(&'static str),
    Error(String),
}

/// Iterator over text deltas of a streaming generation. Generation runs on
/// a background thread; dropping the iterator (or calling `cancel()`)
/// stops it at the next token.
#[pyclass]
pub struct TextStream {
    rx: Mutex<Receiver<StreamMsg>>,
    cancel: Arc<AtomicBool>,
    finish_reason: Option<&'static str>,
}

#[pymethods]
impl TextStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        if self.finish_reason.is_some() {
            return Ok(None);
        }
        let rx = &self.rx;
        match py.allow_threads(|| rx.lock().unwrap_or_else(|e| e.into_inner()).recv()) {
            Ok(StreamMsg::Delta(s)) => Ok(Some(s)),
            Ok(StreamMsg::Done(reason)) => {
                self.finish_reason = Some(reason);
                Ok(None)
            }
            Ok(StreamMsg::Error(e)) => {
                self.finish_reason = Some("error");
                Err(PyRuntimeError::new_err(e))
            }
            Err(_) => {
                self.finish_reason = Some("error");
                Err(PyRuntimeError::new_err(
                    "generation thread ended unexpectedly",
                ))
            }
        }
    }

    /// "stop" / "length" once the stream is exhausted, else None.
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.finish_reason
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for TextStream {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[pyclass]
pub struct XorNetModel {
    generator: Arc<Mutex<Generator>>,
    config: Config,
}

impl XorNetModel {
    fn build(model: LlamaCausalLM, tokenizer_dir: &Path) -> PyResult<Self> {
        let config = model.config().clone();
        let generator = Generator::from_dir(Arc::new(model), tokenizer_dir).map_err(py_err)?;
        Ok(Self {
            generator: Arc::new(Mutex::new(generator)),
            config,
        })
    }

    fn run(&self, py: Python<'_>, prompt: Prompt, params: GenerationParams) -> PyResult<String> {
        let generator = &self.generator;
        py.allow_threads(move || {
            let mut g = lock(generator);
            match prompt {
                Prompt::Text(p) => g.generate(&p, &params),
                Prompt::Chat(m) => g.chat(&m, &params),
            }
        })
        .map(|out| out.text)
        .map_err(py_err)
    }

    fn stream(&self, prompt: Prompt, params: GenerationParams) -> TextStream {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let generator = self.generator.clone();
        let stop = cancel.clone();
        std::thread::spawn(move || {
            let mut g = lock(&generator);
            let on_text = |d: &str| {
                if stop.load(Ordering::Relaxed) || tx.send(StreamMsg::Delta(d.to_owned())).is_err()
                {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let res = match prompt {
                Prompt::Text(p) => g.generate_stream(&p, &params, on_text),
                Prompt::Chat(m) => g.chat_stream(&m, &params, on_text),
            };
            let _ = tx.send(match res {
                Ok(out) => StreamMsg::Done(out.finish_reason.as_openai_str()),
                Err(e) => StreamMsg::Error(format!("{e:#}")),
            });
        });
        TextStream {
            rx: Mutex::new(rx),
            cancel,
            finish_reason: None,
        }
    }
}

enum Prompt {
    Text(String),
    Chat(Vec<ChatMessage>),
}

#[pymethods]
impl XorNetModel {
    #[staticmethod]
    #[pyo3(signature = (model_id, quantization=None, num_threads=None))]
    fn from_pretrained(
        model_id: &str,
        quantization: Option<QuantizationConfigPy>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(n) = num_threads {
            init_threads(n).map_err(py_err)?;
        }
        let quant_config = quantization
            .map(|q| q.inner)
            .unwrap_or(QuantizationConfig::None);
        let model = LlamaCausalLM::from_pretrained(model_id, quant_config).map_err(py_err)?;

        let api = hf_hub::api::sync::Api::new().map_err(py_err)?;
        let repo = api.repo(hf_hub::Repo::new(
            model_id.to_string(),
            hf_hub::RepoType::Model,
        ));
        let tokenizer_file = repo.get("tokenizer.json").map_err(py_err)?;
        // Optional files land in the same snapshot directory.
        for optional in ["tokenizer_config.json", "generation_config.json"] {
            let _ = repo.get(optional);
        }
        let dir = tokenizer_file
            .parent()
            .ok_or_else(|| py_err("bad tokenizer path"))?
            .to_path_buf();
        Self::build(model, &dir)
    }

    #[staticmethod]
    #[pyo3(signature = (model_dir, quantization=None, num_threads=None))]
    fn from_local(
        model_dir: &str,
        quantization: Option<QuantizationConfigPy>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(n) = num_threads {
            init_threads(n).map_err(py_err)?;
        }
        let quant_config = quantization
            .map(|q| q.inner)
            .unwrap_or(QuantizationConfig::None);
        let dir = Path::new(model_dir);
        let model = LlamaCausalLM::from_local(dir, quant_config).map_err(py_err)?;
        Self::build(model, dir)
    }

    /// Returns the completion only (not the prompt).
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None))]
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
    ) -> PyResult<String> {
        let params = build_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        );
        self.run(py, Prompt::Text(prompt), params)
    }

    /// Returns an iterator of text deltas.
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None))]
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
    ) -> TextStream {
        let params = build_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        );
        self.stream(Prompt::Text(prompt), params)
    }

    /// Chat completion over `[{"role": ..., "content": ...}, ...]` using the
    /// model's chat template. Returns the assistant reply.
    #[pyo3(signature = (messages, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None))]
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
    ) -> PyResult<String> {
        let params = build_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        );
        self.run(py, Prompt::Chat(to_messages(messages)?), params)
    }

    /// Streaming chat: an iterator of text deltas.
    #[pyo3(signature = (messages, max_tokens=None, temperature=None, top_p=None, top_k=None,
        min_p=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None,
        stop=None, seed=None))]
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
    ) -> PyResult<TextStream> {
        let params = build_params(
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            stop,
            seed,
        );
        Ok(self.stream(Prompt::Chat(to_messages(messages)?), params))
    }

    /// Render messages with the chat template (adds the generation prompt).
    fn apply_chat_template(&self, messages: Vec<HashMap<String, String>>) -> PyResult<String> {
        let msgs = to_messages(messages)?;
        lock(&self.generator)
            .apply_chat_template(&msgs, true)
            .map_err(py_err)
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        lock(&self.generator).encode(text, true).map_err(py_err)
    }

    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        lock(&self.generator).decode(&tokens, true).map_err(py_err)
    }

    fn config(&self) -> ModelConfig {
        ModelConfig {
            inner: self.config.clone(),
        }
    }
}

#[pyfunction]
fn init_threads_py(num_threads: usize) -> PyResult<()> {
    init_threads(num_threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn xor_net(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_threads_py, m)?)?;
    m.add_class::<QuantizationConfigPy>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<XorNetModel>()?;
    m.add_class::<TextStream>()?;
    Ok(())
}
