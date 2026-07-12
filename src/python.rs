use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::Path;
use std::sync::Arc;

use crate::models::auto::AutoModelForCausalLM;
use crate::models::llama::{Cache, Config, Llama, LlamaConfig};
use crate::bit1_58::quantization::TernaryPackType;
use crate::nn::{QuantizationConfig, LmHeadConfig};
use crate::init_threads;

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
            _ => return Err(PyRuntimeError::new_err(format!("Unknown pack type: {pack_type}"))),
        };
        let lm_head_config = match lm_head.to_lowercase().as_str() {
            "int4" => LmHeadConfig::Int4,
            "int8" => LmHeadConfig::Int8,
            "fp32" => LmHeadConfig::F32,
            _ => return Err(PyRuntimeError::new_err(format!("Unknown lm_head config: {lm_head}"))),
        };
        Ok(Self {
            inner: QuantizationConfig::Bit1_58(pack, lm_head_config),
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
        let llama_config: LlamaConfig = serde_json::from_str(json_str)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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

#[pyclass]
pub struct XorNetModel {
    model: Llama,
    config: Config,
    cache: Cache,
    tokenizer: Arc<tokenizers::Tokenizer>,
    logits_processor: candle_transformers::generation::LogitsProcessor,
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
            init_threads(n).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        let quant_config = quantization.map(|q| q.inner).unwrap_or(QuantizationConfig::None);

        let (model, config) = AutoModelForCausalLM::from_pretrained(model_id, quant_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let api = hf_hub::api::sync::Api::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let repo = api.repo(hf_hub::Repo::new(model_id.to_string(), hf_hub::RepoType::Model));

        let tokenizer_file = repo.get("tokenizer.json").map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        );

        let cache = Cache::new(true, &config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let logits_processor = candle_transformers::generation::LogitsProcessor::new(299792458, Some(0.7_f64), None);

        Ok(Self {
            model,
            config,
            cache,
            tokenizer,
            logits_processor,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (model_dir, quantization=None, num_threads=None))]
    fn from_local(
        model_dir: &str,
        quantization: Option<QuantizationConfigPy>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(n) = num_threads {
            init_threads(n).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        let quant_config = quantization.map(|q| q.inner).unwrap_or(QuantizationConfig::None);

        let model_dir = Path::new(model_dir);
        let (model, config) = AutoModelForCausalLM::from_local(model_dir, quant_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tokenizer_file = model_dir.join("tokenizer.json");
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        );

        let cache = Cache::new(true, &config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let logits_processor = candle_transformers::generation::LogitsProcessor::new(299792458, Some(0.7_f64), None);

        Ok(Self {
            model,
            config,
            cache,
            tokenizer,
            logits_processor,
        })
}
 
#[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None))]
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let max_tokens = max_tokens.unwrap_or(512);
        let temperature = temperature.unwrap_or(0.7) as f64;
        let top_p = top_p.unwrap_or(0.9) as f64;
        let _top_k = top_k.unwrap_or(50);
        let seed = seed.unwrap_or(299792458);

        self.logits_processor = candle_transformers::generation::LogitsProcessor::new(seed, Some(temperature), Some(top_p));

        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut tokens = encoding.get_ids().to_vec();

        let bos_token_id = self.config.bos_token_id.unwrap_or(1);
        let eos_token_ids: Vec<u32> = match &self.config.eos_token_id {
            Some(crate::models::llama::LlamaEosToks::Single(id)) => vec![*id],
            Some(crate::models::llama::LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => vec![2],
        };

        let mut index_pos = 0;
        let mut _generated = 0;
 
        for _ in 0..max_tokens {
            let context_size = if index_pos == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len().saturating_sub(context_size);
 
            let logits = self.model.forward(&tokens[start_pos..], index_pos, &mut self.cache)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
 
            let logits_shape = logits.shape[2];
            let logits_tensor = candle_core::Tensor::from_vec(logits.into_data(), (logits_shape,), &candle_core::Device::Cpu)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
 
            let next_token = self.logits_processor.sample(&logits_tensor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
 
tokens.push(next_token);
            index_pos += context_size;
            _generated += 1;

            if next_token == bos_token_id || eos_token_ids.contains(&next_token) {
                break;
            }
        }

        let output = self.tokenizer.decode(&tokens, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(output)
    }
 
#[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None))]
    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        let max_tokens = max_tokens.unwrap_or(512);
        let temperature = temperature.unwrap_or(0.7) as f64;
        let top_p = top_p.unwrap_or(0.9) as f64;
        let _top_k = top_k.unwrap_or(50);
        let seed = seed.unwrap_or(299792458);

        self.logits_processor = candle_transformers::generation::LogitsProcessor::new(seed, Some(temperature), Some(top_p));

        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut tokens = encoding.get_ids().to_vec();

        let bos_token_id = self.config.bos_token_id.unwrap_or(1);
        let eos_token_ids: Vec<u32> = match &self.config.eos_token_id {
            Some(crate::models::llama::LlamaEosToks::Single(id)) => vec![*id],
            Some(crate::models::llama::LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => vec![2],
        };

        let mut index_pos = 0;
        let mut _generated = 0;
        let mut outputs = Vec::new();

        for _ in 0..max_tokens {
            let context_size = if index_pos == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len().saturating_sub(context_size);

            let logits = self.model.forward(&tokens[start_pos..], index_pos, &mut self.cache)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let logits_shape = logits.shape[2];
            let logits_tensor = candle_core::Tensor::from_vec(logits.into_data(), (logits_shape,), &candle_core::Device::Cpu)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let next_token = self.logits_processor.sample(&logits_tensor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            tokens.push(next_token);
            index_pos += context_size;
            _generated += 1;

            let current_text = self.tokenizer.decode(&tokens, true)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            outputs.push(current_text);

            if next_token == bos_token_id || eos_token_ids.contains(&next_token) {
                break;
            }
        }

        Ok(outputs)
    }

    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.tokenizer.decode(&tokens, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn config(&self) -> ModelConfig {
        ModelConfig { inner: self.config.clone() }
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
    Ok(())
}