//! [`CausalLM`] implementation for [`Llama`] (BitNet / Llama-architecture
//! checkpoints).

use std::path::Path;

use super::{Cache, Config, Llama, LlamaEosToks};
use crate::models::auto::AutoModelForCausalLM;
use crate::models::traits::{CausalLM, Logits, ModelState, downcast_state};
use crate::nn::QuantizationConfig;

impl ModelState for Cache {
    fn reset(&mut self) -> anyhow::Result<()> {
        // The ring KV cache is position-indexed and attention only reads
        // positions `< pos + seq_len`, all of which are rewritten when a new
        // sequence is fed from position 0. Nothing to clear.
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// A loaded [`Llama`] plus the config fields generation needs.
pub struct LlamaCausalLM {
    model: Llama,
    config: Config,
    eos: Vec<u32>,
}

impl LlamaCausalLM {
    pub fn new(model: Llama, config: Config) -> Self {
        let eos = match &config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => vec![*id],
            Some(LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        };
        Self { model, config, eos }
    }

    /// Load from a local HF-style directory (`config.json` + safetensors).
    pub fn from_local(dir: &Path, quantization: QuantizationConfig) -> anyhow::Result<Self> {
        let (model, config) = AutoModelForCausalLM::from_local(dir, quantization)?;
        Ok(Self::new(model, config))
    }

    /// Load from the HF hub.
    pub fn from_pretrained(
        repo_id: &str,
        quantization: QuantizationConfig,
    ) -> anyhow::Result<Self> {
        let (model, config) = AutoModelForCausalLM::from_pretrained(repo_id, quantization)?;
        Ok(Self::new(model, config))
    }

    pub fn model(&self) -> &Llama {
        &self.model
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl CausalLM for LlamaCausalLM {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(Cache::new(true, &self.config)?))
    }

    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        anyhow::ensure!(!tokens.is_empty(), "forward: empty token slice");
        anyhow::ensure!(
            pos + tokens.len() <= self.config.max_position_embeddings,
            "forward: positions {}..{} exceed max_seq_len {}",
            pos,
            pos + tokens.len(),
            self.config.max_position_embeddings
        );
        let cache = downcast_state::<Cache>(state)?;
        // `Llama::forward` already projects only the last position.
        let mut logits = self.model.forward(tokens, pos, cache)?.into_data();
        let vocab = self.config.vocab_size;
        if logits.len() > vocab {
            logits.drain(..logits.len() - vocab);
        }
        anyhow::ensure!(
            logits.len() == vocab,
            "forward: expected {vocab} logits, got {}",
            logits.len()
        );
        Ok(logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_seq_len(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.config.bos_token_id
    }
}
