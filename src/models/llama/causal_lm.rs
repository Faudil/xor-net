//! [`CausalLM`] implementation for [`Llama`] (BitNet / Llama-architecture
//! checkpoints).

use std::path::Path;

use super::{Cache, Config, Llama, LlamaEosToks, RopeTables};
use crate::models::auto::AutoModelForCausalLM;
use crate::models::traits::{CausalLM, Logits, ModelState, StateSnapshot, downcast_state};
use crate::nn::QuantizationConfig;

impl ModelState for Cache {
    fn reset(&mut self) -> anyhow::Result<()> {
        // The ring KV cache is position-indexed and attention only reads
        // positions `< pos + seq_len`, all of which are rewritten when a new
        // sequence is fed from position 0. Nothing to clear.
        Ok(())
    }

    fn truncate(&mut self, _len: usize) -> bool {
        // Position-indexed: positions `>= len` are rewritten before attention
        // reads them (it only reads `< pos + seq_len`).
        true
    }

    /// Compact copy of the f32 K/V rows `< len` of every layer. Not
    /// supported with the optional u8 cache or once positions have wrapped.
    fn snapshot(&self, len: usize) -> Option<StateSnapshot> {
        let rc = self.ring_cache.as_ref()?;
        if !rc.u8_buffers.is_empty() || len > rc.max_seq_len {
            return None;
        }
        let (heads, hd, cap) = (rc.num_kv_heads, rc.head_dim, rc.max_seq_len);
        let take = |bufs: &[Vec<f32>]| -> Vec<Vec<f32>> {
            bufs.iter()
                .map(|b| {
                    let mut out = Vec::with_capacity(heads * len * hd);
                    if !b.is_empty() {
                        for h in 0..heads {
                            out.extend_from_slice(&b[h * cap * hd..(h * cap + len) * hd]);
                        }
                    }
                    out
                })
                .collect()
        };
        let snap = LlamaSnapshot {
            k: take(&rc.k_buffers),
            v: take(&rc.v_buffers),
            heads,
            head_dim: hd,
        };
        let bytes = snap.k.iter().chain(&snap.v).map(|b| b.len() * 4).sum();
        Some(StateSnapshot::new(len, bytes, snap))
    }

    fn restore(&mut self, snap: &StateSnapshot) -> bool {
        let len = snap.len();
        let (Some(s), Some(rc)) = (
            snap.downcast_ref::<LlamaSnapshot>(),
            self.ring_cache.as_mut(),
        ) else {
            return false;
        };
        let (heads, hd) = (rc.num_kv_heads, rc.head_dim);
        let rows = heads * len * hd;
        if !rc.u8_buffers.is_empty()
            || (s.heads, s.head_dim) != (heads, hd)
            || s.k.len() != rc.k_buffers.len()
            || s.v.len() != rc.v_buffers.len()
            || len > rc.max_capacity
            || s.k.iter().chain(&s.v).any(|b| b.len() != rows)
        {
            return false;
        }
        rc.reserve(len);
        let cap = rc.max_seq_len;
        if len > cap {
            return false;
        }
        for (dst, src) in rc
            .k_buffers
            .iter_mut()
            .zip(&s.k)
            .chain(rc.v_buffers.iter_mut().zip(&s.v))
        {
            for h in 0..heads {
                dst[h * cap * hd..(h * cap + len) * hd]
                    .copy_from_slice(&src[h * len * hd..(h + 1) * len * hd]);
            }
        }
        true
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// [`StateSnapshot`] payload: per layer `[heads][len][head_dim]` f32 K / V.
struct LlamaSnapshot {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    heads: usize,
    head_dim: usize,
}

/// A loaded [`Llama`] plus the config fields generation needs.
pub struct LlamaCausalLM {
    model: Llama,
    config: Config,
    eos: Vec<u32>,
    /// Shared by every state this model creates.
    rope: RopeTables,
}

impl LlamaCausalLM {
    pub fn new(model: Llama, config: Config) -> Self {
        let eos = match &config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => vec![*id],
            Some(LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        };
        let rope = RopeTables::new(&config);
        Self {
            model,
            config,
            eos,
            rope,
        }
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
        Ok(Box::new(Cache::with_rope(true, &self.config, &self.rope)?))
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
