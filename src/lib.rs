#![allow(unsafe_op_in_unsafe_fn)]

pub mod batch_session;
pub mod bit1;
pub mod bit1_58;
pub mod generation;
pub mod loader;
pub mod models;
pub mod nn;
pub mod prompt_cache;
pub mod sampler;
pub mod speculative;
pub mod tensor;
pub mod util;
pub mod vision;

#[cfg(feature = "python")]
pub mod python;

pub use bit1::layers::BitLinear;
pub use bit1_58::layers::TernaryLinear;
pub use bit1_58::quantization::TernaryPackType;
pub use generation::{
    ChatMessage, ChatTemplate, FinishReason, Generation, GenerationError, GenerationParams,
    Generator, ReasoningFormat, StreamEvent, Timings, Usage,
};
pub use loader::gguf::{BonsaiConfig, GgufFile, is_prism_dtype};
pub use loader::{SafeTensorLoader, SafeTensorRepo};
pub use models::auto::AutoModelForCausalLM;
pub use models::batched::{BatchItem, BatchedCausalLM};
pub use models::llama::causal_lm::LlamaCausalLM;
pub use models::llama::{Activation, Cache, Config, Llama, LlamaConfig};
pub use models::q1::qwen3::{Qwen3Config, Qwen3Model};
pub use models::q1::qwen35::{Qwen35Config, Qwen35Model};
pub use models::traits::{CausalLM, Logits, ModelState, StateSnapshot};
pub use nn::{DynamicLinear, QuantizationConfig};
pub use prompt_cache::{PromptCache, PromptCacheStats};
pub use sampler::{Sampler, SamplingParams};
pub use tensor::FastTensor;

pub fn init_threads(num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Auto ("0"): cap the pool. The BitNet decode GEMV is memory/cache
    // bound, not core-bound  beyond ~8 threads on a 2-channel DDR5 box we
    // only add L2/L3 thrash + pool-contention and *regress*
    // (measured: 22/40/57/65 tok/s at 1/2/3/4 threads, then flat or
    // slower at 6/8/12). This single change lifts single-stream decode
    // from ~58 to ~65 tok/s on the target 6C/12T Zen5.
    let n = if num_threads == 0 {
        std::cmp::min(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
            8,
        )
        .max(1)
    } else {
        num_threads
    };
    let mut b = rayon::ThreadPoolBuilder::new().num_threads(n);
    // `XORNET_PIN=1`: pin worker i to one CPU, physical cores first (see
    // `util::pin_plan`). Linux only; without a readable topology, or with
    // fewer allowed CPUs than workers, the pool stays unpinned.
    if std::env::var("XORNET_PIN").is_ok_and(|v| v.trim() == "1")
        && let Some(plan) = util::pin_plan(n)
    {
        b = b.start_handler(move |i| {
            if let Some(&cpu) = plan.get(i) {
                util::pin_current_thread(cpu);
            }
        });
    }
    b.build_global()?;
    Ok(())
}
