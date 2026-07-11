pub mod bit1;
pub mod bit1_58;
pub mod loader;
pub mod tensor;

pub use tensor::FastTensor;
pub use loader::{SafeTensorRepo, SafeTensorLoader};

pub use bit1::layers::BitLinear;
pub use bit1_58::layers::TernaryLinear;
pub use bit1_58::quantization::TernaryPackType;

/// Configures the global Rayon thread pool with the specified number of threads.
/// This allows the user to restrict the number of CPU cores used by xor-net during matrix multiplications.
/// If `num_threads` is 0, Rayon will automatically use all available logical cores.
/// 
/// Note: This must be called before any parallel operations are executed.
pub fn init_threads(num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;
    Ok(())
}

pub mod nn;
pub mod models;

pub use models::auto::AutoModelForCausalLM;
pub use nn::{QuantizationConfig, DynamicLinear};

pub use models::llama::{Cache, Config, Llama, LlamaConfig, Activation};
