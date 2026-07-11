#![allow(unsafe_op_in_unsafe_fn)]

pub mod bit1;
pub mod bit1_58;
pub mod loader;
pub mod tensor;
pub mod util;
pub mod nn;
pub mod models;
pub mod sampler;

pub use tensor::FastTensor;
pub use loader::{SafeTensorRepo, SafeTensorLoader};
pub use bit1::layers::BitLinear;
pub use bit1_58::layers::TernaryLinear;
pub use bit1_58::quantization::TernaryPackType;
pub use models::auto::AutoModelForCausalLM;
pub use nn::{QuantizationConfig, DynamicLinear};
pub use models::llama::{Cache, Config, Llama, LlamaConfig, Activation};

pub fn init_threads(num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;
    Ok(())
}
