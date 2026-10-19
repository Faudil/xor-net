pub mod dynamic_linear;
pub use dynamic_linear::{DynamicLinear, LinearKind, LmHeadConfig, QuantizationConfig};

pub mod fast_attention;
pub mod fast_attention_simd;
pub mod fast_rmsnorm;

pub use fast_attention::{CpuRingCache, fast_attention};
pub use fast_rmsnorm::FastRmsNorm;
