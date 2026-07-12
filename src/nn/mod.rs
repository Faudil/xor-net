pub mod dynamic_linear;
pub use dynamic_linear::{DynamicLinear, QuantizationConfig, LinearKind, LmHeadConfig};

pub mod fast_rmsnorm;
pub mod fast_attention;
pub mod fast_attention_simd;

pub use fast_rmsnorm::FastRmsNorm;
pub use fast_attention::{fast_attention, CpuRingCache};
