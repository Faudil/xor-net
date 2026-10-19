//! CPU inference for binarized (1-bit) image classification networks.
//!
//! The first supported family is **Larq Zoo QuickNet** (QuickNet,
//! QuickNetSmall, QuickNetLarge; Apache-2.0, ImageNet top-1 63.3 / 59.4 /
//! 66.9 %). Its 3x3 convolutions are fully binary (sign activations *and*
//! sign weights, no scaling factors), so they run as XNOR + popcount over
//! bit-packed channels; the stem, the downsampling transitions and the
//! classifier stay real-valued, as in the original model.
//!
//! # Pipeline
//!
//! ```text
//! RGB8 --[preprocess: crop + TF bicubic + normalise]--> f32 NHWC
//!      --[graph: conv / binary conv / depthwise / pool / dense, fused
//!         ReLU + BN + residual epilogues]--> logits --> softmax / top-k
//! ```
//!
//! # Model files
//!
//! Models are converted offline by `scripts/vision/convert_larq_zoo.py`
//! into a single safetensors file (format `xornet-vision-v1`): binary conv
//! weights are pre-packed to sign bits (`u64 [cout, kh, kw, ceil(cin/64)]`),
//! batch norms are rewritten as per-channel `scale`/`bias`, and the layer
//! graph, preprocessing parameters and ImageNet labels are stored as JSON
//! metadata. See the script for the exact schema.
//!
//! # Soundness
//!
//! * Every graph is shape-checked end to end at load time.
//! * SIMD kernels are selected at runtime (`avx512f + avx512vpopcntdq`,
//!   `avx2`, scalar) and each one is tested bit-exactly against a plain
//!   float reference implementation.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let model = xor_net::vision::ImageClassifier::load("quicknet.safetensors")?;
//! # let (rgb, w, h) = (vec![0u8; 3 * 64 * 64], 64, 64);
//! let result = model.classify(&rgb, w, h)?;
//! for (class, p) in result.top_k(5) {
//!     println!("{p:.3} {}", model.label(class).unwrap());
//! }
//! # Ok(()) }
//! ```

pub mod classifier;
pub mod graph;
pub mod kernels;
pub mod preprocess;
pub mod tensor;

pub use classifier::{Classification, ImageClassifier, Prediction, softmax, top_k};
pub use graph::{Network, OpSpec, TensorSource};
pub use kernels::binary::{BinaryConvWeights, BinaryKernel, PadValue};
pub use kernels::epilogue::Epilogue;
pub use preprocess::{ImageError, Preprocess};
pub use tensor::{FeatureMap, PackedMap, Padding};
