//! Compute kernels for the vision module.
//!
//! * [`binary`]: bit-packing of activations and XNOR/popcount convolution
//!   (AVX-512 VPOPCNTDQ, AVX2, scalar), plus a float reference.
//! * [`float`]: real-valued conv / depthwise / pooling / dense.
//! * [`gemm`]: small SGEMM (AVX-512 micro-kernel + portable fallback).
//! * [`epilogue`]: fused ReLU / batch-norm / residual tails.

pub mod binary;
pub mod epilogue;
pub mod float;
pub mod gemm;
