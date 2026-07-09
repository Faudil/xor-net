# xor-net

`xor-net` is a high-performance CPU-side inference engine implementing **1-bit (binary)** and **1.58-bit (ternary)** matrix multiplications for the [Hugging Face Candle](https://github.com/huggingface/candle) machine learning framework in Rust. 

By replacing standard floating-point matrix multiplications (`f32`) with bitwise operations and integer arithmetic, `xor-net` dramatically reduces memory consumption and optimizes execution speeds on modern CPU architectures using SIMD intrinsics.

---

## Features

- **1-Bit Quantization (BitNet b1.0)**:
  - Dynamically binarizes activation inputs to sign-based values.
  - Compresses weights and activations into row-aligned bitpacked vectors (`1 bit per value`).
  - Executes matrix multiplication using bitwise **XNOR and Popcount** operations.
  - Leverages AVX2 instructions (`_mm256_xor_si256` and a vectorized `vpshufb` popcount lookup) to process 256 bits per cycle, with a fast fallback to hardware-accelerated scalar `count_ones()`.
  - Embarrassingly parallel execution mapped concurrently across matrix rows via the **Rayon** global thread pool.

- **1.58-Bit Quantization (BitNet b1.58)**:
  - Supports ternary weights `{-1, 0, 1}` using two packing layouts:
    - **Pack4**: 4 weights packed per byte (2 bits per weight). Extremely fast decoding.
    - **Pack5**: 5 weights packed per byte (base-3 arithmetic). Max memory density (20% reduction over Pack4).
  - Employs dynamic **absmax quantization** to scale `f32` input activations to `i8` (`[-127, 127]`).
  - Utilizes the AVX2 `_mm256_sign_epi8` instruction to perform ternary multiplication natively (negating, zeroing, or retaining activation bytes based on ternary signs) without any floating-point arithmetic.
  - Accumulates products directly into `i32` using integer vector addition (`_mm256_maddubs_epi16` and `_mm256_madd_epi16`).
  - Native multi-core scalability distributing chunked dot-products across available CPUs using **Rayon**.

- **Integration**:
  - Exposes drop-in, simplified top-level `BitLinear` and `TernaryLinear` layers implementing Candle's standard `Module` trait.

---

## Project Structure

```text
src/
├── lib.rs                 # Top-level module exports and ThreadPool initializers
├── bit1/                  # 1-bit Quantization Engine
│   ├── mod.rs
│   ├── layers.rs          # BitLinear implementation
│   ├── ops.rs             # Custom CustomOp1 CPU forward pass
│   ├── quantization.rs    # Row-wise bitpacking logic
│   └── simd.rs            # AVX2 XNOR/Popcount kernel
└── bit1_58/               # 1.58-bit Quantization Engine
    ├── mod.rs
    ├── layers.rs          # TernaryLinear implementation
    ├── ops.rs             # Custom CustomOp1 CPU forward pass (absmax + de-quantize)
    ├── quantization.rs    # Ternary packing & i8 activation scaling
    └── simd.rs            # AVX2 _mm256_sign_epi8 kernel
tests/
├── bit1_tests.rs          # Integration tests for 1-bit layers
└── bit1_58_tests.rs       # Integration tests for 1.58-bit layers
examples/
├── simple_usage.rs        # Basic usage of 1-bit and 1.58-bit layers
├── jepa_policy.rs         # Low-precision JEPA transformer block example
├── run_3b.rs              # Running inference on a quantized 3B LLaMA model
└── run_benchmark.rs       # Performance benchmarks comparing Ternary vs Baseline F32 LLaMA
```

---

## Core Performance Kernels

### 1-Bit Dot Product (AVX2 XNOR + Popcount)
```rust
// bit1/simd.rs
let xored = _mm256_xor_si256(va, vb);
let ones = _mm256_set1_epi32(-1);
let xnored = _mm256_xor_si256(xored, ones); // XNOR

let lo = _mm256_and_si256(xnored, low_mask);
let hi = _mm256_and_si256(_mm256_srli_epi16(xnored, 4), low_mask);
let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
let popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
```

### 1.58-Bit Dot Product (AVX2 i8 Sign Negation)
```rust
// bit1_58/simd.rs
// Multiplies i8 activations by ternary weights {-1, 0, 1} instantly
let prod = _mm256_sign_epi8(acts, w_i8);

// Accumulate i8 -> i16 -> i32
let sums_i16 = _mm256_maddubs_epi16(ones_u8, prod);
let sums_i32 = _mm256_madd_epi16(sums_i16, ones_i16);
```

### Parallelization with Rayon
To maximize performance, the matrix multiplication kernel maps dot-products concurrently over the output dimension using Rayon's work-stealing thread pool. This allows `xor-net` to fully saturate modern multi-core CPUs.

If you are integrating `xor-net` into an environment where threads must be restricted, you can manually override the global thread limit:
```rust
// Limit xor-net processing to 4 threads
xor_net::init_threads(4).unwrap();
```

---

## Getting Started

### Prerequisites

- A Rust toolchain supporting the **2024 edition**
- An **x86_64** CPU with AVX2 instruction support. If AVX2 is not available at runtime, execution automatically falls back to scalar implementations.

### Running Examples

- **Simple Layer Usage**:
  ```bash
  cargo run --example simple_usage
  ```

- **Low-Precision JEPA Block**:
  ```bash
  cargo run --example jepa_policy
  ```

- **LLaMA 3B Inference**:
  To execute a generation loop on Microsoft's `1bitLLM/bitnet_b1_58-3B` model (fully packed to 750 MB instead of 12 GB):
  ```bash
  cargo run --release --example run_3b
  ```

- **Inference Speed/Memory Benchmarking**:
  ```bash
  cargo run --release --example run_benchmark -- --mode=ternary
  cargo run --release --example run_benchmark -- --mode=baseline
  ```

---

## Usage Example

```rust
use candle_core::{Device, Tensor, Module};
use xor_net::{BitLinear, TernaryLinear, TernaryPackType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let in_dim = 4;
    let out_dim = 2;

    // Define weights
    let bit_weights = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
    let ternary_weights = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0];

    // Initialize layers (will pack the weights under the hood)
    let bit_layer = BitLinear::new(in_dim, out_dim, &bit_weights)?;
    let ternary_layer = TernaryLinear::new(in_dim, out_dim, &ternary_weights, TernaryPackType::Pack4)?;

    // Define inputs
    let input = Tensor::from_vec(vec![0.5f32, 1.0, -0.5, 2.0], (1, in_dim), &device)?;

    // Run inference
    let out_bit = bit_layer.forward(&input)?;
    let out_ternary = ternary_layer.forward(&input)?;

    println!("BitNet 1-bit output: {}", out_bit);
    println!("BitNet 1.58-bit output: {}", out_ternary);

    Ok(())
}
```

---

## License

This project is licensed under the MIT License.
