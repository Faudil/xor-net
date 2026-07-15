# xor-net

`xor-net` is a high-performance **CPU** inference engine for **1-bit (binary)** and
**1.58-bit (ternary)** quantized LLMs, built on the [Hugging Face Candle](https://github.com/huggingface/candle)
framework in Rust. It replaces floating-point matrix multiplications with bitwise
and integer SIMD ops, drastically cutting memory footprint and latency on modern
x86 CPUs.

It is no longer just a kernel demo: it now ships a full LLaMA-class transformer
runtime, an OpenAI/llama.cpp-compatible HTTP server, a lossless sparse-weight
format, per-phase profiling, and a **comprehensive multi-config benchmark
harness**. See [`OPTIMISATION.md`](OPTIMISATION.md) for the measured throughput
analysis that drove the design.

---

## Features

- **1-bit quantization (BitNet b1.0)** : input binarization + row-aligned
  bitpacked weights; matmul via AVX2 XNOR + popcount (`_mm256_xor_si256`,
  `vpshufb` popcount). Scalar `count_ones()` fallback when AVX2 is absent.
- **1.58-bit quantization (BitNet b1.58)** : ternary `{-1, 0, +1}` weights in two
  packing layouts:
  - **Pack4**: 4 weights/byte (2 bits/weight), fastest decode.
  - **Pack5**: 5 weights/byte (base-3), ~20% smaller on disk than Pack4.
  - Dynamic absmax → `i8` activation scaling; AVX-512 VNNI `_mm512_dpbusd` kernel
    with 16 accumulators, plus an AVX2 `_mm256_sign_epi8` path.
- **Lossless XorSparse weights** an opt-in block-64 sparse container
  (`XORNET_WEIGHT_FMT=sparse`, magic `XORSPARE1`) that stores only non-zero ternary
  signs. Bit-identical to `pack4` (verified by unit tests) and ~18% smaller on disk.
  Built offline with `convert_sparse`, documented in [`SPARSE_FORMAT.md`](SPARSE_FORMAT.md).
- **Configurable LM head** : the language-model head can run in `F32`, `F16`,
  `Int8`, `Int4`, or `Ternary` precision independently of the transformer blocks.
  `Int4` + partial-select sampling is the default single-stream configuration.
- **Fast sampler** : greedy / temperature / top-p with a reused index buffer and
  `select_nth_unstable` partial selection (no per-token allocation or full vocab
  sort).
- **Concurrency**  the decoder is memory/cache-bound, so aggregate throughput
  scales with *physical cores*. The server and `batch_bench` example serve many
  independent streams (each on its own 1-thread rayon pool) to reach ~110 tok/s
  aggregate on a 6-core box vs ~59 tok/s single-stream.
- **OpenAI-compatible server** : `xor-net-server` exposes `/v1/completions`,
  `/v1/chat/completions` (SSE + non-stream), `/v1/models`, and the llama.cpp
  legacy `/completion`, hand-rolled over HTTP/1.1 (no async framework).
- **Speculative decoding (experimental)** : `XORNET_SPEC=N` drafts `N` tokens per
  step; currently slower than greedy on this engine (the verify pass dominates).
- **Python bindings** : optional `pyo3` extension module behind the `python` feature.
- **Multi-config benchmark harness** : `bench_runner` + `scripts/bench_all.sh`
  sweep the full configuration matrix and log every run to JSONL + CSV.

---

## Supported models

| Model | Hub id | Notes |
|-------|--------|-------|
| BitNet b1.58 2B-4T | `microsoft/bitnet-b1.58-2B-4T` | Default; downloaded to `models/bitnet-2b` |
| BitNet b1.58 3B | `1bitLLM/bitnet_b1_58-3B` | |
| LLaMA-3 8B 1.58 | `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | |

More are coming like Bonsai-27B or other based on qwen's architecture

Any local directory with `config.json` + `*.safetensors` (and optionally
`model.safetensors.index.json`) is also accepted in place of a hub id.

---

## Project structure

```text
src/
├── lib.rs                 # Crate root; init_threads() global rayon pool
├── bit1/                  # 1-bit engine
│   ├── layers.rs          # BitLinear (Candle Module)
│   ├── ops.rs             # CPU forward pass
│   ├── quantization.rs    # Row-wise bitpacking
│   └── simd.rs            # AVX2 XNOR + popcount kernel
├── bit1_58/               # 1.58-bit engine
│   ├── layers.rs          # TernaryLinear
│   ├── ops.rs             # absmax + de-quant forward
│   ├── quantization.rs    # Pack4 / Pack5 packing + i8 scaling
│   ├── simd.rs            # AVX2 sign / AVX-512 VNNI kernels + sparse
│   └── sparse.rs          # XorSparse encode/decode
├── nn/                    # QuantizationConfig, LmHeadConfig, DynamicLinear
├── models/                # AutoModelForCausalLM, Llama runtime, loader
├── sampler.rs             # Partial-select sampler
└── tensor.rs              # FastTensor (packed f32/i8 workspace)
server/                    # xor-net-server binary (OpenAI/llama-compatible)
examples/                  # run_2b, run_3b, run_llama3_8b_158, chat,
                           # batch_bench, bench_runner, bench_layer,
                           # bench_memory, check_accuracy, dump_logits, ...
scripts/                   # bench_all.sh, jsonl_to_csv.py, convert_bitnet.py
src/bin/convert_sparse.rs  # Build a XorSparse .sparse container
```

---

## Getting started

### Prerequisites

- A Rust toolchain supporting the **2024 edition**.
- An **x86_64** CPU with **AVX2**. AVX-512 VNNI is used automatically when
  available; otherwise execution falls back to scalar/AVX2 paths.
- (~10 GB free) for the 2B weights if downloading from the Hub.

### Build

```bash
cargo build --release                 # library + examples
cargo build --release --bin xor-net-server
```

### Quick inference

```bash
# 2B BitNet, Int4 LM head, 8 threads (auto-capped)
XORNET_THREADS=8 cargo run --release --example run_2b

# 3B BitNet
cargo run --release --example run_3b

# 8B LLaMA 1.58
cargo run --release --example run_llama3_8b_158

# Interactive chat (local 2B or HF 3B/8B)
XORNET_MODEL=models/bitnet-2b cargo run --release --example chat
```

### Configuration matrix

Every run is parameterized by three enums (see `src/nn/dynamic_linear.rs`):

```rust
pub enum QuantizationConfig {
    None,                                  // F32 baseline
    Bit1(LmHeadConfig),                    // 1-bit
    Bit1_58(TernaryPackType, LmHeadConfig, bool /* inverted scale */),
}

pub enum TernaryPackType { Pack4, Pack5 }

pub enum LmHeadConfig { F32, F16, Int8, Int4, Ternary }
```

A typical high-throughput setup (current default):

```rust
QuantizationConfig::Bit1_58(TernaryPackType::Pack4, LmHeadConfig::Int4, /*inverted*/ true)
```

Thread count is the dominant single-stream lever: `init_threads(0)` auto-caps at 8
(beyond that, L2/L3 thrash regresses throughput on this hardware). Use the
**server** or `batch_bench` for aggregate throughput via concurrency instead.

---

## Environment variables

### Engine / runtime

| Variable | Default | Effect |
|----------|---------|--------|
| `XORNET_THREADS` | `0` (auto, cap 8) | Global rayon thread count for decode. |
| `XORNET_WEIGHT_FMT` | `dense` | Set to `sparse` to load a prebuilt `model.sparse` (XorSparse). |
| `XORNET_VNNI_PREFETCH` | `256` | VNNI decode prefetch distance (bytes). `0` → default. |
| `XORNET_LMHEAD` |  | Override LM-head precision: `f32`/`f16`/`int8`/`int4`/`ternary`. |
| `XORNET_SPEC` |  | Speculative decoding draft length `N` (experimental). |
| `XORNET_DEBUG` |  | Print per-block MLP projection kinds. |

### Server (`xor-net-server`)

| Variable | Default | Effect |
|----------|---------|--------|
| `XORNET_MODEL` | `models/bitnet-2b` | Model dir or hub id. |
| `XORNET_SERVER_HOST` | `127.0.0.1` | Bind host. |
| `XORNET_SERVER_PORT` | `8080` | Bind port. |
| `XORNET_SLOTS` | logical/2 | Worker slots (one per physical core → max aggregate). |
| `XORNET_THREADS_PER_SLOT` | `1` | Rayon threads per slot (`1` → ~110 tok/s aggregate). |

### Benchmark harness (`bench_runner` / `bench_all.sh`)

| Variable | Default | Effect |
|----------|---------|--------|
| `XORNET_MODEL` | `models/bitnet-2b` | Model dir or hub id. |
| `XORNET_QUANT` | `bit1_58_pack4` | `none`/`bit1`/`bit1_58_pack4`/`bit1_58_pack5`. |
| `XORNET_LMHEAD` | `int4` | LM-head precision (ignored when `quant=none`). |
| `XORNET_WEIGHT_FMT` | `dense` | `dense`/`sparse`. |
| `XORNET_THREADS` | `0` | Thread count for this run. |
| `XORNET_MODE` | `single` | `single` (1 stream) or `concurrent` (multi-stream). |
| `XORNET_STREAMS` | `4` | Concurrent streams (concurrent mode). |
| `XORNET_POOL_THREADS` | `0` | Per-stream rayon pool size (`0` = shared global). |
| `XORNET_TOKENS` | `128` | Tokens generated per stream. |
| `XORNET_PROMPT` |  | Prompt text. |
| `XORNET_TOKENIZER_REPO` | `microsoft/bitnet-b1.58-2B-4T` | Tokenizer source when model dir lacks one. |
| `XORNET_WARMUP` | `0` | Warmup tokens before measurement. |
| `XORNET_SEED` | `299792458` | Sampler seed. |
| `XORNET_LOG` |  | JSONL file to append each run's record to. |
| `XORNET_RUN_ID` / `XORNET_GIT_SHA` / `XORNET_RUSTC` / `XORNET_FEATURES` |  | Metadata stamped into the log. |

---

## Server

```bash
# 6 slots (one per physical core), 1 thread each → ~110 tok/s aggregate
XORNET_SLOTS=6 XORNET_THREADS_PER_SLOT=1 cargo run --release --bin xor-net-server
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"bitnet-2b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## Benchmarking

The harness sweeps the **entire** configuration space (quant format × LM-head ×
weight format × thread count × single/concurrent) and logs every run. A fresh
process is spawned per combo because `init_threads` installs a global rayon pool
that can only be set once per process.

```bash
# Quick validation (single mode, threads 4/8)
bash scripts/bench_all.sh --quick

# Full comprehensive sweep (1792 combos  subset for real runs)
QUANTS="bit1_58_pack4" LMHEADS="int4" THREADS="4 8" MODES="single" \
  TOKENS=128 bash scripts/bench_all.sh

# Restrict a single axis
bash scripts/bench_all.sh --quant bit1_58_pack5 --mode concurrent
```

Outputs land in `bench_logs/<timestamp>.jsonl` (one JSON object per run, recording
system metadata, the full config, per-phase profiler deltas, memory, and results)
and a flattened `bench_logs/<timestamp>.jsonl.csv` summary.

XorSparse must be built first to include the sparse axis:

```bash
cargo run --release --bin convert_sparse models/bitnet-2b models/bitnet-2b/model.sparse
SPARSE=1 bash scripts/bench_all.sh
```

Other benchmarks:

```bash
cargo run --release --example bench_layer     # single-layer GEMV throughput
cargo run --release --example bench_memory    # DRAM bandwidth probe
cargo run --release --example batch_bench     # concurrent multi-stream serving
```

---

## Accuracy & verification

```bash
# Compare ternary vs F32 baseline logits on a model
cargo run --release --example check_accuracy models/bitnet-2b
cargo run --release --example check_8b_accuracy
cargo run --release --example dump_logits_for_compare <model>
```

---

## Python bindings

Build the `pyo3` extension module (requires a Python 3.10+ dev environment):

```bash
cargo build --release --features python
```

---

## License

MIT. See [LICENSE](LICENSE).
