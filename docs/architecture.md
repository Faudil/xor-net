# Architecture

xor-net is one Rust crate (`xor-net`, imported as `xor_net`) with a small
number of layers. Everything above the kernels is model-family agnostic.

```text
             xor-net CLI        xor-net-server        Python (pyo3)
            (src/bin/xor-net)     (server/)          (src/python.rs)
                     \               |                   /
                      +---- models::source::load_generator ----+   path / .gguf / .xq1 / dir / Hub id
                                     |
                         generation::Generator                    chat template, decode loop,
                                     |                            sampling, stop strings, reasoning
                          models::traits::CausalLM                (object-safe trait)
                    /                                  \
      models::q1 (Bonsai, 1-bit)              models::llama (BitNet b1.58 / Llama 1.58)
        qwen3.rs   dense                        attention, mlp, block, causal_lm
        qwen35.rs  hybrid DeltaNet              nn::DynamicLinear (pack4/pack5, LM head precision)
        format.rs  .xq1 reader/writer           loader (safetensors, XorSparse)
                    |                                   |
      bit1::q1_0  Q1_0 GEMV/GEMM kernels      bit1_58  ternary kernels (VNNI, AVX2, sparse)
                              \                /
                         util::cpu_features (runtime ISA dispatch)

      vision (independent): ImageClassifier -> graph -> kernels/{binary,float,gemm,epilogue}
```

## Code map

| Path | What it is |
|------|------------|
| `src/generation.rs` | `Generator`: owns `Arc<dyn CausalLM>`, tokenizer, chat template (minijinja, HF-compatible), EOS ids and one reusable state. Chunked prefill, prefix caching across turns (the state is rolled back to the longest common prefix with the new prompt; recurrent states restore a single checkpoint taken at the end of the conversation history), UTF-8-safe streaming detokenisation, stop strings, `<think>` reasoning separation (`StreamEvent::{Text, Reasoning}`), chat-template kwargs such as `enable_thinking`. |
| `src/sampler.rs` | Penalties, temperature, top-k / top-p / min-p (HF / vLLM order), seeded xoshiro RNG, partial selection (no full-vocab sort). |
| `src/models/traits.rs` | `CausalLM` + opaque `ModelState` (KV cache and/or recurrent state, with `truncate` / `checkpoint` for prefix reuse). The only interface the generator sees. |
| `src/models/source.rs` | Resolves a user model reference (file, directory, `owner/repo[:file]` on the Hub) and loads a `Generator`. Used by the CLI and Python. |
| `src/models/q1/` | Bonsai on the 1-bit fast path. `qwen3.rs`: dense Qwen3 (1.7B/4B/8B). `qwen35.rs`: hybrid Qwen3.5 (27B) with gated full attention every N-th layer and Gated DeltaNet linear attention otherwise. `mod.rs`: shared `Linear`, `Embedding` (dequantizes looked-up rows only), RoPE incl. YaRN, growable `KvCache`, attention. `format.rs`: `.xq1`. |
| `src/bit1/q1_0.rs` | `Q1Weight` (split sign bits + f16 scales, owned or borrowed from an mmap) and the Q1_0 x int8 kernels: AVX-512 VNNI, AVX-512 BW, AVX2, scalar. Four weight rows share every activation load. |
| `src/models/llama/` | Llama-architecture runtime used for BitNet b1.58 and Llama3-8B-1.58 safetensors. |
| `src/bit1_58/` | Ternary packing (`pack4`: 2 bits/weight, `pack5`: base-3), AVX-512 VNNI `vpdpbusd` and AVX2 kernels, XorSparse. |
| `src/bit1/` (rest) | Binary (XNOR + popcount) `BitLinear` for BitNet b1.0-style models. |
| `src/nn/` | `QuantizationConfig`, `DynamicLinear`, LM-head precision (`F32`/`F16`/`Int8`/`Int4`/`Ternary`), fused attention and RMSNorm helpers. |
| `src/loader/` | GGUF parser (mmap, tokenizer reconstruction from `tokenizer.ggml.*`), safetensors repo, XorSparse container. |
| `src/vision/` | Binary CNN image classification: preprocessing (TF-legacy bicubic resize, centre crop), graph builder with end-to-end shape checks, binary conv kernels (AVX-512 VPOPCNTDQ, AVX2, scalar; each tested bit-exactly against a float reference). |
| `src/python.rs` | pyo3 bindings (feature `python`). |
| `src/bin/xor-net/` | The CLI (feature `cli`). |
| `server/` | `xor-net-server`, an OpenAI / llama.cpp compatible HTTP server built on `Generator`. |

## Design rules

- **Soundness first.** Every SIMD kernel is selected at runtime
  (`is_x86_feature_detected!`) and is tested for equivalence against the
  scalar path; `XORNET_FORCE_ISA=avx2|scalar` lowers the dispatch level so
  the whole test suite can run through the fallbacks. Untrusted file
  offsets are bounds-checked; the release profile uses `panic = "abort"`,
  so request paths in the server return errors rather than panic.
- **Lossless weights.** The published 1-bit / ternary weights are used as
  is; only activations are quantized (int8 per 32-element block for Q1_0,
  absmax int8 for BitNet).
- **One decode loop.** CLI, server and Python all go through `Generator`,
  so chat templates, stop strings, reasoning and sampling behave the same
  everywhere.
- **Memory-bound decode.** Single-stream decode streams every weight once
  per token, so throughput tracks bytes per token and memory bandwidth. The
  global rayon pool is capped at 8 threads by default (`init_threads(0)`),
  because more threads add cache thrash without bandwidth. Aggregate
  throughput comes from concurrent streams on physical cores (the server's
  slots). Details and measurements: [`../OPTIMISATION.md`](../OPTIMISATION.md).

## Threading

`xor_net::init_threads(n)` builds the global rayon pool once per process
(`0` = auto, capped at 8). The CLI exposes it as `--threads`, Python as
`load(..., threads=N)` (first call wins). The server runs one pool per slot.

## Adding a model family

Implement `CausalLM` (`new_state`, `forward(tokens, pos, state)`,
`vocab_size`, `max_seq_len`, `eos_token_ids`, `bos_token_id`) and its
`ModelState`; wrap it in `Generator::new(model, tokenizer)` (optionally
`with_chat_template`). Everything else (sampling, streaming, the server,
Python) works unchanged. Add a branch in `models/source.rs` so the CLI and
Python can resolve it.
