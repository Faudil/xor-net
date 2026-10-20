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
| `src/speculative/` | Lossless speculative decoding (see [Speculative decoding](#speculative-decoding)): `Drafter` / `DraftModel` traits, `SpecConfig`, the cost-aware `AdaptiveK`, rejection sampling; `lookup.rs`: prompt-lookup (n-gram) drafter. The decode loop itself is `SpecDecode` in `generation.rs`. |
| `src/prompt_cache.rs` | `PromptCache`: thread-safe LRU of model-state snapshots keyed by token prefix, shared by generators / server slots (see [Prompt cache](#prompt-cache)). |
| `src/sampler.rs` | Penalties, temperature, top-k / top-p / min-p (HF / vLLM order), seeded xoshiro RNG, partial selection (no full-vocab sort). |
| `src/models/traits.rs` | `CausalLM` + opaque `ModelState` (KV cache and/or recurrent state, with `truncate` / `checkpoint` for prefix reuse and `snapshot` / `restore` for the prompt cache). The only interface the generator sees. |
| `src/models/source.rs` | Resolves a user model reference (file, directory, `owner/repo[:file]` on the Hub) and loads a `Generator`. Used by the CLI and Python. |
| `src/models/q1/` | Bonsai on the 1-bit fast path. `qwen3.rs`: dense Qwen3 (1.7B/4B/8B). `qwen35.rs`: hybrid Qwen3.5 (27B) with gated full attention every N-th layer and Gated DeltaNet linear attention otherwise. `mod.rs`: shared `Linear`, `Embedding` (dequantizes looked-up rows only), RoPE incl. YaRN, growable `KvCache`, attention. `format.rs`: `.xq1`. `dspark.rs`: PrismML's DSpark drafter for Bonsai 27B, with its Q4_1 kernels in `dspark/q4_1.rs`. |
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

## Prompt cache

Within one `Generator`, a new prompt reuses the longest common prefix with
the tokens already in its state. `PromptCache` extends that across
requests and generators (e.g. every server slot), so a shared system
prompt or an earlier turn is prefilled once:

```rust
use std::sync::Arc;
use xor_net::prompt_cache::PromptCache;

// 512 MiB of snapshots, prefixes of at least 64 tokens.
let cache = Arc::new(PromptCache::new(512 << 20, 64));
let mut a = generator.with_prompt_cache(cache.clone());
let mut b = a.fork(); // forks share the cache
a.chat(&[sys.clone(), ChatMessage::user("q1")], &params)?;
let out = b.chat(&[sys, ChatMessage::user("q2")], &params)?;
assert!(out.usage.cached_prompt_tokens > 0); // B restored the system prompt
```

- **Snapshots.** `ModelState::snapshot(len)` deep-copies the first `len`
  positions into an immutable, `Send + Sync` `StateSnapshot` (only the used
  KV rows, never spare capacity or scratch, plus the fixed recurrent state
  for Qwen3.5); `ModelState::restore` writes one back into a state of the
  same model. Recurrent states can only snapshot their current length or
  their checkpoint, so the generator stops prefill at each snapshot point
  for them. Bonsai-1.7B: 112 KiB/token (56 MiB at 512 tokens, about 6 ms to
  copy).
- **Lookup.** Entries are indexed by the chained hash of every complete
  64-token block; a lookup hashes the prompt block by block and verifies
  the candidates under the deepest matching block, returning the longest
  entry whose tokens are all a prefix of the prompt (at least one prompt
  token is always left to prefill). Restoring an entry is exact.
- **When it is used.** Before prefill the best entry is restored if it is
  longer than what the generator's own state can keep. After prefill a
  snapshot is inserted at each *prefix boundary* not cached yet and at least
  `min_tokens` long. For chat the boundaries are the end of the leading
  system message(s) rendered alone (reused across users) and the end of
  the conversation without the generation prompt (reused by the next turn,
  on any slot); see `Generator::chat_prefix_boundaries`. Raw prompts pass
  boundaries explicitly with `generate_from_tokens_with_boundaries`.
  `Usage::cached_prompt_tokens` counts restored tokens.
- **Budget.** Entries are charged their snapshot bytes; inserting evicts
  least-recently-used entries (lookups and repeated inserts refresh them).
  `PromptCache::default_budget()` is a quarter of the available memory,
  capped at 1 GiB. One cache per model (entries are also keyed by the
  model's `Arc`, so a shared cache never mixes models).
- **Server.** `xor-net-server` shares one cache across all slots
  (`--prompt-cache-mb N`, `0` disables, default auto;
  `--prompt-cache-min-tokens`, default 64), reports
  `usage.prompt_tokens_details.cached_tokens` (llama.cpp:
  `tokens_cached`) and exposes counters under `prompt_cache` in
  `GET /health`. Its queue adds slot affinity: an arriving job goes to the
  idle slot whose state holds the longest prefix of its prompt, and a slot
  that frees up picks the best match among the first 4 queued jobs (FIFO
  beyond; a head job passed over 4 times runs next).

## Speculative decoding

`GenerationParams::with_speculative(SpecConfig::new(drafter, k))` (CLI:
`--draft lookup|<dspark.gguf> [--draft-k N]`, server: `--draft`) decodes
several tokens per target forward without changing the output:

- **Round.** The drafter proposes up to `k` tokens after the last sampled
  token `t`; the target runs `[t, d1..dk]` in one forward
  (`CausalLM::forward_with` with `all_logits` + `provisional`) whose logits
  are bit-identical to decoding them one by one. Greedy accepts `d_i` iff it
  is the target's argmax (after penalties), so the output is
  token-identical to plain greedy decoding. Sampling uses standard
  speculative rejection sampling (accept `d ~ q` with `min(1, p(d)/q(d))`,
  else draw from `norm(max(p - q, 0))`), with `p` and `q` both passed
  through the request's penalties / temperature / top-k / top-p / min-p
  (`Sampler::distribution`), so tokens follow exactly the plain sampling
  distribution (checked by a chi-square test in `tests/speculative_tests.rs`).
  After a fully accepted draft a bonus token comes from the last position.
- **Rollback.** `ModelState::commit(len)` keeps the accepted prefix. KV
  caches just truncate. Qwen3.5's Gated DeltaNet layers do not update their
  recurrent state during the provisional forward: each layer runs the delta
  rule on a scratch copy and saves its per-token inputs (raw conv rows,
  conv outputs, beta, decay: ~80 KB per token and layer). `commit` replays
  the accepted tokens' updates with the very same code (`gdn_heads`), so
  the state is bit-identical to having fed only those tokens, at the cost
  of one extra pass over the 144 MiB recurrent state per round (6 ms on
  27B), instead of storing `k + 1` state copies (720 MiB).
- **Stops.** Accepted drafts go through the same per-token accounting as
  plain decoding (EOS / stop ids, detokenizer, stop strings, streaming,
  abort, `max_tokens`); only tokens that continue are committed, so the
  state always matches `processed_tokens()`.
- **Drafters.** `PromptLookup` matches the last 3..5 tokens against the
  context (any model). `DsparkModel` (Bonsai 27B) is PrismML's six-layer
  block drafter: it reads the target's layer outputs at layers
  1/16/31/46/61 (`HiddenTaps`, copied out of the target forward only when
  a drafter asks), projects them into per-layer context K/V, and drafts a
  block of 4 tokens in one non-causal forward plus a sequential low-rank
  Markov correction; semantics follow PrismML's llama.cpp fork (see the
  module docs). By default it uses the target's 1-bit output head (same
  acceptance as the GGUF's Q4_1 head, 3.3x fewer bytes per round).
- **Adaptive k.** `SpecConfig::with_adaptive(true)` picks each round's
  draft length (0 = no draft) maximising expected tokens per second from
  the measured acceptance per depth and the measured draft and verify
  times.
- **Cost.** Verification is only worthwhile when a `k + 1`-token forward
  costs little more than a 1-token one (`CausalLM::batched_verify`; true
  for the Bonsai models, whose small-batch Q1_0 kernel reads the weights
  once; the generic fallback used by BitNet/Llama feeds tokens one by one
  and gains nothing).

## Adding a model family

Implement `CausalLM` (`new_state`, `forward(tokens, pos, state)`,
`vocab_size`, `max_seq_len`, `eos_token_ids`, `bos_token_id`) and its
`ModelState` (`snapshot` / `restore` are optional; without them the prompt
cache is simply never hit); wrap it in `Generator::new(model, tokenizer)` (optionally
`with_chat_template`). Everything else (sampling, streaming, the server,
Python) works unchanged. Add a branch in `models/source.rs` so the CLI and
Python can resolve it.
