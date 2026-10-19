# Validation against llama.cpp

xor-net's Bonsai (Q1_0) implementation is checked against
[PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)
(`prism` branch, CPU build), the reference runtime for these models, on
**identical token ids**.

## Results (2026-10-19, Ryzen 5 9600X, 6 threads)

Text: `scripts/validation/sample.txt` (English prose, Python code, French; 218
tokens for 1.7B, 221 for 27B). Logits are compared at every position.

| | Bonsai 1.7B | Bonsai 27B |
|---|---|---|
| Tokenization (GGUF-derived tokenizer vs llama.cpp) | identical | identical |
| KL(llama.cpp ‖ xor-net), mean / p95 / max | 3.9e-4 / 1.1e-3 / 1.7e-2 | 2.2e-4 / 8.1e-4 / 3.2e-3 |
| Top-1 agreement | 100% (218/218) | 99.1% (219/221)* |
| Top-5 overlap | 98.9% | 98.6% |
| Max \|Δ centred logit\| (mean, relative) | 0.20 (0.95%) | 0.16 (0.83%) |
| Perplexity on the text, llama.cpp → xor-net | 5.405 → 5.417 (+0.23%) | 3.644 → 3.649 (+0.14%) |

\* Both disagreements are exact near-ties in llama.cpp's own distribution
(top-2 probabilities 0.244 vs 0.242 and 0.119 vs 0.119).

xor-net's prefill path gives the same logits as its token-by-token decode path
(bit-identical). The residual difference vs llama.cpp comes from activation
quantization (int8 per 32-element block in both engines, with different
rounding/accumulation order) and float summation order; the weights are used
exactly.

### Speed on the same machine (6 threads)

| | llama.cpp `llama-bench` | xor-net |
|---|---|---|
| 1.7B prefill (pp512) | 349 tok/s | ~990 tok/s (488-token prompt) |
| 1.7B decode (tg64) | 144 tok/s | ~146 tok/s at ~500-token context, ~174 short |
| 27B prefill (pp512) | 23.2 tok/s | 64.6 tok/s (488-token prompt) |
| 27B decode (tg64) | 12.05 tok/s | 13.4 tok/s |

Decode in both engines is bound by memory bandwidth (27B: 3.53 GiB of weights
per token × 13.4 tok/s ≈ 51 GB/s, this machine's measured ceiling). xor-net's
prefill lead comes from its token-tiled Q1_0 GEMM.

## Reproduce

```bash
# 1. Build the reference (CPU only)
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
ninja -C llama.cpp/build llama llama-bench
g++ -O2 -std=c++17 scripts/validation/llamacpp_dump_logits.cpp \
    -Illama.cpp/include -Illama.cpp/ggml/include -Lllama.cpp/build/bin \
    -lllama -lggml -lggml-base -Wl,-rpath,$PWD/llama.cpp/build/bin -o dump_logits

# 2. Same tokens, both engines
M=Bonsai-27B-Q1_0.gguf
./dump_logits tokenize $M scripts/validation/sample.txt ref.u32
cargo run --release --example dump_logits_q1 -- tokenize $M scripts/validation/sample.txt ours.u32
cmp ref.u32 ours.u32                                   # tokenization must match
./dump_logits logits $M ref.u32 ref.f32 6
cargo run --release --example dump_logits_q1 -- logits $M ref.u32 ours.f32   # add --prefill for the prefill path

# 3. Compare (needs numpy); last args: vocab size, token file
python scripts/validation/compare_logits.py ref.f32 ours.f32 248320 ref.u32
```

The 27B needs ~4-5 GB of RAM in either engine.
