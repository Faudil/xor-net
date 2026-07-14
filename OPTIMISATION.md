# Optimisation notes — 2B BitNet engine

## Goal
Reach **80 tok/s** on the `bitnet-b1.58-2B-4T` model with lossless weights.

## Machine
Ryzen 5 9600X (6C/12T Zen 5), 32 GB DDR5-5600 dual-channel. Has
`avx512f`/`avx512bw`/`avx512vbmi`/`avx512_vnni`. Crate `xor-net`; bins import
as `xor_net`.

## What was tried

### XorSparse (lossless sparse ternary) — DONE, parity
Custom block-64 sparse format (`mask: u64` + compact sign bits), offline
converter (`convert_sparse`), loader (`SparseFile`, magic `XORSPARE1`), and an
opt-in engine path (`XORNET_WEIGHT_FMT=sparse`). Documented in
`SPARSE_FORMAT.md`. Lossless — bit-identical output to `pack4`, verified by
unit tests in `src/bit1_58/simd.rs::sparse_correctness_tests`.

Result: **~49 tok/s sparse vs ~51 tok/s dense**, output identical. On-disk
file ~18% smaller (521 MB → 408 MB). No throughput gain.

### Thread count — no effect
6/12/24 threads all ~51 tok/s. Not core-bound.

### `vpdpbusd` VNNI — already in use
The dense path already uses the AVX-512 VNNI `dpbusd` kernel with 16
accumulators (optimal MAC). No change needed.

### Prefetch distance (`XORNET_VNNI_PREFETCH`) — no effect
256 → 8192 bytes all ~50–51 tok/s. Not memory-latency bound; HW stream
prefetcher already covers the sequential weight read.

### Decode-reduction kernel (`ternary_dot_product_pack4_avx512_vnni_tight`) — not viable
A hand-rolled "tight" unpack (broadcast + `shuffle_epi8` nibble extract
instead of `cvtepu8` + 4 shifts) was drafted to cut unpack ops. It did not
compile (no Rust `_mm512_setr_epi8`) and, on inspection, a *correct* shuffle
unpack needs the same ≥4 field-extracts as the reference — extracting four
independent 2-bit ternary codes from one byte is irreducible. No op-count win,
so no throughput win. Reference VNNI kernel (`ternary_dot_product_pack4_avx512_vnni`)
remains the optimal decode path.

### Thread count — the real single-stream lever
`init_threads(0)` (all cores = 12 on this 6C/12T Zen5) is **suboptimal**:
the decode GEMV is memory/cache-bound, and >~4 threads only add L2/L3
thrash + pool contention. Measured single-stream tok/s vs rayon threads:

| threads | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 |
|--------:|--:|--:|--:|--:|--:|--:|--:|--:|
| tok/s   | 22| 40| 57| 65| 63| 61| 65| 60|

Sweet spot **4–8 threads (~65 tok/s)**; 12 threads *regresses* to 60.
`init_threads(0)` now auto-caps at 8 (see `src/lib.rs`), lifting the
default single-stream from ~58 to **~67 tok/s** out of the box.

### Concurrent multi-stream serving — real but physical-core bound
New benchmark `examples/batch_bench.rs` loads the model once (`Arc<Llama>`)
and serves `N` independent generation streams, each with its own KV `Cache`.
Two modes: a shared global rayon pool, or **per-stream pools** (each stream
gets its own pool of `XORNET_POOL_THREADS` threads, so the OS can place it
on a core subset). 100 tokens/stream:

| mode                  | streams | pool/st | aggregate | per-stream |
|-----------------------|--------:|--------:|----------:|-----------:|
| shared global pool    | 6        | 8       | 82.8      | 14.0       |
| per-stream pool       | 6        | 2       | 88.6      | 15.0       |
| **per-stream pool**   | **6**    | **1**   | **110.5**  | 18.7       |
| per-stream pool       | 12       | 1       | 97.2      | 8.3 (HT)   |

The box is **6 physical cores / 12 logical (HT)**. The throughput ceiling
is the *physical* cores: per-core decode = **22 tok/s**, so
`6 physical × 22 ≈ 132` is the max; 6 single-threaded streams
(one per physical core) achieve **110 tok/s aggregate** (84% efficiency).
Logical cores 7–12 (HT) give nothing — that is exactly why >4 threads
regressed in the single-stream sweep. Shared pools of 8 also waste cores on
HT and contention, capping at ~83.

So:
- **Single-stream ceiling ≈ 65–67 tok/s** (memory/cache + 2-bit-unpack
  bound; 22 tok/s per physical core). **80 tok/s single-stream is NOT
  reachable losslessly** on this hardware.
- **Aggregate 80 tok/s IS reachable**: 6 concurrent single-threaded
  streams → **110 tok/s aggregate** (each stream ~18.7 tok/s). This is the
  practical "beat 80" target.

### Serving — `server/` (`xor-net-server` bin)
A hand-rolled HTTP/1.1 server (no async framework; the engine is
synchronous) exposing an OpenAI-compatible surface:
`GET /health`, `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (SSE stream + non-stream),
plus the llama.cpp legacy `POST /completion`. It runs `XORNET_SLOTS`
worker threads (default = logical/2 = **6**, one per physical core), each
with its own **1-thread rayon pool** (`XORNET_THREADS_PER_SLOT`,
default 1), draining a shared job queue. Up to 6 decodes run
concurrently → the 110 tok/s aggregate ceiling. Verified:
- `6` concurrent `/v1/completions` → **103 tok/s aggregate**
  (6 × 100 tokens / 5.82 s), each request ~22 tok/s when alone.
- single request → ~22 tok/s (1 core); trade-off is latency vs
  aggregate (more slots/threads-per-slot = higher per-request speed,
  lower aggregate — see the throughput table above).

## Conclusion
The engine is **memory/cache + decode-bound**, not raw-bandwidth or
core-bound. The per-block 2-bit→signed unpack is the dominant per-core cost,
and it is irreducible (see decode-reduction note). **80 tok/s single-stream
is not reachable** losslessly on this 6C/12T box (ceiling ~67). **80 tok/s
aggregate IS reachable** by serving 6 concurrent single-threaded streams
(~110 tok/s). The earlier XorSparse / kernel / prefetch work confirmed
weight tricks don't move the needle.

## Path to higher tok/s
- **Concurrent serving** — shipped as `server/` (`xor-net-server` bin,
  OpenAI/llama-compatible). Runs `XORNET_SLOTS` 1-thread worker
  slots (default 6, one per physical core) → **~110 tok/s aggregate**
  (verified 103 in a 6-concurrent curl test). Trade-off: each
  request sees ~22 tok/s; raising `XORNET_THREADS_PER_SLOT`
  lifts per-request speed but lowers aggregate (HT contends).
- **Single-stream**: only a faster/unpack-cheaper kernel could lift the 22
  tok/s-per-core figure; the decode-reduction micro-opt was shown infeasible,
  so this needs a new SIMD strategy (e.g. `vpdpbusd` with pre-unpacked
  weights held in a wider layout) — a larger refactor.
- **Speculative decoding**: draft model proposes k tokens/step → breaks the
  one-token-per-forward bound; only change that can lift the *per-stream*
  ceiling. Already prototyped (`XORNET_SPEC`) and found slower here because
  the verify pass dominates on this engine.
- **Mixed precision** (int4/int2) or structured pruning: `pack5` previously
  rejected on correctness/complexity grounds.
