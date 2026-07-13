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

## Conclusion
The engine is **compute/decode-bound**, not bandwidth-bound. The per-block
2-bit→signed decode is the dominant cost, and weight sparsity doesn't reduce
FLOPs because the SIMD block processes all 64 lanes regardless. 80 tok/s is not
reachable via weight compression on this hardware.

## Path to 80 tok/s (not yet done)
- **Batched GEMV**: compute all output rows of each matrix in one call so the
  activation vector is loaded once and the weight stream stays sequential across
  rows, keeping the 16 accumulators saturated. Largest plausible gain;
  restructures `layers.rs` / `nn/dynamic_linear.rs` GEMV call sites.
- **FLOP reduction**: mixed precision (int4/int2) or structured (channel)
  pruning. `pack5` was previously rejected on correctness/complexity grounds.
