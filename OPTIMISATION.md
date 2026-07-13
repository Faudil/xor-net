# Optimisation Notes — XorNet 1.58-bit Inference

## Model & Hardware

- **Model**: `1bitLLM/bitnet_b1_58-3B` (BitNet b1.58 3B parameters)
  - `hidden_size=3200`, `intermediate_size=8640`, 26 layers, `vocab_size=32002`, tied embeddings
- **CPU**: Ryzen 5 9600X (6C/12T Zen 5), 32GB DDR5 dual-channel (~55 GB/s)
- **AVX-512**: `avx512f`, `avx512bw`, `avx512vbmi`, `avx512_vnni` all available

### Also tested: `bitnet_b1.58-2B` (2B, 30 layers)

Same `hidden_size=3200`, `intermediate_size=8640`, `vocab_size=32002`, `ffn_layernorm=Some`,
Silu activation. All transformer-layer projections load as **Ternary** (Pack4) — including
`down_proj` — so the all-ternary fused path (`fused_mlp_all`) is taken during generation.

---

## Current State (2B, DDR5, post-VNNI kernel)

Optimizations landed (all pushed to `main`):

- **Fused MLP dispatch removed** — `Mlp::forward` routes gate+up+silu+down into a single
  fused forward, eliminating the separate `DynamicLinear` virtual call + re-quantization.
- **Vectorized SiLU/ReLU²** (`silu_inplace_avx512`, Taylor `exp`) replaces the scalar libm loop.
- **VNNI `vpdpbusd` kernel** (`ternary_dot_product_pack4_avx512_vnni`) implemented and
  dispatched ahead of the `avx512bw` kernel when `avx512_vnni` is detected. This was the
  original "Future Direction #1".
- **Per-block MLP-kind diagnostic** (`XORNET_DEBUG=1`) confirms every block is
  `gate=Ternary up=Ternary down=Ternary`.

### Measured decode rate (2B, 12 threads, long generation ~295 tokens)

| Stage | ms/tok | Notes |
|-------|--------|-------|
| Attn | 6.2 | bandwidth-bound (QKV+O, 4 GEMVs/block) |
| MLP gate+up | 9.4 | fused, **~42.6 GB/s ≈ 77% of DDR5** |
| MLP down | ~0.0 (gen) / 0.26 (prefill) | fused during generation; prefill uses the `b_size>1` fallback |
| Norms | 0.15 | |
| SiLU+quant | 3.2 | vectorized; minor |
| LM Head | 5.7 | Int8 |
| **Total** | **~16.1 Blocks** | **≈ 41 tok/s** |

**The 2B model decodes at ~41 tok/s on this DDR5 box** — it is fundamentally
**RAM-bandwidth-bound** at ~77% of the DDR5 ceiling for the dominant MLP GEMV.

### Critical measurement caveat

Short-prompt runs badly under-measure decode rate. The per-token breakdown divides total
block time by *generated* tokens, but the **prefill** forward uses the slower `b_size>1`
fallback (non-fused `TernaryLinear::forward`), and its cost is amortized over few generated
tokens. Example: prompt `"hi"` (10 tokens) shows **23 tok/s** with MLP gate+up 19.5 ms,
whereas a 295-token generation shows **41 tok/s** with MLP gate+up 9.4 ms. **Always
benchmark with a long generation to get the true decode rate.**

---

## 1.58-bit Ternary Format

Weights are ternary (-1, 0, +1). Two packing schemes exist:

### Pack4 (default)
2 bits per weight, 4 weights per byte. Encoding: `00=-1, 01=0, 10=+1`.
- Row size: `(in_dim + 3) / 4` bytes
- 800 bytes per 3200-dim row
- Decode: bit-shift + mask + `vpshufb` LUT (single instruction)

### Pack5 (experimental)
Base-3 encoding, 5 weights per byte (`3^5 = 243 ≤ 256`).
- Row size: `(in_dim + 4) / 5` bytes
- 640 bytes per 3200-dim row (20% less weight volume)
- Decode: 5 iterations of `mulhi(magic)` / `mullo(3)` / `sub` arithmetic (expensive)

---

## Optimisation Phases

### Phase A — Baseline (scalar)
- Naive loop: for each row, iterate every element, decode ternary value, multiply by activation, accumulate
- **~27 tok/s**

### Phase B — Thread parallelism (Rayon)
- Parallelize over output rows with `par_chunks_mut`
- Activation quantization pooled across threads via thread-local or single quantization
- **~27 tok/s** (parallel by itself doesn't help a single-row workload)

### Phase C — Int8 LM Head
The tied lm_head (vocab_size=32002, hidden_size=3200) was the largest single layer. Replaced `f32 × f32` dot product with `i8 × i8`:

1. Pre-quantize weights per row: `i8 = round(f32 / scale)` capped to `[-127, 127]`
2. Quantize activations on-the-fly: `i8 = round(f32 * 127 / max_abs)`
3. Compute dot product as `Σ i8_i × i8_j` → `i32` → scale back to `f32`
4. `dot_product_i8` dispatch: `avx512bw` → `avx2` → scalar

**AVX‑512 kernel** (`dot_product_i8_avx512`):
- 64 values/step: load 2×32 bytes via `_mm256_loadu_si256`, extend to i16 via `_mm512_cvtepi8_epi16`, multiply-add via `_mm512_madd_epi16`
- AVX2 drain for remaining 32 values

**AVX2 fallback** (`dot_product_i8_avx2`):
- 32 values/step: 2× `_mm_loadu_si128` → `_mm256_cvtepi8_epi16` → `_mm256_madd_epi16`

**Result**: LM Head **7.0ms → ~1.9ms** (3.7× faster). Total: **~27 → ~34 tok/s**

### Phase D — AVX-512 Pack4 Ternary Kernel

`ternary_dot_product_pack4_avx512` processes 64 ternary values per step using zmm registers.

#### Weight decode (2-bit → i8)
1. Load 64 values' worth of Pack4 data (16 bytes) via `_mm_loadu_si128`
2. Zero-extend to i32: `_mm512_cvtepu8_epi32`
3. Extract 4×2-bit fields: `and(srli(w32, 0), mask3)` ... `and(srli(w32, 6), mask3)`
4. Interleave into bytes: `p0 | (p1<<8) | (p2<<16) | (p3<<24)`
5. LUT decode: `_mm512_shuffle_epi8(lut, packed)` — 64 parallel lookups, each 16-byte lane maps `{0→-1, 1→0, 2→+1, 3→0}`

#### Sign application (replacing `vpsignb`)
AVX-512 has no `vpsignb` for zmm. Replaced with masked operations:
1. `_mm512_cmp_epi8_mask(w, 0, 6)` — mask where weight > 0 (→ negate activation)
2. `_mm512_cmp_epi8_mask(w, 0, 0)` — mask where weight == 0 (→ output 0)
3. `_mm512_sub_epi8(zero, acts)` → negate all
4. `_mm512_mask_blend_epi8(pos_mask, neg_a, acts)` → -act where w>0, act elsewhere
5. `_mm512_mask_mov_epi8(blend, zero_mask, zero)` → 0 where w==0

#### Dot product reduction
`vpmaddubsw` (`_mm512_maddubs_epi16`) treats first operand as unsigned bytes (set to 1), second as signed bytes (the signed activations). Result: pairwise i16 sums. Then `vpmaddwd` (`_mm512_madd_epi16`) sums i16 pairs into i32. Finally `_mm512_add_epi32` accumulates.

Total reduction for 64 values: 2 instructions (maddubs → madd) + 1 add, vs 4+ in naive.

#### Prefetch
`_mm_prefetch(b_ptr + idx*16 + 256, _MM_HINT_T0)` — prefetch weights 16 chunks ahead. The distance was tuned empirically; 256 bytes (~1/3 of a row) gave the best result on this CPU.

**Result**: Blocks **~27.7ms → ~25.6ms** (8% faster). Total: **~34 → ~36 tok/s** (with 8 threads)

### AVX2 Fallback Kernel

`ternary_dot_product_pack4_avx2` processes 64 values per step using dual ymm accumulators:

- 2×32 values: each step loads 32 activation bytes + 8 weight bytes, decodes 2-bit fields, LUT-decodes via `_mm256_shuffle_epi8`
- Uses `_mm256_sign_epi8` (available in AVX2 but not for zmm in AVX-512) for sign application
- `vpmaddubsw` + `vpmaddwd` reduction chain

### Pack5 AVX-512 Kernel (experimental, not used in production)

`ternary_dot_product_pack5_avx512` processes 320 elements (64 Pack5 bytes) per outer iteration, 5 inner trit iterations:

#### Base-3 decode
Each Pack5 byte encodes weights 0-4 as base-3 digits: `b = w0 + w1*3 + w2*9 + w3*27 + w4*81`. Decoding extracts each trit:
1. `q = mulhi(cur, 0x5556)` — unsigned multiply-high by reciprocal of 3
2. `rem = cur - q*3` — remainder (0, 1, or 2)
3. `weight = rem - 1` — map to -1, 0, +1

This repeats 5 times per 64-byte chunk, progressively dividing by 3 via `cur = q`.

#### Stride-5 activation gather
Activations are contiguous memory, but each trit position k needs values at offsets `[k, k+5, k+10, ..., k+315]`. These span 5×64-byte windows. The gather uses 5× `_mm512_mask_permutexvar_epi8` with precomputed non-overlapping index+mask tables:

1. Precompute (lazy via `OnceLock`): for each trit k (0..4) and window m (0..4), an index vector of 64 bytes and a 64-bit mask. Each valid activation's position is placed in a unique, non-overlapping destination slot across the 5 windows.
2. At runtime: `masked_permute(zero, mask, idx, window)` → OR-blend across 5 windows → 64 gathered activation bytes.

#### Performance verdict
**2× slower than Pack4** (58.5ms vs 25.6ms blocks). Arithmetic base-3 decode is fundamentally more expensive than bit-shift + `vpshufb` LUT. The 20% bandwidth savings do not compensate. **Not viable** for this generation of hardware.

---

## Experiments that Did Not Improve Performance

| Experiment | What | Result |
|-----------|------|--------|
| Row prefetching | `_mm_prefetch` for next weight row in fused QKV/MLP loops (lookahead=6 rows) | No consistent gain |
| Dual accumulator | 128 values/step Pack4 kernel (2× 64) | Slightly slower (register pressure) |
| Pack5 production | See above | 2× slower than Pack4 |
| Thread scaling | 4, 6, 8, 12 threads | **12 threads best post-VNNI** (memory-bound → SMT helps overlap RAM latency). Old "8 best" was pre-VNNI. |
| VNNI prefetch distance | `XORNET_VNNI_PREFETCH` swept 256→4096 bytes | **Flat (41.2–41.6 tok/s)** — HW prefetcher already hides DDR5 latency; 256 is optimal |
| Cache-block / tile GEMV | Tiled micro-kernel for activation reuse | **No-op**: activation is 3.2 KB (already in L1); weights (620 MB/token) ≫ 32 MB L3 must come from RAM every token |
| Speculative decoding | Self-speculation, layer-skipping draft (every-other layer), `XORNET_SPEC=4` (greedy verify) | **Slower: 13.1 tok/s vs 41.3 baseline** (42.7% acceptance). Verify forward over N tokens costs ≈N× weight traffic — no amortization for a per-token bandwidth-bound GEMV.❌ |

---

## Bottleneck Analysis

- **Memory-bound**: IPC=2.25, balanced inner loop, 805MB weights/token at 35 GB/s effective (63% of DDR5 ceiling)
- **Perf profiling** (166K samples): no instruction hotspot — `vpmovzxbd` (weight load+extend) 164-403 samples, `vmovdqu64` (act load) 148-245 samples, `vpshufb` (decode) 329-353 samples. All balanced — the kernel is as efficient as possible for this memory-bound workload.

---

## Future Directions

- **VNNI `vpdpbusd` kernel**: ✅ Done — implemented (`ternary_dot_product_pack4_avx512_vnni`) and dispatched ahead of the `avx512bw` kernel. Engaged on this Zen 5 box.
- **Tile-based matmul**: ❌ No-op for this shape (see experiments table). Activation is 3.2 KB (L1-resident); weight traffic is irreducible.
- **Quantization fusion**: Fuse activation quantization with the preceding RMSNorm to reduce temporary allocations.
- **Fused prefill (b_size>1)**: The prefill forward still uses the non-fused `b_size>1` fallback (re-quantizes per row, `down` via `c_proj.forward`). Extending `fused_mlp_all` to `b_size>1` would remove the prefill-only `MLP down` cost and speed up long-context / many-short-exchange chat. One-time per prompt, so low priority for generation-bound workloads.
- **Speculative decoding**: ❌ **No win on this hardware** (measured: 13.1 tok/s vs 41.3 baseline at N=4). The verify pass over N tokens re-streams the full weight matrices N times — each token's GEMV independently reads all weights from DDR5, so there is **no weight-traffic amortization** (unlike a GPU where weights are cached in HBM). The verify cost therefore scales with N, and even 100% acceptance would still be slower (~27 tok/s). Speculative decoding only helps engines that are compute- or fixed-overhead-bound, not bandwidth-bound ones. The `Llama::forward_layers` / `forward_all` entry points and the `XORNET_SPEC` chat mode remain as experimental infra.

---

## Files

| File | Contents |
|------|----------|
| `src/bit1_58/simd.rs` | Pack4/Pack5 AVX-512 and AVX2 kernels, dispatch |
| `src/bit1_58/layers.rs` | Fused QKV/MLP forward, parallel row dispatch |
| `src/bit1_58/quantization.rs` | Pack4/Pack5 encode/decode, activation quantization |
| `src/bit1_58/ops.rs` | TernaryMatMulOp (candle CustomOp) |
| `src/nn/dynamic_linear.rs` | Int8Linear, dot_product_i8 dispatch, F32Linear |
| `src/models/llama.rs` | Model loading, lm_head Int8 path |
| `Cargo.toml` | `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` |
