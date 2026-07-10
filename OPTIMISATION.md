# Optimisation Notes — XorNet 1.58-bit Inference

## Model & Hardware

- **Model**: `1bitLLM/bitnet_b1_58-3B` (BitNet b1.58 3B parameters)
  - `hidden_size=3200`, `intermediate_size=8640`, 26 layers, `vocab_size=32002`, tied embeddings
- **CPU**: Ryzen 5 9600X (6C/12T Zen 5), 32GB DDR5 dual-channel (~55 GB/s)
- **AVX-512**: `avx512f`, `avx512bw`, `avx512vbmi`, `avx512_vnni` all available

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
| Thread scaling | 4, 6, 8, 12 threads | 8 threads best (6 phys + 2 SMT) |

---

## Bottleneck Analysis

- **Memory-bound**: IPC=2.25, balanced inner loop, 805MB weights/token at 35 GB/s effective (63% of DDR5 ceiling)
- **Perf profiling** (166K samples): no instruction hotspot — `vpmovzxbd` (weight load+extend) 164-403 samples, `vmovdqu64` (act load) 148-245 samples, `vpshufb` (decode) 329-353 samples. All balanced — the kernel is as efficient as possible for this memory-bound workload.

---

## Future Directions

- **VNNI `vpdpbusd` kernel**: Native `u8 × i8 → i32` dot product could replace the 3-instruction `maddubs → madd → add` chain with a single instruction per 64 values. Might free execution ports for better memory-level parallelism.
- **Tile-based matmul**: Process multiple output rows together to improve cache reuse on activations.
- **Quantization fusion**: Fuse activation quantization with the preceding RMSNorm to reduce temporary allocations.

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
