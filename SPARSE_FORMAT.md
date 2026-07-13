# XorSparse — lossless sparse ternary weight format

XorSparse stores BitNet 1.58-bit (`{-1,0,+1}`) weights keeping **only the
non-zero positions and their signs**. It is *lossless*: decoding a row and
expanding it reproduces the original ternary row exactly, so the engine emits
bit-identical logits to the dense `pack4` path.

## Why it helps

Zeros contribute nothing to a dot product. A weight row with density `d`
(non-zero fraction) can be stored as a bitmask of non-zero positions plus one
sign bit per non-zero. With block size 64 the mask is exactly one `u64` and the
sign stream is `popcount(mask)` single bits.

Row byte cost at density `d`, per 64-element block:

```
8 (mask) + ceil(popcount/8)  ≈  8 * (1 + d)   bytes
```

vs `pack4` which uses 16 bytes per 64 weights (`in_dim/4`). Saving:

```
1 - (1 + d)/2      →   ~25% smaller at d = 0.5, ~35% at d = 0.3
```

On the 2B model this shrinks the weight file from ~521 MB (`pack4`) to ~408 MB
(~18% smaller).

## On-disk layout (little-endian)

```
magic      : b"XORSPARE1"            (9 bytes)
version    : u32  (= 1)
num_tensors: u32
repeat num_tensors times:
    name_len : u32
    name     : name_len bytes        (safetensors key, e.g. model.layers.0.self_attn.q_proj.weight)
    out_dim  : u32
    in_dim   : u32
    w_scale  : f32
    blob_len : u32
    blob     : blob_len bytes         (the per-row XorSparse payload below)

per weight row (inside `blob`):
    num_blocks : u32                    // ceil(in_dim / 64)
    for each block of 64 elements:
        mask  : u64                     // bit i set  => position i is non-zero
        signs : ceil(nnz/8) bytes       // bit j (scan order) set => weight is -1
```

The activation vector is *not* sparse, so each block maps to 64 contiguous
activations; the dot product reads the full (quantised) input once and gathers
only the non-zero weights.

## Converter

`convert_sparse <model_dir> <output.sparse> [--no-invert] [--model <dir>]`

Scans every U8 packed-ternary weight that has a companion `*_scale` tensor,
unpacks it to ternary `{-1,0,+1}`, re-encodes only the non-zero signs, and
writes the XorSparse file. The scale is inverted by default (to match
HF1BitLLM checkpoints) — exactly like `get_prepacked_ternary`.

## Engine wiring

Set `XORNET_WEIGHT_FMT=sparse` to opt in.

* If a sibling `model.sparse` exists next to `model.safetensors`, it is loaded
  once at startup and weights are served straight from it (no on-the-fly work).
* Otherwise the engine ternarizes + re-encodes the `pack4` weights as XorSparse
  at load time (needs no separate artifact).

The decode path is verified lossless by `xor_net::bit1_58::sparse` unit tests
(`sparse_decode_roundtrip`, `dense_and_sparse_dot_match`) which assert the
sparse dot product equals the dense `pack4` dot product **exactly**.

## Runtime note — performance is parity, not faster

The engine is **compute/decode-bound, not bandwidth-bound**. The dense `pack4`
path already uses the AVX-512 VNNI `vpdpbusd` kernel with 16 accumulators, and
a prefetch-distance sweep shows no speedup (the HW stream prefetcher already
serves the sequential weight read). Because the activation is dense, the SIMD
sparse block must still process all 64 lanes every block (masked), so weight
sparsity reduces *neither* FLOPs nor decode work.

Two practical consequences:

* The current runtime keeps each block's mask and **lane-order** sign as two
  `u64`s in RAM (16 bytes/block), identical to `pack4`, so the sparse path is
  performance-neutral: **~51 tok/s dense vs ~49 tok/s sparse on the 2B model,
  with bit-identical output** (verified by the correctness unit tests).
* Even reading the *compact* 12-byte block and expanding signs with a vectorised
  gather would not help throughput here: it would only shrink the weight stream,
  and at ~27 GB/s effective the path is not stream-bound. The on-disk file is
  still ~18% smaller.

Reaching the 80 tok/s target needs an architectural change (e.g. **batched
GEMV** — computing all output rows of a matrix in one call so the activation is
loaded once and the weight stream stays sequential across rows, keeping the
accumulators saturated), not weight compression. XorSparse is delivered as a
lossless, on-disk-smaller alternative to `pack4`.
