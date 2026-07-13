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
(~18% smaller) and, when streamed straight from RAM at decode time, cuts the
per-token weight traffic by the same fraction.

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

## Runtime note

For simplicity the current runtime keeps each block's mask and **lane-order**
sign as two `u64`s in RAM (16 bytes/block), so the decoded form is the same
size as `pack4` and the sparse path is performance-neutral versus dense. The
on-disk file is still ~18% smaller. Reading the *compact* 12-byte block and
expanding the signs to lane order with a vectorised gather (instead of the
current `u64` decode) is the follow-up that turns the on-disk saving into a
per-token speedup; it is the remaining piece for the bandwidth-bound engine to
convert the ~25% smaller weight stream into throughput.
