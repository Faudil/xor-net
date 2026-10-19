# Weight formats

xor-net reads three kinds of model files. All of them are **lossless** with
respect to the published 1-bit / 1.58-bit weights: nothing is re-quantized.

| Format | Models | Produced by | Loading |
|--------|--------|-------------|---------|
| GGUF with `Q1_0` tensors | Bonsai 1.7B / 4B / 8B (`qwen3`), Bonsai 27B (`qwen35`) | PrismML (Hugging Face) | parse + repack into the split layout |
| `.xq1` | the same Bonsai models | `xor-net convert` | zero-copy `mmap` |
| HF safetensors directory | BitNet b1.58 2B-4T / 3B, Llama3-8B-1.58 | Microsoft / 1bitLLM / HF1BitLLM | ternarize + pack at load (optionally from a XorSparse `model.sparse`) |
| `xornet-vision-v1` safetensors | QuickNet (Larq Zoo) | `scripts/vision/convert_larq_zoo.py` | read into memory (~4 MB) |

---

## Q1_0 (Bonsai)

Every weight is one sign bit; every block of `B` consecutive weights in a row
shares one f16 scale `d`: `w = bit ? +d : -d`. The GGUF files ship one of
three block layouts, which xor-net normalises into the same in-memory form:

| GGUF type id | block `B` | bytes / block | layout |
|--------------|-----------|---------------|--------|
| 41 `Q1_0` (upstream ggml) | 128 | 18 | `[d: f16][bits: 16 B]` |
| 42 `Q1_0` (PrismML)       | 32  | 6  | `[d: f16][bits: 4 B]`  |
| 43 `Q1_0_G128`            | 128 | 18 | `[bits: 16 B][d: f16]` |

In memory (`Q1Weight`) a row's sign bits are contiguous (`cols / 8` bytes,
bit `i` of byte `k` is element `8k + i`) and the scales are a separate f16
array. This lets the AVX-512 kernel load 64 sign bits straight into a mask
register and 16 scales with one `vcvtph2ps`. Activations are quantized to
int8 per block of 32 (ggml `Q8_0` granularity); the per-block integer dot
product is exact.

## `.xq1`: converted Bonsai files

A GGUF has to be parsed and repacked on every load. `.xq1` moves that work
offline, once:

- Q1_0 blocks are already split into sign bits and f16 scales;
- Q/K/V and gate/up projections are pre-fused (one GEMV each per layer);
- the tokenizer (HF `tokenizer.json`) and the Jinja chat template are
  embedded, so the file is self-contained;
- every tensor starts at a 64-byte aligned offset, so 1-bit weights are used
  **in place** from the `mmap`: near-instant start-up, and the weights are
  clean file-backed pages that the kernel shares between processes (several
  servers on one box pay for the weights once) and can evict under pressure.

### Convert

```bash
xor-net convert Bonsai-8B-Q1_0.gguf Bonsai-8B.xq1
xor-net convert prism-ml/Bonsai-8B-gguf            # from the Hub; writes ./Bonsai-8B.xq1
xor-net convert in.gguf out.xq1 --tokenizer tokenizer.json   # embed another tokenizer
```

(`bonsai_convert <in.gguf> <out.xq1>` is the same converter as a standalone
binary.)

The conversion is **lossless and verified**: after writing, the converter
reloads the `.xq1`, hashes every tensor (shape, sign bits, scales, f32 data)
and compares against the GGUF-loaded model; any difference is an error.
`Q1Weight::to_gguf` restores the original GGUF blocks bit for bit. Converting
`Bonsai-1.7B-Q1_0.gguf` twice gives byte-identical files.

Measured on Bonsai 1.7B (Ryzen 5 9600X, one run, machine under load):
load 0.93 s / peak RSS 483 MB from GGUF vs 0.23 s / 345 MB from `.xq1`.
Conversion takes ~2 s for 1.7B.

### Layout (little endian)

```text
offset 0   magic "XQ1F" (4 bytes), version u32 = 1
       8   header_off u64, header_len u64
      24   zero padding up to 64
      64   tensor data, each tensor at a 64-byte aligned file offset
           JSON header at header_off (written last, so tensors stream to disk)
```

The JSON header holds:

| key | content |
|-----|---------|
| `arch` | `"qwen3"` (dense) or `"qwen35"` (hybrid) |
| `config` | the model config (`Qwen3Config` / `Qwen35Config` as JSON) |
| `tensors` | name → `{"kind":"q1", rows, cols, block, bits:[off,len], scales:[off,len]}` or `{"kind":"f32", shape, data:[off,len]}` |
| `tokenizer` | HF tokenizer JSON (string) |
| `chat_template` | Jinja source (string or null) |

Readers must bounds-check every `(off, len)` against the file size; xor-net's
reader does, and rejects truncated files and unknown versions.

`xor-net info model.xq1` prints the architecture, parameter count, block
size, context length and whether a chat template is embedded.

---

## XorSparse (BitNet b1.58, optional)

XorSparse stores ternary (`{-1, 0, +1}`) weights keeping **only the non-zero
positions and their signs**. It is lossless: decoding a row reproduces the
ternary row exactly, so the engine emits bit-identical logits to the dense
`pack4` path (verified by the `sparse_decode_roundtrip` and
`dense_and_sparse_dot_match` unit tests).

### Why it helps on disk

With block size 64 the non-zero mask is exactly one `u64` and the sign stream
is `popcount(mask)` single bits. Per 64-weight block:

```text
8 (mask) + ceil(popcount / 8)  ≈  8 * (1 + d) bytes      (d = non-zero fraction)
```

versus 16 bytes for `pack4`, i.e. ~25% smaller at `d = 0.5`. On BitNet 2B the
weight file shrinks from ~521 MB (`pack4`) to ~408 MB (~18%).

### Layout (little endian)

```text
magic      : b"XORSPARE1"            (9 bytes)
version    : u32  (= 1)
num_tensors: u32
repeat num_tensors times:
    name_len : u32
    name     : name_len bytes        (safetensors key)
    out_dim  : u32
    in_dim   : u32
    w_scale  : f32
    blob_len : u32
    blob     : blob_len bytes        (per-row payload below)

per weight row (inside blob):
    num_blocks : u32                 // ceil(in_dim / 64)
    for each block of 64 elements:
        mask  : u64                  // bit i set => position i is non-zero
        signs : ceil(nnz / 8) bytes  // bit j (scan order) set => weight is -1
```

### Usage

```bash
cargo run --release --bin convert_sparse models/bitnet-2b models/bitnet-2b/model.sparse
XORNET_WEIGHT_FMT=sparse xor-net run models/bitnet-2b -p "Hello"
```

With `XORNET_WEIGHT_FMT=sparse` and a `model.sparse` next to the
safetensors, weights are served from it; without the file the engine
re-encodes `pack4` weights as XorSparse at load time.

### Performance: parity, not faster

The decode path is compute/decode-bound, not bandwidth-bound, and the
activation is dense, so the SIMD block still processes all 64 lanes. Measured
on BitNet 2B: ~51 tok/s dense vs ~49 tok/s sparse, identical output.
XorSparse is a smaller-on-disk alternative, not a speed-up. See
[`../OPTIMISATION.md`](../OPTIMISATION.md).

---

## `xornet-vision-v1` (binary CNNs)

A single safetensors file produced offline by
`scripts/vision/convert_larq_zoo.py` from the Larq Zoo Keras weights:

- binary 3x3 conv weights pre-packed to sign bits
  (`u64 [cout, kh, kw, ceil(cin / 64)]`);
- batch norms folded into per-channel `scale` / `bias`;
- metadata (JSON strings): `format = "xornet-vision-v1"`, `graph` (layer
  list), `preprocess` (crop, size, TF-legacy bicubic resize, mean/std),
  `labels` (ImageNet class names), `output`, `model`.

Every graph is shape-checked end to end at load time. See `src/vision/mod.rs`.
