# xor-net

**Fast, sound CPU inference for 1-bit and 1.58-bit models.**

xor-net runs models whose weights are *natively* binary or ternary (PrismML
Bonsai Q1_0, Microsoft BitNet b1.58, binary CNNs) on ordinary CPUs, with
hand-written AVX-512 / AVX2 kernels, runtime ISA dispatch and a scalar
fallback everywhere. It is a Rust library with a CLI, an OpenAI-compatible
server and Python bindings.

It is deliberately narrow: it does **not** run fp16 / int4 / int8 models.
If your model is 1-bit or 1.58-bit, xor-net aims to be the fastest and most
memory-frugal way to run it on a CPU.

```text
$ xor-net run Bonsai-1.7B.xq1 -p "The capital of France is" -n 24 --greedy --echo
loaded Bonsai-1.7B.xq1 in 0.15s (8 threads)
The capital of France is Paris, and the capital of Spain is Madrid. Which of the following is the correct answer? A. Paris B.
5 prompt (0 cached) + 24 completion tokens | prefill 7.5 tok/s | decode 164.4 tok/s | finish length | load 0.15s | peak RSS 345 MB
```

## Supported models

Measured on a Ryzen 5 9600X (6 cores Zen 5, AVX-512 VNNI, DDR5-5600 dual
channel), single stream, decode with 6-8 threads.

| Model | Type | File | Decode | Prefill | Peak RSS |
|-------|------|------|-------:|--------:|---------:|
| [Bonsai 1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) | 1-bit Q1_0, Qwen3 dense | 248 MB | **150-164 tok/s** | 300-380 tok/s | 300-350 MB |
| [Bonsai 4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | 1-bit Q1_0, Qwen3 dense | 572 MB | **~64 tok/s** | not measured | not measured |
| [Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | 1-bit Q1_0, Qwen3 dense | 1.16 GB | **41-44 tok/s** | not measured | 1.1-1.4 GB |
| [Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) | 1-bit Q1_0, Qwen3.5 hybrid (attention + Gated DeltaNet) | 3.8 GB | **11-12 tok/s** | not measured | 3.7 GB |
| [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | 1.58-bit ternary, Llama-style | safetensors | **~74 tok/s** | not measured | not measured |
| [BitNet b1.58 3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) | 1.58-bit ternary | safetensors | not measured | | |
| [Llama3-8B-1.58](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | 1.58-bit ternary | safetensors | not measured | | |
| QuickNet (Larq Zoo) | binary CNN, ImageNet top-1 63.3% | 4.4 MB | **~0.7 ms / image** (6 threads, ~10x faster than TF/Larq) | | ~15 MB |

Bonsai models load from the published `.gguf` or from xor-net's `.xq1`
format (same weights, bit for bit, mmapped for near-instant start-up; see
[docs/formats.md](docs/formats.md)). Decode is memory-bandwidth bound, so
throughput scales with bytes per token: expect proportionally lower numbers
on DDR4 or single-channel machines. For aggregate throughput, serve several
streams on separate cores (BitNet 2B: ~110 tok/s aggregate with 6 streams on
this box, see [OPTIMISATION.md](OPTIMISATION.md)).

## Quickstart

```bash
# 1. Install the CLI (Rust >= 1.89; the release build uses fat LTO, so give it a few minutes)
cargo install --git https://github.com/faudil/xor-net xor-net --features cli

# 2. Run: downloads Bonsai 1.7B (248 MB) into the Hugging Face cache on first use
xor-net run prism-ml/Bonsai-1.7B-gguf -p "Explain 1-bit LLMs in one paragraph." --chat

# 3. Chat
xor-net chat prism-ml/Bonsai-1.7B-gguf
```

Optional: convert once to `.xq1` for faster loading and page-cache sharing:

```bash
xor-net convert prism-ml/Bonsai-8B-gguf          # writes ./Bonsai-8B.xq1, verified lossless
xor-net chat Bonsai-8B.xq1
```

## CLI

| command | |
|---------|-|
| `xor-net run <model> -p "..." [-n N] [--temp T] [--top-k K] [--top-p P] [--seed S] [--stop STR] [--chat] [--no-think] [-t N]` | stream a completion to stdout, stats to stderr (`-q` to silence) |
| `xor-net chat <model> [--system "..."] [--no-think] [--show-thinking]` | interactive multi-turn chat; Ctrl-C stops a reply; `/reset`, `/system`, `/think on\|off`, `/exit` |
| `xor-net convert <in.gguf> [out.xq1]` | lossless GGUF -> `.xq1` conversion |
| `xor-net bench <model> [-p 128] [-n 64] [-t N]` | prefill / decode tok/s and peak RSS |
| `xor-net info <model> [--template]` | architecture, layers, context, params, quantization, size, chat template |
| `xor-net classify <img>... [-m quicknet.safetensors] [-k 5]` | ImageNet classification with a binary CNN |
| `xor-net pull <owner/repo>` | download from the Hub, print the path |
| `xor-net serve [...]` | start `xor-net-server` |

`<model>` is a `.gguf` / `.xq1` file, a Llama/BitNet safetensors directory,
or a Hub id (`prism-ml/Bonsai-8B-gguf`, `owner/repo:file.gguf`). Full
reference: [docs/cli.md](docs/cli.md).

## Rust

```toml
[dependencies]
xor-net = { git = "https://github.com/faudil/xor-net" }
```

### Generate and stream

```rust
use std::ops::ControlFlow;
use xor_net::models::source::{load_generator, LoadOptions};
use xor_net::{GenerationParams, SamplingParams};

fn main() -> anyhow::Result<()> {
    xor_net::init_threads(0).map_err(|e| anyhow::anyhow!("{e}"))?; // auto, max 8
    // A path (.gguf / .xq1 / safetensors dir) or a Hub id.
    let mut g = load_generator("prism-ml/Bonsai-1.7B-gguf", &LoadOptions::default())?;

    let params = GenerationParams::default()
        .with_max_tokens(64)
        .with_sampling(SamplingParams::default().with_temperature(0.7).with_top_p(0.8).with_top_k(20))
        .with_stop(["\n\n"]);
    let out = g.generate_stream("The capital of France is", &params, |delta| {
        print!("{delta}");
        ControlFlow::Continue(()) // ControlFlow::Break(()) cancels
    })?;
    println!("\n{:?}, {:.1} tok/s", out.finish_reason, out.timings.decode_tok_per_s());
    Ok(())
}
```

Bonsai files can also be loaded directly with
`xor_net::AutoModelForCausalLM::generator_from_file("Bonsai-8B.xq1", None)`.

### Chat, multi-turn and reasoning

```rust
use xor_net::{ChatMessage, GenerationParams, ReasoningFormat, StreamEvent};

let mut history = vec![
    ChatMessage::system("You are a concise assistant."),
    ChatMessage::user("What is a 1-bit LLM?"),
];
let params = GenerationParams::default()
    .with_max_tokens(256)
    .with_enable_thinking(false)                     // templates that read `enable_thinking`
    .with_reasoning_format(ReasoningFormat::Separate); // <think> content -> out.reasoning
let out = g.chat_stream_events(&history, &params, |ev| {
    match ev {
        StreamEvent::Text(t) => print!("{t}"),
        StreamEvent::Reasoning(r) => eprint!("{r}"),
    }
    std::ops::ControlFlow::Continue(())
})?;
history.push(ChatMessage::assistant(out.text));    // next turn reuses the cached prefix when possible
```

`Generator` owns one model state; `g.fork()` gives another generator that
shares the weights (one per concurrent stream). Any model implementing the
`CausalLM` trait plugs into `Generator`; see
[docs/architecture.md](docs/architecture.md).

### Vision

```rust
let clf = xor_net::vision::ImageClassifier::load("quicknet.safetensors")?;
let img = image::open("cat.jpg")?.to_rgb8();           // any RGB8 source works
let (w, h) = (img.width() as usize, img.height() as usize);
for p in clf.predict(img.as_raw(), w, h, 5)? {
    println!("{:>5.1}%  {}", 100.0 * p.probability, p.label);
}
```

The library takes raw interleaved RGB8; it does not depend on an image
decoder.

## Python

```bash
pip install maturin && maturin develop --release   # from a checkout; abi3 wheel, Python >= 3.10
```

```python
import xor_net

model = xor_net.load("prism-ml/Bonsai-1.7B-gguf")         # or a .gguf/.xq1 path, or a BitNet dir
print(model.chat([{"role": "user", "content": "Hi!"}], max_tokens=64))

for piece in model.generate_stream("Once upon a time", max_tokens=100, temperature=0.7):
    print(piece, end="", flush=True)

c = model.chat([{"role": "user", "content": "2+2?"}], enable_thinking=False, details=True)
print(c.text, c.reasoning, c.decode_tok_per_s)

clf = xor_net.ImageClassifier("quicknet.safetensors")
clf.predict("cat.jpg", k=5)            # also numpy HxWx3 uint8 arrays, PIL images, bytes
```

Full reference: [docs/python.md](docs/python.md).

## Server

`xor-net-server` exposes an OpenAI-compatible API (`/v1/chat/completions`
with SSE streaming, `/v1/completions`, `/v1/models`) plus llama.cpp's
`/completion` and `/health`.

```bash
cargo install --git https://github.com/faudil/xor-net xor-net --bin xor-net-server
xor-net-server -m Bonsai-8B.xq1 --port 8080          # see --help for slots / threads / limits

curl -N http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Any OpenAI client works (`base_url="http://127.0.0.1:8080/v1"`).

## The `.xq1` format

`.xq1` is a converted Bonsai file: the same sign bits and f16 scales as the
GGUF (lossless, verified tensor by tensor after conversion), but pre-split,
pre-fused, 64-byte aligned and with the tokenizer and chat template embedded.
Loading is an `mmap` plus a JSON header parse, so start-up is near-instant
and the weights are shared page cache between processes. Convert with
`xor-net convert in.gguf out.xq1`. Layout and details:
[docs/formats.md](docs/formats.md), which also covers XorSparse (a lossless,
~18% smaller on-disk container for BitNet weights).

## Hardware

| CPU | Status |
|-----|--------|
| x86-64 with AVX-512 VNNI (Zen 4/5, Ice Lake+, Sapphire Rapids) | best: `vpdpbusd` kernels for Q1_0 and ternary |
| x86-64 with AVX-512 BW, no VNNI | supported, `vpmaddubsw` path |
| x86-64 with AVX2 + FMA (Haswell+, Zen 1-3) | supported, AVX2 kernels |
| aarch64 (Apple Silicon, Graviton), other | works with portable scalar kernels; no NEON kernels yet, so much slower |

Kernels are chosen at runtime, so one binary runs everywhere; there is no
need for `-C target-cpu=native`. `XORNET_FORCE_ISA=avx2|scalar` forces a
lower path (testing). Threads: `--threads` / `init_threads(n)`; the default
caps at 8 because decode is memory-bound and more threads only add cache
contention. Use physical cores, not SMT siblings.

## Documentation

- [docs/cli.md](docs/cli.md): command-line reference
- [docs/python.md](docs/python.md): Python API
- [docs/formats.md](docs/formats.md): Q1_0, `.xq1`, XorSparse, vision files
- [docs/architecture.md](docs/architecture.md): code map and design rules
- [OPTIMISATION.md](OPTIMISATION.md): performance investigation and measurements

## Development

```bash
cargo test                                   # unit + integration tests (model tests skip without files)
cargo test --features cli --bin xor-net      # CLI argument parsing
BONSAI_XQ1=Bonsai-1.7B.xq1 cargo test --release --features cli --test cli_e2e -- --ignored
cargo bench --bench kernels
```

## License

MIT. See [LICENSE](LICENSE). Model weights are subject to their own
licenses (Bonsai: see the PrismML model cards; BitNet: MIT; QuickNet:
Apache-2.0).
