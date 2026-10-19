# The `xor-net` command

```bash
cargo install --git https://github.com/faudil/xor-net xor-net --features cli
# or, from a checkout:
cargo install --path . --features cli
# or without installing:
cargo run --release --features cli --bin xor-net -- <command> ...
```

The `cli` feature only adds the binary's own dependencies (JPEG/PNG decoding
for `classify`, a Ctrl-C handler for `chat`); the library stays free of them.
Every command has `--help`.

## Model references

Wherever a command takes `<MODEL>` you can pass:

| form | meaning |
|------|---------|
| `Bonsai-8B.xq1`, `Bonsai-8B-Q1_0.gguf` | a Bonsai 1-bit model file |
| `models/bitnet-2b` | a Llama / BitNet b1.58 safetensors directory (`config.json` + `*.safetensors`); if it has no `tokenizer.json`, the BitNet 2B-4T tokenizer is fetched |
| a directory holding one `.xq1` / `.gguf` | that file (`.xq1` preferred) |
| `prism-ml/Bonsai-1.7B-gguf` | a Hub repo: its Q1_0 `.gguf` is downloaded into the HF cache (`$HF_HOME`, default `~/.cache/huggingface`) |
| `prism-ml/Bonsai-27B-gguf:Bonsai-27B-Q1_0.gguf` | a specific file of a Hub repo |
| `microsoft/bitnet-b1.58-2B-4T` | a Hub safetensors repo (config, weights, tokenizer files) |

Downloads happen once; afterwards the cached file is used (also offline for
`.gguf` repos). Set `HF_TOKEN` for gated or private repos. `xor-net pull
<repo>` downloads and prints the local path.

## `run`

```bash
xor-net run <MODEL> -p "prompt" [options]
xor-net run <MODEL> "prompt"                 # prompt as 2nd positional
echo "prompt" | xor-net run <MODEL>          # prompt from stdin
xor-net run <MODEL> -f prompt.txt
```

The completion streams to stdout (the prompt is not echoed unless `--echo`);
a stats line goes to stderr:

```text
5 prompt (0 cached) + 24 completion tokens | prefill 7.5 tok/s | decode 164.4 tok/s | finish length | load 0.15s | peak RSS 345 MB
```

(Prefill tok/s is only meaningful for longer prompts; very short ones are
dominated by first-call warm-up. Use `bench` for throughput numbers.)

| option | default | |
|--------|---------|-|
| `-n, --max-tokens N` | 512 | |
| `--temp T` | 0.7 | `0` = greedy |
| `--top-k K` | 20 | `0` = off |
| `--top-p P` | 0.8 | |
| `--min-p P` | 0 | |
| `--repeat-penalty R` | 1.0 | |
| `--greedy` | | same as `--temp 0` |
| `--seed N` | random | reproducible sampling |
| `--stop STR` | | repeatable; the stop string is not printed |
| `--chat` | off | wrap the prompt in the model's chat template |
| `-s, --system TEXT` | | system message (implies `--chat`) |
| `--think` / `--no-think` | template default | sets `enable_thinking` for the chat template |
| `--show-thinking` | off | print reasoning (dimmed, to stderr) instead of hiding it |
| `--ignore-eos` | off | |
| `--echo` | off | print the prompt first (raw mode) |
| `-q, --quiet` | off | no load / stats lines |
| `-t, --threads N` | auto (max 8) | |
| `--tokenizer FILE` | embedded | override `tokenizer.json` |

The sampling defaults are the Qwen3 "non-thinking" recommendation (Bonsai is
Qwen3-based).

**Thinking.** Reasoning (`<think>...</think>`) is always separated from the
answer: hidden by default, shown dimmed with `--show-thinking`, never mixed
into stdout. `--think` / `--no-think` only take effect for templates that read
`enable_thinking` (Bonsai 27B / Qwen3.5); the Bonsai 1.7B / 4B / 8B templates
always emit an empty think block, so they never think. `xor-net info <MODEL>`
shows which kind a model has.

## `chat`

```bash
xor-net chat <MODEL> [--system "You are terse."] [--no-think] [--show-thinking]
```

Interactive multi-turn chat through the model's chat template. Accepts the
same generation options as `run`.

- **Ctrl-C** stops the current reply and keeps the chat open (the partial
  reply stays in the history). At the prompt, press Ctrl-C twice or Ctrl-D
  to quit.
- `/reset` forgets the conversation, `/system TEXT` sets the system message
  (and resets), `/think on|off` toggles `enable_thinking`,
  `/show-thinking on|off` shows or hides reasoning, `/exit` quits, `/help`.
- While the model is thinking with reasoning hidden, a dimmed
  `(thinking...)` marker is shown.
- A dimmed stats line follows each reply on a terminal.
- Line editing is whatever your terminal provides (no readline history);
  piping a script of lines into `xor-net chat` works too.

## `convert`

```bash
xor-net convert <IN.gguf | HUB_ID> [OUT.xq1] [--tokenizer tokenizer.json]
```

Lossless, verified conversion to `.xq1`; `OUT` defaults to the input name
without `-Q1_0` (`Bonsai-8B-Q1_0.gguf` -> `Bonsai-8B.xq1`), written to the
current directory for Hub inputs. See [formats.md](formats.md).

## `bench`

```bash
xor-net bench <MODEL> [-p PROMPT_LEN=128] [-n GEN=64] [-r REPS=2] [-t THREADS]
```

Prefills `P` synthetic tokens in one call, then decodes `G` tokens one at a
time (greedy argmax, no sampling overhead); prints the best of `R`
repetitions, load time and peak RSS (Linux):

```text
Bonsai-1.7B.xq1: threads 6 | load 0.02s | prefill 128 tok: ... tok/s | decode 64 tok: ... tok/s | peak RSS 304 MB
```

Run it on an otherwise idle machine; other CPU-heavy processes easily halve
the numbers.

## `info`

```bash
xor-net info <MODEL> [--template]
```

Reads only headers / metadata (no weights): format, architecture, layers,
heads, context, vocab, parameter count, quantization, tokenizer, chat
template (and whether it reads `enable_thinking`), size on disk.
`--template` prints the template source.

```text
model          models/bonsai/Bonsai-1.7B.xq1
format         xq1 v1 (lossless Q1_0, zero-copy mmap)
architecture   qwen3 (dense transformer)
layers         28
hidden size    2048
attention      16 heads, 8 kv heads, head_dim 128
context        32768 tokens
vocab          151669
parameters     1.72 B
quantization   1-bit Q1_0: sign bits + one f16 scale per 128 weights
tokenizer      embedded
chat template  yes (4063 chars, <think> blocks, no enable_thinking switch)
size on disk   248.7 MB
```

## `classify`

```bash
xor-net classify cat.jpg [more.png ...] [-m quicknet.safetensors] [-k 5] [-t THREADS]
```

ImageNet top-k with a binary CNN (`xornet-vision-v1` file, see
[formats.md](formats.md#xornet-vision-v1-binary-cnns)). The model defaults to
`$XORNET_VISION_MODEL`, else `models/vision/quicknet/quicknet.safetensors`.
The time printed per image is a single cold call: preprocessing (crop and
bicubic resize of the full-resolution image) plus the network, JPEG decoding
excluded. The ~0.7 ms/img figure in the README is the warm network-only median
(`cargo run --release --example classify -- img.jpg --bench 100`).

## `serve`

`xor-net serve [server options]` runs `xor-net-server` (installed alongside)
with the same arguments; see `xor-net-server --help` and the README.

## Exit codes

`0` success, `1` runtime error (missing file, bad model, ...), `2` usage
error, `130` quit with Ctrl-C.
