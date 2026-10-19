# Python API

## Install (from source)

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release          # builds the `xor_net` extension (abi3, Python >= 3.10)
```

`maturin build --release` produces a wheel in `target/wheels/` instead.

## Language models

```python
import xor_net

model = xor_net.load("prism-ml/Bonsai-1.7B-gguf")     # Hub id: downloads the Q1_0 GGUF once
# model = xor_net.load("models/bonsai/Bonsai-8B.xq1")  # .xq1 / .gguf file
# model = xor_net.load("models/bitnet-2b")             # Llama / BitNet b1.58 safetensors dir
# model = xor_net.load("...", threads=6)               # size the thread pool (first call wins)

print(model.generate("The capital of France is", max_tokens=16, temperature=0))

reply = model.chat(
    [{"role": "system", "content": "You are terse."},
     {"role": "user", "content": "Name three primary colors."}],
    max_tokens=64,
)

for piece in model.chat_stream([{"role": "user", "content": "Tell me a story"}], max_tokens=200):
    print(piece, end="", flush=True)
```

`xor_net.load(model, threads=None, tokenizer=None, quantization=None)` takes
the same model references as the CLI (see [cli.md](cli.md#model-references))
and returns a `Model` (also reachable as `xor_net.Model.load`). All calls
release the GIL.

### Generation parameters

`generate(prompt, ...)`, `generate_stream(prompt, ...)`,
`chat(messages, ...)`, `chat_stream(messages, ...)` accept:

| keyword | default | |
|---------|---------|-|
| `max_tokens` | 256 | |
| `temperature` | 0.7 | `0` = greedy |
| `top_p` | 0.8 | |
| `top_k` | 20 | `0` = off |
| `min_p` | 0.0 | |
| `repetition_penalty` | 1.0 | HF semantics |
| `frequency_penalty`, `presence_penalty` | 0.0 | OpenAI semantics |
| `stop` | None | list of stop strings (not included in the output) |
| `seed` | None | reproducible sampling |
| `enable_thinking` | None | chat-template variable (Bonsai 27B / Qwen3.5 templates read it) |
| `ignore_eos` | False | |
| `details` | False | (`generate`, `chat`) return a `Completion` instead of `str` |
| `reasoning` | False | (`*_stream`) yield `(kind, text)` tuples, `kind` in `{"text", "reasoning"}` |

The returned text never contains the prompt or `<think>` blocks; reasoning
is available as `Completion.reasoning` or as `"reasoning"` stream events.

### `Completion`

```python
c = model.chat(msgs, max_tokens=64, details=True)
c.text, c.reasoning, c.finish_reason          # "stop" | "length"
c.prompt_tokens, c.cached_prompt_tokens, c.completion_tokens
c.prefill_tok_per_s, c.decode_tok_per_s
```

### Streams

`generate_stream` / `chat_stream` return a `TextStream` iterator. Generation
runs on a background thread; `stream.cancel()` or dropping the iterator stops
it at the next token. After exhaustion, `stream.finish_reason` and
`stream.completion` (a `Completion`) are set.

### Other methods

`apply_chat_template(messages, enable_thinking=None)`, `tokenize(text,
add_special_tokens=True)`, `decode(ids, skip_special_tokens=True)`,
`reset()` (free the KV cache), and the properties `kind` (`"bonsai"` /
`"llama"`), `path`, `vocab_size`, `context_length`, `has_chat_template`,
`supports_thinking`.

A `Model` serialises requests (one generation at a time); load the model
once per process and reuse it. Multi-turn chats that extend the previous
prompt reuse the cached prefix when the template allows it.

### Legacy names

`XorNetModel` (= `Model`) with `from_local(dir, quantization=None,
num_threads=None)` / `from_pretrained(repo_id, ...)`,
`QuantizationConfig(pack_type="pack4", lm_head="int4")` and
`init_threads(n)` still work.

## Vision

```python
clf = xor_net.ImageClassifier("models/vision/quicknet/quicknet.safetensors")
clf.predict("cat.jpg", k=5)
# [{'label': 'tiger_cat', 'probability': 0.64, 'class_index': 282}, ...]
```

`predict(image, k=5, width=None, height=None)` accepts:

- a path (`str` / `pathlib.Path`), or encoded JPEG / PNG `bytes`;
- a numpy `uint8` array of shape `HxWx3` (RGB), `HxWx4` (alpha dropped) or
  `HxW` (grayscale) - no numpy dependency, any object with `shape`, `dtype`
  and `tobytes()` works;
- a PIL image (converted to RGB);
- raw interleaved RGB `bytes` with `width=` and `height=`.

`logits(image)` returns the 1000 raw logits; `labels`, `num_classes` and
`name` describe the model.

## Tests

```bash
pip install pytest numpy pillow
pytest python/tests -q      # model-dependent tests skip when files are missing
```

`BONSAI_MODEL`, `XORNET_VISION_MODEL` and `XORNET_VISION_IMAGE` override the
default model and image locations.
