"""Smoke tests for the xor_net Python bindings.

Build first:  maturin develop --release
Run:          pytest python/tests -q

Model-dependent tests are skipped when the files are missing. Override the
locations with BONSAI_MODEL (a .gguf/.xq1 path) and XORNET_VISION_MODEL /
XORNET_VISION_IMAGE.
"""

import os
import pathlib

import pytest

xor_net = pytest.importorskip("xor_net")

ROOT = pathlib.Path(__file__).resolve().parents[2]
BONSAI = pathlib.Path(
    os.environ.get("BONSAI_MODEL", ROOT / "models/bonsai/Bonsai-1.7B.xq1")
)
VISION = pathlib.Path(
    os.environ.get(
        "XORNET_VISION_MODEL", ROOT / "models/vision/quicknet/quicknet.safetensors"
    )
)
IMAGE = pathlib.Path(
    os.environ.get("XORNET_VISION_IMAGE", ROOT / "models/vision/images/zebra.jpg")
)

needs_bonsai = pytest.mark.skipif(not BONSAI.is_file(), reason=f"{BONSAI} missing")
needs_vision = pytest.mark.skipif(
    not (VISION.is_file() and IMAGE.is_file()), reason="vision model/image missing"
)


def test_module_surface():
    for name in ["load", "Model", "ImageClassifier", "Completion", "TextStream",
                 "XorNetModel", "QuantizationConfig", "__version__"]:
        assert hasattr(xor_net, name), name


def test_load_missing_path_raises():
    with pytest.raises(RuntimeError, match="does not exist"):
        xor_net.load("definitely/not/here.gguf")


@pytest.fixture(scope="module")
def model():
    return xor_net.load(str(BONSAI))


@needs_bonsai
def test_generate_greedy_is_deterministic(model):
    assert model.kind == "bonsai"
    assert model.vocab_size > 100_000 and model.context_length >= 4096
    a = model.generate("The capital of France is", max_tokens=8, temperature=0)
    b = model.generate("The capital of France is", max_tokens=8, temperature=0)
    assert a == b and "Paris" in a


@needs_bonsai
def test_details_and_stop(model):
    c = model.generate("1, 2, 3, 4,", max_tokens=24, temperature=0, stop=["7"],
                       details=True)
    assert isinstance(c, xor_net.Completion)
    assert "7" not in c.text and c.finish_reason == "stop"
    assert c.prompt_tokens > 0 and c.completion_tokens > 0
    assert c.decode_tok_per_s > 0


@needs_bonsai
def test_stream_matches_generate(model):
    full = model.generate("Once upon a time", max_tokens=16, temperature=0)
    stream = model.generate_stream("Once upon a time", max_tokens=16, temperature=0)
    pieces = list(stream)
    assert "".join(pieces) == full
    assert stream.finish_reason in ("stop", "length")
    assert stream.completion.text == full


@needs_bonsai
def test_chat_and_chat_stream(model):
    msgs = [{"role": "user", "content": "What is 2 + 2? Answer with a number."}]
    reply = model.chat(msgs, max_tokens=32, temperature=0, enable_thinking=False)
    assert "4" in reply
    events = list(model.chat_stream(msgs, max_tokens=32, temperature=0, reasoning=True))
    assert all(kind in ("text", "reasoning") for kind, _ in events)
    assert "".join(t for k, t in events if k == "text") == reply
    assert "<|im_start|>assistant" in model.apply_chat_template(msgs)


@needs_bonsai
def test_stream_cancel(model):
    s = model.generate_stream("Count to one thousand:", max_tokens=200, temperature=0)
    next(s)
    s.cancel()
    rest = list(s)
    assert len(rest) < 200


@needs_vision
def test_image_classifier_inputs():
    clf = xor_net.ImageClassifier(str(VISION))
    assert clf.num_classes == 1000 and len(clf.labels) == 1000
    top = clf.predict(str(IMAGE), k=3)
    assert len(top) == 3 and top[0]["label"] == "zebra"
    assert top[0]["probability"] >= top[1]["probability"]
    # Encoded bytes give the same answer as the path.
    assert clf.predict(IMAGE.read_bytes(), k=1)[0]["label"] == "zebra"
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    pil = Image.open(IMAGE).convert("RGB")
    arr = np.asarray(pil, dtype=np.uint8)
    assert clf.predict(arr, k=1)[0]["label"] == "zebra"
    assert clf.predict(pil, k=1)[0]["label"] == "zebra"
    h, w, _ = arr.shape
    assert clf.predict(arr.tobytes(), k=1, width=w, height=h)[0]["label"] == "zebra"
    with pytest.raises(ValueError):
        clf.predict(arr.astype(np.float32))
    assert len(clf.logits(arr)) == 1000
