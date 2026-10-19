#!/usr/bin/env python3
"""Convert a pretrained Larq Zoo binary network into xor-net's vision format.

Supported models: QuickNet, QuickNetSmall, QuickNetLarge (larq/zoo releases,
Apache-2.0). The script builds the Keras model with `larq_zoo`, loads the
official ImageNet weights, walks the functional graph and writes one
`.safetensors` file that `xor_net::vision::ImageClassifier::load` reads.

Usage (inside a venv with tensorflow-cpu, tf-keras, larq, larq-zoo,
safetensors; see scripts/vision/README in the module docs):

    TF_USE_LEGACY_KERAS=1 python -I scripts/vision/convert_larq_zoo.py \
        --model quicknet --out models/vision/quicknet/quicknet.safetensors

Weights are downloaded into the output directory and checked against the
SHA-256 hashes that larq_zoo pins. Downloaded files are only ever read as
data (HDF5 via Keras `load_weights`, JSON via `json.load`).

File format ("xornet-vision-v1")
--------------------------------
safetensors tensors (all little endian):
  * float conv        `<layer>.weight`      f32 [kh, kw, cin, cout]   (HWIO)
  * depthwise conv    `<layer>.weight`      f32 [kh, kw, c]
  * binary conv       `<layer>.weight_bits` u64 [cout, kh, kw, ceil(cin/64)]
        bit (c % 64) of word (c / 64) is 1 iff sign(w) = +1 (w >= 0, larq's
        sign convention); unused high bits of the last word are 0.
  * batch norm        `<layer>.scale`, `<layer>.bias` f32 [c]
        y = x * scale + bias, scale = gamma / sqrt(var + eps),
        bias = beta - mean * scale  (exact rewrite of inference-mode BN).
  * dense             `<layer>.weight` f32 [in, out], `<layer>.bias` f32 [out]
metadata (string -> string):
  * "format"  = "xornet-vision-v1"
  * "model"   = model name
  * "graph"   = JSON list of ops (topological order), each
        {"op", "name", "inputs": [value names], "output": value name, ...}
        ops: conv2d, binary_conv2d, depthwise_conv2d, scale_bias, relu,
             add, max_pool2d, avg_pool2d, global_avg_pool, dense.
        The value "input" is the preprocessed NHWC image; the graph output
        is named by "output".
  * "output"  = name of the logits value
  * "preprocess" = JSON {"size", "crop_padding", "mean", "std", "resize"}
  * "labels"  = JSON list of 1000 human readable ImageNet class names
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

RELEASE = "https://github.com/larq/zoo/releases/download/quicknet-v1.0/"
MODELS = {
    "quicknet": (
        "QuickNet",
        "quicknet_weights.h5",
        "8aba9e4e5f8d342faef04a0b2ae8e562da57dbb7d15162e8b3e091c951ba756c",
    ),
    "quicknet_small": (
        "QuickNetSmall",
        "quicknet_small_weights.h5",
        "1ac3b07df7f5a911dd0b49febb2486428ddf1ca130297c403815dfae5a1c71a2",
    ),
    "quicknet_large": (
        "QuickNetLarge",
        "quicknet_large_weights.h5",
        "c5158e8a59147b31370becd937825f4db8a5cdf308314874f678f596629be45c",
    ),
}
LABELS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)
LABELS_SHA256 = None  # informational; the file is parsed as plain JSON only


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, expected_sha=None):
    if not os.path.exists(path):
        print(f"downloading {url}", file=sys.stderr)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    if expected_sha is not None:
        got = sha256(path)
        if got != expected_sha:
            raise SystemExit(f"{path}: sha256 {got} != expected {expected_sha}")
    return path


def build_model(model_key, weights_path):
    import larq_zoo as lqz

    ctor = getattr(lqz.sota, MODELS[model_key][0])
    return ctor(weights=weights_path)


def inbound_names(layer):
    nodes = layer._inbound_nodes
    if not nodes:
        return []
    ins = nodes[0].inbound_layers
    if not isinstance(ins, (list, tuple)):
        ins = [ins]
    return [l.name for l in ins]


def is_ste_sign(q):
    import larq as lq

    return isinstance(q, lq.quantizers.SteSign)


def pack_bits(kernel_hwio):
    """[kh, kw, cin, cout] float -> [cout, kh, kw, words] u64 sign bits."""
    kh, kw, cin, cout = kernel_hwio.shape
    words = (cin + 63) // 64
    bits = (kernel_hwio >= 0).astype(np.uint64)  # larq sign(0) = +1
    bits = np.transpose(bits, (3, 0, 1, 2))  # cout, kh, kw, cin
    padded = np.zeros((cout, kh, kw, words * 64), dtype=np.uint64)
    padded[..., :cin] = bits
    padded = padded.reshape(cout, kh, kw, words, 64)
    shifts = np.arange(64, dtype=np.uint64)
    return np.bitwise_or.reduce(padded << shifts, axis=-1).astype(np.uint64)


def padding_of(cfg):
    p = cfg["padding"]
    if p not in ("same", "valid"):
        raise SystemExit(f"unsupported padding {p}")
    return p


def convert(model):
    import larq as lq
    import tensorflow as tf

    tensors = {}
    graph = []
    value_of = {}  # keras layer name -> value name

    for layer in model.layers:
        cls = type(layer).__name__
        name = layer.name
        ins = [value_of[n] for n in inbound_names(layer)]
        cfg = layer.get_config()

        def emit(op, **kw):
            graph.append({"op": op, "name": name, "inputs": ins, "output": name, **kw})
            value_of[name] = name

        if cls == "InputLayer":
            value_of[name] = "input"
            input_shape = layer.output_shape[0] if isinstance(layer.output_shape, list) else layer.output_shape
            continue
        if cls == "Flatten":
            value_of[name] = ins[0]  # [1,1,C] -> [C]: no-op in NHWC
            continue
        if cls == "Activation":
            act = cfg["activation"]
            if act == "softmax":
                continue  # the runtime returns logits and applies softmax itself
            if act != "relu":
                raise SystemExit(f"unsupported activation {act}")
            emit("relu")
            continue
        if cls == "TFOpLambda":
            if cfg["function"] != "__operators__.add":
                raise SystemExit(f"unsupported op {cfg['function']}")
            emit("add")
            continue
        if cls == "BatchNormalization":
            eps = cfg["epsilon"]
            mean = layer.moving_mean.numpy().astype(np.float64)
            var = layer.moving_variance.numpy().astype(np.float64)
            gamma = layer.gamma.numpy().astype(np.float64) if layer.scale else np.ones_like(mean)
            beta = layer.beta.numpy().astype(np.float64) if layer.center else np.zeros_like(mean)
            scale = gamma / np.sqrt(var + eps)
            bias = beta - mean * scale
            tensors[f"{name}.scale"] = scale.astype(np.float32)
            tensors[f"{name}.bias"] = bias.astype(np.float32)
            emit("scale_bias", scale=f"{name}.scale", bias=f"{name}.bias")
            continue
        if cls == "MaxPooling2D":
            emit(
                "max_pool2d",
                pool=list(cfg["pool_size"]),
                stride=list(cfg["strides"]),
                padding=padding_of(cfg),
            )
            continue
        if cls == "AveragePooling2D":
            in_hw = list(layer.input_shape[1:3])
            if list(cfg["pool_size"]) == in_hw and cfg["padding"] == "valid":
                emit("global_avg_pool")
            else:
                emit(
                    "avg_pool2d",
                    pool=list(cfg["pool_size"]),
                    stride=list(cfg["strides"]),
                    padding=padding_of(cfg),
                )
            continue
        if cls == "GlobalAveragePooling2D":
            emit("global_avg_pool")
            continue
        if cls in ("QuantConv2D", "Conv2D"):
            kernel = layer.kernel.numpy().astype(np.float32)
            if cfg.get("use_bias"):
                raise SystemExit(f"{name}: conv bias not supported")
            if tuple(cfg.get("dilation_rate", (1, 1))) != (1, 1):
                raise SystemExit(f"{name}: dilation not supported")
            act = cfg.get("activation", "linear")
            if act not in ("linear", "relu"):
                raise SystemExit(f"{name}: unsupported activation {act}")
            iq = getattr(layer, "input_quantizer", None)
            kq = getattr(layer, "kernel_quantizer", None)
            common = dict(
                kernel=list(kernel.shape[:2]),
                stride=list(cfg["strides"]),
                padding=padding_of(cfg),
                in_channels=int(kernel.shape[2]),
                out_channels=int(kernel.shape[3]),
                activation=act,
            )
            if iq is None and kq is None:
                if float(cfg.get("pad_values", 0.0)) != 0.0:
                    raise SystemExit(f"{name}: float conv with pad_values")
                tensors[f"{name}.weight"] = kernel
                emit("conv2d", weight=f"{name}.weight", **common)
            elif is_ste_sign(iq) and is_ste_sign(kq):
                tensors[f"{name}.weight_bits"] = pack_bits(kernel)
                emit(
                    "binary_conv2d",
                    weight=f"{name}.weight_bits",
                    pad_value=float(cfg.get("pad_values", 0.0)),
                    **common,
                )
            else:
                raise SystemExit(f"{name}: unsupported quantizer combination")
            continue
        if cls in ("QuantDepthwiseConv2D", "DepthwiseConv2D"):
            if getattr(layer, "input_quantizer", None) or getattr(
                layer, "depthwise_quantizer", None
            ):
                raise SystemExit(f"{name}: quantized depthwise not supported")
            if cfg.get("use_bias") or cfg.get("depth_multiplier", 1) != 1:
                raise SystemExit(f"{name}: depthwise bias/multiplier not supported")
            if cfg.get("activation", "linear") != "linear":
                raise SystemExit(f"{name}: depthwise activation not supported")
            k = layer.depthwise_kernel.numpy().astype(np.float32)  # kh, kw, c, 1
            tensors[f"{name}.weight"] = np.ascontiguousarray(k[..., 0])
            emit(
                "depthwise_conv2d",
                weight=f"{name}.weight",
                kernel=list(k.shape[:2]),
                stride=list(cfg["strides"]),
                padding=padding_of(cfg),
                channels=int(k.shape[2]),
            )
            continue
        if cls in ("QuantDense", "Dense"):
            if getattr(layer, "input_quantizer", None) or getattr(
                layer, "kernel_quantizer", None
            ):
                raise SystemExit(f"{name}: quantized dense not supported")
            if cfg.get("activation", "linear") != "linear":
                raise SystemExit(f"{name}: dense activation not supported")
            w = layer.kernel.numpy().astype(np.float32)
            b = layer.bias.numpy().astype(np.float32)
            tensors[f"{name}.weight"] = w
            tensors[f"{name}.bias"] = b
            emit(
                "dense",
                weight=f"{name}.weight",
                bias=f"{name}.bias",
                in_features=int(w.shape[0]),
                out_features=int(w.shape[1]),
            )
            continue
        raise SystemExit(f"unsupported layer {cls} ({name})")

    output = graph[-1]["output"]
    return tensors, graph, output, list(input_shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="quicknet", choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    _, fname, digest = MODELS[args.model]
    weights = fetch(RELEASE + fname, os.path.join(out_dir, fname), digest)
    labels_path = fetch(LABELS_URL, os.path.join(out_dir, "imagenet_class_index.json"))
    with open(labels_path) as f:
        idx = json.load(f)
    labels = [str(idx[str(i)][1]) for i in range(1000)]

    model = build_model(args.model, weights)
    tensors, graph, output, in_shape = convert(model)
    if in_shape != [224, 224, 3]:
        raise SystemExit(f"unexpected input shape {in_shape}")

    meta = {
        "format": "xornet-vision-v1",
        "model": args.model,
        "source": RELEASE + fname,
        "license": "Apache-2.0 (larq/zoo)",
        "graph": json.dumps(graph),
        "output": output,
        "preprocess": json.dumps(
            {
                "size": 224,
                "crop_padding": 32,
                "mean": [0.485 * 255, 0.456 * 255, 0.406 * 255],
                "std": [0.229 * 255, 0.224 * 255, 0.225 * 255],
                "resize": "tf_legacy_bicubic",
            }
        ),
        "labels": json.dumps(labels),
    }
    from safetensors.numpy import save_file

    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, args.out, metadata=meta)
    nbin = sum(1 for op in graph if op["op"] == "binary_conv2d")
    print(
        f"wrote {args.out}: {len(graph)} ops ({nbin} binary convs), "
        f"{len(tensors)} tensors, {os.path.getsize(args.out) / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
