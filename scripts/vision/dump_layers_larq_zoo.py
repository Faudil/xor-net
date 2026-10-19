#!/usr/bin/env python3
"""Dump every layer output of a Larq Zoo model for one preprocessed input.

Reads input `<index>.input` from a reference file written by
reference_larq_zoo.py and writes `{layer_name: f32 [h, w, c] or [n]}` to a
safetensors file, for layer-by-layer comparison with the Rust graph
(`tests/vision_quicknet.rs::quicknet_layerwise`).
"""

import argparse
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

CTORS = {"quicknet": "QuickNet", "quicknet_small": "QuickNetSmall", "quicknet_large": "QuickNetLarge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="quicknet", choices=sorted(CTORS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    import larq_zoo as lqz
    from safetensors.numpy import load_file, save_file

    model = getattr(lqz.sota, CTORS[args.model])(weights=args.weights)
    layers = [l for l in model.layers if type(l).__name__ not in ("InputLayer",)]
    probe = tf.keras.Model(model.input, [l.output for l in layers])
    x = load_file(args.reference)[f"{args.index}.input"]
    outs = probe(x[None], training=False)
    save_file({l.name: np.ascontiguousarray(o.numpy()[0].astype(np.float32)) for l, o in zip(layers, outs)}, args.out)


if __name__ == "__main__":
    main()
