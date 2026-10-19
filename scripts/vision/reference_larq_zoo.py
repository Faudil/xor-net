#!/usr/bin/env python3
"""Run the Larq/TensorFlow reference for a Larq Zoo model and dump test vectors.

For each image it stores, in one safetensors file:
  * `<i>.rgb`    u8  [h, w, 3]   decoded RGB (PIL), the raw input to preprocessing
  * `<i>.input`  f32 [224,224,3] larq_zoo `preprocess_input` output
  * `<i>.logits` f32 [1000]      pre-softmax logits from the Keras model
metadata "names" = JSON list of image file names.

Optionally benchmarks Keras inference latency (batch 1) with a fixed number
of intra-op threads.

    TF_USE_LEGACY_KERAS=1 python -I scripts/vision/reference_larq_zoo.py \
        --model quicknet --weights models/vision/quicknet/quicknet_weights.h5 \
        --out models/vision/quicknet/reference.safetensors models/vision/images/*.jpg
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

CTORS = {"quicknet": "QuickNet", "quicknet_small": "QuickNetSmall", "quicknet_large": "QuickNetLarge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="quicknet", choices=sorted(CTORS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="timed iterations")
    ap.add_argument("images", nargs="*")
    args = ap.parse_args()

    import tensorflow as tf

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    import larq_zoo as lqz
    from PIL import Image

    model = getattr(lqz.sota, CTORS[args.model])(weights=args.weights)
    dense = [l for l in model.layers if type(l).__name__ == "QuantDense"][-1]
    logits_model = tf.keras.Model(model.input, dense.output)

    out = {}
    names = []
    for i, path in enumerate(args.images):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        x = lqz.preprocess_input(rgb).astype(np.float32)
        logits = logits_model(x[None], training=False).numpy()[0]
        out[f"{i}.rgb"] = rgb
        out[f"{i}.input"] = x
        out[f"{i}.logits"] = logits.astype(np.float32)
        names.append(os.path.basename(path))
        top = np.argsort(-logits)[:5]
        print(os.path.basename(path), top.tolist(), [round(float(logits[t]), 3) for t in top])

    if args.out:
        from safetensors.numpy import save_file

        save_file(out, args.out, metadata={"names": json.dumps(names), "model": args.model})
        print(f"wrote {args.out}", file=sys.stderr)

    if args.bench:
        x = tf.constant(np.random.default_rng(0).normal(size=(1, 224, 224, 3)).astype(np.float32))
        fn = tf.function(lambda t: logits_model(t, training=False))
        for _ in range(5):
            fn(x)
        t0 = time.perf_counter()
        for _ in range(args.bench):
            fn(x).numpy()
        dt = (time.perf_counter() - t0) / args.bench
        print(f"tf/larq {args.model} threads={args.threads or 'default'}: {dt * 1e3:.2f} ms/img, {1 / dt:.1f} img/s")


if __name__ == "__main__":
    main()
