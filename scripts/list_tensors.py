import safetensors.torch
import sys
import glob
for f in glob.glob("models/bitnet-2b/*.safetensors"):
    t = safetensors.torch.load_file(f)
    for k, v in t.items():
        if "weight" in k and "norm" not in k and "embed" not in k and "layers.0" in k:
            print(f"  {k}: {v.dtype} min={v.min().item()} max={v.max().item()} mean={v.float().mean().item()}")
