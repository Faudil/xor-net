#!/usr/bin/env python3
"""
Convert a BitNet model from HuggingFace format to standard f32 safetensors
compatible with XorNet's Llama loader.

Usage:
    python scripts/convert_bitnet.py \\
        --model-id microsoft/bitnet-b1.58-2B-4T \\
        --output-dir /tmp/bitnet_converted

The loaded model (via transformers + autobitlinear) has standard f32 weights,
not packed ternary. This script renames tensors to match XorNet conventions
and writes a compatible config.json.
"""

import argparse
import json
import os
import sys
from typing import Dict

import torch
from safetensors.torch import save_file as safe_save
from transformers import AutoModelForCausalLM


MAX_FILE_SIZE = 2 * 1024 ** 3  # 2 GB per safetensors shard


def rename_tensor(key: str) -> str:
    key = key.replace("self_attn.attn_sub_norm", "self_attn.inner_attn_ln")
    key = key.replace("mlp.ffn_sub_norm", "mlp.ffn_layernorm")
    return key


def should_skip(key: str, tie_word_embeddings: bool) -> bool:
    if key.endswith("weight_scale"):
        return True
    if tie_word_embeddings and key == "lm_head.weight":
        return True
    return False


def tensor_byte_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * 4  # f32


def build_output_config(orig: dict) -> dict:
    out = {
        "hidden_size": orig.get("hidden_size", 2560),
        "intermediate_size": orig.get("intermediate_size", 6912),
        "num_attention_heads": orig.get("num_attention_heads", 20),
        "num_key_value_heads": orig.get("num_key_value_heads", 5),
        "num_hidden_layers": orig.get("num_hidden_layers", 28),
        "vocab_size": orig.get("vocab_size", 32000),
        "max_position_embeddings": orig.get("max_position_embeddings", 4096),
        "rms_norm_eps": orig.get("rms_norm_eps", 1e-5),
        "rope_theta": orig.get("rope_theta", 500000.0),
        "bos_token_id": orig.get("bos_token_id"),
        "eos_token_id": orig.get("eos_token_id"),
        "tie_word_embeddings": orig.get("tie_word_embeddings", True),
        "hidden_act": "relu2",
    }
    return {k: v for k, v in out.items() if v is not None}


def convert(model_id: str, output_dir: str) -> None:
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model type: {type(model).__name__}")

    orig_config = model.config.to_dict()
    print(f"Hidden size: {orig_config.get('hidden_size')}")
    print(f"Num layers: {orig_config.get('num_hidden_layers')}")

    tie_embeddings = bool(orig_config.get("tie_word_embeddings", True))
    output_config = build_output_config(orig_config)

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(output_config, f, indent=2)
    print(f"Wrote {config_path}")

    state_dict = model.state_dict()
    print(f"\nProcessing {len(state_dict)} raw tensors...")

    renamed: Dict[str, torch.Tensor] = {}
    for old_name, tensor in state_dict.items():
        if should_skip(old_name, tie_embeddings):
            print(f"  Skip: {old_name}")
            continue
        new_name = rename_tensor(old_name)
        renamed[new_name] = tensor.contiguous().float()
        shape_str = "x".join(str(s) for s in tensor.shape)
        print(f"  {old_name} -> {new_name}  [{shape_str}]")

    print(f"\nTotal tensors to save: {len(renamed)}")

    sorted_items = sorted(renamed.items(), key=lambda x: x[0])
    shards: list = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0

    for name, tensor in sorted_items:
        sz = tensor_byte_size(tensor)
        if current_size + sz > MAX_FILE_SIZE and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += sz
    if current_shard:
        shards.append(current_shard)

    print(f"Saving {len(shards)} safetensors shard(s)...")

    weight_map = {}
    if len(shards) == 1:
        path = os.path.join(output_dir, "model.safetensors")
        safe_save(shards[0], path)
        for name in shards[0]:
            weight_map[name] = "model.safetensors"
        print(f"  {path}  ({len(shards[0])} tensors)")
    else:
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            path = os.path.join(output_dir, fname)
            safe_save(shard, path)
            for name in shard:
                weight_map[name] = fname
            print(f"  {path}  ({len(shard)} tensors)")

        total_size = sum(
            tensor_byte_size(t) for shard in shards for t in shard.values()
        )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Wrote {index_path}")

    readme = """# Converted BitNet Model

This directory contains a BitNet model converted from HuggingFace safetensors
to standard f32 format compatible with XorNet's Llama loader.

## Conversion notes

- All weights are stored as float32 (f32) tensors
  - Original model loaded via transformers (autobitlinear unpacking)
  - weight_scale tensors discarded (f32 weights are dequantized)
- Tensor names follow XorNet Llama convention:
  - attn_sub_norm renamed to inner_attn_ln
  - ffn_sub_norm renamed to ffn_layernorm
- lm_head.weight excluded (tied with embed_tokens.weight)
- Config adjusted for XorNet:
  - hidden_act = "relu2"
  - rope_theta = 500000.0
  - rms_norm_eps = 1e-05
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme.strip() + "\n")
    print(f"Wrote {readme_path}")

    print("\nConversion complete!")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BitNet model to XorNet-compatible f32 safetensors"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/bitnet-b1.58-2B-4T",
        help="HuggingFace model ID (default: microsoft/bitnet-b1.58-2B-4T)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted model",
    )
    args = parser.parse_args()

    try:
        convert(args.model_id, args.output_dir)
    except Exception:
        print(f"Fatal error:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
