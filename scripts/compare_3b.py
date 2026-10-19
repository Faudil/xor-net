"""Compare 1bitLLM/bitnet_b1_58-3B logits between HuggingFace and XorNet.

Usage:
  python scripts/compare_3b.py <prompt_text>

Outputs reference logits from HuggingFace to a file, for comparison.
"""
import json
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "1bitLLM/bitnet_b1_58-3B"

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Input tokens: {inputs.input_ids.tolist()}")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids)
        logits = outputs.logits[0, -1, :]  # last token logits
    
    logits_np = logits.numpy()
    
    # Find top-10 tokens
    top_indices = np.argsort(logits_np)[-10:][::-1]
    print("\nTop-10 tokens from HuggingFace:")
    for idx in top_indices:
        print(f"  token {idx:5}: {logits_np[idx]:.4f}")
    
    # Save full logits to file
    np.save("/tmp/hf_logits_3b.npy", logits_np)
    print(f"\nSaved full logits ({len(logits_np)} values) to /tmp/hf_logits_3b.npy")
    
    # Also save the first 20 entries
    print(f"\nFirst 20 logits: {logits_np[:20].tolist()}")
    print(f"Logits min: {logits_np.min():.4f}, max: {logits_np.max():.4f}, mean: {logits_np.mean():.4f}")

if __name__ == "__main__":
    main()
