use std::collections::HashMap;
use memmap2::Mmap;

fn main() {
    let path = "/home/faudil/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127/model.safetensors";
    let file = std::fs::File::open(path).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().populate().map(&file).unwrap() };
    let safetensors = safetensors::SafeTensors::deserialize(&mmap).unwrap();
    let view = safetensors.tensor("model.layers.0.mlp.down_proj.weight").unwrap();
    println!("Shape: {:?}", view.shape());
    println!("Dtype: {:?}", view.dtype());
}
