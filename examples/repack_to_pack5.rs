use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use memmap2::MmapOptions;
use safetensors::tensor::{Dtype, SafeTensors, TensorView, View};
use xor_net::bit1_58::quantization::{pack_1_58bit_5pack, unpack_1_58bit_4pack, TernaryPackType};
use std::borrow::Cow;

fn main() -> anyhow::Result<()> {
    // This script converts a Pack4 safetensors model into a Pack5 safetensors model.
    let input_path = std::env::args().nth(1).unwrap_or_else(|| "model.safetensors".to_string());
    let output_path = std::env::args().nth(2).unwrap_or_else(|| "model_pack5.safetensors".to_string());

    if !Path::new(&input_path).exists() {
        println!("Input file {} not found. Please provide a path to a HuggingFace Pack4 model.", input_path);
        return Ok(());
    }

    println!("Repacking {} to {}", input_path, output_path);
    let file = File::open(&input_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut out_tensors: HashMap<String, TensorView> = HashMap::new();
    let mut pack5_buffers: Vec<Vec<u8>> = Vec::new(); // keep data alive

    for name in tensors.names() {
        let tensor = tensors.tensor(name)?;
        
        if name.ends_with(".weight") && tensor.dtype() == Dtype::I8 {
            // Found a Pack4 quantized weight block!
            // In Hugging Face, weight block sizes are [out_dim, in_dim] but stored in Pack4 format
            // Meaning the data length is out_dim * in_dim / 4
            let expected_len = tensor.shape()[0] * tensor.shape()[1];
            
            // 1. Unpack from Pack4 using Striped logic
            // Note: Hugging Face BitNet uses striped packing. We will unpack to raw F32 first.
            // Wait, for repack, we can just use our `unpack_1_58bit_4pack` directly.
            let unpacked = unpack_1_58bit_4pack(tensor.data(), expected_len);
            
            // 2. Repack into Pack5
            let packed5 = pack_1_58bit_5pack(&unpacked, 1.0); // scale is applied separately
            
            pack5_buffers.push(packed5);
            let buf_ref = pack5_buffers.last().unwrap();
            
            out_tensors.insert(name.to_string(), TensorView::new(
                Dtype::U8,
                tensor.shape().to_vec(),
                buf_ref.as_slice(),
            ).unwrap());
            println!("Repacked {} to Pack5 format.", name);
        } else {
            // Keep unchanged
            out_tensors.insert(name.to_string(), TensorView::new(
                tensor.dtype(),
                tensor.shape().to_vec(),
                tensor.data(),
            ).unwrap());
        }
    }

    let mut out_file = File::create(&output_path)?;
    // Use proper tuple iterator matching for safetensors <= 0.3.1
    let iter = out_tensors.iter().map(|(k, v)| (k.as_str(), v));
    safetensors::serialize_to_file(iter, &None, Path::new(&output_path))?;
    println!("Successfully saved Pack5 model to {}", output_path);
    Ok(())
}
