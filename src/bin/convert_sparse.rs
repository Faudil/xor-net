//! `convert_sparse`  convert a 1.58-bit BitNet checkpoint's ternary weights
//! into the XorSparse (`.sparse`) container.
//!
//! Usage:
//!   convert_sparse <model_dir> <output.sparse> [--no-invert]
//!
//! The converter scans every U8 weight tensor that has a companion
//! `*_scale` tensor (i.e. a packed-ternary projection), unpacks it to ternary
//! `{-1,0,+1}` values, re-encodes only the non-zero signs (lossless), and writes
//! the XorSparse file. The engine loads that file with `XORNET_WEIGHT_FMT=sparse`.
//!
//! `--no-invert` keeps the stored scale as-is; by default the scale is inverted
//! (1/γ) to match HF1BitLLM checkpoints, exactly like `get_prepacked_ternary`.

use std::path::PathBuf;

use xor_net::bit1_58::quantization::unpack_1_58bit_4pack;
use xor_net::bit1_58::sparse::encode_sparse_tensor;
use xor_net::loader::{SafeTensorRepo, SafeTensorLoader, sparse_loader::SparseFile};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut positional = Vec::new();
    let mut invert = true;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--no-invert" {
            invert = false;
        } else if a == "--model" {
            i += 1;
            positional.push(args[i].clone());
        } else {
            positional.push(a.clone());
        }
        i += 1;
    }
    if positional.len() < 2 {
        eprintln!("usage: convert_sparse <model_dir> <output.sparse> [--no-invert] [--model <dir>]");
        std::process::exit(2);
    }
    let model_dir = PathBuf::from(&positional[0]);
    let out_path = PathBuf::from(&positional[1]);

    // Discover safetensors files (mirrors AutoModelForCausalLM::from_local).
    let mut filenames = Vec::new();
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;
        if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
            let mut unique: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            unique.sort();
            unique.dedup();
            for f in unique {
                filenames.push(model_dir.join(f));
            }
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            filenames.push(single);
        }
    }
    if filenames.is_empty() {
        anyhow::bail!("no safetensors found in {}", model_dir.display());
    }

    let repo = SafeTensorRepo::load(&filenames)?;
    let _loader = SafeTensorLoader::new(&repo);

    let mut entries: Vec<(String, usize, usize, f32, Vec<u8>)> = Vec::new();
    let mut total_dense = 0usize;
    let mut total_sparse = 0usize;

    // Iterate tensor directory; only U8 packed-ternary weight tensors qualify.
    let names: Vec<String> = repo.tensors.keys().cloned().collect();
    for name in names {
        if !name.ends_with(".weight") {
            continue;
        }
        let info = &repo.tensors[&name];
        if info.dtype != safetensors::Dtype::U8 {
            continue;
        }
        if info.shape.len() != 2 {
            continue;
        }
        let scale_key = format!("{}_scale", name);
        let Some(sinfo) = repo.tensors.get(&scale_key) else {
            continue;
        };
        let stored_out = info.shape[0];
        let in_dim = info.shape[1];
        let out_dim = stored_out * 4;
        if out_dim == 0 || in_dim == 0 {
            continue;
        }

        let raw = &repo.buffers[info.file_idx][info.start..info.end];
        let ternary = unpack_1_58bit_4pack(raw, out_dim * in_dim);
        let ternary_i8: Vec<i8> = ternary.iter().map(|&v| v as i8).collect();
        let (blob, _offsets) = encode_sparse_tensor(&ternary_i8, in_dim, out_dim);

        let sbytes = &repo.buffers[sinfo.file_idx][sinfo.start..sinfo.end];
        let scale_data = xor_net::loader::convert_scale_bytes(sbytes, sinfo.dtype)?;
        let w_scale = if invert { 1.0 / scale_data } else { scale_data };

        total_dense += raw.len();
        total_sparse += blob.len();

        entries.push((name.clone(), out_dim, in_dim, w_scale, blob));
        eprintln!(
            "  {}  [{}, {}]  -> {} bytes",
            name,
            out_dim,
            in_dim,
            entries.last().unwrap().4.len()
        );
    }

    if entries.is_empty() {
        anyhow::bail!("no packed-ternary weight tensors found to convert");
    }

    let bytes = SparseFile::serialize(&entries);
    std::fs::write(&out_path, &bytes)?;
    eprintln!(
        "\nwrote {} tensors, {} -> {} bytes ({:.1}% of dense pack4)",
        entries.len(),
        total_dense,
        total_sparse,
        if total_dense > 0 {
            100.0 * total_sparse as f64 / total_dense as f64
        } else {
            0.0
        }
    );
    Ok(())
}
