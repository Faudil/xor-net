//! Convert a Bonsai (PrismML Q1_0) GGUF into the `.xq1` format: pre-split
//! sign bits / f16 scales, pre-fused projections, embedded tokenizer and
//! chat template, mmap-able for near-instant zero-copy loading. Lossless;
//! the output is reloaded and verified tensor by tensor.
//!
//! Usage: bonsai_convert <in.gguf> <out.xq1> [--tokenizer tokenizer.json]
use std::path::PathBuf;
use std::time::Instant;

use xor_net::loader::gguf::GgufFile;
use xor_net::models::q1::{ModelAssets, format};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut pos: Vec<PathBuf> = Vec::new();
    let mut tokenizer: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokenizer" => {
                tokenizer = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--tokenizer needs a path"))?
                        .into(),
                )
            }
            _ => pos.push(a.into()),
        }
    }
    let [input, output] = <[PathBuf; 2]>::try_from(pos).map_err(|_| {
        anyhow::anyhow!("usage: bonsai_convert <in.gguf> <out.xq1> [--tokenizer tokenizer.json]")
    })?;
    let t0 = Instant::now();
    let g = GgufFile::open(&input)?;
    let mut assets = ModelAssets::from_gguf(&g)?;
    if let Some(p) = tokenizer {
        assets.tokenizer = tokenizers::Tokenizer::from_file(&p)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", p.display()))?;
    }
    let arch = format::convert_gguf(&g, &output, &assets)?;
    let size = std::fs::metadata(&output)?.len();
    println!(
        "{} ({arch}) -> {} ({:.1} MB) in {:.1}s, verified bit-identical",
        input.display(),
        output.display(),
        size as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
