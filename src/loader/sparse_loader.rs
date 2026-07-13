//! Loader for the XorSparse on-disk container (see `SPARSE_FORMAT.md`).
//!
//! The container is a flat, header+`blob` file: little-endian metadata followed
//! by each tensor's sparse blob (the exact per-row layout produced by
//! [`crate::bit1_58::sparse::encode_sparse_row_into`]). Tensors are addressed by
//! their safetensors name (e.g. `model.layers.0.self_attn.q_proj.weight`), so the
//! engine can look them up with the same key it would use against the original
//! checkpoint.
//!
//! Reading the smaller file into RAM shrinks the per-token weight stream
//! (the bottleneck on this engine); the blobs are copied once and then streamed
//! untouched on every decode step.

use std::collections::HashMap;
use std::path::Path;

use crate::bit1_58::sparse::SparseTernary;

const MAGIC: &[u8; 9] = b"XORSPARE1";
const VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct Entry {
    blob_off: usize,
    blob_len: usize,
    in_dim: usize,
    out_dim: usize,
    w_scale: f32,
}

/// A parsed XorSparse container. Cheap to construct (just parses the directory);
/// the weight bytes are referenced in-place out of `data`.
#[derive(Debug, Clone)]
pub struct SparseFile {
    data: Vec<u8>,
    tensors: HashMap<String, Entry>,
}

impl SparseFile {
    /// Parse an in-memory copy of a `.sparse` file.
    pub fn from_bytes(data: Vec<u8>) -> anyhow::Result<Self> {
        if data.len() < 16 {
            anyhow::bail!("XorSparse file too small");
        }
        if &data[0..9] != MAGIC {
            anyhow::bail!("XorSparse bad magic (expected XORSPARE1)");
        }
        let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if version != VERSION {
            anyhow::bail!("XorSparse unsupported version {}", version);
        }
        let num_tensors = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut tensors = HashMap::new();
        let mut p = 16usize;
        for _ in 0..num_tensors {
            let name_len =
                u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            p += 4;
            let name = String::from_utf8(data[p..p + name_len].to_vec())
                .map_err(|_| anyhow::anyhow!("XorSparse tensor name not utf8"))?;
            p += name_len;
            let out_dim =
                u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            p += 4;
            let in_dim =
                u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            p += 4;
            let w_scale = f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
            p += 4;
            let blob_len =
                u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            p += 4;
            let blob_off = p;
            p += blob_len;
            tensors.insert(
                name,
                Entry {
                    blob_off,
                    blob_len,
                    in_dim,
                    out_dim,
                    w_scale,
                },
            );
        }
        Ok(Self { data, tensors })
    }

    /// Read and parse a `.sparse` file from disk.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(data)
    }

    /// Look up a tensor by its safetensors name and return its sparse weights.
    /// Errors if the name is absent (callers fall back to the original checkpoint).
    pub fn get_sparse(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> anyhow::Result<(SparseTernary, f32)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("XorSparse: tensor '{}' not found", name))?;
        if entry.in_dim != in_dim || entry.out_dim != out_dim {
            anyhow::bail!(
                "XorSparse '{}' shape mismatch: expected [{}, {}], got [{}, {}]",
                name,
                out_dim,
                in_dim,
                entry.out_dim,
                entry.in_dim
            );
        }
        let blob = self.data[entry.blob_off..entry.blob_off + entry.blob_len].to_vec();
        let st = SparseTernary::from_blob(blob, out_dim);
        Ok((st, entry.w_scale))
    }

    /// Encode the directory back to bytes (used by the `convert_sparse` tool).
    pub fn serialize(entries: &[(String, usize, usize, f32, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, out_dim, in_dim, w_scale, blob) in entries {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(*out_dim as u32).to_le_bytes());
            out.extend_from_slice(&(*in_dim as u32).to_le_bytes());
            out.extend_from_slice(&w_scale.to_le_bytes());
            out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
            out.extend_from_slice(blob);
        }
        out
    }
}
