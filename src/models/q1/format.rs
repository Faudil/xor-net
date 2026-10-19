//! `.xq1`: a converted, mmap-friendly file format for Bonsai 1-bit models.
//!
//! Converting a GGUF once (`xor-net convert in.gguf out.xq1`) moves all the
//! load-time work offline: the Q1_0 blocks are already split into sign bits
//! and f16 scales, Q/K/V and gate/up are pre-fused, and the tokenizer and
//! chat template are embedded. Loading is an `mmap` plus a JSON header parse:
//! 1-bit weights are used in place (zero copy, 64-byte aligned), so start-up
//! is near-instant and the weights are clean file-backed pages the kernel
//! can share between processes and evict under pressure.
//!
//! The conversion is lossless: every sign bit and f16 scale is stored
//! verbatim ([`crate::bit1::q1_0::Q1Weight::to_gguf`] restores the GGUF
//! blocks bit for bit).
//!
//! Layout (little endian):
//!
//! ```text
//! 0   magic "XQ1F" (4 bytes), version u32 = 1
//! 8   header_off u64, header_len u64
//! 24  zero padding up to 64
//! 64  tensor data, each tensor at a 64-byte aligned file offset
//!     header (JSON) at header_off
//! ```
//!
//! The JSON header (written last, so tensors stream straight to disk) holds
//! `arch`, `config`, `tensors` (name → entry with absolute file offsets),
//! `tokenizer` (HF tokenizer JSON) and `chat_template`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use super::{Embedding, Linear, ModelAssets, qwen3, qwen35};
use crate::bit1::q1_0::{ByteBuf, Q1Weight};
use crate::models::traits::CausalLM;

const MAGIC: &[u8; 4] = b"XQ1F";
const VERSION: u32 = 1;
const ALIGN: usize = 64;
const DATA_START: usize = 64;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Entry {
    Q1 {
        rows: usize,
        cols: usize,
        block: usize,
        bits: (usize, usize),
        scales: (usize, usize),
    },
    F32 {
        shape: Vec<usize>,
        data: (usize, usize),
    },
}

#[derive(Serialize, Deserialize)]
struct Header {
    arch: String,
    config: serde_json::Value,
    tensors: BTreeMap<String, Entry>,
    tokenizer: Option<String>,
    chat_template: Option<String>,
}

/// Receives a model's named tensors (see `write_xq1` on the models).
pub trait TensorSink {
    fn put_vec(&mut self, name: &str, v: &[f32]);
    fn put_linear(&mut self, name: &str, l: &Linear);
}

/// A [`TensorSink`] that records a content hash per tensor (name, shape and
/// every byte), used to check that a reloaded model is bit-identical.
#[derive(Default, PartialEq, Eq, Debug)]
pub struct TensorDigest(pub BTreeMap<String, u64>);

impl TensorDigest {
    fn hash(parts: &[&[u8]], dims: &[usize]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        dims.hash(&mut h);
        for p in parts {
            p.hash(&mut h);
        }
        h.finish()
    }
}

impl TensorSink for TensorDigest {
    fn put_vec(&mut self, name: &str, v: &[f32]) {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.0.insert(name.into(), Self::hash(&[&b], &[v.len()]));
    }
    fn put_linear(&mut self, name: &str, l: &Linear) {
        let d = match l {
            Linear::Q1(w) => Self::hash(
                &[w.bits().as_slice(), w.scales().as_slice()],
                &[w.rows(), w.cols(), w.block()],
            ),
            Linear::F32 { rows, cols, w } => {
                let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                Self::hash(&[&b], &[*rows, *cols])
            }
        };
        self.0.insert(name.into(), d);
    }
}

/// Streams named tensors into an `.xq1` file; [`XqWriter::finish`] writes
/// the header.
pub struct XqWriter {
    tensors: BTreeMap<String, Entry>,
    file: std::io::BufWriter<std::fs::File>,
    pos: usize,
    error: Option<std::io::Error>,
}

impl XqWriter {
    pub fn create(path: &Path) -> anyhow::Result<Self> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        file.write_all(&[0u8; DATA_START])?; // prefix patched in `finish`
        Ok(Self {
            tensors: BTreeMap::new(),
            file,
            pos: DATA_START,
            error: None,
        })
    }

    fn push(&mut self, bytes: &[u8]) -> (usize, usize) {
        let off = self.pos.next_multiple_of(ALIGN);
        let pad = [0u8; ALIGN];
        let res = self
            .file
            .write_all(&pad[..off - self.pos])
            .and_then(|_| self.file.write_all(bytes));
        if let Err(e) = res {
            self.error.get_or_insert(e);
        }
        self.pos = off + bytes.len();
        (off, bytes.len())
    }

    fn put_f32(&mut self, name: &str, shape: Vec<usize>, v: &[f32]) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let data = self.push(&bytes);
        self.tensors
            .insert(name.to_string(), Entry::F32 { shape, data });
    }
}

impl TensorSink for XqWriter {
    fn put_vec(&mut self, name: &str, v: &[f32]) {
        self.put_f32(name, vec![v.len()], v);
    }

    fn put_linear(&mut self, name: &str, l: &Linear) {
        match l {
            Linear::Q1(w) => {
                let bits = self.push(w.bits().as_slice());
                let scales = self.push(w.scales().as_slice());
                self.tensors.insert(
                    name.to_string(),
                    Entry::Q1 {
                        rows: w.rows(),
                        cols: w.cols(),
                        block: w.block(),
                        bits,
                        scales,
                    },
                );
            }
            Linear::F32 { rows, cols, w } => self.put_f32(name, vec![*rows, *cols], w),
        }
    }
}

impl XqWriter {
    /// Write the header and the file prefix. `config` is the model config
    /// serialized as JSON.
    pub fn finish(
        mut self,
        arch: &str,
        config: serde_json::Value,
        assets: &ModelAssets,
    ) -> anyhow::Result<()> {
        if let Some(e) = self.error.take() {
            return Err(e.into());
        }
        let header = Header {
            arch: arch.to_string(),
            config,
            tensors: std::mem::take(&mut self.tensors),
            tokenizer: Some(
                assets
                    .tokenizer
                    .to_string(false)
                    .map_err(|e| anyhow::anyhow!("serializing tokenizer: {e}"))?,
            ),
            chat_template: assets.chat_template.clone(),
        };
        let json = serde_json::to_vec(&header)?;
        let header_off = self.pos;
        self.file.write_all(&json)?;
        let mut f = self.file.into_inner().map_err(|e| e.into_error())?;
        use std::io::{Seek, SeekFrom};
        f.seek(SeekFrom::Start(0))?;
        let mut prefix = Vec::with_capacity(24);
        prefix.extend_from_slice(MAGIC);
        prefix.extend_from_slice(&VERSION.to_le_bytes());
        prefix.extend_from_slice(&(header_off as u64).to_le_bytes());
        prefix.extend_from_slice(&(json.len() as u64).to_le_bytes());
        f.write_all(&prefix)?;
        f.sync_all()?;
        Ok(())
    }
}

/// A mapped `.xq1` file; hands out zero-copy tensors.
pub struct XqReader {
    map: Arc<Mmap>,
    header: Header,
}

impl XqReader {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only and mapped privately; as for
        // any mmap, modifying it underneath a running process is not
        // supported (the same contract as GgufFile).
        let map = unsafe { Mmap::map(&file)? };
        anyhow::ensure!(
            map.len() >= DATA_START && &map[..4] == MAGIC,
            "not an .xq1 file"
        );
        let version = u32::from_le_bytes(map[4..8].try_into()?);
        anyhow::ensure!(version == VERSION, "unsupported .xq1 version {version}");
        let hoff = u64::from_le_bytes(map[8..16].try_into()?) as usize;
        let hlen = u64::from_le_bytes(map[16..24].try_into()?) as usize;
        let hend = hoff
            .checked_add(hlen)
            .filter(|&e| e <= map.len() && hoff >= DATA_START)
            .ok_or_else(|| anyhow::anyhow!(".xq1 header out of bounds (truncated file?)"))?;
        let header: Header = serde_json::from_slice(&map[hoff..hend])?;
        Ok(Self {
            map: Arc::new(map),
            header,
        })
    }

    pub fn arch(&self) -> &str {
        &self.header.arch
    }

    pub fn config<T: for<'de> Deserialize<'de>>(&self) -> anyhow::Result<T> {
        Ok(serde_json::from_value(self.header.config.clone())?)
    }

    fn entry(&self, name: &str) -> anyhow::Result<&Entry> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!(".xq1: missing tensor {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }

    /// Number of stored tensors.
    pub fn tensor_count(&self) -> usize {
        self.header.tensors.len()
    }

    /// Total number of weights over all stored tensors (fused projections
    /// count once per original weight; a tied LM head is not stored).
    pub fn param_count(&self) -> u64 {
        self.header
            .tensors
            .values()
            .map(|e| match e {
                Entry::Q1 { rows, cols, .. } => (*rows * *cols) as u64,
                Entry::F32 { shape, .. } => shape.iter().product::<usize>() as u64,
            })
            .sum()
    }

    /// Q1 block size (elements per f16 scale) of the 1-bit tensors, if any.
    pub fn q1_block(&self) -> Option<usize> {
        self.header.tensors.values().find_map(|e| match e {
            Entry::Q1 { block, .. } => Some(*block),
            _ => None,
        })
    }

    /// The embedded HF Jinja chat template, if any.
    pub fn chat_template(&self) -> Option<&str> {
        self.header.chat_template.as_deref()
    }

    /// Whether a tokenizer is embedded.
    pub fn has_tokenizer(&self) -> bool {
        self.header.tokenizer.is_some()
    }

    fn buf(&self, (off, len): (usize, usize)) -> anyhow::Result<ByteBuf> {
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = self.map.clone();
        ByteBuf::from_shared(owner, off, len)
    }

    fn f32s(&self, r: (usize, usize)) -> anyhow::Result<Vec<f32>> {
        let b = self.buf(r)?;
        anyhow::ensure!(r.1.is_multiple_of(4), ".xq1: f32 tensor length");
        Ok(b.as_slice()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn vec(&self, name: &str, len: usize) -> anyhow::Result<Vec<f32>> {
        match self.entry(name)? {
            Entry::F32 { shape, data } if shape.iter().product::<usize>() == len => {
                self.f32s(*data)
            }
            _ => anyhow::bail!(".xq1: {name} is not an f32 vector of {len}"),
        }
    }

    pub fn linear(&self, name: &str) -> anyhow::Result<Linear> {
        match self.entry(name)?.clone() {
            Entry::Q1 {
                rows,
                cols,
                block,
                bits,
                scales,
            } => Ok(Linear::Q1(Q1Weight::new(
                rows,
                cols,
                block,
                self.buf(bits)?,
                self.buf(scales)?,
            )?)),
            Entry::F32 { shape, data } => {
                anyhow::ensure!(shape.len() == 2, ".xq1: {name} is not 2-D");
                let w = self.f32s(data)?;
                anyhow::ensure!(w.len() == shape[0] * shape[1], ".xq1: {name} size");
                Ok(Linear::F32 {
                    rows: shape[0],
                    cols: shape[1],
                    w: Arc::new(w),
                })
            }
        }
    }

    pub fn embedding(&self, name: &str) -> anyhow::Result<Embedding> {
        Ok(Embedding(self.linear(name)?))
    }

    pub fn assets(&self) -> anyhow::Result<ModelAssets> {
        let tok = self
            .header
            .tokenizer
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!(".xq1 has no tokenizer"))?;
        Ok(ModelAssets {
            tokenizer: tokenizers::Tokenizer::from_bytes(tok.as_bytes())
                .map_err(|e| anyhow::anyhow!(".xq1 tokenizer: {e}"))?,
            chat_template: self.header.chat_template.clone(),
        })
    }
}

/// Whether `path` starts with the `.xq1` magic.
pub fn is_xq1(path: &Path) -> anyhow::Result<bool> {
    use std::io::Read;
    let mut m = [0u8; 4];
    let mut f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
    Ok(f.read(&mut m)? == 4 && &m == MAGIC)
}

/// Load the model in an `.xq1` file.
pub fn load(path: &Path) -> anyhow::Result<Box<dyn CausalLM>> {
    let r = XqReader::open(path)?;
    load_model(&r)
}

fn load_model(r: &XqReader) -> anyhow::Result<Box<dyn CausalLM>> {
    match r.arch() {
        "qwen3" => Ok(Box::new(qwen3::Qwen3Model::read_xq1(r)?)),
        "qwen35" => Ok(Box::new(qwen35::Qwen35Model::read_xq1(r)?)),
        other => anyhow::bail!(".xq1: unsupported arch {other}"),
    }
}

/// Load the model plus its embedded tokenizer and chat template.
pub fn load_with_assets(path: &Path) -> anyhow::Result<(Box<dyn CausalLM>, ModelAssets)> {
    let r = XqReader::open(path)?;
    Ok((load_model(&r)?, r.assets()?))
}

/// Convert a Bonsai GGUF to `.xq1`, then reload the result and check that
/// every tensor is bit-identical to the GGUF-loaded model. Returns the
/// architecture name.
pub fn convert_gguf(
    gguf: &crate::loader::gguf::GgufFile,
    out: &Path,
    assets: &ModelAssets,
) -> anyhow::Result<String> {
    let arch = gguf.architecture()?;
    let mut w = XqWriter::create(out)?;
    let mut want = TensorDigest::default();
    let config = match arch.as_str() {
        "qwen3" => {
            let m = qwen3::Qwen3Model::from_gguf(gguf)?;
            m.write_xq1(&mut w);
            m.write_xq1(&mut want);
            serde_json::to_value(&m.cfg)?
        }
        "qwen35" => {
            let m = qwen35::Qwen35Model::load_gguf(gguf)?;
            m.write_xq1(&mut w);
            m.write_xq1(&mut want);
            serde_json::to_value(&m.cfg)?
        }
        other => anyhow::bail!("unsupported GGUF architecture {other:?}"),
    };
    w.finish(&arch, config, assets)?;

    let r = XqReader::open(out)?;
    let mut got = TensorDigest::default();
    match arch.as_str() {
        "qwen3" => qwen3::Qwen3Model::read_xq1(&r)?.write_xq1(&mut got),
        _ => qwen35::Qwen35Model::read_xq1(&r)?.write_xq1(&mut got),
    }
    anyhow::ensure!(
        got == want,
        "verification failed: {} reloads with different tensors",
        out.display()
    );
    Ok(arch)
}
