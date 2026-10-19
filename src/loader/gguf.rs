use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_MAGIC_SWAPPED: u32 = 0x47475546;
const DEFAULT_ALIGNMENT: u64 = 32;
const GGUF_MAX_STRING_LEN: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;

// Upstream Q1_0 (128-block, renamed from Q1_0_g128)
pub const PRISM_Q1_0_UPSTREAM: u32 = 41;
// PrismML fork Q1_0 (32-block)
pub const PRISM_Q1_0: u32 = 42;
// PrismML fork Q1_0_G128 (128-block)
pub const PRISM_Q1_0_G128: u32 = 43;

// ---------------------------------------------------------------------------
// GGUF value types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Value::U8(v) => Some(*v as u32),
            Value::U16(v) => Some(*v as u32),
            Value::U32(v) => Some(*v),
            Value::I8(v) if *v >= 0 => Some(*v as u32),
            Value::I16(v) if *v >= 0 => Some(*v as u32),
            Value::I32(v) if *v >= 0 => Some(*v as u32),
            Value::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            Value::I64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn to_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            Value::Array(arr) => Some(arr.len()),
            _ => None,
        }
    }

    /// Extract string array elements (e.g. tokenizer.ggml.tokens / merges).
    pub fn to_string_vec(&self) -> Option<Vec<&str>> {
        match self {
            Value::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        Value::String(s) => out.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF parsing helpers
// ---------------------------------------------------------------------------

trait ReadLE: Read {
    fn read_u8_le(&mut self) -> anyhow::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16_le(&mut self) -> anyhow::Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16_le(&mut self) -> anyhow::Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32_le(&mut self) -> anyhow::Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32_le(&mut self) -> anyhow::Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64_le(&mut self) -> anyhow::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64_le(&mut self) -> anyhow::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32_le(&mut self) -> anyhow::Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64_le(&mut self) -> anyhow::Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> anyhow::Result<bool> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0] != 0)
    }

    fn read_string_len(&mut self, len: usize) -> anyhow::Result<String> {
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }
}

impl<R: Read + ?Sized> ReadLE for R {}

fn read_gguf_string<R: Read>(reader: &mut R, version: VersionedMagic) -> anyhow::Result<String> {
    let len = match version {
        VersionedMagic::V1 => reader.read_u32_le()? as u64,
        VersionedMagic::V2 | VersionedMagic::V3 => reader.read_u64_le()?,
    };
    if len > GGUF_MAX_STRING_LEN {
        anyhow::bail!("GGUF string too long: {len}");
    }
    reader.read_string_len(len as usize)
}

fn read_value<R: Read>(reader: &mut R, version: VersionedMagic) -> anyhow::Result<Value> {
    let type_id = reader.read_u32_le()?;
    read_value_body(reader, type_id, version)
}

fn read_value_body<R: Read>(reader: &mut R, type_id: u32, version: VersionedMagic) -> anyhow::Result<Value> {
    match type_id {
        0 => Ok(Value::U8(reader.read_u8_le()?)),
        1 => Ok(Value::I8(reader.read_u8_le()? as i8)),
        2 => Ok(Value::U16(reader.read_u16_le()?)),
        3 => Ok(Value::I16(reader.read_i16_le()?)),
        4 => Ok(Value::U32(reader.read_u32_le()?)),
        5 => Ok(Value::I32(reader.read_i32_le()?)),
        6 => Ok(Value::F32(reader.read_f32_le()?)),
        7 => Ok(Value::Bool(reader.read_bool()?)),
        8 => {
            let s = read_gguf_string(reader, version)?;
            Ok(Value::String(s))
        }
        9 => {
            let elem_type = reader.read_u32_le()?;
            let len = match version {
                VersionedMagic::V1 => reader.read_u32_le()? as u64,
                VersionedMagic::V2 | VersionedMagic::V3 => reader.read_u64_le()?,
            };
            if len > GGUF_MAX_ARRAY_ELEMENTS {
                anyhow::bail!("GGUF array too long: {len}");
            }
            let mut arr = Vec::with_capacity(len as usize);
            for _ in 0..len {
                arr.push(read_value_body(reader, elem_type, version)?);
            }
            Ok(Value::Array(arr))
        }
        10 => Ok(Value::U64(reader.read_u64_le()?)),
        11 => Ok(Value::I64(reader.read_i64_le()?)),
        12 => Ok(Value::F64(reader.read_f64_le()?)),
        _ => anyhow::bail!("unknown GGUF value type: {type_id}"),
    }
}

// ---------------------------------------------------------------------------
// Versioned magic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VersionedMagic {
    V1,
    V2,
    V3,
}

impl VersionedMagic {
    fn read<R: Read>(reader: &mut R) -> anyhow::Result<Self> {
        let magic = reader.read_u32_le()?;
        if magic != GGUF_MAGIC && magic != GGUF_MAGIC_SWAPPED {
            anyhow::bail!("not a GGUF file (magic: 0x{magic:08x})");
        }
        let version = reader.read_u32_le()?;
        match version {
            1 => Ok(VersionedMagic::V1),
            2 => Ok(VersionedMagic::V2),
            3 => Ok(VersionedMagic::V3),
            _ => anyhow::bail!("unsupported GGUF version: {version}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor info
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype_u32: u32,
    pub offset: u64,
    pub elem_count: usize,
    pub byte_size: u64,
}

impl GgufTensorInfo {
    fn block_size(dtype: u32) -> Option<usize> {
        match dtype {
            0 | 1 | 30 => Some(1),
            2 | 3 | 6 | 7 | 8 | 9 => Some(32),
            PRISM_Q1_0_UPSTREAM => Some(Q1_0_G128_BLOCK_ELEMS),
            PRISM_Q1_0 => Some(Q1_0_BLOCK_ELEMS),
            PRISM_Q1_0_G128 => Some(Q1_0_G128_BLOCK_ELEMS),
            _ => None,
        }
    }

    fn type_size(dtype: u32) -> Option<usize> {
        match dtype {
            0 => Some(4),
            1 => Some(2),
            30 => Some(2),
            2 => Some(20),
            3 => Some(24),
            6 => Some(20),
            7 => Some(24),
            8 => Some(34),
            9 => Some(36),
            PRISM_Q1_0_UPSTREAM => Some(Q1_0_G128_BLOCK_BYTES),
            PRISM_Q1_0 => Some(6),
            PRISM_Q1_0_G128 => Some(18),
            _ => None,
        }
    }

    pub fn compute_byte_size(shape: &[usize], dtype: u32) -> anyhow::Result<u64> {
        let elem_count: usize = shape.iter().product();
        let bs = Self::block_size(dtype)
            .ok_or_else(|| anyhow::anyhow!("unknown GGUF dtype {dtype}"))?;
        let ts = Self::type_size(dtype)
            .ok_or_else(|| anyhow::anyhow!("unknown GGUF dtype {dtype}"))?;
        if elem_count % bs != 0 {
            anyhow::bail!("dtype {dtype} block size {bs} does not divide elem_count {elem_count}");
        }
        Ok((elem_count / bs) as u64 * ts as u64)
    }
}

// ---------------------------------------------------------------------------
// PRISM 1-bit block types
// ---------------------------------------------------------------------------

const Q1_0_BLOCK_ELEMS: usize = 32;
const Q1_0_BLOCK_BYTES: usize = 6;
const Q1_0_G128_BLOCK_ELEMS: usize = 128;
const Q1_0_G128_BLOCK_BYTES: usize = 18;

pub fn fp16_to_f32(half: u16) -> f32 {
    let sign = (half & 0x8000) as u32;
    let exp = (half & 0x7c00) >> 10;
    let mant = (half & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 16)
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let new_exp = (127 - 15 + e + 1) as u32;
            f32::from_bits((sign << 16) | (new_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits((sign << 16) | 0x7f800000)
        } else {
            f32::from_bits((sign << 16) | 0x7fc00000 | (mant << 13))
        }
    } else {
        let new_exp = ((exp as i32) - 15 + 127) as u32;
        f32::from_bits((sign << 16) | (new_exp << 23) | (mant << 13))
    }
}

/// Dequantize PRISM Q1_0 (type 42): 32 elements, 6 bytes per block.
/// Layout: [fp16_scale: 2B LE][signs: 4B (32 bits)]
/// Each bit: 0 → −scale, 1 → +scale
pub fn dequantize_q1_0(data: &[u8], elem_count: usize) -> Vec<f32> {
    let num_blocks = elem_count / Q1_0_BLOCK_ELEMS;
    let mut out = vec![0.0f32; elem_count];
    for b in 0..num_blocks {
        let bo = b * Q1_0_BLOCK_BYTES;
        let scale_half = u16::from_le_bytes([data[bo], data[bo + 1]]);
        let scale = fp16_to_f32(scale_half);
        let signs = &data[bo + 2..bo + 6];
        let base_out = b * Q1_0_BLOCK_ELEMS;
        for i in 0..Q1_0_BLOCK_ELEMS {
            let byte_idx = i >> 3;
            let bit_idx = i & 7;
            let bit = (signs[byte_idx] >> bit_idx) & 1;
            out[base_out + i] = if bit == 0 { -scale } else { scale };
        }
    }
    out
}

/// Dequantize upstream Q1_0 (type 41): 128 elements, 18 bytes per block.
/// Layout: [fp16_scale: 2B LE][signs: 16B (128 bits)]
/// Note: scale comes FIRST, then signs.
/// Encoding: bit=0 → +scale, bit=1 → −scale
pub fn dequantize_q1_0_upstream(data: &[u8], elem_count: usize) -> Vec<f32> {
    const BLOCK_ELEMS: usize = 128;
    const BLOCK_BYTES: usize = 18;
    let num_blocks = elem_count / BLOCK_ELEMS;
    let mut out = vec![0.0f32; elem_count];
    for b in 0..num_blocks {
        let bo = b * BLOCK_BYTES;
        let scale_half = u16::from_le_bytes([data[bo], data[bo + 1]]);
        let scale = fp16_to_f32(scale_half);
        let signs = &data[bo + 2..bo + 18];
        let base_out = b * BLOCK_ELEMS;
        for i in 0..BLOCK_ELEMS {
            let byte_idx = i >> 3;
            let bit_idx = i & 7;
            let bit = (signs[byte_idx] >> bit_idx) & 1;
            out[base_out + i] = if bit == 0 { scale } else { -scale };
        }
    }
    out
}

/// Dequantize PRISM_Q1_0_G128 (type 43): 128 elements, 18 bytes per block.
/// Layout: [signs: 16B (128 bits)][fp16_scale: 2B LE]
/// Note: signs come FIRST, then scale.
/// Each bit: 0 → −scale, 1 → +scale
pub fn dequantize_q1_0_g128(data: &[u8], elem_count: usize) -> Vec<f32> {
    let num_blocks = elem_count / Q1_0_G128_BLOCK_ELEMS;
    let mut out = vec![0.0f32; elem_count];
    for b in 0..num_blocks {
        let bo = b * Q1_0_G128_BLOCK_BYTES;
        let signs = &data[bo..bo + 16];
        let scale_half = u16::from_le_bytes([data[bo + 16], data[bo + 17]]);
        let scale = fp16_to_f32(scale_half);
        let base_out = b * Q1_0_G128_BLOCK_ELEMS;
        for i in 0..Q1_0_G128_BLOCK_ELEMS {
            let byte_idx = i >> 3;
            let bit_idx = i & 7;
            let bit = (signs[byte_idx] >> bit_idx) & 1;
            out[base_out + i] = if bit == 0 { -scale } else { scale };
        }
    }
    out
}

/// Dequantize a PRISM Q1_0 / Q1_0_G128 block back to the raw signs + scale
/// represented as f32 values. Returns (sign_bits, scale_f32) per block.
/// Used when we want to feed the signs directly into xor-net's Bit1 engine
/// (Phase 2) while keeping the scale per-group.
pub fn dequantize_q1_0_g128_blocks(data: &[u8], elem_count: usize) -> Vec<Vec<u8>> {
    let num_blocks = elem_count / Q1_0_G128_BLOCK_ELEMS;
    let mut blocks = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let bo = b * Q1_0_G128_BLOCK_BYTES;
        let signs = data[bo..bo + 16].to_vec();
        blocks.push(signs);
    }
    blocks
}

pub fn dequantize_q1_0_g128_scales(data: &[u8], elem_count: usize) -> Vec<f32> {
    let num_blocks = elem_count / Q1_0_G128_BLOCK_ELEMS;
    let mut scales = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let bo = b * Q1_0_G128_BLOCK_BYTES;
        let half = u16::from_le_bytes([data[bo + 16], data[bo + 17]]);
        scales.push(fp16_to_f32(half));
    }
    scales
}

// ---------------------------------------------------------------------------
// GGUF content
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, GgufTensorInfo>,
    pub alignment: u64,
    pub tensor_data_offset: u64,
}

pub struct GgufFile {
    pub _file: File,
    pub mmap: Mmap,
    pub metadata: GgufMetadata,
}

impl GgufFile {
    /// Open and parse a GGUF file. Reads the header + metadata + tensor infos,
    /// then mmaps the entire file for random-access tensor reads.
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut file = File::open(path.as_ref())?;
        let _file_size = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;

        let version = VersionedMagic::read(&mut file)?;

        let tensor_count = match version {
            VersionedMagic::V1 => file.read_u32_le()? as u64,
            VersionedMagic::V2 | VersionedMagic::V3 => file.read_u64_le()?,
        };
        let metadata_kv_count = match version {
            VersionedMagic::V1 => file.read_u32_le()? as u64,
            VersionedMagic::V2 | VersionedMagic::V3 => file.read_u64_le()?,
        };

        if tensor_count > GGUF_MAX_ARRAY_ELEMENTS {
            anyhow::bail!("tensor count {tensor_count} exceeds maximum");
        }
        if metadata_kv_count > GGUF_MAX_ARRAY_ELEMENTS {
            anyhow::bail!("metadata KV count {metadata_kv_count} exceeds maximum");
        }

        // Read metadata KV pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut file, version)?;
            let value = read_value(&mut file, version)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file, version)?;
            let n_dim = file.read_u32_le()?;
            if n_dim > GGUF_MAX_TENSOR_DIMS {
                anyhow::bail!("tensor '{name}' has {n_dim} dimensions, max is {GGUF_MAX_TENSOR_DIMS}");
            }

            let mut shape = match version {
                VersionedMagic::V1 => {
                    let mut dims = Vec::with_capacity(n_dim as usize);
                    for _ in 0..n_dim {
                        dims.push(file.read_u32_le()? as usize);
                    }
                    dims
                }
                VersionedMagic::V2 | VersionedMagic::V3 => {
                    let mut dims = Vec::with_capacity(n_dim as usize);
                    for _ in 0..n_dim {
                        dims.push(file.read_u64_le()? as usize);
                    }
                    dims
                }
            };
            // GGUF stores dims in reverse order (last dim first)
            shape.reverse();

            let dtype_u32 = file.read_u32_le()?;
            let offset = match version {
                VersionedMagic::V1 => file.read_u32_le()? as u64,
                VersionedMagic::V2 | VersionedMagic::V3 => file.read_u64_le()?,
            };

            let elem_count: usize = shape.iter().product();
            let byte_size = GgufTensorInfo::compute_byte_size(&shape, dtype_u32)?;

            tensor_infos.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    dtype_u32,
                    offset,
                    elem_count,
                    byte_size,
                },
            );
        }

        // Compute alignment and tensor_data_offset
        let position = file.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.to_u32())
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let tensor_data_offset = position.div_ceil(alignment) * alignment;

        let mmap = unsafe { Mmap::map(&file)? };
        let _ = file.seek(SeekFrom::Start(0)); // reset but not needed after mmap

        Ok(GgufFile {
            _file: file,
            mmap,
            metadata: GgufMetadata {
                metadata,
                tensor_infos,
                alignment,
                tensor_data_offset,
            },
        })
    }

    /// Get raw bytes of a tensor (the packed quantized data).
    pub fn tensor_bytes(&self, name: &str) -> anyhow::Result<&[u8]> {
        let info = self.metadata.tensor_infos.get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' not found"))?;
        let start = self.metadata.tensor_data_offset + info.offset;
        let end = start + info.byte_size;
        Ok(&self.mmap[start as usize..end as usize])
    }

    /// Dequantize a tensor to f32. Handles standard dtypes (F32, F16, BF16)
    /// and PRISM 1-bit types (42, 43).
    pub fn dequantize_tensor(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        let info = self.metadata.tensor_infos.get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' not found"))?;
        let bytes = self.tensor_bytes(name)?;
        dequantize_data(bytes, info.dtype_u32, info.elem_count)
    }

    /// Get tensor metadata (shape + dtype)
    pub fn tensor_info(&self, name: &str) -> anyhow::Result<&GgufTensorInfo> {
        self.metadata.tensor_infos.get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' not found"))
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.metadata.tensor_infos.contains_key(name)
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.metadata.get(key)
    }

    /// Extract the general.architecture name from metadata
    pub fn architecture(&self) -> anyhow::Result<String> {
        self.metadata.metadata
            .get("general.architecture")
            .and_then(|v| v.to_string())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))
    }
}

// ---------------------------------------------------------------------------
// Top-level dequantization dispatcher
// ---------------------------------------------------------------------------

pub fn is_prism_dtype(dtype: u32) -> bool {
    dtype == PRISM_Q1_0_UPSTREAM || dtype == PRISM_Q1_0 || dtype == PRISM_Q1_0_G128
}

pub fn dequantize_data(data: &[u8], dtype: u32, elem_count: usize) -> anyhow::Result<Vec<f32>> {
    match dtype {
        0 => {
            if data.len() % 4 != 0 {
                anyhow::bail!("F32 data length {} not multiple of 4", data.len());
            }
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let offset = i * 4;
                out.push(f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]));
            }
            Ok(out)
        }
        1 => {
            if data.len() % 2 != 0 {
                anyhow::bail!("F16 data length {} not multiple of 2", data.len());
            }
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let offset = i * 2;
                let half = u16::from_le_bytes([data[offset], data[offset + 1]]);
                out.push(fp16_to_f32(half));
            }
            Ok(out)
        }
        30 => {
            if data.len() % 2 != 0 {
                anyhow::bail!("BF16 data length {} not multiple of 2", data.len());
            }
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let offset = i * 2;
                let bf16 = u16::from_le_bytes([data[offset], data[offset + 1]]);
                out.push(f32::from_bits((bf16 as u32) << 16));
            }
            Ok(out)
        }
        PRISM_Q1_0_UPSTREAM => Ok(dequantize_q1_0_upstream(data, elem_count)),
        PRISM_Q1_0 => Ok(dequantize_q1_0(data, elem_count)),
        PRISM_Q1_0_G128 => Ok(dequantize_q1_0_g128(data, elem_count)),
        _ => anyhow::bail!("unsupported GGUF dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer extraction from GGUF metadata
// ---------------------------------------------------------------------------

/// Build a BPE tokenizer from the tokenizer data embedded in the GGUF
/// (tokenizer.ggml.tokens / merges / special token ids).
pub fn build_tokenizer(gguf: &GgufFile) -> anyhow::Result<tokenizers::Tokenizer> {
    use tokenizers::models::bpe::BpeBuilder;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    let meta = &gguf.metadata.metadata;

    let tokens = meta.get("tokenizer.ggml.tokens")
        .and_then(|v| v.to_string_vec())
        .ok_or_else(|| anyhow::anyhow!("missing tokenizer.ggml.tokens"))?;

    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::default();
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.to_string(), i as u32);
    }

    let mut merges: Vec<(String, String)> = Vec::new();
    if let Some(merge_strs) = meta.get("tokenizer.ggml.merges").and_then(|v| v.to_string_vec()) {
        for m in merge_strs {
            // merges are stored as "a b"
            if let Some((a, b)) = m.split_once(' ') {
                merges.push((a.to_string(), b.to_string()));
            }
        }
    }

    let bpe = BpeBuilder::default()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("BPE build: {e}"))?;

    // Qwen-style: byte-level pretokenizer with prefix space
    let byte_level = ByteLevel::new(true, true, true);
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(byte_level));

    // Add special tokens (for decode round-trips)
    let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
    let added: Vec<tokenizers::AddedToken> = specials
        .iter()
        .map(|s| tokenizers::AddedToken::from(s.to_string(), true))
        .collect();
    let _ = tokenizer.add_special_tokens(added);

    Ok(tokenizer)
}

// ---------------------------------------------------------------------------
// Configuration extraction from GGUF metadata (for Qwen3 / Bonsai models)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BonsaiConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub architecture: String,
}

impl BonsaiConfig {
    /// Extract config from GGUF metadata.
    /// The arch prefix is read from `general.architecture`.
    pub fn from_gguf_metadata(metadata: &HashMap<String, Value>) -> anyhow::Result<Self> {
        let arch = metadata
            .get("general.architecture")
            .and_then(|v| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))?;

        let prefix = |key: &str| format!("{arch}.{key}");

        let get_u32 = |key: &str| -> anyhow::Result<usize> {
            metadata
                .get(&prefix(key))
                .and_then(|v| v.to_u32())
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("missing metadata {}{key}", prefix("")))
        };

        let get_f32 = |key: &str| -> anyhow::Result<f32> {
            metadata
                .get(&prefix(key))
                .and_then(|v| v.to_f32())
                .ok_or_else(|| anyhow::anyhow!("missing metadata {}{key}", prefix("")))
        };

        let hidden_size = get_u32("embedding_length")?;
        let num_attention_heads = get_u32("attention.head_count")?;
        let num_kv_heads = get_u32("attention.head_count_kv")?;
        let head_dim = match metadata.get(&prefix("attention.key_length")).and_then(|v| v.to_u32()) {
            Some(v) => v as usize,
            None => hidden_size / num_attention_heads,
        };
        let num_layers = get_u32("block_count")?;
        let context_length = get_u32("context_length")?;
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon")? as f64;
        let rope_theta = metadata
            .get(&prefix("rope.freq_base"))
            .and_then(|v| v.to_f32())
            .unwrap_or(10000.0);

        let vocab_size = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.array_len())
            .ok_or_else(|| anyhow::anyhow!("missing metadata tokenizer.ggml.tokens"))?;

        // BOS/EOS from tokenizer metadata
        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32());
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32());

        let intermediate_size = get_u32("feed_forward_length")?;

        // Tie word embeddings: if no "output.weight" tensor, assume tied
        // We'll determine this at load time, not from metadata.
        let tie_word_embeddings = false;

        Ok(BonsaiConfig {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers: num_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings: context_length,
            bos_token_id,
            eos_token_id,
            tie_word_embeddings,
            architecture: arch.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q1_0_upstream() {
        // One block: 128 elements, scale=2.0, all bits=0 → +scale
        let mut data = vec![0u8; 18];
        data[0] = 0x00; // fp16 scale = 2.0
        data[1] = 0x40;
        // bits all 0 → +scale
        let out = dequantize_q1_0_upstream(&data, 128);
        assert_eq!(out.len(), 128);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-6, "expected +2.0 got {val}");
        }

        // All bits=1 → -scale
        let mut data = vec![0xFFu8; 18];
        data[0] = 0x00; // fp16 scale = 2.0
        data[1] = 0x40;
        let out = dequantize_q1_0_upstream(&data, 128);
        for &val in &out {
            assert!((val - (-2.0)).abs() < 1e-6, "expected -2.0 got {val}");
        }
    }

    #[test]
    fn test_fp16_to_f32() {
        // 1.0 in fp16
        assert!((fp16_to_f32(0x3c00) - 1.0).abs() < 1e-6);
        // -1.0 in fp16
        assert!((fp16_to_f32(0xbc00) - (-1.0)).abs() < 1e-6);
        // 0.5 in fp16
        assert!((fp16_to_f32(0x3800) - 0.5).abs() < 1e-6);
        // 0.0
        assert!((fp16_to_f32(0x0000) - 0.0).abs() < 1e-6);
        // -0.0
        assert!((fp16_to_f32(0x8000) - (-0.0)).abs() < 1e-6);
        // 65504 (max fp16 normal)
        assert!((fp16_to_f32(0x7bff) - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_dequantize_q1_0() {
        // One block: 32 elements, scale=1.0 (0x3c00), signs alternating
        let mut data = vec![0u8; 6];
        data[0] = 0x00; // fp16 scale = 1.0
        data[1] = 0x3c;
        data[2] = 0x55; // 01010101
        data[3] = 0x55; // 01010101
        data[4] = 0x55; // 01010101
        data[5] = 0x55; // 01010101

        let out = dequantize_q1_0(&data, 32);
        assert_eq!(out.len(), 32);
        for (i, &val) in out.iter().enumerate() {
            let expected = if (i % 2) == 0 { 1.0 } else { -1.0 };
            assert!((val - expected).abs() < 1e-6, "mismatch at {i}: {val} vs {expected}");
        }
    }

    #[test]
    fn test_dequantize_q1_0_g128() {
        // One block: 128 elements, scale=2.0, all bits=1 (+scale)
        let mut data = vec![0xFFu8; 18];
        data[16] = 0x00; // fp16 scale = 2.0
        data[17] = 0x40;

        let out = dequantize_q1_0_g128(&data, 128);
        assert_eq!(out.len(), 128);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-6, "expected +2.0 got {val}");
        }

        // All bits=0 (-scale)
        let mut data = vec![0x00u8; 18];
        data[16] = 0x00;
        data[17] = 0x40;

        let out = dequantize_q1_0_g128(&data, 128);
        for &val in &out {
            assert!((val - (-2.0)).abs() < 1e-6, "expected -2.0 got {val}");
        }
    }

    #[test]
    fn test_dequantize_q1_0_g128_alternating() {
        let mut data = vec![0u8; 18];
        data[16] = 0x00; // fp16 scale = 1.0
        data[17] = 0x3c;
        data[0] = 0x55; // 01010101 -> + - + - + - + -
        data[2] = 0xAA; // 10101010 -> - + - + - + - +

        let out = dequantize_q1_0_g128(&data, 128);
        for i in 0..128 {
            let byte_idx = i >> 3;
            let bit_idx = i & 7;
            let bit = (data[byte_idx] >> bit_idx) & 1;
            let expected = if bit == 0 { -1.0 } else { 1.0 };
            assert!((out[i] - expected).abs() < 1e-6, "mismatch at {i}: {} vs {}", out[i], expected);
        }
    }

    #[test]
    fn test_dequantize_data_f32() {
        let f32_data: [f32; 4] = [1.5, -2.5, 3.0, -4.0];
        let mut bytes = Vec::with_capacity(16);
        for &v in &f32_data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequantize_data(&bytes, 0, 4).unwrap();
        assert_eq!(out.len(), 4);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - f32_data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequantize_data_prism() {
        let mut bytes = vec![0u8; Q1_0_G128_BLOCK_BYTES];
        bytes[16] = 0x00;
        bytes[17] = 0x3c; // scale = 1.0
        bytes[0] = 0xFF; // first 8 bits all +1
        let out = dequantize_data(&bytes, PRISM_Q1_0_G128, 128).unwrap();
        assert_eq!(out.len(), 128);
        for i in 0..8 {
            assert!((out[i] - 1.0).abs() < 1e-6);
        }
    }
}
