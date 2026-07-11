use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use safetensors::Dtype;
use memmap2::Mmap;
use crate::tensor::FastTensor;
use crate::bit1_58::quantization::{pack_1_58bit_4pack, pack_1_58bit_5pack, unpack_1_58bit_4pack, unpack_1_58bit_5pack, TernaryPackType};

fn convert_to_f32_vec(bytes: &[u8], dtype: Dtype) -> anyhow::Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                anyhow::bail!("F32 bytes length not multiple of 4: {}", bytes.len());
            }
            let count = bytes.len() / 4;
            let mut out = vec![0.0f32; count];
            for i in 0..count {
                let offset = i * 4;
                out[i] = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                anyhow::bail!("BF16 bytes length not multiple of 2: {}", bytes.len());
            }
            let count = bytes.len() / 2;
            let mut out = vec![0.0f32; count];
            for i in 0..count {
                let offset = i * 2;
                let bf16_val = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                out[i] = f32::from_bits((bf16_val as u32) << 16);
            }
            Ok(out)
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                anyhow::bail!("F16 bytes length not multiple of 2: {}", bytes.len());
            }
            let count = bytes.len() / 2;
            let mut out = vec![0.0f32; count];
            for i in 0..count {
                let offset = i * 2;
                let f16_val = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                let sign = (f16_val & 0x8000) as u32;
                let exp = (f16_val & 0x7c00) >> 10;
                let mant = f16_val & 0x03ff;
                
                let f32_bits = if exp == 0 {
                    if mant == 0 {
                        sign << 16
                    } else {
                        let mut m = mant as u32;
                        let mut e = 0i32;
                        while (m & 0x0400) == 0 {
                            m <<= 1;
                            e -= 1;
                        }
                        m &= 0x03ff;
                        let new_exp = (127 - 15 + e + 1) as u32;
                        (sign << 16) | (new_exp << 23) | (m << 13)
                    }
                } else if exp == 31 {
                    if mant == 0 {
                        (sign << 16) | 0x7f800000
                    } else {
                        (sign << 16) | 0x7fc00000
                    }
                } else {
                    let new_exp = (exp as i32 - 15 + 127) as u32;
                    (sign << 16) | (new_exp << 23) | ((mant as u32) << 13)
                };
                out[i] = f32::from_bits(f32_bits);
            }
            Ok(out)
        }
        Dtype::U8 => {
            let mut out = vec![0.0f32; bytes.len()];
            for (i, &b) in bytes.iter().enumerate() {
                out[i] = b as f32;
            }
            Ok(out)
        }
        _ => anyhow::bail!("Unsupported safetensors dtype: {:?}", dtype),
    }
}

/// Load, unpack and transpose 1.58-bit ternary weights stored in U8 format.
/// The stored tensor has shape [packed_out_dim, in_dim] where packed_out_dim = out_dim / 4,
/// with each U8 byte packing 4 ternary values along the OUTPUT dimension
/// (4 consecutive output rows at the same input position).
///
/// Returns unpacked and transposed tensor with shape [out_dim, in_dim] in standard
/// row-major order, with weight_scale applied.
pub fn load_packed_ternary_weight(
    repo: &SafeTensorRepo,
    weight_name: &str,
    expected_shape: &[usize],
    pack_type: TernaryPackType,
) -> anyhow::Result<FastTensor> {
    let info = repo.tensors.get(weight_name)
        .ok_or_else(|| anyhow::anyhow!("Packed weight tensor not found: {}", weight_name))?;
    
    let scale_key = format!("{}.weight_scale", weight_name.strip_suffix(".weight").unwrap_or(weight_name));
    let sinfo = repo.tensors.get(&scale_key)
        .ok_or_else(|| anyhow::anyhow!("Packed weight '{}' has no companion weight_scale '{}'", weight_name, scale_key))?;
    
    let weight_bytes = &repo.buffers[info.file_idx][info.start..info.end];
    let scale_bytes = &repo.buffers[sinfo.file_idx][sinfo.start..sinfo.end];
    let scale_data = convert_to_f32_vec(scale_bytes, sinfo.dtype)?;
    let w_scale = scale_data[0];

    let packed_data: Vec<u8> = weight_bytes.iter().map(|&b| b).collect();
    let expected_out_dim = expected_shape[0];
    let expected_in_dim = expected_shape[1];
    let expected_len = expected_out_dim * expected_in_dim;

    // Unpack the ternary values (still in output-dim order)
    let ternary_values = match pack_type {
        TernaryPackType::Pack4 => unpack_1_58bit_4pack(&packed_data, expected_len),
        TernaryPackType::Pack5 => unpack_1_58bit_5pack(&packed_data, expected_len),
    };

    // Transpose from output-dim packing order to standard row-major [out_dim, in_dim].
    // The model packs 4 output rows per byte, striped: byte[p][i] contains bits for
    // w[p][i], w[p + stored_rows][i], w[p + 2*stored_rows][i], w[p + 3*stored_rows][i]
    let stored_in_dim = expected_in_dim;
    let stored_rows = expected_out_dim / 4;
    let mut data = vec![0.0f32; expected_len];
    let inv_w_scale = 1.0 / w_scale;
    for k in 0..expected_len {
        let byte_idx = k / 4;
        let j = k % 4;
        let row_p = byte_idx / stored_in_dim;
        let col = byte_idx % stored_in_dim;
        let o = j * stored_rows + row_p;
        let i = col;
        data[o * expected_in_dim + i] = ternary_values[k] * inv_w_scale;
    }

    Ok(FastTensor::new(data, expected_shape.to_vec()))
}

pub struct RawTensorInfo {
    pub file_idx: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct SafeTensorRepo {
    pub buffers: Vec<Mmap>,
    pub tensors: HashMap<String, RawTensorInfo>,
}

impl SafeTensorRepo {
    pub fn load<P: AsRef<Path>>(paths: &[P]) -> anyhow::Result<Self> {
        let mut buffers = Vec::new();
        let mut tensors = HashMap::new();
        for (file_idx, path) in paths.iter().enumerate() {
            let file = File::open(path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };
            let file_start_ptr = mmap.as_ptr() as usize;
            let safetensors = safetensors::SafeTensors::deserialize(&mmap)?;
            for (name, view) in safetensors.tensors() {
                let view_start_ptr = view.data().as_ptr() as usize;
                let start = view_start_ptr - file_start_ptr;
                let end = start + view.data().len();
                tensors.insert(name.clone(), RawTensorInfo {
                    file_idx,
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    start,
                    end,
                });
            }
            buffers.push(mmap);
        }
        Ok(Self { buffers, tensors })
    }
    
    pub fn get(&self, name: &str) -> anyhow::Result<FastTensor> {
        let info = self.tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;
        let bytes = &self.buffers[info.file_idx][info.start..info.end];
        let mut data = convert_to_f32_vec(bytes, info.dtype)?;

        // Auto-dequantize U8 weight tensors using companion weight_scale
        if info.dtype == Dtype::U8 {
            let scale_key = if let Some(stripped) = name.strip_suffix(".weight") {
                format!("{}.weight_scale", stripped)
            } else {
                format!("{}.weight_scale", name)
            };
            let sinfo = self.tensors.get(&scale_key)
                .ok_or_else(|| anyhow::anyhow!("U8 tensor '{}' has no companion weight_scale '{}'", name, scale_key))?;
            let sbytes = &self.buffers[sinfo.file_idx][sinfo.start..sinfo.end];
            let scale_data = convert_to_f32_vec(sbytes, sinfo.dtype)?;
            let inv_scale = 1.0 / scale_data[0];
            for v in data.iter_mut() {
                *v *= inv_scale;
            }
        }

        Ok(FastTensor::new(data, info.shape.clone()))
    }
}

#[derive(Clone)]
pub struct SafeTensorLoader<'a> {
    repo: &'a SafeTensorRepo,
    prefix: String,
}

impl<'a> SafeTensorLoader<'a> {
    pub fn new(repo: &'a SafeTensorRepo) -> Self {
        Self {
            repo,
            prefix: String::new(),
        }
    }

    pub fn pp(&self, namespace: impl AsRef<str>) -> Self {
        let ns = namespace.as_ref();
        let new_prefix = if self.prefix.is_empty() {
            ns.to_string()
        } else {
            format!("{}.{}", self.prefix, ns)
        };
        Self {
            repo: self.repo,
            prefix: new_prefix,
        }
    }

    pub fn get(&self, shape: &[usize], name: &str) -> anyhow::Result<FastTensor> {
        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        let t = self.repo.get(&key)?;
        if t.shape != shape {
            anyhow::bail!(
                "Tensor '{}' shape mismatch: expected {:?}, got {:?}",
                key,
                shape,
                t.shape
            );
        }
        Ok(t)
    }

    pub fn get_vector(&self, size: usize, name: &str) -> anyhow::Result<FastTensor> {
        self.get(&[size], name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.repo.tensors.contains_key(&key)
    }

    /// Load pre-packed 1.58-bit ternary weight data with its original weight_scale.
    /// Only succeeds when the stored tensor is U8 packed ternary (out_dim stored as out_dim/4).
    ///
    /// The stored format packs 4 consecutive OUTPUT rows per byte (output-dim packing).
    /// This function converts to input-dim packing (4 consecutive input columns per byte)
    /// which is what the dot product expects, and returns the converted packed bytes
    /// along with the original weight_scale.
    ///
    /// Returns (packed_weights_in_input_dim_order, w_scale, in_dim, out_dim).
    pub fn get_prepacked_ternary(
        &self,
        expected_shape: &[usize],
        name: &str,
        pack_type: TernaryPackType,
    ) -> anyhow::Result<(Vec<u8>, f32, usize, usize)> {
        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        let info = self.repo.tensors.get(&key)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found", key))?;
        let stored_shape = &info.shape;
        let expected_out = expected_shape[0];
        let expected_in = expected_shape[1];

        let pack_factor = match pack_type {
            TernaryPackType::Pack4 => 4usize,
            TernaryPackType::Pack5 => 5usize,
        };
        let is_packed = info.dtype == Dtype::U8
            && stored_shape.len() == 2
            && stored_shape[1] == expected_in
            && stored_shape[0] * pack_factor == expected_out;

        if !is_packed {
            anyhow::bail!("Tensor '{}' is not in pre-packed U8 ternary format", key);
        }

        let weight_bytes = &self.repo.buffers[info.file_idx][info.start..info.end];
        let packed_data: Vec<u8> = weight_bytes.to_vec();

        let scale_key = format!("{}.weight_scale", key.strip_suffix(".weight").unwrap_or(&key));
        let sinfo = self.repo.tensors.get(&scale_key)
            .ok_or_else(|| anyhow::anyhow!("Packed weight '{}' has no companion weight_scale '{}'", key, scale_key))?;
        let sbytes = &self.repo.buffers[sinfo.file_idx][sinfo.start..sinfo.end];
        let scale_data = convert_to_f32_vec(sbytes, sinfo.dtype)?;
        let w_scale = scale_data[0];
        let inv_w_scale = 1.0 / w_scale;

        let out_dim = expected_out;
        let in_dim = expected_in;
        let expected_len = out_dim * in_dim;

        // Unpack from output-dim order
        let ternary_values = match pack_type {
            TernaryPackType::Pack4 => unpack_1_58bit_4pack(&packed_data, expected_len),
            TernaryPackType::Pack5 => unpack_1_58bit_5pack(&packed_data, expected_len),
        };

        let stored_rows = out_dim / pack_factor;

        // Transpose to standard row-major [out_dim, in_dim]
        let mut row_major = vec![0.0f32; expected_len];
        for k in 0..expected_len {
            let byte_idx = k / pack_factor;
            let j = k % pack_factor;
            let row_p = byte_idx / in_dim;
            let col = byte_idx % in_dim;
            let o = j * stored_rows + row_p;
            let i = col;
            row_major[o * in_dim + i] = ternary_values[k];
        }

        // Repack in input-dim order (4 consecutive input values per byte).
        // Values are already ternary {-1, 0, 1}, so use scale=1.0 to preserve them.
        let repacked = match pack_type {
            TernaryPackType::Pack4 => {
                let mut out = Vec::with_capacity((expected_len + 3) / 4);
                for row in row_major.chunks(in_dim) {
                    out.extend_from_slice(&pack_1_58bit_4pack(row, 1.0));
                }
                out
            }
            TernaryPackType::Pack5 => {
                let mut out = Vec::with_capacity((expected_len + 4) / 5);
                for row in row_major.chunks(in_dim) {
                    out.extend_from_slice(&pack_1_58bit_5pack(row, 1.0));
                }
                out
            }
        };

        Ok((repacked, inv_w_scale, in_dim, out_dim))
    }

    /// Load a weight tensor that may be packed in 1.58-bit ternary format (Pack4).
    /// If the stored tensor has output dimension 4x smaller than expected and is U8, unpack it.
    pub fn get_packed_ternary(
        &self,
        expected_shape: &[usize],
        name: &str,
        pack_type: TernaryPackType,
    ) -> anyhow::Result<FastTensor> {
        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        
        // Try to get the tensor - it might be packed or full size
        if let Some(info) = self.repo.tensors.get(&key) {
            let stored_shape = &info.shape;
            let expected_out = expected_shape[0];
            let expected_in = expected_shape[1];
            
            // Check if it's packed ternary: U8 dtype, stored_out * 4 == expected_out, stored_in == expected_in
            let is_packed = info.dtype == Dtype::U8
                && stored_shape.len() == 2 
                && stored_shape[1] == expected_in 
                && stored_shape[0] * 4 == expected_out;
            
            if is_packed {
                // It's packed! Load and unpack
                return load_packed_ternary_weight(self.repo, &key, expected_shape, pack_type);
            }
            
            // Not packed, try normal load
            let t = self.repo.get(&key)?;
            if t.shape == expected_shape {
                return Ok(t);
            }
        }
        
        anyhow::bail!(
            "Tensor '{}' not found or shape mismatch: expected {:?}",
            key,
            expected_shape
        )
    }
}