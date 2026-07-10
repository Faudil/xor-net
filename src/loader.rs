use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use safetensors::Dtype;
use memmap2::Mmap;
use crate::tensor::FastTensor;

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
        _ => anyhow::bail!("Unsupported safetensors dtype: {:?}", dtype),
    }
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
        let data = convert_to_f32_vec(bytes, info.dtype)?;
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
}
