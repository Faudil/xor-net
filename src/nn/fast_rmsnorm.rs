use crate::loader::SafeTensorLoader;
use crate::tensor::FastTensor;

#[derive(Debug, Clone)]
pub struct FastRmsNorm {
    pub eps: f32,
    pub weight: FastTensor,
}

impl FastRmsNorm {
    pub fn new(weight: FastTensor, eps: f32) -> Self {
        Self { eps, weight }
    }

    pub fn forward(&self, t: &FastTensor) -> anyhow::Result<FastTensor> {
        t.rmsnorm(&self.weight, self.eps)
    }

    pub fn load(size: usize, eps: f32, loader: &SafeTensorLoader) -> anyhow::Result<Self> {
        let weight = loader.get_vector(size, "weight")?;
        Ok(Self::new(weight, eps))
    }
}
