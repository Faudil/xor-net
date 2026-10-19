//! High-level image classification API.

use super::graph::{Network, OpSpec, TensorSource};
use super::kernels::binary::BinaryKernel;
use super::preprocess::Preprocess;
use super::tensor::FeatureMap;
use anyhow::{Context, Result, anyhow, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::path::Path;

/// Model file format tag (safetensors metadata key `"format"`).
pub const FORMAT: &str = "xornet-vision-v1";

struct StSource<'a>(&'a SafeTensors<'a>);

impl TensorSource for StSource<'_> {
    fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self
            .0
            .tensor(name)
            .with_context(|| format!("tensor `{name}`"))?;
        ensure!(
            t.dtype() == Dtype::F32,
            "tensor `{name}`: dtype {:?}, expected F32",
            t.dtype()
        );
        let v = t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((v, t.shape().to_vec()))
    }

    fn u64(&self, name: &str) -> Result<(Vec<u64>, Vec<usize>)> {
        let t = self
            .0
            .tensor(name)
            .with_context(|| format!("tensor `{name}`"))?;
        ensure!(
            t.dtype() == Dtype::U64,
            "tensor `{name}`: dtype {:?}, expected U64",
            t.dtype()
        );
        let v = t
            .data()
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok((v, t.shape().to_vec()))
    }
}

#[derive(Deserialize)]
struct PreprocessSpec {
    size: usize,
    crop_padding: usize,
    mean: [f64; 3],
    std: [f64; 3],
    resize: String,
}

/// Output of [`ImageClassifier::classify`].
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// Raw class scores (pre-softmax).
    pub logits: Vec<f32>,
    /// Softmax of `logits`.
    pub probabilities: Vec<f32>,
}

impl Classification {
    pub fn from_logits(logits: Vec<f32>) -> Self {
        let probabilities = softmax(&logits);
        Self {
            logits,
            probabilities,
        }
    }

    /// The `k` most probable classes as `(class_index, probability)`,
    /// highest first (ties broken by lower index).
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        top_k(&self.probabilities, k)
    }
}

/// A labelled prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub class_index: usize,
    pub label: String,
    pub probability: f32,
    pub logit: f32,
}

/// Numerically stable softmax.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Indices of the `k` largest values, descending (NaNs sort last).
pub fn top_k(x: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| {
        let (va, vb) = (x[a], x[b]);
        match (va.is_nan(), vb.is_nan()) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => vb
                .partial_cmp(&va)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b)),
        }
    });
    idx.into_iter().take(k).map(|i| (i, x[i])).collect()
}

/// A pretrained image classifier (e.g. Larq Zoo QuickNet) loaded from an
/// `xornet-vision-v1` safetensors file.
///
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// use xor_net::vision::ImageClassifier;
/// let model = ImageClassifier::load("models/vision/quicknet/quicknet.safetensors")?;
/// # let (rgb, w, h) = (vec![0u8; 640 * 480 * 3], 640, 480);
/// for p in model.predict(&rgb, w, h, 5)? {
///     println!("{:>5.1}%  {}", 100.0 * p.probability, p.label);
/// }
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct ImageClassifier {
    name: String,
    network: Network,
    preprocess: Preprocess,
    labels: Vec<String>,
}

impl ImageClassifier {
    /// Load a model file. The whole file is read into memory (QuickNet is
    /// ~4.4 MB); no memory mapping, so concurrent modification of the file
    /// cannot cause undefined behaviour.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("loading {}", path.display()))
    }

    /// Parse a model from the bytes of an `xornet-vision-v1` safetensors file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, meta) = SafeTensors::read_metadata(bytes)?;
        let md = meta
            .metadata()
            .as_ref()
            .ok_or_else(|| anyhow!("safetensors file has no metadata"))?;
        let get = |k: &str| {
            md.get(k)
                .ok_or_else(|| anyhow!("metadata key `{k}` missing"))
        };
        ensure!(
            get("format")? == FORMAT,
            "unsupported format {:?} (expected {FORMAT})",
            get("format")?
        );
        let specs: Vec<OpSpec> = serde_json::from_str(get("graph")?).context("parsing graph")?;
        let pp: PreprocessSpec =
            serde_json::from_str(get("preprocess")?).context("parsing preprocess")?;
        if pp.resize != "tf_legacy_bicubic" {
            bail!("unsupported resize method {}", pp.resize);
        }
        ensure!(pp.size > 0 && pp.size <= 4096, "bad input size {}", pp.size);
        let preprocess = Preprocess {
            size: pp.size,
            crop_padding: pp.crop_padding,
            mean: pp.mean.map(|v| v as f32),
            std: pp.std.map(|v| v as f32),
        };
        let labels: Vec<String> = match md.get("labels") {
            Some(s) => serde_json::from_str(s).context("parsing labels")?,
            None => Vec::new(),
        };
        let st = SafeTensors::deserialize(bytes)?;
        let network = Network::build(
            &specs,
            &StSource(&st),
            "input",
            get("output")?,
            [pp.size, pp.size, 3],
        )?;
        let [oh, ow, nc] = network.output_shape();
        ensure!(
            oh == 1 && ow == 1,
            "network output is {oh}x{ow}x{nc}, expected a class vector"
        );
        let labels = if labels.is_empty() {
            (0..nc).map(|i| format!("class {i}")).collect()
        } else {
            ensure!(
                labels.len() == nc,
                "{} labels for {nc} classes",
                labels.len()
            );
            labels
        };
        let name = md.get("model").cloned().unwrap_or_else(|| "unknown".into());
        Ok(Self {
            name,
            network,
            preprocess,
            labels,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_classes(&self) -> usize {
        self.labels.len()
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn label(&self, class_index: usize) -> Option<&str> {
        self.labels.get(class_index).map(String::as_str)
    }

    pub fn preprocess_params(&self) -> &Preprocess {
        &self.preprocess
    }

    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Force a binary-conv kernel (for benchmarking / cross-checking).
    pub fn set_binary_kernel(&mut self, k: BinaryKernel) -> Result<()> {
        self.network.set_binary_kernel(k)
    }

    /// Crop/resize/normalise an interleaved RGB8 image (`width * height * 3`
    /// bytes, row-major) into the network input.
    pub fn preprocess(&self, rgb: &[u8], width: usize, height: usize) -> Result<FeatureMap> {
        Ok(self.preprocess.apply(rgb, width, height)?)
    }

    /// Logits for an already preprocessed input of shape `[size, size, 3]`.
    pub fn logits(&self, input: FeatureMap) -> Result<Vec<f32>> {
        Ok(self.network.forward(input)?.data)
    }

    /// Classify an interleaved RGB8 image.
    pub fn classify(&self, rgb: &[u8], width: usize, height: usize) -> Result<Classification> {
        let x = self.preprocess(rgb, width, height)?;
        Ok(Classification::from_logits(self.logits(x)?))
    }

    /// Classify several RGB8 images `(rgb, width, height)` in parallel (one
    /// rayon task per image, each still using intra-layer parallelism when
    /// threads are idle). Best throughput for many images.
    pub fn classify_batch(&self, images: &[(&[u8], usize, usize)]) -> Result<Vec<Classification>> {
        use rayon::prelude::*;
        images
            .par_iter()
            .map(|&(rgb, w, h)| self.classify(rgb, w, h))
            .collect()
    }

    /// Classify and return the `k` most likely labelled classes.
    pub fn predict(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        k: usize,
    ) -> Result<Vec<Prediction>> {
        let c = self.classify(rgb, width, height)?;
        Ok(self.label_top_k(&c, k))
    }

    /// Attach labels to the top `k` of a classification.
    pub fn label_top_k(&self, c: &Classification, k: usize) -> Vec<Prediction> {
        c.top_k(k)
            .into_iter()
            .map(|(i, p)| Prediction {
                class_index: i,
                label: self.labels.get(i).cloned().unwrap_or_default(),
                probability: p,
                logit: c.logits[i],
            })
            .collect()
    }
}
