//! A tiny inference graph for image models: parsing, validation, fusion and
//! execution.
//!
//! The graph is a topologically ordered list of nodes over named values
//! (see [`OpSpec`] for the JSON schema stored in the model file). At load
//! time it is
//!
//! 1. resolved against the weight tensors (dtype/shape checked),
//! 2. **fused**: a conv/dense node absorbs a following chain of
//!    `relu` / `scale_bias` / `add` nodes into its [`Epilogue`] when each
//!    intermediate value has a single consumer,
//! 3. **shape-checked** end to end for the declared input shape, so that
//!    [`Network::forward`] cannot hit a kernel shape assertion for a correctly
//!    sized input.

use super::kernels::binary::{self, BinaryConvWeights, BinaryKernel, PadValue};
use super::kernels::epilogue::Epilogue;
use super::kernels::float;
use super::tensor::{FeatureMap, Geometry, Padding};
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use std::collections::HashMap;

/// One op as stored in the model file's `"graph"` metadata (JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct OpSpec {
    pub op: String,
    #[serde(default)]
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub bias: Option<String>,
    #[serde(default)]
    pub scale: Option<String>,
    #[serde(default)]
    pub kernel: Option<[usize; 2]>,
    #[serde(default)]
    pub stride: Option<[usize; 2]>,
    #[serde(default)]
    pub pool: Option<[usize; 2]>,
    #[serde(default)]
    pub padding: Option<String>,
    #[serde(default)]
    pub in_channels: Option<usize>,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default)]
    pub channels: Option<usize>,
    #[serde(default)]
    pub in_features: Option<usize>,
    #[serde(default)]
    pub out_features: Option<usize>,
    #[serde(default)]
    pub activation: Option<String>,
    #[serde(default)]
    pub pad_value: Option<f32>,
}

/// Access to named weight tensors.
pub trait TensorSource {
    /// An `f32` tensor and its shape.
    fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)>;
    /// A `u64` tensor and its shape.
    fn u64(&self, name: &str) -> Result<(Vec<u64>, Vec<usize>)>;
}

/// A resolved layer.
#[derive(Debug, Clone)]
pub enum Layer {
    Conv2d {
        weight: Vec<f32>,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: Padding,
        cin: usize,
        cout: usize,
    },
    BinaryConv2d {
        weight: BinaryConvWeights,
        stride: [usize; 2],
        padding: Padding,
        pad_value: PadValue,
    },
    DepthwiseConv2d {
        weight: Vec<f32>,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: Padding,
        channels: usize,
    },
    Dense {
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    },
    ScaleBias {
        scale: Vec<f32>,
        bias: Vec<f32>,
    },
    Relu,
    Add,
    MaxPool2d {
        pool: [usize; 2],
        stride: [usize; 2],
        padding: Padding,
    },
    AvgPool2d {
        pool: [usize; 2],
        stride: [usize; 2],
        padding: Padding,
    },
    GlobalAvgPool,
}

impl Layer {
    fn fusable_host(&self) -> bool {
        matches!(
            self,
            Layer::Conv2d { .. }
                | Layer::BinaryConv2d { .. }
                | Layer::DepthwiseConv2d { .. }
                | Layer::Dense { .. }
        )
    }
}

#[derive(Debug, Clone)]
struct Node {
    name: String,
    layer: Layer,
    inputs: Vec<usize>,
    output: usize,
    /// Name of the output value (after fusion: the last fused op's output).
    output_name: String,
    epi: Epilogue,
    residual: Option<usize>,
}

/// An executable, validated network.
#[derive(Debug, Clone)]
pub struct Network {
    nodes: Vec<Node>,
    n_values: usize,
    input: usize,
    output: usize,
    input_shape: [usize; 3],
    output_shape: [usize; 3],
    /// `last_use[v]` = index of the last node reading value `v`.
    last_use: Vec<usize>,
    kernel: BinaryKernel,
}

fn padding(s: &Option<String>) -> Result<Padding> {
    match s.as_deref() {
        Some("same") => Ok(Padding::Same),
        Some("valid") => Ok(Padding::Valid),
        other => bail!("unsupported padding {other:?}"),
    }
}

fn req<T: Copy>(v: Option<T>, what: &str) -> Result<T> {
    v.ok_or_else(|| anyhow!("missing field `{what}`"))
}

fn tensor_f32(src: &dyn TensorSource, name: &Option<String>, shape: &[usize]) -> Result<Vec<f32>> {
    let name = name
        .as_deref()
        .ok_or_else(|| anyhow!("missing tensor reference"))?;
    let (v, s) = src.f32(name)?;
    ensure!(s == shape, "tensor {name}: shape {s:?}, expected {shape:?}");
    Ok(v)
}

fn activation_is_relu(spec: &OpSpec) -> Result<bool> {
    match spec.activation.as_deref() {
        None | Some("linear") => Ok(false),
        Some("relu") => Ok(true),
        Some(a) => bail!("unsupported activation {a}"),
    }
}

impl Network {
    /// Build from op specs. `input` names the graph input value, `output`
    /// the value returned by [`forward`](Self::forward), `input_shape` the
    /// `[h, w, c]` the network will be run on.
    pub fn build(
        specs: &[OpSpec],
        src: &dyn TensorSource,
        input: &str,
        output: &str,
        input_shape: [usize; 3],
    ) -> Result<Self> {
        let mut ids: HashMap<String, usize> = HashMap::new();
        ids.insert(input.to_string(), 0);
        let mut nodes = Vec::with_capacity(specs.len());
        for spec in specs {
            let node = Self::resolve(spec, src, &ids)
                .with_context(|| format!("op `{}` ({})", spec.name, spec.op))?;
            ensure!(
                !ids.contains_key(&spec.output),
                "value `{}` defined twice",
                spec.output
            );
            let id = ids.len();
            ids.insert(spec.output.clone(), id);
            nodes.push(Node {
                output: id,
                output_name: spec.output.clone(),
                ..node
            });
        }
        let out_id = *ids
            .get(output)
            .ok_or_else(|| anyhow!("graph output `{output}` is never produced"))?;
        let mut net = Self {
            nodes,
            n_values: ids.len(),
            input: 0,
            output: out_id,
            input_shape,
            output_shape: [0; 3],
            last_use: Vec::new(),
            kernel: BinaryKernel::detect(),
        };
        net.fuse();
        net.compute_last_use();
        net.output_shape = net.infer_shapes()?;
        Ok(net)
    }

    fn resolve(
        spec: &OpSpec,
        src: &dyn TensorSource,
        ids: &HashMap<String, usize>,
    ) -> Result<Node> {
        let inputs = spec
            .inputs
            .iter()
            .map(|n| {
                ids.get(n)
                    .copied()
                    .ok_or_else(|| anyhow!("unknown input value `{n}`"))
            })
            .collect::<Result<Vec<_>>>()?;
        let arity = if spec.op == "add" { 2 } else { 1 };
        ensure!(
            inputs.len() == arity,
            "expected {arity} input(s), got {}",
            inputs.len()
        );
        let mut epi = Epilogue::default();
        let layer = match spec.op.as_str() {
            "conv2d" => {
                let kernel = req(spec.kernel, "kernel")?;
                let (cin, cout) = (
                    req(spec.in_channels, "in_channels")?,
                    req(spec.out_channels, "out_channels")?,
                );
                epi.pre_relu = activation_is_relu(spec)?;
                Layer::Conv2d {
                    weight: tensor_f32(src, &spec.weight, &[kernel[0], kernel[1], cin, cout])?,
                    kernel,
                    stride: req(spec.stride, "stride")?,
                    padding: padding(&spec.padding)?,
                    cin,
                    cout,
                }
            }
            "binary_conv2d" => {
                let kernel = req(spec.kernel, "kernel")?;
                let (cin, cout) = (
                    req(spec.in_channels, "in_channels")?,
                    req(spec.out_channels, "out_channels")?,
                );
                let name = spec
                    .weight
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing weight"))?;
                let (bits, shape) = src.u64(name)?;
                let words = cin.div_ceil(64);
                ensure!(
                    shape == [cout, kernel[0], kernel[1], words],
                    "tensor {name}: shape {shape:?}, expected {:?}",
                    [cout, kernel[0], kernel[1], words]
                );
                let pv = req(spec.pad_value, "pad_value")?;
                let padding = padding(&spec.padding)?;
                let pad_value = match PadValue::from_f32(pv) {
                    Some(p) => p,
                    // With VALID padding (or a 1x1 kernel) nothing is ever
                    // padded, so the value is irrelevant.
                    None if padding == Padding::Valid || kernel == [1, 1] => PadValue::PlusOne,
                    None => bail!("binary conv pad_value {pv} unsupported (only +1/-1)"),
                };
                epi.pre_relu = activation_is_relu(spec)?;
                Layer::BinaryConv2d {
                    weight: BinaryConvWeights::from_packed_ohwi(
                        &bits, cout, kernel[0], kernel[1], cin,
                    )
                    .map_err(|e| anyhow!(e))?,
                    stride: req(spec.stride, "stride")?,
                    padding,
                    pad_value,
                }
            }
            "depthwise_conv2d" => {
                let kernel = req(spec.kernel, "kernel")?;
                let c = req(spec.channels, "channels")?;
                epi.pre_relu = activation_is_relu(spec)?;
                Layer::DepthwiseConv2d {
                    weight: tensor_f32(src, &spec.weight, &[kernel[0], kernel[1], c])?,
                    kernel,
                    stride: req(spec.stride, "stride")?,
                    padding: padding(&spec.padding)?,
                    channels: c,
                }
            }
            "dense" => {
                let (fi, fo) = (
                    req(spec.in_features, "in_features")?,
                    req(spec.out_features, "out_features")?,
                );
                epi.pre_relu = activation_is_relu(spec)?;
                Layer::Dense {
                    weight: tensor_f32(src, &spec.weight, &[fi, fo])?,
                    bias: match &spec.bias {
                        Some(_) => Some(tensor_f32(src, &spec.bias, &[fo])?),
                        None => None,
                    },
                    in_features: fi,
                    out_features: fo,
                }
            }
            "scale_bias" => {
                let s = spec
                    .scale
                    .as_deref()
                    .ok_or_else(|| anyhow!("missing scale"))?;
                let (scale, ss) = src.f32(s)?;
                let (bias, bs) = src.f32(
                    spec.bias
                        .as_deref()
                        .ok_or_else(|| anyhow!("missing bias"))?,
                )?;
                ensure!(
                    ss.len() == 1 && bs == ss,
                    "scale/bias must be 1-D and equal length"
                );
                Layer::ScaleBias { scale, bias }
            }
            "relu" => Layer::Relu,
            "add" => Layer::Add,
            "max_pool2d" => Layer::MaxPool2d {
                pool: req(spec.pool, "pool")?,
                stride: req(spec.stride, "stride")?,
                padding: padding(&spec.padding)?,
            },
            "avg_pool2d" => Layer::AvgPool2d {
                pool: req(spec.pool, "pool")?,
                stride: req(spec.stride, "stride")?,
                padding: padding(&spec.padding)?,
            },
            "global_avg_pool" => Layer::GlobalAvgPool,
            other => bail!("unsupported op `{other}`"),
        };
        Ok(Node {
            name: spec.name.clone(),
            layer,
            inputs,
            output: usize::MAX,
            output_name: String::new(),
            epi,
            residual: None,
        })
    }

    /// Peephole fusion of element-wise tails into conv/dense epilogues.
    fn fuse(&mut self) {
        let mut producer = vec![usize::MAX; self.nodes.len() + 1];
        for (i, n) in self.nodes.iter().enumerate() {
            producer[n.output] = i;
        }
        let mut removed = vec![false; self.nodes.len()];
        for i in 0..self.nodes.len() {
            if removed[i] || !self.nodes[i].layer.fusable_host() {
                continue;
            }
            // 0 none, 1 pre-relu, 2 scale-bias, 3 residual, 4 post-relu.
            let mut stage = if self.nodes[i].epi.pre_relu { 1 } else { 0 };
            loop {
                let v = self.nodes[i].output;
                if v == self.output {
                    break;
                }
                let consumers: Vec<usize> = (0..self.nodes.len())
                    .filter(|&j| !removed[j] && j != i)
                    .filter(|&j| {
                        self.nodes[j].inputs.contains(&v) || self.nodes[j].residual == Some(v)
                    })
                    .collect();
                if consumers.len() != 1 {
                    break;
                }
                let j = consumers[0];
                let next = &self.nodes[j];
                let ok = match &next.layer {
                    Layer::Relu if stage == 0 => {
                        stage = 1;
                        Some((false, true, None))
                    }
                    Layer::Relu if stage < 4 => {
                        stage = 4;
                        Some((true, false, None))
                    }
                    Layer::ScaleBias { .. } if stage < 2 => {
                        stage = 2;
                        Some((false, false, None))
                    }
                    Layer::Add
                        if stage < 3 && !matches!(self.nodes[i].layer, Layer::Dense { .. }) =>
                    {
                        let other = if next.inputs[0] == v {
                            next.inputs[1]
                        } else {
                            next.inputs[0]
                        };
                        // The residual must exist before node i runs.
                        let avail = other == self.input || producer[other] < i;
                        if other != v && avail {
                            stage = 3;
                            Some((false, false, Some(other)))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let Some((post_relu, pre_relu, residual)) = ok else {
                    break;
                };
                let next_layer = self.nodes[j].layer.clone();
                let next_out = self.nodes[j].output;
                let next_name = self.nodes[j].output_name.clone();
                let host = &mut self.nodes[i];
                if pre_relu {
                    host.epi.pre_relu = true;
                }
                if post_relu {
                    host.epi.post_relu = true;
                }
                if let Layer::ScaleBias { scale, bias } = next_layer {
                    host.epi.scale_bias = Some((scale, bias));
                }
                if let Some(r) = residual {
                    host.epi.residual = true;
                    host.residual = Some(r);
                }
                host.output = next_out;
                host.output_name = next_name;
                producer[next_out] = i;
                removed[j] = true;
            }
        }
        let mut k = 0;
        self.nodes.retain(|_| {
            k += 1;
            !removed[k - 1]
        });
    }

    fn compute_last_use(&mut self) {
        let mut last = vec![0usize; self.n_values];
        for (i, n) in self.nodes.iter().enumerate() {
            for &v in n.inputs.iter().chain(n.residual.iter()) {
                last[v] = i;
            }
        }
        last[self.output] = usize::MAX;
        self.last_use = last;
    }

    /// Propagate shapes from `input_shape`, validating every layer.
    fn infer_shapes(&self) -> Result<[usize; 3]> {
        let mut shapes: Vec<Option<[usize; 3]>> = vec![None; self.n_values];
        shapes[self.input] = Some(self.input_shape);
        for n in &self.nodes {
            let get = |v: usize| shapes[v].ok_or_else(|| anyhow!("value used before definition"));
            let [h, w, c] = get(n.inputs[0])?;
            let geo = |k: [usize; 2], s: [usize; 2], p: Padding| {
                Geometry::new(h, w, k, s, p)
                    .ok_or_else(|| anyhow!("window {k:?} does not fit {h}x{w}"))
            };
            let out = (|| -> Result<[usize; 3]> {
                Ok(match &n.layer {
                    Layer::Conv2d {
                        kernel,
                        stride,
                        padding,
                        cin,
                        cout,
                        ..
                    } => {
                        ensure!(c == *cin, "input has {c} channels, conv expects {cin}");
                        let g = geo(*kernel, *stride, *padding)?;
                        [g.oh, g.ow, *cout]
                    }
                    Layer::BinaryConv2d {
                        weight,
                        stride,
                        padding,
                        ..
                    } => {
                        ensure!(
                            c == weight.cin,
                            "input has {c} channels, conv expects {}",
                            weight.cin
                        );
                        let g = geo([weight.kh, weight.kw], *stride, *padding)?;
                        [g.oh, g.ow, weight.cout]
                    }
                    Layer::DepthwiseConv2d {
                        kernel,
                        stride,
                        padding,
                        channels,
                        ..
                    } => {
                        ensure!(
                            c == *channels,
                            "input has {c} channels, depthwise expects {channels}"
                        );
                        let g = geo(*kernel, *stride, *padding)?;
                        [g.oh, g.ow, c]
                    }
                    Layer::Dense {
                        in_features,
                        out_features,
                        ..
                    } => {
                        ensure!(
                            h * w * c == *in_features,
                            "dense expects {in_features} inputs, got {}",
                            h * w * c
                        );
                        [1, 1, *out_features]
                    }
                    Layer::ScaleBias { scale, .. } => {
                        ensure!(
                            scale.len() == c,
                            "scale_bias has {} channels, input {c}",
                            scale.len()
                        );
                        [h, w, c]
                    }
                    Layer::Relu => [h, w, c],
                    Layer::Add => {
                        let other = get(n.inputs[1])?;
                        ensure!(other == [h, w, c], "add of {other:?} and {:?}", [h, w, c]);
                        [h, w, c]
                    }
                    Layer::MaxPool2d {
                        pool,
                        stride,
                        padding,
                    }
                    | Layer::AvgPool2d {
                        pool,
                        stride,
                        padding,
                    } => {
                        let g = geo(*pool, *stride, *padding)?;
                        [g.oh, g.ow, c]
                    }
                    Layer::GlobalAvgPool => [1, 1, c],
                })
            })()
            .with_context(|| format!("layer `{}`", n.name))?;
            if let Some((s, _)) = &n.epi.scale_bias {
                ensure!(
                    s.len() == out[2],
                    "layer `{}`: fused scale_bias length {} != {}",
                    n.name,
                    s.len(),
                    out[2]
                );
            }
            if let Some(r) = n.residual {
                ensure!(
                    get(r)? == out,
                    "layer `{}`: residual shape mismatch",
                    n.name
                );
            }
            shapes[n.output] = Some(out);
        }
        shapes[self.output].ok_or_else(|| anyhow!("graph output never computed"))
    }

    pub fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    pub fn output_shape(&self) -> [usize; 3] {
        self.output_shape
    }

    /// Number of executed (post-fusion) layers.
    pub fn num_layers(&self) -> usize {
        self.nodes.len()
    }

    /// Short description of the fused layer sequence.
    pub fn summary(&self) -> Vec<String> {
        self.nodes
            .iter()
            .map(|n| {
                let kind = match &n.layer {
                    Layer::Conv2d { .. } => "conv2d",
                    Layer::BinaryConv2d { .. } => "binary_conv2d",
                    Layer::DepthwiseConv2d { .. } => "depthwise_conv2d",
                    Layer::Dense { .. } => "dense",
                    Layer::ScaleBias { .. } => "scale_bias",
                    Layer::Relu => "relu",
                    Layer::Add => "add",
                    Layer::MaxPool2d { .. } => "max_pool2d",
                    Layer::AvgPool2d { .. } => "avg_pool2d",
                    Layer::GlobalAvgPool => "global_avg_pool",
                };
                let e = &n.epi;
                format!(
                    "{:<28} {kind}{}{}{}{}",
                    n.name,
                    if e.pre_relu { " +relu" } else { "" },
                    if e.scale_bias.is_some() { " +bn" } else { "" },
                    if e.residual { " +add" } else { "" },
                    if e.post_relu { " +relu" } else { "" },
                )
            })
            .collect()
    }

    /// Force a specific binary kernel (e.g. to compare SIMD levels).
    /// Errors if the CPU does not support it.
    pub fn set_binary_kernel(&mut self, k: BinaryKernel) -> Result<()> {
        ensure!(
            k.is_supported(),
            "{k:?} binary kernel not supported on this CPU"
        );
        self.kernel = k;
        Ok(())
    }

    pub fn binary_kernel(&self) -> BinaryKernel {
        self.kernel
    }

    /// Run the network. `input` must have shape [`input_shape`](Self::input_shape).
    pub fn forward(&self, input: FeatureMap) -> Result<FeatureMap> {
        self.forward_inspect(input, |_, _| {})
    }

    /// Like [`forward`](Self::forward), calling `inspect(value_name, output)`
    /// after every executed (fused) layer, where `value_name` is the graph
    /// value it produces (for debugging/validation).
    pub fn forward_inspect(
        &self,
        input: FeatureMap,
        mut inspect: impl FnMut(&str, &FeatureMap),
    ) -> Result<FeatureMap> {
        ensure!(
            input.shape() == self.input_shape,
            "input shape {:?} != network input {:?}",
            input.shape(),
            self.input_shape
        );
        let mut vals: Vec<Option<FeatureMap>> = vec![None; self.n_values];
        vals[self.input] = Some(input);
        for (i, n) in self.nodes.iter().enumerate() {
            let x = vals[n.inputs[0]]
                .as_ref()
                .expect("validated: input computed");
            let res = n
                .residual
                .map(|r| vals[r].as_ref().expect("validated: residual computed"));
            let y = match &n.layer {
                Layer::Conv2d {
                    weight,
                    kernel,
                    stride,
                    padding,
                    cout,
                    ..
                } => float::conv2d(x, weight, *kernel, *stride, *padding, *cout, &n.epi, res),
                Layer::BinaryConv2d {
                    weight,
                    stride,
                    padding,
                    pad_value,
                } => {
                    let packed = binary::pack_signs(x);
                    binary::binary_conv2d_with(
                        self.kernel,
                        &packed,
                        weight,
                        *stride,
                        *padding,
                        *pad_value,
                        &n.epi,
                        res,
                    )
                }
                Layer::DepthwiseConv2d {
                    weight,
                    kernel,
                    stride,
                    padding,
                    ..
                } => float::depthwise_conv2d(x, weight, *kernel, *stride, *padding, &n.epi, res),
                Layer::Dense {
                    weight,
                    bias,
                    out_features,
                    ..
                } => {
                    // Fusion never attaches a residual to a dense layer.
                    float::dense(x, weight, bias.as_deref(), *out_features, &n.epi)
                }
                Layer::ScaleBias { scale, bias } => {
                    let mut y = x.clone();
                    for px in y.data.chunks_exact_mut(y.c) {
                        for ((v, &s), &b) in px.iter_mut().zip(scale).zip(bias) {
                            *v = *v * s + b;
                        }
                    }
                    y
                }
                Layer::Relu => {
                    let mut y = x.clone();
                    y.data.iter_mut().for_each(|v| *v = v.max(0.0));
                    y
                }
                Layer::Add => {
                    let o = vals[n.inputs[1]].as_ref().expect("validated");
                    let mut y = x.clone();
                    y.data.iter_mut().zip(&o.data).for_each(|(a, b)| *a += b);
                    y
                }
                Layer::MaxPool2d {
                    pool,
                    stride,
                    padding,
                } => float::max_pool2d(x, *pool, *stride, *padding),
                Layer::AvgPool2d {
                    pool,
                    stride,
                    padding,
                } => float::avg_pool2d(x, *pool, *stride, *padding),
                Layer::GlobalAvgPool => float::global_avg_pool(x),
            };
            inspect(&n.output_name, &y);
            for &v in n.inputs.iter().chain(n.residual.iter()) {
                if self.last_use[v] == i {
                    vals[v] = None;
                }
            }
            vals[n.output] = Some(y);
        }
        Ok(vals[self.output]
            .take()
            .expect("validated: output computed"))
    }
}
