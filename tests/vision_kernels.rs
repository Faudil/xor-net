//! Commit-safe tests for `xor_net::vision`: SIMD kernels against scalar /
//! float references, plus a synthetic end-to-end model (no downloads).

use std::collections::HashMap;
use xor_net::vision::kernels::binary::{
    binary_conv2d_reference, binary_conv2d_with, pack_pixel_scalar, pack_signs,
};
use xor_net::vision::kernels::float;
use xor_net::vision::{
    BinaryConvWeights, BinaryKernel, Epilogue, FeatureMap, ImageClassifier, PadValue, Padding,
};

/// Small deterministic PRNG (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
    fn map(&mut self, h: usize, w: usize, c: usize) -> FeatureMap {
        FeatureMap::from_vec(h, w, c, self.vec(h * w * c))
    }
}

fn random_epilogue(rng: &mut Rng, c: usize, residual: bool) -> Epilogue {
    Epilogue {
        pre_relu: rng.next().is_multiple_of(2),
        scale_bias: if !rng.next().is_multiple_of(3) {
            Some((rng.vec(c), rng.vec(c)))
        } else {
            None
        },
        residual,
        post_relu: rng.next().is_multiple_of(2),
    }
}

// ---------------------------------------------------------------------------
// Binary conv
// ---------------------------------------------------------------------------

#[test]
fn pack_signs_matches_scalar_and_sign_convention() {
    let mut rng = Rng(7);
    for c in [1, 3, 16, 63, 64, 65, 127, 128, 200] {
        let mut m = rng.map(3, 5, c);
        // Exercise the edge values of larq's sign: 0 and -0 are +1, NaN -1.
        m.data[0] = 0.0;
        if m.data.len() > 2 {
            m.data[1] = -0.0;
            m.data[2] = f32::NAN;
        }
        let packed = pack_signs(&m);
        assert_eq!(packed.words, c.div_ceil(64));
        for y in 0..3 {
            for x in 0..5 {
                let mut expect = vec![0u64; packed.words];
                pack_pixel_scalar(m.pixel(y, x), &mut expect);
                assert_eq!(packed.pixel(y, x), &expect[..], "c={c} ({y},{x})");
                for (ci, &v) in m.pixel(y, x).iter().enumerate() {
                    let bit = (expect[ci / 64] >> (ci % 64)) & 1 == 1;
                    assert_eq!(bit, v >= 0.0, "c={c} ch={ci} v={v}");
                }
                // Tail bits beyond c must be zero.
                if c % 64 != 0 {
                    assert_eq!(expect[packed.words - 1] >> (c % 64), 0);
                }
            }
        }
    }
}

#[test]
fn binary_conv_all_kernels_match_float_reference() {
    let mut rng = Rng(0x1234_5678);
    let kernels = BinaryKernel::available();
    assert!(kernels.contains(&BinaryKernel::Scalar));
    // (h, w, cin, cout, k, stride, padding)
    type Case = (usize, usize, usize, usize, [usize; 2], [usize; 2], Padding);
    let cases: &[Case] = &[
        (5, 7, 1, 1, [3, 3], [1, 1], Padding::Same),
        (6, 6, 3, 5, [3, 3], [2, 2], Padding::Same),
        (7, 5, 63, 33, [3, 3], [1, 1], Padding::Same),
        (8, 9, 64, 32, [3, 3], [2, 2], Padding::Same),
        (9, 8, 65, 70, [3, 3], [1, 1], Padding::Same),
        (4, 11, 130, 31, [3, 3], [1, 2], Padding::Same),
        (7, 7, 96, 64, [1, 1], [1, 1], Padding::Same),
        (7, 7, 40, 17, [5, 5], [1, 1], Padding::Same),
        (6, 8, 64, 40, [3, 3], [1, 1], Padding::Valid),
        (9, 9, 33, 9, [2, 3], [2, 1], Padding::Valid),
        (1, 1, 200, 3, [3, 3], [1, 1], Padding::Same),
        (2, 13, 128, 96, [3, 3], [1, 1], Padding::Same),
    ];
    for &(h, w, cin, cout, k, stride, padding) in cases {
        for pad in [PadValue::PlusOne, PadValue::MinusOne] {
            let pad_f = if pad == PadValue::PlusOne { 1.0 } else { -1.0 };
            let input = rng.map(h, w, cin);
            let wt = rng.vec(k[0] * k[1] * cin * cout);
            let bw = BinaryConvWeights::from_f32_hwio(&wt, k[0], k[1], cin, cout).unwrap();
            let packed = pack_signs(&input);
            // Output shape for the residual.
            let probe = binary_conv2d_reference(
                &input,
                &wt,
                k,
                cout,
                stride,
                padding,
                pad_f,
                &Epilogue::default(),
                None,
            );
            let use_res = rng.next().is_multiple_of(2);
            let res = use_res.then(|| rng.map(probe.h, probe.w, cout));
            let epi = random_epilogue(&mut rng, cout, use_res);
            let reference = binary_conv2d_reference(
                &input,
                &wt,
                k,
                cout,
                stride,
                padding,
                pad_f,
                &epi,
                res.as_ref(),
            );
            for &kern in &kernels {
                let got = binary_conv2d_with(
                    kern,
                    &packed,
                    &bw,
                    stride,
                    padding,
                    pad,
                    &epi,
                    res.as_ref(),
                );
                assert_eq!(got.shape(), reference.shape());
                assert_eq!(
                    got.data, reference.data,
                    "{kern:?}: h={h} w={w} cin={cin} cout={cout} k={k:?} s={stride:?} {padding:?} {pad:?}"
                );
            }
        }
    }
}

#[test]
fn padding_counts_as_plus_one_not_zero() {
    // 1x1 input, 3x3 SAME conv: 8 of the 9 taps are padding. With all
    // weights +1 and pad +1 the sum is 9 * cin regardless of the input sign
    // at the centre (+1) — zero-padding would give 1 * cin.
    let cin = 70;
    let input = FeatureMap::from_vec(1, 1, cin, vec![0.5; cin]);
    let wt = vec![1.0; 9 * cin];
    let bw = BinaryConvWeights::from_f32_hwio(&wt, 3, 3, cin, 1).unwrap();
    let packed = pack_signs(&input);
    for kern in BinaryKernel::available() {
        let plus = binary_conv2d_with(
            kern,
            &packed,
            &bw,
            [1, 1],
            Padding::Same,
            PadValue::PlusOne,
            &Epilogue::default(),
            None,
        );
        let minus = binary_conv2d_with(
            kern,
            &packed,
            &bw,
            [1, 1],
            Padding::Same,
            PadValue::MinusOne,
            &Epilogue::default(),
            None,
        );
        assert_eq!(plus.data, vec![(9 * cin) as f32]);
        assert_eq!(minus.data, vec![(cin as f32) - (8 * cin) as f32]);
    }
}

#[test]
fn binary_weights_reject_bad_input() {
    // Wrong length.
    assert!(BinaryConvWeights::from_packed_ohwi(&[0; 3], 1, 3, 3, 64).is_err());
    // Bits set beyond cin.
    let mut w = vec![0u64; 9];
    w[4] = 1 << 10;
    assert!(BinaryConvWeights::from_packed_ohwi(&w, 1, 3, 3, 10).is_err());
    w[4] = 1 << 9;
    assert!(BinaryConvWeights::from_packed_ohwi(&w, 1, 3, 3, 10).is_ok());
    // Oversized kernel.
    assert!(BinaryConvWeights::from_packed_ohwi(&vec![0; 64], 1, 8, 8, 1).is_err());
}

// ---------------------------------------------------------------------------
// Float kernels vs naive references
// ---------------------------------------------------------------------------

/// TF SAME/VALID geometry, written independently of the library.
fn geom(i: usize, k: usize, s: usize, same: bool) -> (usize, isize) {
    if same {
        let o = i.div_ceil(s);
        let total = ((o - 1) * s + k).saturating_sub(i);
        (o, (total / 2) as isize)
    } else {
        ((i - k) / s + 1, 0)
    }
}

#[allow(clippy::too_many_arguments)]
fn naive_conv(
    x: &FeatureMap,
    w: &[f32],
    k: [usize; 2],
    s: [usize; 2],
    same: bool,
    cout: usize,
    depthwise: bool,
) -> FeatureMap {
    let (oh, pt) = geom(x.h, k[0], s[0], same);
    let (ow, pl) = geom(x.w, k[1], s[1], same);
    let cout = if depthwise { x.c } else { cout };
    let mut out = FeatureMap::zeros(oh, ow, cout);
    for oy in 0..oh {
        for ox in 0..ow {
            for o in 0..cout {
                let mut acc = 0.0f64;
                for ky in 0..k[0] {
                    for kx in 0..k[1] {
                        let iy = (oy * s[0] + ky) as isize - pt;
                        let ix = (ox * s[1] + kx) as isize - pl;
                        if iy < 0 || ix < 0 || iy >= x.h as isize || ix >= x.w as isize {
                            continue;
                        }
                        let px = x.pixel(iy as usize, ix as usize);
                        if depthwise {
                            acc += (px[o] * w[(ky * k[1] + kx) * x.c + o]) as f64;
                        } else {
                            for ci in 0..x.c {
                                acc +=
                                    (px[ci] * w[((ky * k[1] + kx) * x.c + ci) * cout + o]) as f64;
                            }
                        }
                    }
                }
                out.data[(oy * ow + ox) * cout + o] = acc as f32;
            }
        }
    }
    out
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

#[test]
fn float_conv_and_depthwise_match_naive() {
    let mut rng = Rng(99);
    for &(h, w, cin, cout, k, s, same) in &[
        (
            9usize,
            9usize,
            3usize,
            16usize,
            [3usize, 3usize],
            [2usize, 2usize],
            true,
        ),
        (8, 8, 3, 5, [3, 3], [2, 2], true),
        (7, 6, 16, 64, [1, 1], [1, 1], false),
        (11, 5, 7, 9, [5, 3], [1, 2], true),
        (6, 6, 4, 3, [3, 3], [1, 1], false),
    ] {
        let x = rng.map(h, w, cin);
        let wt = rng.vec(k[0] * k[1] * cin * cout);
        let p = if same { Padding::Same } else { Padding::Valid };
        let got = float::conv2d(&x, &wt, k, s, p, cout, &Epilogue::default(), None);
        let want = naive_conv(&x, &wt, k, s, same, cout, false);
        assert_eq!(got.shape(), want.shape());
        assert_close(&got.data, &want.data, 1e-5, "conv2d");

        let wd = rng.vec(k[0] * k[1] * cin);
        let got = float::depthwise_conv2d(&x, &wd, k, s, p, &Epilogue::default(), None);
        let want = naive_conv(&x, &wd, k, s, same, 0, true);
        assert_eq!(got.shape(), want.shape());
        assert_close(&got.data, &want.data, 1e-5, "depthwise");
    }
}

#[test]
fn pooling_and_dense_match_naive() {
    let mut rng = Rng(5);
    let x = rng.map(7, 6, 5);
    // max pool 2x2 stride 1 valid: QuickNet transition.
    let mp = float::max_pool2d(&x, [2, 2], [1, 1], Padding::Valid);
    assert_eq!(mp.shape(), [6, 5, 5]);
    for y in 0..6 {
        for xx in 0..5 {
            for c in 0..5 {
                let m = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    .iter()
                    .map(|&(dy, dx)| x.pixel(y + dy, xx + dx)[c])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert_eq!(mp.pixel(y, xx)[c], m);
            }
        }
    }
    // SAME avg pool excludes padding: the bottom-right corner of a 3x3 s2
    // window on 7x6 averages only in-bounds pixels.
    let ap = float::avg_pool2d(&x, [3, 3], [2, 2], Padding::Same);
    assert_eq!(ap.shape(), [4, 3, 5]);
    let corner: f32 = [(5, 4), (5, 5), (6, 4), (6, 5)]
        .iter()
        .map(|&(y, xx)| x.pixel(y, xx)[2])
        .sum::<f32>()
        / 4.0;
    assert!((ap.pixel(3, 2)[2] - corner).abs() < 1e-6);

    let g = float::global_avg_pool(&x);
    let mean0: f32 = x.data.chunks(5).map(|p| p[0]).sum::<f32>() / 42.0;
    assert!((g.data[0] - mean0).abs() < 1e-6);

    let wt = rng.vec(5 * 11);
    let b = rng.vec(11);
    let d = float::dense(&g, &wt, Some(&b), 11, &Epilogue::default());
    for o in 0..11 {
        let want: f32 = (0..5).map(|i| g.data[i] * wt[i * 11 + o]).sum::<f32>() + b[o];
        assert!((d.data[o] - want).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Synthetic end-to-end model
// ---------------------------------------------------------------------------

struct Builder {
    rng: Rng,
    tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
    ops: Vec<serde_json::Value>,
    /// f32 weights by tensor name (for the naive reference).
    f32s: HashMap<String, Vec<f32>>,
    n: usize,
}

impl Builder {
    fn f32_tensor(&mut self, name: &str, shape: &[usize], v: Vec<f32>) {
        let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.tensors
            .push((name.into(), safetensors::Dtype::F32, shape.to_vec(), bytes));
        self.f32s.insert(name.into(), v);
    }
    fn next_name(&mut self, p: &str) -> String {
        self.n += 1;
        format!("{p}_{}", self.n)
    }
    fn op(&mut self, mut v: serde_json::Value, input: &[&str]) -> String {
        let name = self.next_name(v["op"].as_str().unwrap());
        v["name"] = name.clone().into();
        v["inputs"] = input.iter().map(|s| serde_json::Value::from(*s)).collect();
        v["output"] = name.clone().into();
        self.ops.push(v);
        name
    }
    fn conv(&mut self, x: &str, k: usize, s: usize, cin: usize, cout: usize) -> String {
        let t = self.next_name("w");
        let v = self
            .rng
            .vec(k * k * cin * cout)
            .iter()
            .map(|v| v * 0.5)
            .collect();
        self.f32_tensor(&t, &[k, k, cin, cout], v);
        self.op(
            serde_json::json!({"op": "conv2d", "weight": t, "kernel": [k, k], "stride": [s, s],
            "padding": "same", "in_channels": cin, "out_channels": cout, "activation": "linear"}),
            &[x],
        )
    }
    fn dw(&mut self, x: &str, s: usize, c: usize) -> String {
        let t = self.next_name("dw");
        let v = self.rng.vec(9 * c);
        self.f32_tensor(&t, &[3, 3, c], v);
        self.op(serde_json::json!({"op": "depthwise_conv2d", "weight": t, "kernel": [3, 3], "stride": [s, s],
            "padding": "same", "channels": c}), &[x])
    }
    fn bn(&mut self, x: &str, c: usize) -> String {
        let (s, b) = (self.next_name("s"), self.next_name("b"));
        let sv = self.rng.vec(c).iter().map(|v| 0.5 + v.abs()).collect();
        let bv = self.rng.vec(c);
        self.f32_tensor(&s, &[c], sv);
        self.f32_tensor(&b, &[c], bv);
        self.op(
            serde_json::json!({"op": "scale_bias", "scale": s, "bias": b}),
            &[x],
        )
    }
    fn bconv(&mut self, x: &str, cin: usize, cout: usize) -> String {
        let t = self.next_name("bw");
        let wf = self.rng.vec(9 * cin * cout);
        let words = cin.div_ceil(64);
        let mut packed = vec![0u64; cout * 9 * words];
        for k in 0..9 {
            for ci in 0..cin {
                for o in 0..cout {
                    if wf[(k * cin + ci) * cout + o] >= 0.0 {
                        packed[(o * 9 + k) * words + ci / 64] |= 1 << (ci % 64);
                    }
                }
            }
        }
        let bytes = packed.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.tensors.push((
            t.clone(),
            safetensors::Dtype::U64,
            vec![cout, 3, 3, words],
            bytes,
        ));
        self.f32s.insert(t.clone(), wf);
        self.op(serde_json::json!({"op": "binary_conv2d", "weight": t, "kernel": [3, 3], "stride": [1, 1],
            "padding": "same", "in_channels": cin, "out_channels": cout, "activation": "relu", "pad_value": 1.0}), &[x])
    }
}

/// A QuickNet-shaped model: float stem, binary residual blocks with
/// relu -> BN -> add, a blur-pool transition and a dense head.
fn synthetic_model(size: usize) -> (Vec<u8>, Builder) {
    let mut b = Builder {
        rng: Rng(42),
        tensors: vec![],
        ops: vec![],
        f32s: HashMap::new(),
        n: 0,
    };
    let (c0, c1, c2) = (8, 24, 70);
    let x = b.conv("input", 3, 2, 3, c0);
    let x = b.bn(&x, c0);
    let x = b.op(serde_json::json!({"op": "relu"}), &[&x]);
    let x = b.dw(&x, 2, c0);
    let x = b.bn(&x, c0);
    let x = b.conv(&x, 1, 1, c0, c1);
    let mut x = b.bn(&x, c1);
    for _ in 0..2 {
        let y = b.bconv(&x, c1, c1);
        let y = b.bn(&y, c1);
        x = b.op(serde_json::json!({"op": "add"}), &[&y, &x]);
    }
    let x = b.op(serde_json::json!({"op": "relu"}), &[&x]);
    let x = b.op(serde_json::json!({"op": "max_pool2d", "pool": [2, 2], "stride": [1, 1], "padding": "valid"}), &[&x]);
    let x = b.dw(&x, 2, c1);
    let x = b.conv(&x, 1, 1, c1, c2);
    let x = b.bn(&x, c2);
    let y = b.bconv(&x, c2, c2);
    let y = b.bn(&y, c2);
    let x = b.op(serde_json::json!({"op": "add"}), &[&y, &x]);
    let x = b.op(serde_json::json!({"op": "relu"}), &[&x]);
    let x = b.op(serde_json::json!({"op": "global_avg_pool"}), &[&x]);
    let (wn, bn) = (b.next_name("dense_w"), b.next_name("dense_b"));
    let wv = b.rng.vec(c2 * 10);
    let bv = b.rng.vec(10);
    b.f32_tensor(&wn, &[c2, 10], wv);
    b.f32_tensor(&bn, &[10], bv);
    let out = b.op(serde_json::json!({"op": "dense", "weight": wn, "bias": bn, "in_features": c2, "out_features": 10}), &[&x]);

    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "xornet-vision-v1".to_string());
    meta.insert("model".to_string(), "synthetic".to_string());
    meta.insert("graph".to_string(), serde_json::to_string(&b.ops).unwrap());
    meta.insert("output".to_string(), out);
    meta.insert(
        "preprocess".to_string(),
        serde_json::json!({"size": size, "crop_padding": 4, "mean": [120.0, 110.0, 100.0],
            "std": [60.0, 55.0, 50.0], "resize": "tf_legacy_bicubic"})
        .to_string(),
    );
    let labels: Vec<String> = (0..10).map(|i| format!("label{i}")).collect();
    meta.insert(
        "labels".to_string(),
        serde_json::to_string(&labels).unwrap(),
    );
    let views: Vec<(String, safetensors::tensor::TensorView)> = b
        .tensors
        .iter()
        .map(|(n, d, s, bytes)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(*d, s.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let bytes = safetensors::serialize(views, &Some(meta)).unwrap();
    (bytes, b)
}

/// Unfused op-by-op interpreter built only from reference functions.
fn naive_forward(b: &Builder, input: FeatureMap) -> Vec<f32> {
    let mut vals: HashMap<String, FeatureMap> = HashMap::new();
    vals.insert("input".into(), input);
    let mut last = String::new();
    for op in &b.ops {
        let ins: Vec<&FeatureMap> = op["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| &vals[v.as_str().unwrap()])
            .collect();
        let x = ins[0];
        let wt = |key: &str| &b.f32s[op[key].as_str().unwrap()];
        let arr = |key: &str| -> [usize; 2] {
            let a = op[key].as_array().unwrap();
            [
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            ]
        };
        let y = match op["op"].as_str().unwrap() {
            "conv2d" => naive_conv(
                x,
                wt("weight"),
                arr("kernel"),
                arr("stride"),
                true,
                op["out_channels"].as_u64().unwrap() as usize,
                false,
            ),
            "depthwise_conv2d" => {
                naive_conv(x, wt("weight"), arr("kernel"), arr("stride"), true, 0, true)
            }
            "binary_conv2d" => {
                let epi = Epilogue {
                    pre_relu: true,
                    ..Default::default()
                };
                let cout = op["out_channels"].as_u64().unwrap() as usize;
                binary_conv2d_reference(
                    x,
                    wt("weight"),
                    [3, 3],
                    cout,
                    [1, 1],
                    Padding::Same,
                    1.0,
                    &epi,
                    None,
                )
            }
            "scale_bias" => {
                let (s, bb) = (wt("scale"), wt("bias"));
                let mut y = x.clone();
                for px in y.data.chunks_mut(x.c) {
                    for c in 0..x.c {
                        px[c] = px[c] * s[c] + bb[c];
                    }
                }
                y
            }
            "relu" => {
                FeatureMap::from_vec(x.h, x.w, x.c, x.data.iter().map(|v| v.max(0.0)).collect())
            }
            "add" => FeatureMap::from_vec(
                x.h,
                x.w,
                x.c,
                x.data
                    .iter()
                    .zip(&ins[1].data)
                    .map(|(a, b)| a + b)
                    .collect(),
            ),
            "max_pool2d" => float::max_pool2d(x, [2, 2], [1, 1], Padding::Valid),
            "global_avg_pool" => float::global_avg_pool(x),
            "dense" => {
                let (w, bb) = (wt("weight"), wt("bias"));
                let n = bb.len();
                let v = (0..n)
                    .map(|o| {
                        (0..x.data.len())
                            .map(|i| x.data[i] * w[i * n + o])
                            .sum::<f32>()
                            + bb[o]
                    })
                    .collect();
                FeatureMap::from_vec(1, 1, n, v)
            }
            other => panic!("{other}"),
        };
        last = op["output"].as_str().unwrap().to_string();
        vals.insert(last.clone(), y);
    }
    vals.remove(&last).unwrap().data
}

#[test]
fn synthetic_model_end_to_end() {
    let (bytes, b) = synthetic_model(32);
    let mut model = ImageClassifier::from_bytes(&bytes).expect("load synthetic model");
    assert_eq!(model.name(), "synthetic");
    assert_eq!(model.num_classes(), 10);
    // Fusion: conv+bn+relu, binary conv+relu+bn+add, etc.
    let summary = model.network().summary();
    assert!(
        summary
            .iter()
            .any(|s| s.contains("binary_conv2d +relu +bn +add")),
        "{summary:#?}"
    );
    assert!(
        summary.iter().any(|s| s.contains("conv2d +bn +relu")),
        "{summary:#?}"
    );
    assert!(model.network().num_layers() < b.ops.len());

    // A synthetic 45x37 RGB gradient image.
    let (w, h) = (45, 37);
    let rgb: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 37 + (i / 3) * 11) % 256) as u8)
        .collect();
    let x = model.preprocess(&rgb, w, h).unwrap();
    assert_eq!(x.shape(), [32, 32, 3]);
    let want = naive_forward(&b, x.clone());

    let mut outs = Vec::new();
    for k in BinaryKernel::available() {
        model.set_binary_kernel(k).unwrap();
        let got = model.logits(x.clone()).unwrap();
        assert_close(&got, &want, 1e-3, "synthetic logits");
        outs.push(got);
    }
    for o in &outs {
        assert_eq!(o, &outs[0], "kernels disagree");
    }

    let c = model.classify(&rgb, w, h).unwrap();
    let psum: f32 = c.probabilities.iter().sum();
    assert!((psum - 1.0).abs() < 1e-5);
    let top = model.predict(&rgb, w, h, 3).unwrap();
    assert_eq!(top.len(), 3);
    assert!(top[0].probability >= top[1].probability && top[1].probability >= top[2].probability);
    assert_eq!(top[0].label, format!("label{}", top[0].class_index));

    // Wrong buffer sizes and wrong input shapes are errors, not panics.
    assert!(model.classify(&rgb[1..], w, h).is_err());
    assert!(model.logits(FeatureMap::zeros(31, 32, 3)).is_err());
}

#[test]
fn malformed_models_are_rejected() {
    let (bytes, _) = synthetic_model(32);
    // Input too small for the graph: VALID max-pool / dense shape mismatch.
    let (small, _) = synthetic_model(4);
    assert!(ImageClassifier::from_bytes(&small).is_err());
    // Truncated file.
    assert!(ImageClassifier::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    // Garbage.
    assert!(ImageClassifier::from_bytes(b"not a model").is_err());
}
