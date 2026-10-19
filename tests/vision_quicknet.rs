//! Real-weights validation of the vision module against Larq/TensorFlow.
//!
//! Needs files produced by the conversion scripts (not committed):
//!
//! ```text
//! models/vision/quicknet/quicknet.safetensors            (convert_larq_zoo.py)
//! models/vision/quicknet/reference_quicknet.safetensors  (reference_larq_zoo.py)
//! ```
//!
//! Run with `cargo test --release --test vision_quicknet -- --ignored --nocapture`.
//! Paths can be overridden with `XORNET_VISION_MODEL` / `XORNET_VISION_REF`.

use safetensors::SafeTensors;
use xor_net::vision::{BinaryKernel, FeatureMap, ImageClassifier, top_k};

fn path(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| format!("{}/{default}", env!("CARGO_MANIFEST_DIR")))
}

fn f32s(st: &SafeTensors, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = st.tensor(name).unwrap();
    (
        t.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect(),
        t.shape().to_vec(),
    )
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs converted QuickNet weights and reference dumps under models/vision"]
fn quicknet_matches_larq_reference() {
    let mut model = ImageClassifier::load(path(
        "XORNET_VISION_MODEL",
        "models/vision/quicknet/quicknet.safetensors",
    ))
    .expect("load model");
    let ref_bytes = std::fs::read(path(
        "XORNET_VISION_REF",
        "models/vision/quicknet/reference_quicknet.safetensors",
    ))
    .unwrap();
    let (_, meta) = SafeTensors::read_metadata(&ref_bytes).unwrap();
    let names: Vec<String> =
        serde_json::from_str(meta.metadata().as_ref().unwrap().get("names").unwrap()).unwrap();
    let st = SafeTensors::deserialize(&ref_bytes).unwrap();

    let (mut exact, mut top1) = (0, 0);
    for (i, name) in names.iter().enumerate() {
        let (input, ishape) = f32s(&st, &format!("{i}.input"));
        let (ref_logits, _) = f32s(&st, &format!("{i}.logits"));
        let rgb_t = st.tensor(&format!("{i}.rgb")).unwrap();
        let (h, w) = (rgb_t.shape()[0], rgb_t.shape()[1]);
        let rgb = rgb_t.data().to_vec();

        // 1. Network only, on the reference's own preprocessed tensor, for
        //    every available binary kernel.
        let mut per_kernel = Vec::new();
        for k in BinaryKernel::available() {
            model.set_binary_kernel(k).unwrap();
            let x = FeatureMap::from_vec(ishape[0], ishape[1], ishape[2], input.clone());
            per_kernel.push((k, model.logits(x).unwrap()));
        }
        let logits = &per_kernel.last().unwrap().1;
        for (k, l) in &per_kernel {
            assert_eq!(l, logits, "{k:?} kernel differs from the others");
        }
        let net_diff = max_abs_diff(logits, &ref_logits);
        let ref_top5: Vec<usize> = top_k(&ref_logits, 5).into_iter().map(|t| t.0).collect();
        let our_top5: Vec<usize> = top_k(logits, 5).into_iter().map(|t| t.0).collect();

        // 2. Preprocessing.
        let ours = model.preprocess(&rgb, w, h).unwrap();
        let pre_diff = max_abs_diff(&ours.data, &input);

        // 3. End to end from RGB.
        let e2e = model.classify(&rgb, w, h).unwrap();
        let e2e_diff = max_abs_diff(&e2e.logits, &ref_logits);
        let e2e_top5: Vec<usize> = e2e.top_k(5).into_iter().map(|t| t.0).collect();

        println!(
            "{name:<22} net max|dlogit|={net_diff:.2e} pre max|dx|={pre_diff:.2e} e2e max|dlogit|={e2e_diff:.2e} \
             top5 ref={ref_top5:?} ours={our_top5:?} e2e={e2e_top5:?} top1={}",
            model.label(our_top5[0]).unwrap()
        );
        // A binary network is discontinuous: float summation-order noise
        // (~1e-5) in the real-valued stem can flip the sign of an activation
        // that is within ~1e-6 of zero, and that flip propagates. TensorFlow
        // itself shows this between its oneDNN and Eigen CPU backends (on
        // this image set: 11/17 images bit-close for QuickNet, 1 top-1 flip
        // on a 0.08-logit margin). So: most images must match to float
        // noise, and top-1 must agree whenever the reference is not a near
        // tie. `quicknet_layerwise` shows where a divergence starts.
        let ref_sorted = top_k(&ref_logits, 2);
        let margin = ref_sorted[0].1 - ref_sorted[1].1;
        if margin > 0.5 {
            assert_eq!(
                our_top5[0], ref_top5[0],
                "{name}: top-1 differs (reference margin {margin})"
            );
            assert_eq!(e2e_top5[0], ref_top5[0], "{name}: end-to-end top-1 differs");
        }
        let overlap = our_top5.iter().filter(|c| ref_top5.contains(c)).count();
        assert!(overlap >= 3, "{name}: top-5 overlap {overlap}/5");
        assert!(
            pre_diff < 1e-3,
            "{name}: preprocessing differs by {pre_diff}"
        );
        top1 += (our_top5[0] == ref_top5[0]) as usize;
        if net_diff < 1e-3 {
            exact += 1;
        }
    }
    println!(
        "{exact}/{} images with max|dlogit| < 1e-3, top-1 agreement {top1}/{}",
        names.len(),
        names.len()
    );
    assert!(
        2 * exact >= names.len(),
        "only {exact}/{} images match the reference logits",
        names.len()
    );
}

/// Layer-by-layer comparison for one image. Produce the dump with
/// `scripts/vision/dump_layers_larq_zoo.py --index N --out FILE` and run with
/// `XORNET_VISION_LAYERS=FILE XORNET_VISION_INDEX=N`.
#[test]
#[ignore = "needs a per-layer dump from scripts/vision/dump_layers_larq_zoo.py"]
fn quicknet_layerwise() {
    let Ok(layers_path) = std::env::var("XORNET_VISION_LAYERS") else {
        eprintln!("XORNET_VISION_LAYERS not set; skipping layer-wise comparison");
        return;
    };
    let model = ImageClassifier::load(path(
        "XORNET_VISION_MODEL",
        "models/vision/quicknet/quicknet.safetensors",
    ))
    .expect("load model");
    let ref_bytes = std::fs::read(path(
        "XORNET_VISION_REF",
        "models/vision/quicknet/reference_quicknet.safetensors",
    ))
    .unwrap();
    let idx: usize = std::env::var("XORNET_VISION_INDEX")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    let layers_bytes = std::fs::read(layers_path).unwrap();
    let st = SafeTensors::deserialize(&ref_bytes).unwrap();
    let layers = SafeTensors::deserialize(&layers_bytes).unwrap();
    let (input, s) = f32s(&st, &format!("{idx}.input"));
    let x = FeatureMap::from_vec(s[0], s[1], s[2], input);
    model
        .network()
        .forward_inspect(x, |name, y| {
            let Ok(t) = layers.tensor(name) else {
                println!("{name:<28} (no reference)");
                return;
            };
            let r: Vec<f32> = t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            let d = max_abs_diff(&y.data, &r);
            // Sign disagreements (what the next binary layer would see).
            let flips = y
                .data
                .iter()
                .zip(&r)
                .filter(|(a, b)| (**a >= 0.0) != (**b >= 0.0))
                .count();
            let min_abs_flip = y
                .data
                .iter()
                .zip(&r)
                .filter(|(a, b)| (**a >= 0.0) != (**b >= 0.0))
                .map(|(a, b)| a.abs().max(b.abs()))
                .fold(f32::INFINITY, f32::min);
            println!("{name:<28} max|d|={d:.3e} sign_flips={flips} (|v|<={min_abs_flip:.1e})");
        })
        .unwrap();
}
