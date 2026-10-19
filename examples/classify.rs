//! Classify an image with a binarized ImageNet network (Larq Zoo QuickNet).
//!
//! ```text
//! cargo run --release --example classify -- <image.jpg|png> [model.safetensors]
//!     [--threads N] [--bench ITERS] [--kernel scalar|avx2|avx512] [--profile]
//!     [--batch N]   (throughput of classify_batch over N copies of the image)
//! ```
//!
//! The model file is produced by `scripts/vision/convert_larq_zoo.py`
//! (default path: `models/vision/quicknet/quicknet.safetensors`).

use std::time::Instant;
use xor_net::vision::{BinaryKernel, ImageClassifier};

fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

/// Minor page faults of this process so far (Linux).
fn minor_faults() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesised command name; minflt is field 10.
    let rest = &s[s.rfind(')')? + 2..];
    rest.split_whitespace().nth(7)?.parse().ok()
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut positional = Vec::new();
    let (mut threads, mut bench, mut kernel, mut profile, mut batch) =
        (0usize, 0usize, None, false, 0usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().expect("--threads N").parse()?,
            "--profile" => profile = true,
            "--batch" => batch = args.next().expect("--batch N").parse()?,
            "--bench" => bench = args.next().expect("--bench ITERS").parse()?,
            "--kernel" => {
                kernel = Some(match args.next().as_deref() {
                    Some("scalar") => BinaryKernel::Scalar,
                    Some("avx2") => BinaryKernel::Avx2,
                    Some("avx512") => BinaryKernel::Avx512,
                    k => anyhow::bail!("unknown kernel {k:?}"),
                })
            }
            _ => positional.push(a),
        }
    }
    let Some(image_path) = positional.first() else {
        eprintln!(
            "usage: classify <image> [model.safetensors] [--threads N] [--bench ITERS] [--kernel K]"
        );
        std::process::exit(2);
    };
    let model_path = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "models/vision/quicknet/quicknet.safetensors".into());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    let t = Instant::now();
    let mut model = ImageClassifier::load(&model_path)?;
    if let Some(k) = kernel {
        model.set_binary_kernel(k)?;
    }
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    let img = image::open(image_path)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let t = Instant::now();
    let top = pool.install(|| model.predict(&rgb, w, h, 5))?;
    let ms = t.elapsed().as_secs_f64() * 1e3;

    println!(
        "{} ({w}x{h}) with {} [{} threads, {:?} kernel, load {load_ms:.1} ms, first run {ms:.2} ms]",
        image_path,
        model.name(),
        pool.current_num_threads(),
        model.network().binary_kernel()
    );
    for p in &top {
        println!(
            "  {:>6.2}%  {:>4}  {}",
            100.0 * p.probability,
            p.class_index,
            p.label
        );
    }

    if bench > 0 {
        // Network only (preprocessing excluded) and end to end.
        let x = model.preprocess(&rgb, w, h)?;
        let mut net = Vec::with_capacity(bench);
        let mut e2e = Vec::with_capacity(bench);
        let mut faults = 0;
        pool.install(|| -> anyhow::Result<()> {
            for _ in 0..3 {
                model.logits(x.clone())?;
            }
            let f0 = minor_faults().unwrap_or(0);
            for _ in 0..10 {
                model.logits(x.clone())?;
            }
            faults = (minor_faults().unwrap_or(0) - f0) / 10;
            for _ in 0..bench {
                let t = Instant::now();
                std::hint::black_box(model.logits(x.clone())?);
                net.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                std::hint::black_box(model.classify(&rgb, w, h)?);
                e2e.push(t.elapsed().as_secs_f64() * 1e3);
            }
            Ok(())
        })?;
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[v.len() / 2], v[0], v.iter().sum::<f64>() / v.len() as f64)
        };
        let (nmed, nmin, nmean) = stats(&mut net);
        let (emed, emin, _) = stats(&mut e2e);
        println!(
            "network: median {nmed:.3} ms (min {nmin:.3}, mean {nmean:.3}) = {:.0} img/s; \
             end-to-end incl. preprocessing: median {emed:.3} ms (min {emin:.3}) = {:.0} img/s; \
             {faults} page faults/inference",
            1e3 / nmed,
            1e3 / emed
        );
    }
    if batch > 0 {
        let imgs: Vec<(&[u8], usize, usize)> = (0..batch).map(|_| (&rgb[..], w, h)).collect();
        pool.install(|| model.classify_batch(&imgs[..imgs.len().min(8)]))?;
        let t = Instant::now();
        let out = pool.install(|| model.classify_batch(&imgs))?;
        let s = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), batch);
        println!(
            "batch of {batch}: {:.1} ms total = {:.0} img/s (end to end)",
            s * 1e3,
            batch as f64 / s
        );
    }
    if profile {
        // Per-layer wall time (median of 50 runs) of the fused graph.
        let x = model.preprocess(&rgb, w, h)?;
        let mut times: Vec<(String, Vec<f64>)> = Vec::new();
        pool.install(|| -> anyhow::Result<()> {
            for it in 0..50 {
                let mut i = 0;
                let mut t = Instant::now();
                model.network().forward_inspect(x.clone(), |name, y| {
                    let dt = t.elapsed().as_secs_f64() * 1e3;
                    if it == 0 {
                        times.push((format!("{name} {:?}", y.shape()), Vec::new()));
                    }
                    times[i].1.push(dt);
                    i += 1;
                    t = Instant::now();
                })?;
            }
            Ok(())
        })?;
        let mut total = 0.0;
        for (name, v) in &mut times {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            total += v[v.len() / 2];
            println!("  {:>8.3} ms  {name}", v[v.len() / 2]);
        }
        println!("  {total:>8.3} ms  total");
    }
    if let Some(rss) = peak_rss_mb() {
        println!("peak RSS: {rss:.1} MB");
    }
    Ok(())
}
