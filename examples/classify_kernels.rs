//! Micro-benchmark of the vision kernels at QuickNet's layer shapes.
//!
//! ```text
//! cargo run --release --example classify_kernels -- [--threads N]
//! ```
//!
//! Reports, per shape and binary kernel, the time of one binary 3x3 conv
//! (packing + XNOR/popcount + fused relu/BN/residual epilogue) and the
//! achieved binary multiply-accumulates per second.

use std::time::Instant;
use xor_net::vision::kernels::binary::{binary_conv2d_with, pack_signs};
use xor_net::vision::kernels::float;
use xor_net::vision::{BinaryConvWeights, BinaryKernel, Epilogue, FeatureMap, PadValue, Padding};

fn median_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut v: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut threads = 1;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--threads" {
            threads = args.next().unwrap().parse().unwrap();
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    println!("threads={threads}");
    for &(hw, c) in &[(56usize, 64usize), (28, 128), (14, 256), (7, 512)] {
        let x = FeatureMap::from_vec(hw, hw, c, (0..hw * hw * c).map(|_| rnd()).collect());
        let res = FeatureMap::from_vec(hw, hw, c, (0..hw * hw * c).map(|_| rnd()).collect());
        let wt: Vec<f32> = (0..9 * c * c).map(|_| rnd()).collect();
        let w = BinaryConvWeights::from_f32_hwio(&wt, 3, 3, c, c).unwrap();
        let epi = Epilogue {
            pre_relu: true,
            scale_bias: Some((
                (0..c).map(|_| rnd()).collect(),
                (0..c).map(|_| rnd()).collect(),
            )),
            residual: true,
            post_relu: false,
        };
        let macs = (hw * hw * c * c * 9) as f64;
        let pack = pool.install(|| median_ms(200, || drop(std::hint::black_box(pack_signs(&x)))));
        let packed = pack_signs(&x);
        for k in BinaryKernel::available() {
            let ms = pool.install(|| {
                median_ms(200, || {
                    let y = binary_conv2d_with(
                        k,
                        &packed,
                        &w,
                        [1, 1],
                        Padding::Same,
                        PadValue::PlusOne,
                        &epi,
                        Some(&res),
                    );
                    std::hint::black_box(y);
                })
            });
            println!(
                "  binary conv {hw:>2}x{hw:<2}x{c:<3} {k:>7?}: {ms:.4} ms (+pack {pack:.4} ms) = {:.0} G binary-MAC/s",
                macs / ms / 1e6
            );
        }
    }
    // Real-valued layers of QuickNet.
    let mut f = |h: usize, cin: usize, cout: usize, k: usize, s: usize| {
        let x = FeatureMap::from_vec(h, h, cin, (0..h * h * cin).map(|_| rnd()).collect());
        let wt: Vec<f32> = (0..k * k * cin * cout).map(|_| rnd()).collect();
        let ms = pool.install(|| {
            median_ms(100, || {
                std::hint::black_box(float::conv2d(
                    &x,
                    &wt,
                    [k, k],
                    [s, s],
                    Padding::Same,
                    cout,
                    &Epilogue::default(),
                    None,
                ));
            })
        });
        let oh = h.div_ceil(s);
        let flops = 2.0 * (oh * oh * k * k * cin * cout) as f64;
        println!(
            "  float conv {h}x{h}x{cin} -> {cout} k{k} s{s}: {ms:.4} ms = {:.1} GFLOP/s",
            flops / ms / 1e6
        );
    };
    f(224, 3, 16, 3, 2);
    f(56, 16, 64, 1, 1);
    f(28, 64, 128, 1, 1);
    f(7, 256, 512, 1, 1);
    for &(h, c) in &[(112usize, 16usize), (55, 64)] {
        let x = FeatureMap::from_vec(h, h, c, (0..h * h * c).map(|_| rnd()).collect());
        let wt: Vec<f32> = (0..9 * c).map(|_| rnd()).collect();
        let ms = pool.install(|| {
            median_ms(100, || {
                std::hint::black_box(float::depthwise_conv2d(
                    &x,
                    &wt,
                    [3, 3],
                    [2, 2],
                    Padding::Same,
                    &Epilogue::default(),
                    None,
                ));
            })
        });
        println!("  depthwise {h}x{h}x{c} k3 s2: {ms:.4} ms");
    }
    {
        let x = FeatureMap::from_vec(56, 56, 64, (0..56 * 56 * 64).map(|_| rnd()).collect());
        let ms = pool.install(|| {
            median_ms(100, || {
                drop(std::hint::black_box(float::max_pool2d(
                    &x,
                    [2, 2],
                    [1, 1],
                    Padding::Valid,
                )))
            })
        });
        println!("  max_pool 56x56x64 2x2 s1: {ms:.4} ms");
    }
}
