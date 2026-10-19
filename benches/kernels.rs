//! Micro-benchmarks for the decode hot kernels (synthetic data only).
//!
//! Run with `cargo bench --bench kernels`. Each line reports the median and
//! minimum wall time over many repetitions after a warm-up, so the numbers are
//! stable enough to catch a >2% regression between two builds.
//!
//! Optional filter: `cargo bench --bench kernels -- gemv` runs only the cases
//! whose name contains `gemv`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use xor_net::bit1_58::layers::{TernaryLinear, TernaryRep};
use xor_net::bit1_58::quantization::{TernaryPackType, quantize_f32_to_i8, silu_inplace};
use xor_net::bit1_58::simd::ternary_dot_product_pack4;
use xor_net::nn::dynamic_linear::Int4Linear;
use xor_net::nn::fast_attention_simd::rms_norm;
use xor_net::tensor::FastTensor;

/// Deterministic xorshift generator so runs are reproducible without deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }
}

fn run<F: FnMut()>(filter: &Option<String>, name: &str, iters: usize, mut f: F) {
    if let Some(flt) = filter
        && !name.contains(flt.as_str())
    {
        return;
    }
    for _ in 0..(iters / 10).max(3) {
        f();
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples[0];
    println!(
        "{:<44} median {:>10.2} us   min {:>10.2} us",
        name,
        median.as_secs_f64() * 1e6,
        min.as_secs_f64() * 1e6
    );
}

/// Like [`run`], also reporting effective bandwidth for `bytes` per call.
fn run_bw<F: FnMut()>(filter: &Option<String>, name: &str, iters: usize, bytes: usize, mut f: F) {
    if let Some(flt) = filter
        && !name.contains(flt.as_str())
    {
        return;
    }
    for _ in 0..(iters / 10).max(3) {
        f();
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    println!(
        "{:<60} median {:>10.2} us   {:>7.1} GB/s",
        name,
        median.as_secs_f64() * 1e6,
        bytes as f64 / median.as_secs_f64() / 1e9
    );
}

/// Random ternary weights packed as pack4 (codes 0b00/0b01/0b10 only).
fn random_pack4(rng: &mut Rng, in_dim: usize, out_dim: usize) -> Vec<u8> {
    let bpr = in_dim.div_ceil(4);
    let mut v = vec![0u8; bpr * out_dim];
    for b in v.iter_mut() {
        let mut byte = 0u8;
        for k in 0..4 {
            let code = rng.byte() % 3; // 0 -> -1, 1 -> 0, 2 -> +1
            byte |= code << (2 * k);
        }
        *b = byte;
    }
    v
}

fn ternary_linear(rng: &mut Rng, in_dim: usize, out_dim: usize) -> TernaryLinear {
    TernaryLinear {
        rep: TernaryRep::Dense(random_pack4(rng, in_dim, out_dim)),
        in_dim,
        out_dim,
        pack_type: TernaryPackType::Pack4,
        w_scales: vec![0.01; out_dim],
    }
}

fn main() {
    let filter = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .filter(|a| !a.is_empty());
    let _ = xor_net::init_threads(0);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // ── Ternary pack4 GEMV (rayon, as used in decode) ────────────────────
    for &(in_dim, out_dim) in &[(2560usize, 2560usize), (2560, 6912), (6912, 2560)] {
        let lin = ternary_linear(&mut rng, in_dim, out_dim);
        let x: Vec<f32> = (0..in_dim).map(|_| rng.f32()).collect();
        let mut q = vec![0i8; in_dim];
        let inv = quantize_f32_to_i8(&x, &mut q);
        let xs = FastTensor::new(x.clone(), vec![1, in_dim]);
        run(
            &filter,
            &format!("gemv_ternary_pack4 {}x{} (mt)", out_dim, in_dim),
            400,
            || {
                let out = lin.forward_with_quantized(&xs, &q, inv).unwrap();
                black_box(&out);
            },
        );
        // Kernel only, single thread: one full pass over every row.
        let TernaryRep::Dense(ref w) = lin.rep else {
            unreachable!()
        };
        let bpr = in_dim.div_ceil(4);
        run(
            &filter,
            &format!("dot_ternary_pack4 {}x{} (1t)", out_dim, in_dim),
            200,
            || {
                let mut acc = 0i64;
                for r in 0..out_dim {
                    acc += ternary_dot_product_pack4(&q, &w[r * bpr..(r + 1) * bpr], in_dim) as i64;
                }
                black_box(acc);
            },
        );
    }

    // ── Int4 LM-head GEMV (vocab 128256 x 2560, ~164 MB of weights) ──────
    {
        let (in_dim, out_dim) = (2560usize, 128_256usize);
        let mut weight_i4 = vec![0u8; in_dim.div_ceil(2) * out_dim];
        for b in weight_i4.iter_mut() {
            *b = rng.byte();
        }
        let lin = Int4Linear {
            weight_i4,
            scales: vec![0.01; out_dim],
            in_dim,
            out_dim,
        };
        let x: Vec<f32> = (0..in_dim).map(|_| rng.f32()).collect();
        let mut q = vec![0i8; in_dim];
        let inv = quantize_f32_to_i8(&x, &mut q);
        let xs = FastTensor::new(x.clone(), vec![1, in_dim]);
        run(&filter, "gemv_int4_lm_head 128256x2560 (mt)", 40, || {
            let out = lin.forward_with_quantized(&xs, &q, inv).unwrap();
            black_box(&out);
        });
    }

    // ── Q1_0 (1-bit Bonsai) GEMV / GEMM ──────────────────────────────────
    {
        use xor_net::bit1::q1_0::{ByteBuf, Isa, Q1Weight, Q8Acts, dot_rows_isa, gemm, gemv};
        let q1 = |rng: &mut Rng, rows: usize, cols: usize, block: usize| -> Q1Weight {
            let bits: Vec<u8> = (0..rows * cols / 8).map(|_| rng.byte()).collect();
            let scales: Vec<u8> = (0..rows * cols / block)
                .flat_map(|_| (0x2000u16 | (rng.byte() as u16)).to_le_bytes())
                .collect();
            Q1Weight::new(
                rows,
                cols,
                block,
                ByteBuf::from_vec(bits),
                ByteBuf::from_vec(scales),
            )
            .unwrap()
        };
        // Single matrices (L3-resident after warm-up): Bonsai-1.7B shapes.
        for &(name, rows, cols, block) in &[
            ("qkv 4096x2048", 4096usize, 2048usize, 128usize),
            ("gate_up 12288x2048", 12288, 2048, 128),
            ("down 2048x6144", 2048, 6144, 128),
            ("gate_up b32 12288x2048", 12288, 2048, 32),
            ("lm_head 151669x2048", 151669, 2048, 128),
        ] {
            let w = q1(&mut rng, rows, cols, block);
            let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
            let act_blocks: &[usize] = if block == 128 { &[32, 128] } else { &[32] };
            for &ab in act_blocks {
                let mut acts = Q8Acts::new();
                acts.quantize(&x, 1, cols, ab);
                let mut out = vec![0.0f32; rows];
                run_bw(&filter, &format!("q1_gemv {name} act{ab} (mt)"), 200, w.nbytes(), || {
                    gemv(&w, &acts, &mut out);
                    black_box(&out);
                });
                if rows <= 4096 {
                    for isa in Isa::all_supported() {
                        run_bw(
                            &filter,
                            &format!("q1_rows {name} act{ab} {isa:?} (1 thread)"),
                            50,
                            w.nbytes(),
                            || {
                                dot_rows_isa(isa, &w, 0, acts.row(0), &mut out);
                                black_box(&out);
                            },
                        );
                    }
                }
                if rows == 12288 && block == 128 {
                    let n = 64;
                    let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
                    let mut acts = Q8Acts::new();
                    acts.quantize(&xs, n, cols, ab);
                    let mut out = vec![0.0f32; n * rows];
                    run_bw(
                        &filter,
                        &format!("q1_gemm {name} act{ab} n=64 (mt, weight bytes x n)"),
                        20,
                        w.nbytes() * n,
                        || {
                            gemm(&w, &acts, &mut out);
                            black_box(&out);
                        },
                    );
                }
            }
        }
        // DRAM-bound decode sweep: 28 distinct layers (qkv, o, gate_up, down)
        // of Bonsai-1.7B = 0.21 GB of weights, far beyond L3.
        let layers: Vec<[Q1Weight; 4]> = (0..28)
            .map(|_| {
                [
                    q1(&mut rng, 4096, 2048, 128),
                    q1(&mut rng, 2048, 2048, 128),
                    q1(&mut rng, 12288, 2048, 128),
                    q1(&mut rng, 2048, 6144, 128),
                ]
            })
            .collect();
        let total: usize = layers.iter().flatten().map(|w| w.nbytes()).sum();
        let x: Vec<f32> = (0..6144).map(|_| rng.f32()).collect();
        let (mut a2k, mut a6k) = (Q8Acts::new(), Q8Acts::new());
        a2k.quantize(&x[..2048], 1, 2048, 32);
        a6k.quantize(&x, 1, 6144, 32);
        let mut out = vec![0.0f32; 12288];
        run_bw(
            &filter,
            "q1_gemv 1.7B 28-layer sweep act32 (mt, DRAM)",
            30,
            total,
            || {
                for l in &layers {
                    for w in l {
                        let a = if w.cols() == 2048 { &a2k } else { &a6k };
                        gemv(w, a, &mut out[..w.rows()]);
                    }
                }
                black_box(&out);
            },
        );
    }

    // ── Element-wise kernels ─────────────────────────────────────────────
    for &n in &[2560usize, 6912] {
        let x: Vec<f32> = (0..n).map(|_| rng.f32() * 4.0).collect();
        let mut q = vec![0i8; n];
        run(
            &filter,
            &format!("quantize_f32_to_i8 n={}", n),
            20000,
            || {
                black_box(quantize_f32_to_i8(black_box(&x), &mut q));
            },
        );
    }
    {
        let n = 2560;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let mut buf = x.clone();
        run(&filter, "rms_norm n=2560", 20000, || {
            buf.copy_from_slice(&x);
            rms_norm(black_box(&mut buf), &w, 1e-5);
        });
    }
    {
        let n = 6912;
        let g: Vec<f32> = (0..n).map(|_| rng.f32() * 6.0).collect();
        let u: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let mut buf = g.clone();
        run(&filter, "silu_inplace n=6912", 20000, || {
            buf.copy_from_slice(&g);
            silu_inplace(black_box(&mut buf), &u, true);
        });
    }
}
