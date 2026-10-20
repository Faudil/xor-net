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
    // `cargo test --all-targets` also runs this (harness = false) binary,
    // without `--bench`; the timings are meaningless there, so do nothing.
    if !std::env::args().any(|a| a == "--bench") {
        println!("kernels: run with `cargo bench --bench kernels`");
        return;
    }
    let filter = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .filter(|a| !a.is_empty());
    // `XORNET_BENCH_THREADS=N` overrides the default pool size.
    let threads = std::env::var("XORNET_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let _ = xor_net::init_threads(threads);
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
        use xor_net::bit1::q1_0::{
            ByteBuf, GemmKernel, Isa, Q1Weight, Q8Acts, dot_rows_isa, gemm, gemm_with, gemv,
        };
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
                run_bw(
                    &filter,
                    &format!("q1_gemv {name} act{ab} (mt)"),
                    200,
                    w.nbytes(),
                    || {
                        gemv(&w, &acts, &mut out);
                        black_box(&out);
                    },
                );
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
                if rows == 4096 && ab == 32 {
                    // Short multi-token calls (chat turns, speculative
                    // chunks): tiled vs per-token row kernels.
                    for n in [2usize, 4, 5, 8, 16] {
                        let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
                        let mut acts = Q8Acts::new();
                        acts.quantize(&xs, n, cols, ab);
                        let mut out = vec![0.0f32; n * rows];
                        for k in [
                            GemmKernel::best_small(),
                            GemmKernel::best(),
                            GemmKernel::Rows(Isa::best()),
                        ] {
                            run_bw(
                                &filter,
                                &format!("q1_gemm_small {name} n={n} {k:?} (mt, x n)"),
                                100,
                                w.nbytes() * n,
                                || {
                                    gemm_with(k, &w, &acts, &mut out);
                                    black_box(&out);
                                },
                            );
                        }
                    }
                }
                if rows == 12288 && block == 128 {
                    let one = rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap();
                    for n in [64usize, 512] {
                        let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
                        let mut acts = Q8Acts::new();
                        acts.quantize(&xs, n, cols, ab);
                        let mut out = vec![0.0f32; n * rows];
                        let iters = if n > 64 { 5 } else { 20 };
                        run_bw(
                            &filter,
                            &format!("q1_gemm {name} act{ab} n={n} (mt, weight bytes x n)"),
                            iters,
                            w.nbytes() * n,
                            || {
                                gemm(&w, &acts, &mut out);
                                black_box(&out);
                            },
                        );
                        // Per-thread throughput of every multi-token kernel.
                        for k in GemmKernel::all_supported() {
                            if k == GemmKernel::Rows(Isa::Scalar) {
                                continue;
                            }
                            run_bw(
                                &filter,
                                &format!("q1_gemm {name} act{ab} n={n} {k:?} (1 thread, x n)"),
                                3,
                                w.nbytes() * n,
                                || {
                                    one.install(|| gemm_with(k, &w, &acts, &mut out));
                                    black_box(&out);
                                },
                            );
                        }
                    }
                }
            }
        }
        // Prefill GEMM at the projection shapes of 1.7B / 8B / 27B (n=256):
        // GMAC/s (1 MAC = one 1-bit weight x int8 activation product).
        {
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let n = 256usize;
            for &(name, rows, cols) in &[
                ("1.7B gate_up", 12288usize, 2048usize),
                ("1.7B down", 2048, 6144),
                ("8B gate_up", 24576, 4096),
                ("8B down", 4096, 12288),
                ("27B gate_up", 34816, 5120),
                ("27B down", 5120, 17408),
            ] {
                let w = q1(&mut rng, rows, cols, 128);
                let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
                let mut acts = Q8Acts::new();
                acts.quantize(&xs, n, cols, 32);
                let mut out = vec![0.0f32; n * rows];
                let macs = (rows * cols * n) as f64;
                let mut gmacs = |label: &str, iters: usize, f: &mut dyn FnMut(&mut [f32])| {
                    if let Some(flt) = &filter
                        && !label.contains(flt.as_str())
                    {
                        return;
                    }
                    f(&mut out);
                    let mut s: Vec<f64> = (0..iters)
                        .map(|_| {
                            let t = Instant::now();
                            f(&mut out);
                            t.elapsed().as_secs_f64()
                        })
                        .collect();
                    s.sort_by(f64::total_cmp);
                    println!(
                        "{label:<60} median {:>10.2} us   {:>7.1} GMAC/s",
                        s[s.len() / 2] * 1e6,
                        macs / s[s.len() / 2] / 1e9
                    );
                };
                gmacs(
                    &format!("q1_prefill_gemm {name} {rows}x{cols} n={n} (mt)"),
                    5,
                    &mut |o| gemm(&w, &acts, o),
                );
                if rows * cols <= 12288 * 2048 {
                    gmacs(
                        &format!("q1_prefill_gemm {name} {rows}x{cols} n={n} (1 thread)"),
                        3,
                        &mut |o| one.install(|| gemm(&w, &acts, o)),
                    );
                }
            }
        }
        // Small batches at Bonsai-27B shapes (speculative verification,
        // batched decode): `gemm` on T tokens over DRAM-resident weights
        // (gate_up: 6 x 34816x5120 = 140 MB; down: 8 x 5120x17408 = 90 MB)
        // and an L3-resident matrix (compute only); cost relative to T = 1.
        for &(name, rows, cols, mats) in &[
            ("27B gate_up DRAM", 34816usize, 5120usize, 6usize),
            ("27B down DRAM", 5120, 17408, 8),
            ("27B gate_up-slice L3", 4096, 5120, 1),
        ] {
            let label = format!("q1_small27 {name}");
            if filter.as_ref().is_some_and(|f| !label.contains(f.as_str())) {
                continue;
            }
            let ws: Vec<Q1Weight> = (0..mats).map(|_| q1(&mut rng, rows, cols, 128)).collect();
            let mut out = vec![0.0f32; 8 * rows];
            let mut t1 = 0.0f64;
            for t in 1..=8usize {
                let xs: Vec<f32> = (0..t * cols).map(|_| rng.f32()).collect();
                let mut acts = Q8Acts::new();
                acts.quantize(&xs, t, cols, 32);
                let reps = if mats == 1 { 400 } else { 15 };
                let mut sweep = || {
                    let st = Instant::now();
                    for w in &ws {
                        gemm(w, &acts, &mut out[..t * rows]);
                    }
                    st.elapsed().as_secs_f64()
                };
                sweep();
                let mut v: Vec<f64> = (0..reps).map(|_| sweep()).collect();
                v.sort_by(f64::total_cmp);
                let (mn, med) = (v[0], v[v.len() / 2]);
                if t == 1 {
                    t1 = mn;
                }
                println!(
                    "{label} T={t} min {:>8.1} us median {:>8.1} us  cost x{:.2} (min vs T=1)",
                    mn * 1e6,
                    med * 1e6,
                    mn / t1
                );
            }
        }
        // One Bonsai-1.7B layer's projections at prefill sizes (quantize +
        // GEMM of qkv, o, gate_up, down).
        {
            let shapes = [
                (4096usize, 2048usize),
                (2048, 2048),
                (12288, 2048),
                (2048, 6144),
            ];
            let ws: Vec<Q1Weight> = shapes
                .iter()
                .map(|&(r, c)| q1(&mut rng, r, c, 128))
                .collect();
            for n in [128usize, 512] {
                let xs: Vec<Vec<f32>> = shapes
                    .iter()
                    .map(|&(_, c)| (0..n * c).map(|_| rng.f32()).collect())
                    .collect();
                let mut outs: Vec<Vec<f32>> =
                    shapes.iter().map(|&(r, _)| vec![0.0; n * r]).collect();
                let mut acts = Q8Acts::new();
                let bytes: usize = ws.iter().map(|w| w.nbytes() * n).sum();
                run_bw(
                    &filter,
                    &format!("q1_layer 1.7B quantize+gemm x4 n={n} (mt, x n)"),
                    10,
                    bytes,
                    || {
                        for ((w, x), o) in ws.iter().zip(&xs).zip(outs.iter_mut()) {
                            acts.quantize(x, n, w.cols(), 32);
                            gemm(w, &acts, o);
                        }
                        black_box(&outs);
                    },
                );
                run(
                    &filter,
                    &format!("q1_layer 1.7B quantize only x4 n={n}"),
                    10,
                    || {
                        for (w, x) in ws.iter().zip(&xs) {
                            acts.quantize(x, n, w.cols(), 32);
                        }
                        black_box(&acts);
                    },
                );
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
        // Transparent huge pages: the same sweep with all weights in one
        // 2 MiB-aligned MADV_HUGEPAGE arena (`util::HugeBuf`, what
        // XORNET_HUGEPAGES=1 uses) vs the plain 4 KiB-page buffers, timed
        // alternately in one process so load drift hits both alike.
        let thp = "q1_gemv 1.7B 28-layer sweep THP A/B";
        if filter.as_ref().is_none_or(|f| thp.contains(f.as_str())) {
            use std::sync::Arc;
            let mut arena = xor_net::util::HugeBuf::new(total).unwrap();
            let mut off = 0;
            let mut spans = Vec::new();
            for w in layers.iter().flatten() {
                let (b, sc) = (w.bits().as_slice(), w.scales().as_slice());
                arena.as_mut_slice()[off..off + b.len()].copy_from_slice(b);
                arena.as_mut_slice()[off + b.len()..off + b.len() + sc.len()].copy_from_slice(sc);
                spans.push((off, b.len(), sc.len()));
                off += b.len() + sc.len();
            }
            let arena: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(arena);
            let huge: Vec<Q1Weight> = layers
                .iter()
                .flatten()
                .zip(&spans)
                .map(|(w, &(o, nb, ns))| {
                    Q1Weight::new(
                        w.rows(),
                        w.cols(),
                        w.block(),
                        ByteBuf::from_shared(arena.clone(), o, nb).unwrap(),
                        ByteBuf::from_shared(arena.clone(), o + nb, ns).unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            let plain: Vec<&Q1Weight> = layers.iter().flatten().collect();
            let mut sweep = |ws: &mut dyn Iterator<Item = &Q1Weight>| {
                let t = Instant::now();
                for w in ws {
                    let a = if w.cols() == 2048 { &a2k } else { &a6k };
                    gemv(w, a, &mut out[..w.rows()]);
                }
                t.elapsed().as_secs_f64()
            };
            let (mut tp, mut th) = (Vec::new(), Vec::new());
            for i in 0..60 {
                let p = sweep(&mut plain.iter().copied());
                let h = sweep(&mut huge.iter());
                if i >= 5 {
                    tp.push(p);
                    th.push(h);
                }
            }
            tp.sort_by(f64::total_cmp);
            th.sort_by(f64::total_cmp);
            let anon_huge = std::fs::read_to_string("/proc/self/smaps_rollup")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("AnonHugePages:"))
                        .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
                })
                .unwrap_or_else(|| "?".into());
            for (name, t) in [("4K pages", &tp), ("THP", &th)] {
                println!(
                    "{thp} {name:<8} min {:>8.1} us  median {:>8.1} us  {:.1} GB/s (min)",
                    t[0] * 1e6,
                    t[t.len() / 2] * 1e6,
                    total as f64 / t[0] / 1e9
                );
            }
            println!(
                "{thp} THP/4K: min x{:.3}, median x{:.3}; AnonHugePages {anon_huge} kB",
                th[0] / tp[0],
                th[th.len() / 2] / tp[tp.len() / 2]
            );
        }
        // Speculative verification / batched decode: T tokens through the
        // same DRAM-bound sweep with `gemm` (its default kernel for T).
        // "cost" is the time relative to T = 1.
        let verify = "q1_verify 1.7B 28-layer sweep";
        if filter.as_ref().is_none_or(|f| verify.contains(f.as_str())) {
            let mut t1 = 0.0f64;
            let mut cases: Vec<(usize, Option<GemmKernel>)> = (1..=9).map(|t| (t, None)).collect();
            for t in [9usize, 12, 16, 24, 32] {
                cases.push((t, Some(GemmKernel::best_small())));
                cases.push((t, Some(GemmKernel::best())));
            }
            for (t, kern) in cases {
                let xs: Vec<f32> = (0..t * 6144).map(|_| rng.f32()).collect();
                let (mut b2k, mut b6k) = (Q8Acts::new(), Q8Acts::new());
                b2k.quantize(&xs[..t * 2048], t, 2048, 32);
                b6k.quantize(&xs, t, 6144, 32);
                let mut o = vec![0.0f32; t * 12288];
                let mut sweep = || {
                    for l in &layers {
                        for w in l {
                            let a = if w.cols() == 2048 { &b2k } else { &b6k };
                            let o = &mut o[..t * w.rows()];
                            match kern {
                                Some(k) => gemm_with(k, w, a, o),
                                None => gemm(w, a, o),
                            }
                        }
                    }
                };
                for _ in 0..3 {
                    sweep();
                }
                let mut s: Vec<f64> = (0..15)
                    .map(|_| {
                        let st = Instant::now();
                        sweep();
                        st.elapsed().as_secs_f64()
                    })
                    .collect();
                s.sort_by(f64::total_cmp);
                let med = s[s.len() / 2];
                if t == 1 {
                    t1 = med;
                }
                let kname = kern.map_or("gemm".to_string(), |k| format!("{k:?}"));
                println!(
                    "{verify} T={t:<2} {kname:<18} median {:>9.1} us  cost x{:.2} (vs T=1)  {:.1} GB/s",
                    med * 1e6,
                    med / t1,
                    total as f64 / med / 1e9
                );
            }
        }
    }

    // ── End-to-end synthetic models (non-kernel code: attention, RoPE, ──
    // ── norms, SwiGLU, Gated DeltaNet, sampling) ──────────────────────────
    model_benches(&filter, &mut rng);

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

/// Random Q1_0 matrix (f16 scales around 2^-7) wrapped as a model `Linear`.
fn q1_linear(rng: &mut Rng, rows: usize, cols: usize, block: usize) -> xor_net::models::q1::Linear {
    use xor_net::bit1::q1_0::{ByteBuf, Q1Weight};
    let bits: Vec<u8> = (0..rows * cols / 8).map(|_| rng.byte()).collect();
    let scales: Vec<u8> = (0..rows * cols / block)
        .flat_map(|_| (0x2000u16 | (rng.byte() as u16)).to_le_bytes())
        .collect();
    xor_net::models::q1::Linear::Q1(
        Q1Weight::new(
            rows,
            cols,
            block,
            ByteBuf::from_vec(bits),
            ByteBuf::from_vec(scales),
        )
        .unwrap(),
    )
}

fn rand_vec(rng: &mut Rng, n: usize, center: f32, spread: f32) -> Vec<f32> {
    (0..n).map(|_| center + spread * rng.f32()).collect()
}

/// Times one decode step (1 token at position `ctx`) after a `ctx`-token
/// prefill, plus the prefill itself. Weights are random, so the numbers are
/// what a real model of the same shape costs without the page-cache effects.
fn bench_decode(
    filter: &Option<String>,
    name: &str,
    model: &dyn xor_net::models::traits::CausalLM,
    ctx: usize,
) {
    let wanted = |n: &str| filter.as_ref().is_none_or(|f| n.contains(f.as_str()));
    let prefill_name = format!("model {name} prefill {ctx}");
    let decode_name = format!("model {name} decode @ctx {ctx}");
    if !wanted(&prefill_name) && !wanted(&decode_name) {
        return;
    }
    let toks: Vec<u32> = (0..ctx as u32).map(|i| (i * 37 + 11) % 1000).collect();
    let mut st = model.new_state().unwrap();
    run(filter, &prefill_name, 3, || {
        st.reset().unwrap();
        black_box(model.forward(&toks, 0, st.as_mut()).unwrap());
    });
    st.reset().unwrap();
    model.forward(&toks, 0, st.as_mut()).unwrap();
    // Re-running position `ctx` overwrites the same KV slot each time.
    run(filter, &decode_name, 40, || {
        black_box(model.forward(&[7], ctx, st.as_mut()).unwrap());
    });
}

fn model_benches(filter: &Option<String>, rng: &mut Rng) {
    use xor_net::models::q1::Embedding;
    use xor_net::models::q1::qwen3::{Qwen3Config, Qwen3Layer, Qwen3Model};
    use xor_net::models::q1::qwen35::{
        AttnMixer, GdnMixer, Mixer, Qwen35Config, Qwen35Layer, Qwen35Model,
    };

    // Bonsai-1.7B (Qwen3) shape: 28 layers, 2048 hidden, 16/8 heads of 128.
    {
        let (h, inter, nh, nkv, hd, vocab) = (2048, 6144, 16, 8, 128, 151_669);
        let cfg = Qwen3Config {
            hidden: h,
            intermediate: inter,
            layers: 28,
            heads: nh,
            kv_heads: nkv,
            head_dim: hd,
            eps: 1e-6,
            rope_theta: 1_000_000.0,
            yarn: None,
            context_length: 4096,
            vocab,
            bos: None,
            eos: vec![1],
        };
        let layers = (0..cfg.layers)
            .map(|_| Qwen3Layer {
                attn_norm: rand_vec(rng, h, 1.0, 0.2),
                qkv: q1_linear(rng, (nh + 2 * nkv) * hd, h, 128),
                q_norm: rand_vec(rng, hd, 1.0, 0.2),
                k_norm: rand_vec(rng, hd, 1.0, 0.2),
                o: q1_linear(rng, h, nh * hd, 128),
                ffn_norm: rand_vec(rng, h, 1.0, 0.2),
                gate_up: q1_linear(rng, 2 * inter, h, 128),
                down: q1_linear(rng, h, inter, 128),
            })
            .collect();
        let embed = Embedding(q1_linear(rng, vocab, h, 128));
        let norm = rand_vec(rng, h, 1.0, 0.2);
        let m = Qwen3Model::new(cfg, embed, layers, norm, None).unwrap();
        bench_decode(filter, "qwen3-1.7B", &m, 128);
        bench_decode(filter, "qwen3-1.7B", &m, 2048);
    }

    // Qwen3.5-style hybrid (3 Gated DeltaNet : 1 attention), 8 layers of a
    // 4B-class shape so the recurrent path is measured without a big model.
    {
        let (h, inter, vocab) = (2560, 9216, 248_320);
        let cfg = Qwen35Config {
            hidden: h,
            intermediate: inter,
            layers: 8,
            recurrent: (0..8).map(|i| i % 4 != 3).collect(),
            heads: 16,
            kv_heads: 4,
            head_dim: 256,
            rot_dim: 64,
            eps: 1e-6,
            rope_theta: 10_000_000.0,
            yarn: None,
            context_length: 4096,
            vocab,
            bos: None,
            eos: vec![1],
            conv_kernel: 4,
            k_heads: 16,
            v_heads: 32,
            head_k_dim: 128,
            head_v_dim: 128,
        };
        let (cd, vd, nv) = (cfg.conv_dim(), cfg.value_dim(), cfg.v_heads);
        let qd = cfg.heads * cfg.head_dim;
        let kvd = cfg.kv_heads * cfg.head_dim;
        let layers = cfg
            .recurrent
            .clone()
            .into_iter()
            .map(|rec| {
                let mixer = if rec {
                    Mixer::Gdn(GdnMixer {
                        qkvz: q1_linear(rng, cd + vd, h, 128),
                        ba: q1_linear(rng, 2 * nv, h, 128),
                        conv: rand_vec(rng, cd * 4, 0.0, 0.5),
                        dt_bias: rand_vec(rng, nv, 0.0, 1.0),
                        ssm_a: rand_vec(rng, nv, -1.0, 0.5),
                        norm: rand_vec(rng, cfg.head_v_dim, 1.0, 0.2),
                        out: q1_linear(rng, h, vd, 128),
                    })
                } else {
                    Mixer::Attn(AttnMixer {
                        qkv: q1_linear(rng, 2 * qd + 2 * kvd, h, 128),
                        q_norm: rand_vec(rng, cfg.head_dim, 1.0, 0.2),
                        k_norm: rand_vec(rng, cfg.head_dim, 1.0, 0.2),
                        o: q1_linear(rng, h, qd, 128),
                    })
                };
                Qwen35Layer {
                    attn_norm: rand_vec(rng, h, 1.0, 0.2),
                    mixer,
                    post_norm: rand_vec(rng, h, 1.0, 0.2),
                    gate_up: q1_linear(rng, 2 * inter, h, 128),
                    down: q1_linear(rng, h, inter, 128),
                }
            })
            .collect();
        let embed = Embedding(q1_linear(rng, vocab, h, 128));
        let norm = rand_vec(rng, h, 1.0, 0.2);
        let m = Qwen35Model::new(cfg, embed, layers, norm, None).unwrap();
        bench_decode(filter, "qwen35-hybrid-8L", &m, 128);
    }

    // Pieces of one decode step (Bonsai-1.7B shapes).
    {
        use xor_net::bit1::q1_0::Q8Acts;
        use xor_net::models::q1::{KvCache, attention};
        for &(cols, ab) in &[(2048usize, 32usize), (6144, 32), (2048, 128)] {
            let x = rand_vec(rng, cols, 0.0, 2.0);
            let mut acts = Q8Acts::new();
            run(
                filter,
                &format!("q8acts quantize n=1x{cols} act{ab}"),
                20000,
                || {
                    acts.quantize(black_box(&x), 1, cols, ab);
                    black_box(&acts);
                },
            );
        }
        let (nh, nkv, hd) = (16usize, 8usize, 128usize);
        for &ctx in &[128usize, 2048] {
            let mut kv = KvCache::new(nkv, hd, 4096);
            kv.reserve(ctx + 1, 0).unwrap();
            for p in 0..=ctx {
                for h in 0..nkv {
                    let k = rand_vec(rng, hd, 0.0, 1.0);
                    let v = rand_vec(rng, hd, 0.0, 1.0);
                    kv.store(h, p, &k, &v);
                }
            }
            let q = rand_vec(rng, nh * hd, 0.0, 1.0);
            let mut out = vec![0.0f32; nh * hd];
            run(
                filter,
                &format!("q1 attention decode ctx={ctx} (16/8 heads)"),
                2000,
                || {
                    attention(black_box(&q), 1, ctx, nh, &kv, 0.088, &mut out);
                    black_box(&out);
                },
            );
        }
        // Causal prefill of n tokens from position 0 (one layer).
        for &n in &[512usize, 2048] {
            let mut kv = KvCache::new(nkv, hd, 4096);
            kv.reserve(n, 0).unwrap();
            for p in 0..n {
                for h in 0..nkv {
                    let k = rand_vec(rng, hd, 0.0, 1.0);
                    let v = rand_vec(rng, hd, 0.0, 1.0);
                    kv.store(h, p, &k, &v);
                }
            }
            let q = rand_vec(rng, n * nh * hd, 0.0, 1.0);
            let mut out = vec![0.0f32; n * nh * hd];
            run(
                filter,
                &format!("q1 attention prefill n={n} (16/8 heads)"),
                if n > 512 { 5 } else { 20 },
                || {
                    attention(black_box(&q), n, 0, nh, &kv, 0.088, &mut out);
                    black_box(&out);
                },
            );
        }
    }

    // Sampling over a Qwen-sized vocabulary.
    {
        use xor_net::sampler::{Sampler, SamplingParams};
        let logits: Vec<f32> = (0..151_669).map(|_| rng.f32() * 8.0).collect();
        let ctx: Vec<u32> = (0..256).collect();
        let mut buf = logits.clone();
        let mut s = Sampler::from_params(SamplingParams::greedy());
        run(filter, "sampler greedy vocab 151669", 2000, || {
            buf.copy_from_slice(&logits);
            black_box(s.sample(&mut buf, &ctx).unwrap());
        });
        let p = SamplingParams::default()
            .with_temperature(0.7)
            .with_top_k(40)
            .with_top_p(0.9)
            .with_repetition_penalty(1.1)
            .with_seed(Some(1));
        let mut s = Sampler::from_params(p);
        run(
            filter,
            "sampler top-k/top-p/penalty vocab 151669",
            2000,
            || {
                buf.copy_from_slice(&logits);
                black_box(s.sample(&mut buf, &ctx).unwrap());
            },
        );
    }
}
