//! Equivalence tests for the Q1_0 (1-bit) GEMV/GEMM kernels: every ISA
//! variant supported by the CPU against the scalar reference, for both block
//! sizes, odd row counts, row offsets and column counts that exercise the
//! 32-element tail of the AVX-512 kernel. All kernels follow the canonical
//! operation order of `dot_rows_scalar`, so results must be bit-identical
//! (independent of ISA, row grouping and the number of tokens per call).

use xor_net::bit1::q1_0::{
    ActRow, GemmKernel, Isa, Q1Weight, Q8Acts, TileIsa, dot_rows_isa, dot_rows_scalar, gemm,
    gemm_with, gemv, perm32,
};
use xor_net::loader::gguf::{PRISM_Q1_0, PRISM_Q1_0_G128, PRISM_Q1_0_UPSTREAM};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random GGUF Q1 tensor with finite scales (including a few zero and
/// subnormal scales).
fn random_weight(rng: &mut Rng, dtype: u32, rows: usize, cols: usize) -> Q1Weight {
    let (block, bbytes, soff) = match dtype {
        PRISM_Q1_0_UPSTREAM => (128, 18, 0),
        PRISM_Q1_0 => (32, 6, 0),
        _ => (128, 18, 16),
    };
    let mut data: Vec<u8> = (0..rows * cols / block * bbytes)
        .map(|_| rng.next() as u8)
        .collect();
    for blk in data.chunks_exact_mut(bbytes) {
        let r = rng.next();
        let h: u16 = if r.is_multiple_of(17) {
            0
        } else if r.is_multiple_of(13) {
            // f16 subnormal scale (exponent field 0).
            ((r >> 16) as u16 & 0x3ff).max(1) | (((r >> 30) & 1) as u16) << 15
        } else {
            ((((r >> 8) % 12 + 8) as u16) << 10)
                | ((r >> 16) as u16 & 0x3ff)
                | (((r >> 30) & 1) as u16) << 15
        };
        blk[soff..soff + 2].copy_from_slice(&h.to_le_bytes());
    }
    Q1Weight::from_gguf(&data, dtype, rows, cols).unwrap()
}

fn random_acts(rng: &mut Rng, n: usize, cols: usize, block: usize) -> (Vec<f32>, Q8Acts) {
    let mut x: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
    // Outliers and an all-zero block.
    for (i, v) in x.iter_mut().enumerate() {
        if i % 97 == 5 {
            *v *= 50.0;
        }
        if i < 32 {
            *v = 0.0;
        }
    }
    let mut a = Q8Acts::new();
    a.quantize(&x, n, cols, block);
    (x, a)
}

/// Bit-for-bit equality (the kernels share one operation order).
fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.to_bits() == w.to_bits(), "{what}[{i}]: got {g}, want {w}");
    }
}

const SHAPES: &[(usize, usize)] = &[
    (1, 128),
    (3, 256),
    (5, 384),
    (7, 640),
    (13, 2048),
    (4, 1024),
    (9, 5120),
];
const B32_ONLY_COLS: &[usize] = &[32, 96, 160, 544];

#[test]
fn every_isa_matches_scalar() {
    let mut rng = Rng(42);
    let isas = Isa::all_supported();
    eprintln!("testing Q1 ISAs: {isas:?}");
    for dtype in [PRISM_Q1_0_UPSTREAM, PRISM_Q1_0, PRISM_Q1_0_G128] {
        let mut shapes = SHAPES.to_vec();
        if dtype == PRISM_Q1_0 {
            shapes.extend(B32_ONLY_COLS.iter().map(|&c| (5, c)));
        }
        for &(rows, cols) in &shapes {
            let w = random_weight(&mut rng, dtype, rows, cols);
            for ab in [32, 128]
                .into_iter()
                .filter(|ab| w.block().is_multiple_of(*ab))
            {
                let (_, acts) = random_acts(&mut rng, 1, cols, ab);
                let a: ActRow = acts.row(0);
                let mut want = vec![0.0; rows];
                dot_rows_scalar(&w, 0, a, &mut want);
                for &isa in &isas {
                    // Whole matrix, and every (offset, count) window.
                    let mut got = vec![0.0; rows];
                    dot_rows_isa(isa, &w, 0, a, &mut got);
                    assert_same(&got, &want, &format!("{isa:?} dtype {dtype} {rows}x{cols}"));
                    for r0 in 0..rows.min(6) {
                        for cnt in 1..=(rows - r0).min(6) {
                            let mut g = vec![0.0; cnt];
                            dot_rows_isa(isa, &w, r0, a, &mut g);
                            assert_same(
                                &g,
                                &want[r0..r0 + cnt],
                                &format!("{isa:?} window {r0}+{cnt}"),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gemv_and_gemm_match_scalar() {
    let mut rng = Rng(7);
    for dtype in [PRISM_Q1_0_UPSTREAM, PRISM_Q1_0] {
        for &(rows, cols) in &[(1usize, 128usize), (37, 256), (515, 512), (2049, 384)] {
            let w = random_weight(&mut rng, dtype, rows, cols);
            for (n, ab) in [(1usize, 32usize), (2, 128), (5, 32), (17, 128)] {
                if !w.block().is_multiple_of(ab) {
                    continue;
                }
                let (_, acts) = random_acts(&mut rng, n, cols, ab);
                let mut want = vec![0.0; n * rows];
                for t in 0..n {
                    dot_rows_scalar(&w, 0, acts.row(t), &mut want[t * rows..(t + 1) * rows]);
                }
                let mut got = vec![0.0; n * rows];
                gemm(&w, &acts, &mut got);
                assert_same(&got, &want, &format!("gemm {rows}x{cols} n={n}"));
                if n == 1 {
                    let mut g = vec![0.0; rows];
                    gemv(&w, &acts, &mut g);
                    assert_same(&g, &want, "gemv");
                }
            }
        }
    }
}

/// Every multi-token kernel (row kernels and token-tiled kernels of every
/// supported ISA) against the scalar reference: odd row counts (partial
/// slabs of 8 / 16 lanes), token counts that are not multiples of the token
/// tile (including 1), both weight block sizes and both activation blocks.
#[test]
fn every_gemm_kernel_matches_scalar() {
    let mut rng = Rng(11);
    let kernels = GemmKernel::all_supported();
    eprintln!("testing Q1 GEMM kernels: {kernels:?}");
    let shapes: &[(u32, usize, usize)] = &[
        (PRISM_Q1_0_UPSTREAM, 1, 128),
        (PRISM_Q1_0_UPSTREAM, 7, 256),
        (PRISM_Q1_0_UPSTREAM, 17, 384),
        (PRISM_Q1_0_UPSTREAM, 33, 1152),
        (PRISM_Q1_0_G128, 24, 512),
        (PRISM_Q1_0, 9, 32),
        (PRISM_Q1_0, 15, 96),
        (PRISM_Q1_0, 16, 160),
        (PRISM_Q1_0, 31, 544),
        (PRISM_Q1_0, 40, 2048),
    ];
    for &(dtype, rows, cols) in shapes {
        let w = random_weight(&mut rng, dtype, rows, cols);
        for ab in [32, 128]
            .into_iter()
            .filter(|ab| w.block().is_multiple_of(*ab))
        {
            for n in [1usize, 2, 3, 4, 5, 8, 9, 13, 17] {
                let (_, acts) = random_acts(&mut rng, n, cols, ab);
                let mut want = vec![0.0; n * rows];
                for t in 0..n {
                    dot_rows_scalar(&w, 0, acts.row(t), &mut want[t * rows..(t + 1) * rows]);
                }
                for &k in &kernels {
                    let mut got = vec![f32::NAN; n * rows];
                    gemm_with(k, &w, &acts, &mut got);
                    assert_same(
                        &got,
                        &want,
                        &format!("{k:?} dtype {dtype} {rows}x{cols} act{ab} n={n}"),
                    );
                }
            }
        }
    }
}

#[test]
fn tiled_gemm_parallel_split_matches_scalar() {
    // Enough rows and tokens for several slabs per task and several token
    // blocks (> 256 tokens).
    let mut rng = Rng(5);
    let (rows, cols) = (203, 256);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, rows, cols);
    let n = 530;
    let (_, acts) = random_acts(&mut rng, n, cols, 32);
    let mut want = vec![0.0; n * rows];
    for t in 0..n {
        dot_rows_scalar(&w, 0, acts.row(t), &mut want[t * rows..(t + 1) * rows]);
    }
    for t in [
        TileIsa::Avx2,
        TileIsa::AvxVnni,
        TileIsa::Avx512Bw,
        TileIsa::Avx512Vnni,
    ] {
        if !t.supported() {
            continue;
        }
        let mut got = vec![f32::NAN; n * rows];
        gemm_with(GemmKernel::Tiled(t), &w, &acts, &mut got);
        assert_same(&got, &want, &format!("{t:?} split"));
    }
}

#[test]
fn activation_side_data_is_consistent() {
    let mut rng = Rng(9);
    for ab in [32, 128] {
        let (_, acts) = random_acts(&mut rng, 3, 384, ab);
        for t in 0..3 {
            let a = acts.row(t);
            for k in 0..384 / 32 {
                let blk = &a.q[32 * k..32 * (k + 1)];
                for (e, &v) in blk.iter().enumerate() {
                    assert_eq!(a.qp[32 * k + perm32(e)], v);
                }
                let s: i32 = blk.iter().map(|&v| v as i32).sum();
                assert_eq!(a.nsum[k], -s);
            }
        }
    }
}

/// `Q8Acts::quantize` (SIMD path where available) against a scalar
/// statement of the block quantizer, bit for bit: zero blocks, ties at
/// .5, NaN and infinite elements, tiny and huge magnitudes.
#[test]
fn quantize_matches_reference_bitwise() {
    let mut rng = Rng(21);
    for block in [32usize, 128] {
        let cols = 1024;
        let n = 3;
        let mut x: Vec<f32> = (0..n * cols).map(|_| rng.f32() * 3.0).collect();
        // Block 0: zeros; block 1: exact ties (amax 127 -> id 1.0);
        // block 2: a NaN; block 3: an infinity; block 4: tiny values.
        x[..block].fill(0.0);
        for (i, v) in x[block..2 * block].iter_mut().enumerate() {
            *v = (i as f32 - 64.0) + 0.5;
        }
        x[block] = 127.0;
        x[2 * block + 3] = f32::NAN;
        x[3 * block + 1] = f32::INFINITY;
        x[4 * block..5 * block].iter_mut().for_each(|v| *v *= 1e-30);
        x[5 * block + 7] = -0.0;
        let mut a = Q8Acts::new();
        a.quantize(&x, n, cols, block);
        for t in 0..n {
            let r = a.row(t);
            for (b, xb) in x[t * cols..(t + 1) * cols].chunks_exact(block).enumerate() {
                let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let (want_d, want_q): (f32, Vec<i8>) = if amax == 0.0 || !amax.is_finite() {
                    (if amax == 0.0 { 0.0 } else { f32::NAN }, vec![0; block])
                } else {
                    let id = 127.0 / amax;
                    let q = xb
                        .iter()
                        .map(|&v| {
                            let r = (v * id).round_ties_even();
                            if r.is_nan() {
                                0
                            } else {
                                r.clamp(-127.0, 127.0) as i8
                            }
                        })
                        .collect();
                    (amax / 127.0, q)
                };
                assert_eq!(r.d[b].to_bits(), want_d.to_bits(), "d block {b} row {t}");
                assert_eq!(
                    &r.q[b * block..(b + 1) * block],
                    &want_q[..],
                    "q block {b} row {t}"
                );
            }
        }
    }
}

#[test]
#[should_panic]
fn gemm_rejects_wrong_output_length() {
    let mut rng = Rng(1);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, 4, 256);
    let (_, acts) = random_acts(&mut rng, 3, 256, 32);
    let mut out = vec![0.0; 11];
    gemm_with(GemmKernel::best(), &w, &acts, &mut out);
}

#[test]
#[should_panic]
fn gemm_rejects_wrong_activation_cols() {
    let mut rng = Rng(1);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, 4, 256);
    let (_, acts) = random_acts(&mut rng, 3, 128, 32);
    let mut out = vec![0.0; 12];
    gemm_with(GemmKernel::best(), &w, &acts, &mut out);
}

#[test]
fn quantized_result_tracks_f32_reference() {
    // Error of int8 block activations vs the exact f32 product.
    let mut rng = Rng(3);
    let (rows, cols) = (64, 2048);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, rows, cols);
    let (x, acts) = random_acts(&mut rng, 1, cols, 32);
    let mut got = vec![0.0; rows];
    gemv(&w, &acts, &mut got);
    let mut row = vec![0.0; cols];
    for (r, &g) in got.iter().enumerate() {
        w.dequantize_row_into(r, &mut row);
        let exact: f64 = row.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let l1: f64 = row
            .iter()
            .zip(&x)
            .map(|(a, b)| (*a as f64 * *b as f64).abs())
            .sum();
        assert!(
            (g as f64 - exact).abs() <= 1e-2 * l1,
            "row {r}: {g} vs {exact} (l1 {l1})"
        );
    }
}

#[test]
#[should_panic]
fn kernel_rejects_rows_out_of_range() {
    let mut rng = Rng(1);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, 4, 128);
    let (_, acts) = random_acts(&mut rng, 1, 128, 128);
    let mut out = vec![0.0; 3];
    dot_rows_isa(Isa::best(), &w, 2, acts.row(0), &mut out);
}

#[test]
#[should_panic]
fn kernel_rejects_wrong_activation_length() {
    let mut rng = Rng(1);
    let w = random_weight(&mut rng, PRISM_Q1_0_UPSTREAM, 4, 256);
    let (_, acts) = random_acts(&mut rng, 1, 128, 128);
    let mut out = vec![0.0; 4];
    dot_rows_isa(Isa::best(), &w, 0, acts.row(0), &mut out);
}
