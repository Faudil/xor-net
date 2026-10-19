//! Equivalence tests for the Q1_0 (1-bit) GEMV/GEMM kernels: every ISA
//! variant supported by the CPU against the scalar reference, for both block
//! sizes, odd row counts, row offsets and column counts that exercise the
//! 32-element tail of the AVX-512 kernel.

use xor_net::bit1::q1_0::{
    ActRow, Isa, Q1Weight, Q8Acts, dot_rows_isa, dot_rows_scalar, gemm, gemv,
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

/// Random GGUF Q1 tensor with finite scales (including a few zero scales).
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

/// Σ_c |W[r, c] · x̂[c]| per row (x̂ = dequantized activations): the scale
/// of the rounding error of any summation order.
fn l1_rows(w: &Q1Weight, acts: &Q8Acts, t: usize, r0: usize, n: usize) -> Vec<f32> {
    let cols = w.cols();
    let a = acts.row(t);
    let mut row = vec![0.0; cols];
    (r0..r0 + n)
        .map(|r| {
            w.dequantize_row_into(r, &mut row);
            (0..cols)
                .map(|c| (row[c] * a.q[c] as f32 * a.d[c / a.block]).abs())
                .sum()
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], l1: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 2e-6 * l1[i] + 1e-6;
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g}, want {w}");
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
                let l1 = l1_rows(&w, &acts, 0, 0, rows);
                for &isa in &isas {
                    // Whole matrix, and every (offset, count) window.
                    let mut got = vec![0.0; rows];
                    dot_rows_isa(isa, &w, 0, a, &mut got);
                    assert_close(
                        &got,
                        &want,
                        &l1,
                        &format!("{isa:?} dtype {dtype} {rows}x{cols}"),
                    );
                    for r0 in 0..rows.min(6) {
                        for cnt in 1..=(rows - r0).min(6) {
                            let mut g = vec![0.0; cnt];
                            dot_rows_isa(isa, &w, r0, a, &mut g);
                            assert_close(
                                &g,
                                &want[r0..r0 + cnt],
                                &l1[r0..r0 + cnt],
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
                let l1: Vec<f32> = (0..n)
                    .flat_map(|t| l1_rows(&w, &acts, t, 0, rows))
                    .collect();
                for t in 0..n {
                    dot_rows_scalar(&w, 0, acts.row(t), &mut want[t * rows..(t + 1) * rows]);
                }
                let mut got = vec![0.0; n * rows];
                gemm(&w, &acts, &mut got);
                assert_close(&got, &want, &l1, &format!("gemm {rows}x{cols} n={n}"));
                if n == 1 {
                    let mut g = vec![0.0; rows];
                    gemv(&w, &acts, &mut g);
                    assert_close(&g, &want, &l1, "gemv");
                }
            }
        }
    }
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
