//! Scalar-reference equivalence tests for every SIMD kernel.
//!
//! Each ISA variant is called *directly* (behind a runtime feature check, so a
//! machine without e.g. AVX-512 skips that variant rather than crashing) and
//! compared against an independent scalar reference over odd lengths, lengths
//! that are not multiples of 16/32/64, length 0/1, and edge values. Integer
//! kernels must match exactly; f32 kernels within a tight tolerance.

use xor_net::bit1_58::quantization::{
    pack_1_58bit_5pack, quantize_f32_to_i8, quantize_f32_to_i8_scalar, silu_inplace,
    silu_inplace_scalar,
};
use xor_net::bit1_58::simd::{
    ternary_dot_product_pack4, ternary_dot_product_pack4_scalar, ternary_dot_product_pack5,
    ternary_dot_product_sparse, ternary_dot_product_sparse_scalar,
};
use xor_net::bit1_58::sparse::{SparseTernary, encode_sparse_tensor};
use xor_net::nn::dynamic_linear::{dot_product_i8, dot_product_i8_scalar, dot_product_u8_zp_i8};
use xor_net::nn::fast_attention_simd as fa;

// ───────────────────────────── helpers ─────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    /// Uniform in [-1, 1).
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    /// Activation in [-127, 127], biased towards the extremes.
    fn act(&mut self) -> i8 {
        match self.below(8) {
            0 => 127,
            1 => -127,
            2 => 0,
            _ => (self.below(255) as i32 - 127) as i8,
        }
    }
}

/// Lengths that hit every tail/remainder path of 8/16/32/64/1024-wide loops.
const LENS: &[usize] = &[
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    1023,
    1024,
    1025,
    1088,
    2047,
    2560,
    2561,
    4096 + 64 * 3 + 7,
];

#[cfg(target_arch = "x86_64")]
fn has(feats: &[&str]) -> bool {
    feats.iter().all(|f| match *f {
        "avx2" => is_x86_feature_detected!("avx2"),
        "fma" => is_x86_feature_detected!("fma"),
        "avx512f" => is_x86_feature_detected!("avx512f"),
        "avx512bw" => is_x86_feature_detected!("avx512bw"),
        "avx512vnni" => is_x86_feature_detected!("avx512vnni"),
        "avx" => is_x86_feature_detected!("avx"),
        "f16c" => is_x86_feature_detected!("f16c"),
        other => panic!("unknown feature {other}"),
    })
}

fn assert_close(got: f32, want: f32, mag: f32, rel: f32, ctx: &str) {
    let tol = rel * mag.max(want.abs()) + 1e-6;
    assert!(
        (got - want).abs() <= tol || (got.is_nan() && want.is_nan()),
        "{ctx}: got {got}, want {want} (tol {tol})"
    );
}

/// Independent pack4 reference: decode each code with a match on the value.
fn pack4_reference(a: &[i8], w: &[u8], n: usize) -> i32 {
    (0..n)
        .map(|i| {
            let code = (w[i / 4] >> ((i % 4) * 2)) & 3;
            let wv = match code {
                0 => -1,
                2 => 1,
                _ => 0,
            };
            a[i] as i32 * wv
        })
        .sum()
}

// ───────────────────────── ternary pack4 ─────────────────────────

/// Weight generators: uniform random bytes (all four codes, including the
/// unused 0b11), all -1, all +1.
fn weight_patterns(rng: &mut Rng, nbytes: usize) -> Vec<Vec<u8>> {
    vec![
        (0..nbytes).map(|_| rng.byte()).collect(),
        vec![0b0000_0000; nbytes],
        vec![0b1010_1010; nbytes],
    ]
}

fn act_patterns(rng: &mut Rng, n: usize) -> Vec<Vec<i8>> {
    vec![
        (0..n).map(|_| rng.act()).collect(),
        vec![127; n],
        vec![-127; n],
    ]
}

#[test]
fn pack4_all_variants_match_reference() {
    let mut rng = Rng(1);
    for &n in LENS {
        let nbytes = n.div_ceil(4);
        for w in weight_patterns(&mut rng, nbytes) {
            for a in act_patterns(&mut rng, n) {
                let want = pack4_reference(&a, &w, n);
                assert_eq!(
                    ternary_dot_product_pack4_scalar(&a, &w, n),
                    want,
                    "scalar n={n}"
                );
                assert_eq!(ternary_dot_product_pack4(&a, &w, n), want, "dispatch n={n}");

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use xor_net::bit1_58::lut::*;
                    use xor_net::bit1_58::simd::*;
                    if has(&["avx512f", "avx512bw", "avx512vnni"]) {
                        assert_eq!(
                            ternary_dot_product_pack4_avx512_vnni(&a, &w, n),
                            want,
                            "vnni n={n}"
                        );
                    }
                    if has(&["avx512f", "avx512bw"]) {
                        assert_eq!(
                            ternary_dot_product_pack4_avx512(&a, &w, n),
                            want,
                            "avx512 n={n}"
                        );
                    }
                    if has(&["avx2"]) {
                        assert_eq!(
                            ternary_dot_product_pack4_avx2(&a, &w, n),
                            want,
                            "avx2 n={n}"
                        );
                        assert_eq!(
                            ternary_dot_product_pack4_avx2_hybrid(&a, &w, n),
                            want,
                            "avx2 hybrid n={n}"
                        );
                        assert_eq!(
                            ternary_dot_product_pack4_avx2_lut(&a, &w, n),
                            want,
                            "avx2 lut n={n}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pack4_uses_only_first_total_elems() {
    // Longer slices than `total_elems`: trailing data must be ignored.
    let mut rng = Rng(2);
    for &n in &[0usize, 1, 33, 64, 100, 1030] {
        let a: Vec<i8> = (0..n + 200).map(|_| rng.act()).collect();
        let w: Vec<u8> = (0..n.div_ceil(4) + 50).map(|_| rng.byte()).collect();
        let want = pack4_reference(&a, &w, n);
        assert_eq!(ternary_dot_product_pack4(&a, &w, n), want, "n={n}");
    }
}

#[test]
#[should_panic(expected = "weight slice too short")]
fn pack4_rejects_short_weights() {
    let a = vec![1i8; 128];
    let w = vec![0u8; 31]; // needs 32
    ternary_dot_product_pack4(&a, &w, 128);
}

#[test]
#[should_panic(expected = "activation slice too short")]
fn pack4_rejects_short_activations() {
    let a = vec![1i8; 127];
    let w = vec![0u8; 32];
    ternary_dot_product_pack4(&a, &w, 128);
}

// ───────────────────────── ternary pack5 ─────────────────────────

#[test]
fn pack5_matches_reference() {
    let mut rng = Rng(3);
    for &n in LENS {
        let w_f: Vec<f32> = (0..n).map(|_| rng.below(3) as f32 - 1.0).collect();
        let packed = pack_1_58bit_5pack(&w_f, 1.0);
        for a in act_patterns(&mut rng, n) {
            let want: i32 = a.iter().zip(&w_f).map(|(&x, &w)| x as i32 * w as i32).sum();
            assert_eq!(ternary_dot_product_pack5(&a, &packed, n), want, "n={n}");
        }
    }
}

// ───────────────────────── sparse ternary ─────────────────────────

#[test]
fn sparse_all_variants_match_dense_reference() {
    let mut rng = Rng(4);
    for &in_dim in &[1usize, 5, 63, 64, 65, 127, 128, 200, 1000, 2560] {
        let out_dim = 3;
        let w: Vec<i8> = (0..in_dim * out_dim)
            .map(|_| rng.below(3) as i8 - 1)
            .collect();
        let (blob, _) = encode_sparse_tensor(&w, in_dim, out_dim);
        let st = SparseTernary::from_blob(blob, out_dim);
        for a in act_patterns(&mut rng, in_dim) {
            for r in 0..out_dim {
                let want: i32 = a
                    .iter()
                    .zip(&w[r * in_dim..(r + 1) * in_dim])
                    .map(|(&x, &y)| x as i32 * y as i32)
                    .sum();
                let (m, s) = st.blocks(r);
                let nb = st.num_blocks;
                assert_eq!(
                    ternary_dot_product_sparse_scalar(&a, m, s, nb, in_dim),
                    want,
                    "scalar in_dim={in_dim}"
                );
                assert_eq!(
                    ternary_dot_product_sparse(&a, m, s, nb, in_dim),
                    want,
                    "dispatch in_dim={in_dim}"
                );
                #[cfg(target_arch = "x86_64")]
                if has(&["avx512f", "avx512bw"]) {
                    let got = unsafe {
                        xor_net::bit1_58::simd::ternary_dot_product_sparse_avx512(
                            &a, m, s, nb, in_dim,
                        )
                    };
                    assert_eq!(got, want, "avx512 in_dim={in_dim}");
                }
            }
        }
    }
}

#[test]
fn sparse_ignores_mask_bits_past_in_dim() {
    // Garbage mask bits beyond `in_dim` in the final block must not be read.
    let in_dim = 70;
    let a: Vec<i8> = (0..in_dim).map(|i| (i as i8) - 30).collect();
    let masks = [u64::MAX, u64::MAX];
    let signs = [0u64, 0];
    let want: i32 = a.iter().map(|&x| x as i32).sum();
    assert_eq!(
        ternary_dot_product_sparse_scalar(&a, &masks, &signs, 2, in_dim),
        want
    );
    assert_eq!(
        ternary_dot_product_sparse(&a, &masks, &signs, 2, in_dim),
        want
    );
}

#[test]
#[should_panic(expected = "num_blocks must be ceil")]
fn sparse_rejects_wrong_block_count() {
    let a = vec![1i8; 64];
    let masks = [0u64; 3];
    ternary_dot_product_sparse(&a, &masks, &masks, 3, 64);
}

// ───────────────────────────── xnor ─────────────────────────────

#[test]
fn xnor_all_variants_match_reference() {
    use xor_net::bit1::simd::{xnor_dot_product, xnor_dot_product_scalar};
    let mut rng = Rng(5);
    for &n in LENS {
        let a_bits: Vec<bool> = (0..n).map(|_| rng.below(2) == 1).collect();
        let b_bits: Vec<bool> = (0..n).map(|_| rng.below(2) == 1).collect();
        let pack = |bits: &[bool], rng: &mut Rng| -> Vec<u8> {
            // Fill padding bits with garbage: they must be ignored.
            let mut v: Vec<u8> = (0..n.div_ceil(8)).map(|_| rng.byte()).collect();
            for (i, &b) in bits.iter().enumerate() {
                if b {
                    v[i / 8] |= 1 << (i % 8);
                } else {
                    v[i / 8] &= !(1 << (i % 8));
                }
            }
            v
        };
        let a = pack(&a_bits, &mut rng);
        let b = pack(&b_bits, &mut rng);
        let want: f32 = a_bits
            .iter()
            .zip(&b_bits)
            .map(|(&x, &y)| if x == y { 1.0 } else { -1.0 })
            .sum();
        assert_eq!(xnor_dot_product_scalar(&a, &b, n), want, "scalar n={n}");
        assert_eq!(xnor_dot_product(&a, &b, n), want, "dispatch n={n}");
        #[cfg(target_arch = "x86_64")]
        if has(&["avx2"]) {
            let got = unsafe { xor_net::bit1::simd::xnor_dot_product_avx2(&a, &b, n) };
            assert_eq!(got, want, "avx2 n={n}");
        }
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn xnor_rejects_mismatched_lengths() {
    xor_net::bit1::simd::xnor_dot_product(&[0u8; 4], &[0u8; 3], 32);
}

#[test]
#[should_panic(expected = "ceil(total_bits/8)")]
fn xnor_rejects_too_many_bits() {
    // Used to underflow `a.len()*8 - total_bits`.
    xor_net::bit1::simd::xnor_dot_product(&[0u8; 2], &[0u8; 2], 17);
}

// ───────────────────────────── int8 dot ─────────────────────────────

#[test]
fn dot_i8_all_variants_match_reference() {
    let mut rng = Rng(6);
    for &n in LENS {
        let patterns: Vec<(Vec<i8>, Vec<i8>)> = vec![
            (
                (0..n).map(|_| rng.byte() as i8).collect(),
                (0..n).map(|_| rng.byte() as i8).collect(),
            ),
            (vec![-128; n], vec![-128; n]),
            (vec![-128; n], vec![127; n]),
            (vec![127; n], vec![127; n]),
        ];
        for (a, b) in patterns {
            let want: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
            assert_eq!(dot_product_i8_scalar(&a, &b), want, "scalar n={n}");
            assert_eq!(dot_product_i8(&a, &b), want, "dispatch n={n}");

            // VNNI path with the +128 zero-point correction.
            let a_u8: Vec<u8> = a.iter().map(|&x| (x as u8) ^ 0x80).collect();
            let row_sum: i32 = b.iter().map(|&x| x as i32).sum();
            assert_eq!(dot_product_u8_zp_i8(&a_u8, &b, row_sum), want, "zp n={n}");

            #[cfg(target_arch = "x86_64")]
            unsafe {
                use xor_net::nn::dynamic_linear::*;
                if has(&["avx2"]) {
                    assert_eq!(dot_product_i8_avx2(&a, &b), want, "avx2 n={n}");
                }
                if has(&["avx512f", "avx512bw"]) {
                    assert_eq!(dot_product_i8_avx512(&a, &b), want, "avx512 n={n}");
                }
                if has(&["avx512f", "avx512bw", "avx512vnni"]) {
                    assert_eq!(
                        dot_product_i8_avx512_vnni(&a_u8, &b, row_sum),
                        want,
                        "vnni n={n}"
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn dot_i8_rejects_mismatched_lengths() {
    dot_product_i8(&[1i8; 64], &[1i8; 63]);
}

// ───────────────────────────── f32 dot ─────────────────────────────

#[test]
fn dot_f32_all_variants_match_reference() {
    let mut rng = Rng(7);
    for &n in LENS {
        let a: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
        let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let mag: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
        let want = want as f32;
        let ctx = format!("n={n}");
        assert_close(fa::dot_product_f32_scalar(&a, &b), want, mag, 1e-5, &ctx);
        assert_close(fa::dot_product_f32(&a, &b), want, mag, 1e-5, &ctx);
        assert_close(
            xor_net::nn::dynamic_linear::dot_product_f32(&a, &b),
            want,
            mag,
            1e-5,
            &ctx,
        );
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has(&["avx2", "fma"]) {
                assert_close(fa::dot_product_f32_avx2(&a, &b), want, mag, 1e-5, &ctx);
                assert_close(
                    xor_net::nn::dynamic_linear::dot_product_f32_avx2(&a, &b),
                    want,
                    mag,
                    1e-5,
                    &ctx,
                );
            }
            if has(&["avx512f"]) {
                assert_close(fa::dot_product_f32_avx512(&a, &b), want, mag, 1e-5, &ctx);
            }
        }
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn dot_f32_rejects_mismatched_lengths() {
    fa::dot_product_f32(&[1.0; 32], &[1.0; 31]);
}

// ───────────────────────────── softmax ─────────────────────────────

#[test]
fn softmax_all_variants_match_reference() {
    let mut rng = Rng(8);
    for &n in LENS {
        let mut base: Vec<f32> = (0..n).map(|_| rng.f32() * 20.0).collect();
        if n > 2 {
            base[1] = -1e30; // effectively -inf after the max shift
            base[n - 1] = 50.0;
        }
        let mut want = base.clone();
        fa::softmax_scalar(&mut want);
        let check = |got: &[f32], tag: &str| {
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_close(g, w, w, 1e-5, &format!("{tag} n={n} i={i}"));
            }
            if n > 0 {
                let s: f32 = got.iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "{tag} n={n} sum={s}");
            }
        };
        let mut got = base.clone();
        fa::softmax_inplace(&mut got);
        check(&got, "dispatch");
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has(&["avx2"]) {
                let mut g = base.clone();
                fa::softmax_avx2(&mut g);
                check(&g, "avx2");
            }
            if has(&["avx512f"]) {
                let mut g = base.clone();
                fa::softmax_avx512(&mut g);
                check(&g, "avx512");
            }
        }
    }
}

// ───────────────────────────── rms_norm ─────────────────────────────

#[test]
fn rms_norm_all_variants_match_reference() {
    let mut rng = Rng(9);
    for &n in LENS.iter().filter(|&&n| n > 0) {
        let x: Vec<f32> = (0..n).map(|_| rng.f32() * 5.0).collect();
        let w: Vec<f32> = (0..n).map(|_| rng.f32() + 1.5).collect();
        let mut want = x.clone();
        fa::rms_norm_scalar(&mut want, &w, 1e-5);
        let check = |got: &[f32], tag: &str| {
            for (i, (&g, &r)) in got.iter().zip(&want).enumerate() {
                assert_close(g, r, r, 1e-5, &format!("{tag} n={n} i={i}"));
            }
        };
        let mut got = x.clone();
        fa::rms_norm(&mut got, &w, 1e-5);
        check(&got, "dispatch");
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has(&["avx2"]) {
                let mut g = x.clone();
                fa::rms_norm_avx2(&mut g, &w, 1e-5);
                check(&g, "avx2");
            }
            if has(&["avx512f"]) {
                let mut g = x.clone();
                fa::rms_norm_avx512(&mut g, &w, 1e-5);
                check(&g, "avx512");
            }
        }
    }
}

#[test]
#[should_panic(expected = "weight shorter than x")]
fn rms_norm_rejects_short_weight() {
    let mut x = vec![1.0f32; 64];
    fa::rms_norm(&mut x, &[1.0; 48], 1e-5);
}

// ─────────────────── weighted_sum / batched_dot_product ───────────────────

const HEAD_DIMS: &[usize] = &[1, 7, 8, 9, 15, 16, 17, 33, 64, 80, 128];

#[test]
fn weighted_sum_all_variants_match_reference() {
    let mut rng = Rng(10);
    let max_seq_len = 13;
    let kv_heads = 3;
    for &hd in HEAD_DIMS {
        let v_buf: Vec<f32> = (0..kv_heads * max_seq_len * hd)
            .map(|_| rng.f32())
            .collect();
        for &total in &[0usize, 1, 5, 13, 20] {
            for h_kv in 0..kv_heads {
                let scores: Vec<f32> = (0..total).map(|_| rng.f32().abs()).collect();
                let mut want = vec![f32::NAN; hd];
                fa::weighted_sum_scalar(&mut want, &scores, &v_buf, h_kv, max_seq_len, hd, total);
                let mag: f32 = scores.iter().sum::<f32>() + 1e-3;
                let check = |got: &[f32], tag: &str| {
                    for i in 0..hd {
                        assert_close(
                            got[i],
                            want[i],
                            mag,
                            1e-5,
                            &format!("{tag} hd={hd} t={total}"),
                        );
                    }
                };
                let mut got = vec![f32::NAN; hd];
                fa::weighted_sum(&mut got, &scores, &v_buf, h_kv, max_seq_len, hd, total);
                check(&got, "dispatch");
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if has(&["avx2", "fma"]) {
                        let mut g = vec![f32::NAN; hd];
                        fa::weighted_sum_avx2(
                            &mut g,
                            &scores,
                            &v_buf,
                            h_kv,
                            max_seq_len,
                            hd,
                            total,
                        );
                        check(&g, "avx2");
                    }
                    if has(&["avx512f"]) {
                        let mut g = vec![f32::NAN; hd];
                        fa::weighted_sum_avx512(
                            &mut g,
                            &scores,
                            &v_buf,
                            h_kv,
                            max_seq_len,
                            hd,
                            total,
                        );
                        check(&g, "avx512");
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "v_buf too short")]
fn weighted_sum_rejects_short_v_buf() {
    let v_buf = vec![0.0f32; 2 * 16 * 64 - 1];
    let scores = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 64];
    fa::weighted_sum(&mut out, &scores, &v_buf, 1, 16, 64, 4);
}

#[test]
#[should_panic(expected = "out shorter than head_dim")]
fn weighted_sum_rejects_short_out() {
    let v_buf = vec![0.0f32; 16 * 64];
    let scores = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 63];
    fa::weighted_sum(&mut out, &scores, &v_buf, 0, 16, 64, 4);
}

#[test]
fn batched_dot_product_all_variants_match_reference() {
    let mut rng = Rng(11);
    let max_seq_len = 11;
    let kv_heads = 2;
    for &hd in HEAD_DIMS {
        let k_buf: Vec<f32> = (0..kv_heads * max_seq_len * hd)
            .map(|_| rng.f32())
            .collect();
        let q: Vec<f32> = (0..hd).map(|_| rng.f32()).collect();
        for &(start, end) in &[(0usize, 0usize), (0, 1), (3, 19), (10, 26)] {
            for h_kv in 0..kv_heads {
                let len = end - start;
                let mut want = vec![0.0f32; len];
                fa::batched_dot_product_scalar(
                    &q,
                    &k_buf,
                    h_kv,
                    max_seq_len,
                    hd,
                    start,
                    end,
                    &mut want,
                );
                let check = |got: &[f32], tag: &str| {
                    for i in 0..len {
                        assert_close(got[i], want[i], hd as f32, 1e-5, &format!("{tag} hd={hd}"));
                    }
                };
                let mut got = vec![0.0f32; len];
                fa::batched_dot_product(&q, &k_buf, h_kv, max_seq_len, hd, start, end, &mut got);
                check(&got, "dispatch");
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if has(&["avx2", "fma"]) {
                        let mut g = vec![0.0f32; len];
                        fa::batched_dot_product_avx2(
                            &q,
                            &k_buf,
                            h_kv,
                            max_seq_len,
                            hd,
                            start,
                            end,
                            &mut g,
                        );
                        check(&g, "avx2");
                    }
                    if has(&["avx512f"]) {
                        let mut g = vec![0.0f32; len];
                        fa::batched_dot_product_avx512(
                            &q,
                            &k_buf,
                            h_kv,
                            max_seq_len,
                            hd,
                            start,
                            end,
                            &mut g,
                        );
                        check(&g, "avx512");
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "q shorter than head_dim")]
fn batched_dot_product_rejects_short_q() {
    let k_buf = vec![0.0f32; 4 * 64];
    let mut out = [0.0f32; 4];
    fa::batched_dot_product(&[0.0; 63], &k_buf, 0, 4, 64, 0, 4, &mut out);
}

// ───────────────────────── quantize_f32_to_i8 ─────────────────────────

#[test]
fn quantize_all_variants_match_reference_exactly() {
    let mut rng = Rng(12);
    for &n in LENS {
        let mut patterns: Vec<Vec<f32>> = vec![
            (0..n).map(|_| rng.f32() * 10.0).collect(),
            vec![0.0; n],
            // max_abs = 127 -> scale = 1: exact .5 ties exercise rounding.
            (0..n)
                .map(|i| match i % 6 {
                    0 => 127.0,
                    1 => 2.5,
                    2 => -3.5,
                    3 => 0.5,
                    4 => -0.5,
                    _ => 126.5,
                })
                .collect(),
            (0..n).map(|_| rng.f32() * 1e-30).collect(), // tiny / subnormal-ish
        ];
        if n > 0 {
            patterns[3][0] = -1e-30;
        }
        for x in patterns {
            let mut want = vec![0i8; n];
            let want_inv = quantize_f32_to_i8_scalar(&x, &mut want);
            assert!(want.iter().all(|&q| q != i8::MIN), "never -128");
            let mut got = vec![0i8; n];
            let got_inv = quantize_f32_to_i8(&x, &mut got);
            assert_eq!(got, want, "dispatch n={n}");
            assert_eq!(got_inv.to_bits(), want_inv.to_bits(), "dispatch inv n={n}");
            #[cfg(target_arch = "x86_64")]
            if has(&["avx512f"]) {
                let mut g = vec![0i8; n];
                let inv = unsafe {
                    xor_net::bit1_58::quantization::quantize_f32_to_i8_avx512(&x, &mut g)
                };
                assert_eq!(g, want, "avx512 n={n}");
                assert_eq!(inv.to_bits(), want_inv.to_bits(), "avx512 inv n={n}");
            }
        }
    }
}

#[test]
#[should_panic(expected = "output shorter than input")]
fn quantize_rejects_short_output() {
    let x = vec![1.0f32; 64];
    let mut q = vec![0i8; 63];
    quantize_f32_to_i8(&x, &mut q);
}

// ───────────────────────────── silu_inplace ─────────────────────────────

#[test]
fn silu_all_variants_match_reference() {
    let mut rng = Rng(13);
    for &n in LENS {
        let mut g: Vec<f32> = (0..n).map(|_| rng.f32() * 12.0).collect();
        if n >= 4 {
            g[0] = 0.0;
            g[1] = -100.0;
            g[2] = 100.0;
            g[3] = -88.5;
        }
        let u: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0).collect();
        for use_silu in [true, false] {
            let mut want = g.clone();
            silu_inplace_scalar(&mut want, &u, use_silu);
            // ReLU² uses identical operations in every path: exact.
            let rel = if use_silu { 1e-4 } else { 0.0 };
            let check = |got: &[f32], tag: &str| {
                for i in 0..n {
                    assert_close(
                        got[i],
                        want[i],
                        want[i],
                        rel,
                        &format!("{tag} silu={use_silu} n={n} i={i}"),
                    );
                }
            };
            let mut got = g.clone();
            silu_inplace(&mut got, &u, use_silu);
            check(&got, "dispatch");
            #[cfg(target_arch = "x86_64")]
            if has(&["avx512f"]) {
                let mut g2 = g.clone();
                unsafe {
                    xor_net::bit1_58::quantization::silu_inplace_avx512(&mut g2, &u, use_silu)
                };
                check(&g2, "avx512");
            }
        }
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn silu_rejects_mismatched_lengths() {
    let mut g = vec![1.0f32; 64];
    silu_inplace(&mut g, &[1.0; 32], true);
}

/// Every f16 bit pattern converts exactly like `half` (NaNs stay NaN), in
/// every kernel variant and at every tail length.
#[test]
fn f16_to_f32_all_variants_match_half() {
    use half::f16;
    let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
    let check = |got: &[f32], src: &[f16], tag: &str| {
        for (i, (&g, s)) in got.iter().zip(src).enumerate() {
            let w = s.to_f32();
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{tag}: {:#06x} -> {g} want {w} (i={i})",
                s.to_bits()
            );
        }
    };
    for n in [0usize, 1, 7, 8, 15, 16, 17, 33, all.len()] {
        let src = &all[all.len() - n..];
        let mut got = vec![0.0f32; n];
        fa::f16_to_f32_slice(src, &mut got);
        check(&got, src, "dispatch");
        #[cfg(target_arch = "x86_64")]
        {
            if has(&["avx512f"]) {
                got.fill(0.0);
                unsafe { fa::f16_to_f32_slice_avx512(src, &mut got) };
                check(&got, src, "avx512");
            }
            if has(&["avx", "f16c"]) {
                got.fill(0.0);
                unsafe { fa::f16_to_f32_slice_f16c(src, &mut got) };
                check(&got, src, "f16c");
            }
        }
    }
}

/// f32 -> f16 rounds like `half::f16::from_f32` (nearest even), including
/// subnormals, overflow to infinity and NaN.
#[test]
fn f32_to_f16_all_variants_match_half() {
    use half::f16;
    let mut rng = Rng(0xF16C);
    let mut src: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        65504.0,
        65520.0,
        1e9,
        -1e9,
        6e-8,
        3e-8,
        1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
    ];
    // Values whose 13 dropped mantissa bits sit exactly on a rounding tie.
    src.extend((0..64).map(|i| f32::from_bits(0x3f80_1000 + (i << 13))));
    src.extend((0..3000).map(|_| rng.f32() * 70000.0));
    src.extend((0..3000).map(|_| rng.f32() * 1e-4));
    let want: Vec<f16> = src.iter().map(|&x| f16::from_f32(x)).collect();
    let check = |got: &[f16], tag: &str| {
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{tag}: {} -> {:#06x} want {:#06x}",
                src[i],
                g.to_bits(),
                w.to_bits()
            );
        }
    };
    for n in [0usize, 1, 7, 8, 9, src.len()] {
        let mut got = vec![f16::ZERO; n];
        fa::f32_to_f16_slice(&src[..n], &mut got);
        check(&got, "dispatch");
        #[cfg(target_arch = "x86_64")]
        if has(&["avx", "f16c"]) {
            got.fill(f16::ZERO);
            unsafe { fa::f32_to_f16_slice_f16c(&src[..n], &mut got) };
            check(&got, "f16c");
        }
    }
}
