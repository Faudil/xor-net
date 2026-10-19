//! Real-valued layers: dense conv, depthwise conv, pooling, dense.
//!
//! Dense convolutions run as per-output-row im2col + [`sgemm`] (1x1
//! stride-1 convs skip the im2col copy). Depthwise conv and pooling are NHWC
//! loops whose innermost loop runs contiguously over channels, which LLVM
//! vectorises; each output row runs under `with_best_simd!` so that happens
//! for the host's best ISA. Parallelism is over output rows.

use super::epilogue::Epilogue;
use super::gemm::sgemm;
use crate::util::with_best_simd;
use crate::vision::tensor::{FeatureMap, Geometry, Padding};
use rayon::prelude::*;

fn check_residual(res: Option<&FeatureMap>, h: usize, w: usize, c: usize) {
    if let Some(r) = res {
        assert_eq!(
            r.shape(),
            [h, w, c],
            "residual shape does not match layer output"
        );
    }
}

/// Dense 2-D convolution, zero padding, no bias (use the epilogue).
///
/// `weight` is HWIO: `[kh, kw, cin, cout]`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &FeatureMap,
    weight: &[f32],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: Padding,
    cout: usize,
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
) -> FeatureMap {
    let cin = input.c;
    assert_eq!(
        weight.len(),
        kernel[0] * kernel[1] * cin * cout,
        "conv2d: weight length does not match [kh, kw, cin, cout]"
    );
    let g = Geometry::new(input.h, input.w, kernel, stride, padding)
        .expect("conv2d: kernel does not fit the input");
    epi.check(cout);
    check_residual(residual, g.oh, g.ow, cout);
    let mut out = FeatureMap::zeros(g.oh, g.ow, cout);
    let kdim = g.kh * g.kw * cin;
    // A 1x1, stride-1 conv reads input rows directly as the GEMM's A matrix.
    let direct =
        g.kh == 1 && g.kw == 1 && g.sh == 1 && g.sw == 1 && g.pad_top == 0 && g.pad_left == 0;
    out.data
        .par_chunks_mut(g.ow * cout)
        .enumerate()
        .for_each_init(Vec::new, |cols: &mut Vec<f32>, (oy, row)| {
            let a: &[f32] = if direct {
                &input.data[oy * g.iw * cin..(oy + 1) * g.iw * cin]
            } else {
                // im2col for one output row: [ow][kh*kw*cin], zero padded.
                // For a fixed (ox, ky) the in-bounds kx taps are one
                // contiguous run of the input row, copied in one go.
                cols.clear();
                cols.resize(g.ow * kdim, 0.0);
                for ox in 0..g.ow {
                    let x0 = (ox * g.sw) as isize - g.pad_left as isize;
                    let kx_lo = (-x0).clamp(0, g.kw as isize) as usize;
                    let kx_hi = (g.iw as isize - x0).clamp(0, g.kw as isize) as usize;
                    if kx_lo >= kx_hi {
                        continue;
                    }
                    let dst = &mut cols[ox * kdim..(ox + 1) * kdim];
                    for ky in 0..g.kh {
                        let Some(iy) = g.in_y(oy, ky) else { continue };
                        let ix_lo = (x0 + kx_lo as isize) as usize;
                        let src = &input.data[(iy * g.iw + ix_lo) * cin..][..(kx_hi - kx_lo) * cin];
                        dst[(ky * g.kw + kx_lo) * cin..(ky * g.kw + kx_hi) * cin]
                            .copy_from_slice(src);
                    }
                }
                cols
            };
            sgemm(a, weight, row, g.ow, kdim, cout);
            epi.apply_row(residual, oy, cout, row);
        });
    out
}

/// Depthwise 2-D convolution (multiplier 1), zero padding, no bias.
///
/// `weight` is `[kh, kw, c]`.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d(
    input: &FeatureMap,
    weight: &[f32],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: Padding,
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
) -> FeatureMap {
    let c = input.c;
    assert_eq!(
        weight.len(),
        kernel[0] * kernel[1] * c,
        "depthwise_conv2d: weight length does not match [kh, kw, c]"
    );
    let g = Geometry::new(input.h, input.w, kernel, stride, padding)
        .expect("depthwise_conv2d: kernel does not fit the input");
    epi.check(c);
    check_residual(residual, g.oh, g.ow, c);
    let mut out = FeatureMap::zeros(g.oh, g.ow, c);
    out.data
        .par_chunks_mut(g.ow * c)
        .enumerate()
        .for_each(|(oy, row)| {
            with_best_simd!(depthwise_row(&g, input, weight, epi, residual, oy, row))
        });
    out
}

/// One output row of [`depthwise_conv2d`]. A function (not a closure) so the
/// slices arrive as non-aliasing arguments and the channel loop vectorises.
#[inline(always)]
fn depthwise_row(
    g: &Geometry,
    input: &FeatureMap,
    weight: &[f32],
    epi: &Epilogue,
    residual: Option<&FeatureMap>,
    oy: usize,
    row: &mut [f32],
) {
    let c = input.c;
    for ox in 0..g.ow {
        let acc = &mut row[ox * c..(ox + 1) * c];
        for ky in 0..g.kh {
            let Some(iy) = g.in_y(oy, ky) else { continue };
            for kx in 0..g.kw {
                let Some(ix) = g.in_x(ox, kx) else { continue };
                let px = input.pixel(iy, ix);
                let wt = &weight[(ky * g.kw + kx) * c..][..c];
                for ((o, &a), &wv) in acc.iter_mut().zip(px).zip(wt) {
                    *o += a * wv;
                }
            }
        }
        epi.apply(acc, residual.map(|r| r.pixel(oy, ox)));
    }
}

/// Max pooling. Padded positions are ignored (TensorFlow semantics).
pub fn max_pool2d(
    input: &FeatureMap,
    pool: [usize; 2],
    stride: [usize; 2],
    padding: Padding,
) -> FeatureMap {
    let c = input.c;
    let g = Geometry::new(input.h, input.w, pool, stride, padding)
        .expect("max_pool2d: window does not fit the input");
    let mut out = FeatureMap::zeros(g.oh, g.ow, c);
    out.data
        .par_chunks_mut(g.ow * c)
        .enumerate()
        .for_each(|(oy, row)| with_best_simd!(max_pool_row(&g, input, oy, row)));
    out
}

#[inline(always)]
fn max_pool_row(g: &Geometry, input: &FeatureMap, oy: usize, row: &mut [f32]) {
    let c = input.c;
    for ox in 0..g.ow {
        let acc = &mut row[ox * c..(ox + 1) * c];
        acc.fill(f32::NEG_INFINITY);
        for ky in 0..g.kh {
            let Some(iy) = g.in_y(oy, ky) else { continue };
            for kx in 0..g.kw {
                let Some(ix) = g.in_x(ox, kx) else { continue };
                for (o, &a) in acc.iter_mut().zip(input.pixel(iy, ix)) {
                    *o = o.max(a);
                }
            }
        }
    }
}

/// Average pooling. Padded positions are excluded from the divisor
/// (TensorFlow semantics).
pub fn avg_pool2d(
    input: &FeatureMap,
    pool: [usize; 2],
    stride: [usize; 2],
    padding: Padding,
) -> FeatureMap {
    let c = input.c;
    let g = Geometry::new(input.h, input.w, pool, stride, padding)
        .expect("avg_pool2d: window does not fit the input");
    let mut out = FeatureMap::zeros(g.oh, g.ow, c);
    out.data
        .par_chunks_mut(g.ow * c)
        .enumerate()
        .for_each(|(oy, row)| with_best_simd!(avg_pool_row(&g, input, oy, row)));
    out
}

#[inline(always)]
fn avg_pool_row(g: &Geometry, input: &FeatureMap, oy: usize, row: &mut [f32]) {
    let c = input.c;
    for ox in 0..g.ow {
        let acc = &mut row[ox * c..(ox + 1) * c];
        let mut n = 0usize;
        for ky in 0..g.kh {
            let Some(iy) = g.in_y(oy, ky) else { continue };
            for kx in 0..g.kw {
                let Some(ix) = g.in_x(ox, kx) else { continue };
                n += 1;
                for (o, &a) in acc.iter_mut().zip(input.pixel(iy, ix)) {
                    *o += a;
                }
            }
        }
        let inv = 1.0 / n.max(1) as f32;
        acc.iter_mut().for_each(|o| *o *= inv);
    }
}

/// Global average pooling: `[h, w, c] -> [1, 1, c]`.
pub fn global_avg_pool(input: &FeatureMap) -> FeatureMap {
    let c = input.c;
    let mut acc = vec![0.0f32; c];
    for px in input.data.chunks_exact(c) {
        for (o, &a) in acc.iter_mut().zip(px) {
            *o += a;
        }
    }
    let inv = 1.0 / (input.h * input.w).max(1) as f32;
    acc.iter_mut().for_each(|o| *o *= inv);
    FeatureMap::from_vec(1, 1, c, acc)
}

/// Fully connected layer on a flattened map. `weight` is `[in, out]`.
pub fn dense(
    input: &FeatureMap,
    weight: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    epi: &Epilogue,
) -> FeatureMap {
    let n_in = input.data.len();
    assert_eq!(
        weight.len(),
        n_in * out_features,
        "dense: weight length does not match [in, out]"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features, "dense: bias length mismatch");
    }
    epi.check(out_features);
    let mut acc = vec![0.0f32; out_features];
    sgemm(&input.data, weight, &mut acc, 1, n_in, out_features);
    // Bias after the matmul, like Keras (`matmul` then `bias_add`).
    if let Some(b) = bias {
        acc.iter_mut().zip(b).for_each(|(o, &bv)| *o += bv);
    }
    epi.apply(&mut acc, None);
    FeatureMap::from_vec(1, 1, out_features, acc)
}
