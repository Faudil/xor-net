//! Image preprocessing matching the reference training pipelines.
//!
//! Larq Zoo evaluates ImageNet models with the EfficientNet-style pipeline:
//!
//! 1. centre-crop a square of side `floor(size / (size + crop_padding) *
//!    min(h, w))` (offsets rounded up),
//! 2. resize it to `size x size` with TensorFlow's *legacy* bicubic
//!    (`tf.compat.v1.image.resize_bicubic`, `align_corners=False`,
//!    `half_pixel_centers=False`: Keys cubic with `a = -0.75`, sample
//!    position `x * scale`, clamped edges, coefficients quantised to a
//!    1024-entry table),
//! 3. normalise per channel: `(x - mean) / std` (values in 0..255 units).
//!
//! [`tf_legacy_resize_bicubic`] reproduces TensorFlow's kernel operation for
//! operation (same table, same rounding, same summation order) so the
//! preprocessed tensor matches the reference to float rounding.

use crate::vision::tensor::FeatureMap;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Preprocessing parameters (stored in the model file).
#[derive(Debug, Clone, PartialEq)]
pub struct Preprocess {
    /// Output side length (the network input is `size x size x 3`).
    pub size: usize,
    /// EfficientNet-style crop padding (32 for 224 inputs).
    pub crop_padding: usize,
    /// Per-channel mean, in 0..255 units.
    pub mean: [f32; 3],
    /// Per-channel standard deviation, in 0..255 units.
    pub std: [f32; 3],
}

impl Default for Preprocess {
    fn default() -> Self {
        Self {
            size: 224,
            crop_padding: 32,
            mean: [
                (0.485f64 * 255.0) as f32,
                (0.456f64 * 255.0) as f32,
                (0.406f64 * 255.0) as f32,
            ],
            std: [
                (0.229f64 * 255.0) as f32,
                (0.224f64 * 255.0) as f32,
                (0.225f64 * 255.0) as f32,
            ],
        }
    }
}

/// Errors for malformed image buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// `rgb.len() != width * height * 3`.
    BadLength { expected: usize, got: usize },
    /// Zero-sized image, or one so small that the centre crop is empty.
    TooSmall { width: usize, height: usize },
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadLength { expected, got } => {
                write!(
                    f,
                    "RGB buffer has {got} bytes, expected width*height*3 = {expected}"
                )
            }
            Self::TooSmall { width, height } => write!(f, "image {width}x{height} is too small"),
        }
    }
}

impl std::error::Error for ImageError {}

/// Validate an interleaved RGB8 buffer.
pub fn check_rgb(rgb: &[u8], width: usize, height: usize) -> Result<(), ImageError> {
    let expected = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(3))
        .ok_or(ImageError::TooSmall { width, height })?;
    if rgb.len() != expected {
        return Err(ImageError::BadLength {
            expected,
            got: rgb.len(),
        });
    }
    if width == 0 || height == 0 {
        return Err(ImageError::TooSmall { width, height });
    }
    Ok(())
}

impl Preprocess {
    /// Centre crop + legacy bicubic resize + normalisation of an interleaved
    /// RGB8 image (`rgb.len() == width * height * 3`, row-major).
    pub fn apply(&self, rgb: &[u8], width: usize, height: usize) -> Result<FeatureMap, ImageError> {
        check_rgb(rgb, width, height)?;
        let (crop, off_y, off_x) = self.center_crop_window(width, height);
        if crop == 0 {
            return Err(ImageError::TooSmall { width, height });
        }
        let mut out =
            tf_legacy_resize_bicubic(rgb, width, off_y, off_x, crop, crop, self.size, self.size);
        out.data.par_chunks_mut(3 * 1024).for_each(|chunk| {
            for px in chunk.as_chunks_mut::<3>().0 {
                for ((v, &m), &s) in px.iter_mut().zip(&self.mean).zip(&self.std) {
                    *v = (*v - m) / s;
                }
            }
        });
        Ok(out)
    }

    /// `(side, offset_y, offset_x)` of the square centre crop, computed like
    /// larq_zoo (`int32(float32(size / (size + pad)) * float32(min_side))`).
    pub fn center_crop_window(&self, width: usize, height: usize) -> (usize, usize, usize) {
        let ratio = (self.size as f64 / (self.size + self.crop_padding) as f64) as f32;
        let min_side = width.min(height) as f32;
        let crop = (ratio * min_side) as usize; // truncation, like tf.cast
        let crop = crop.min(width.min(height));
        let off_y = (height - crop).div_ceil(2);
        let off_x = (width - crop).div_ceil(2);
        (crop, off_y, off_x)
    }
}

const TABLE_SIZE: usize = 1024;

/// TensorFlow's bicubic coefficient table for `a = -0.75` (legacy mode).
fn coeffs_table() -> &'static [f32] {
    static T: OnceLock<Vec<f32>> = OnceLock::new();
    T.get_or_init(|| {
        let a = -0.75f64;
        let mut t = vec![0.0f32; (TABLE_SIZE + 1) * 2];
        for i in 0..=TABLE_SIZE {
            // C++: `float x = i * 1.0 / kTableSize;` then double arithmetic
            // because `a` is a double.
            let x = (i as f64 / TABLE_SIZE as f64) as f32;
            let xd = x as f64;
            t[i * 2] = (((a + 2.0) * xd - (a + 3.0)) * xd * xd + 1.0) as f32;
            let x1 = (x as f64 + 1.0) as f32 as f64;
            t[i * 2 + 1] = (((a * x1 - 5.0 * a) * x1 + 8.0 * a) * x1 - 4.0 * a) as f32;
        }
        t
    })
}

#[derive(Clone, Copy)]
struct Taps {
    w: [f32; 4],
    idx: [usize; 4],
}

fn taps(in_size: usize, out_size: usize) -> Vec<Taps> {
    let table = coeffs_table();
    let scale = in_size as f32 / out_size as f32;
    let limit = in_size as i64;
    let bound = |v: i64| v.clamp(0, limit - 1) as usize;
    (0..out_size)
        .map(|o| {
            let in_f = o as f32 * scale;
            let in_loc = in_f.floor() as i64;
            let delta = in_f - in_loc as f32;
            let off = (delta * TABLE_SIZE as f32).round_ties_even() as usize;
            Taps {
                w: [
                    table[off * 2 + 1],
                    table[off * 2],
                    table[(TABLE_SIZE - off) * 2],
                    table[(TABLE_SIZE - off) * 2 + 1],
                ],
                idx: [
                    bound(in_loc - 1),
                    bound(in_loc),
                    bound(in_loc + 1),
                    bound(in_loc + 2),
                ],
            }
        })
        .collect()
}

#[inline(always)]
fn interp(w: &[f32; 4], v: [f32; 4]) -> f32 {
    v[0] * w[0] + v[1] * w[1] + v[2] * w[2] + v[3] * w[3]
}

/// Resize the `crop_h x crop_w` window at `(off_y, off_x)` of an RGB8 image
/// (row stride `width * 3`) to `out_h x out_w` with TensorFlow 1.x
/// `resize_bicubic` semantics. Output is `f32` in 0..255 units (values may
/// overshoot slightly, as in TensorFlow).
#[allow(clippy::too_many_arguments)]
pub fn tf_legacy_resize_bicubic(
    rgb: &[u8],
    width: usize,
    off_y: usize,
    off_x: usize,
    crop_h: usize,
    crop_w: usize,
    out_h: usize,
    out_w: usize,
) -> FeatureMap {
    assert!(
        crop_h > 0 && crop_w > 0 && out_h > 0 && out_w > 0,
        "resize: empty size"
    );
    assert!(off_x + crop_w <= width, "resize: crop exceeds image width");
    assert!(
        (off_y + crop_h) * width * 3 <= rgb.len(),
        "resize: crop exceeds image height"
    );
    let ys = taps(crop_h, out_h);
    let xs = taps(crop_w, out_w);
    let stride = width * 3;
    let mut out = FeatureMap::zeros(out_h, out_w, 3);
    out.data
        .par_chunks_mut(out_w * 3)
        .zip(ys.par_iter())
        .for_each(|(orow, ty)| {
            // The 4 source rows of this output row, starting at the crop's left edge.
            let r = ty.idx.map(|y| {
                &rgb[(off_y + y) * stride + off_x * 3..(off_y + y) * stride + (off_x + crop_w) * 3]
            });
            for (o, tx) in orow.as_chunks_mut::<3>().0.iter_mut().zip(&xs) {
                for (c, oc) in o.iter_mut().enumerate() {
                    // TF interpolates along y for each of the 4 x taps, then
                    // along x, in exactly this order.
                    let mut col = [0.0f32; 4];
                    for (cv, &xi) in col.iter_mut().zip(&tx.idx) {
                        let i = xi * 3 + c;
                        *cv = interp(
                            &ty.w,
                            [
                                r[0][i] as f32,
                                r[1][i] as f32,
                                r[2][i] as f32,
                                r[3][i] as f32,
                            ],
                        );
                    }
                    *oc = interp(&tx.w, col);
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_window_matches_larq_zoo() {
        let p = Preprocess::default();
        // 960x640: crop = int(0.875 * 640) = 560, off_y = (80+1)//2 = 40, off_x = (400+1)//2 = 200.
        assert_eq!(p.center_crop_window(960, 640), (560, 40, 200));
        assert_eq!(p.center_crop_window(800, 965), (700, 133, 50));
    }

    #[test]
    fn constant_image_stays_constant() {
        let rgb = vec![77u8; 50 * 30 * 3];
        let out = tf_legacy_resize_bicubic(&rgb, 50, 0, 0, 30, 50, 17, 23);
        for v in out.data {
            assert!((v - 77.0).abs() < 1e-3, "{v}");
        }
    }

    #[test]
    fn rejects_bad_buffers() {
        let p = Preprocess::default();
        assert!(matches!(
            p.apply(&[0; 10], 2, 2),
            Err(ImageError::BadLength { .. })
        ));
        assert!(matches!(
            p.apply(&[], 0, 0),
            Err(ImageError::TooSmall { .. })
        ));
        // 1x1 image: crop = int(0.875) = 0.
        assert!(matches!(
            p.apply(&[1, 2, 3], 1, 1),
            Err(ImageError::TooSmall { .. })
        ));
        assert!(p.apply(&[1; 2 * 2 * 3], 2, 2).is_ok());
    }
}
