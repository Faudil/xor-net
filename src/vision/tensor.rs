//! Activation containers and convolution geometry.
//!
//! All activations are a single image in **NHWC** order with `N = 1`:
//! element `(y, x, c)` lives at `(y * w + x) * c_total + c`. Channels are the
//! fastest axis, so every per-pixel inner loop (conv taps, depthwise,
//! pooling, bias/BN epilogues) is a contiguous, vectorisable run, and
//! binarised activations pack the channels of one pixel into consecutive
//! `u64` words.

/// A dense `f32` feature map `[h, w, c]` (NHWC, batch 1).
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMap {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl FeatureMap {
    /// A zero-filled map.
    pub fn zeros(h: usize, w: usize, c: usize) -> Self {
        let n = checked_volume(h, w, c);
        Self {
            h,
            w,
            c,
            data: vec![0.0; n],
        }
    }

    /// Wrap existing NHWC data. Panics if `data.len() != h * w * c`.
    pub fn from_vec(h: usize, w: usize, c: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            checked_volume(h, w, c),
            "FeatureMap::from_vec: data length does not match {h}x{w}x{c}"
        );
        Self { h, w, c, data }
    }

    /// The `c` channel values of pixel `(y, x)`.
    #[inline]
    pub fn pixel(&self, y: usize, x: usize) -> &[f32] {
        let o = (y * self.w + x) * self.c;
        &self.data[o..o + self.c]
    }

    #[inline]
    pub fn shape(&self) -> [usize; 3] {
        [self.h, self.w, self.c]
    }
}

/// A binarised feature map: one bit per activation, `sign(x) = +1 <=> bit = 1`.
///
/// Pixel `(y, x)` occupies `words` consecutive `u64`s starting at
/// `(y * w + x) * words`; channel `c` is bit `c % 64` of word `c / 64`.
/// Bits for channels `>= c` in the last word are always **zero** — the
/// binary-conv kernels rely on this (weights have zero tail bits too, so the
/// tail never produces an XOR mismatch).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedMap {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub words: usize,
    pub data: Vec<u64>,
}

impl PackedMap {
    pub fn zeros(h: usize, w: usize, c: usize) -> Self {
        let words = c.div_ceil(64);
        let n = checked_volume(h, w, words);
        Self {
            h,
            w,
            c,
            words,
            data: vec![0; n],
        }
    }

    #[inline]
    pub fn pixel(&self, y: usize, x: usize) -> &[u64] {
        let o = (y * self.w + x) * self.words;
        &self.data[o..o + self.words]
    }
}

pub(crate) fn checked_volume(a: usize, b: usize, c: usize) -> usize {
    a.checked_mul(b)
        .and_then(|v| v.checked_mul(c))
        .expect("feature map size overflows usize")
}

/// TensorFlow/Keras padding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// `out = ceil(in / stride)`, padding split with the smaller half first.
    Same,
    /// No padding: `out = (in - k) / stride + 1`.
    Valid,
}

/// Output size and leading padding along one spatial axis, with exactly
/// TensorFlow's `SAME`/`VALID` semantics (asymmetric SAME padding puts the
/// extra row/column at the bottom/right).
///
/// Returns `None` when the window does not fit (`VALID` with `in < k`) or
/// `k`/`stride` are zero.
pub fn conv_out_dim(
    input: usize,
    k: usize,
    stride: usize,
    padding: Padding,
) -> Option<(usize, usize)> {
    if k == 0 || stride == 0 || input == 0 {
        return None;
    }
    match padding {
        Padding::Same => {
            let out = input.div_ceil(stride);
            let needed = (out - 1) * stride + k;
            let total = needed.saturating_sub(input);
            Some((out, total / 2))
        }
        Padding::Valid => {
            if input < k {
                None
            } else {
                Some(((input - k) / stride + 1, 0))
            }
        }
    }
}

/// Full 2-D window geometry for a conv/pool layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub ih: usize,
    pub iw: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub oh: usize,
    pub ow: usize,
    pub pad_top: usize,
    pub pad_left: usize,
}

impl Geometry {
    pub fn new(
        ih: usize,
        iw: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: Padding,
    ) -> Option<Self> {
        let (oh, pad_top) = conv_out_dim(ih, kernel[0], stride[0], padding)?;
        let (ow, pad_left) = conv_out_dim(iw, kernel[1], stride[1], padding)?;
        Some(Self {
            ih,
            iw,
            kh: kernel[0],
            kw: kernel[1],
            sh: stride[0],
            sw: stride[1],
            oh,
            ow,
            pad_top,
            pad_left,
        })
    }

    /// Input row for output row `oy` and kernel row `ky`, or `None` if it
    /// falls in the padding.
    #[inline(always)]
    pub fn in_y(&self, oy: usize, ky: usize) -> Option<usize> {
        let y = (oy * self.sh + ky).checked_sub(self.pad_top)?;
        (y < self.ih).then_some(y)
    }

    #[inline(always)]
    pub fn in_x(&self, ox: usize, kx: usize) -> Option<usize> {
        let x = (ox * self.sw + kx).checked_sub(self.pad_left)?;
        (x < self.iw).then_some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_padding_matches_tensorflow() {
        // 224 -> 112 with k3 s2: total pad 1, all at the bottom.
        assert_eq!(conv_out_dim(224, 3, 2, Padding::Same), Some((112, 0)));
        // 55 -> 28 with k3 s2: total pad 2, split 1/1.
        assert_eq!(conv_out_dim(55, 3, 2, Padding::Same), Some((28, 1)));
        assert_eq!(conv_out_dim(56, 3, 1, Padding::Same), Some((56, 1)));
        assert_eq!(conv_out_dim(7, 1, 1, Padding::Same), Some((7, 0)));
        assert_eq!(conv_out_dim(56, 2, 1, Padding::Valid), Some((55, 0)));
        assert_eq!(conv_out_dim(1, 3, 1, Padding::Valid), None);
    }
}
