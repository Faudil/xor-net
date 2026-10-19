//! Fused per-channel post-processing applied to a layer's accumulator.

/// Element-wise tail fused into a conv/dense layer, applied in this fixed
/// order to each output channel `o` of each pixel:
///
/// ```text
/// v = acc
/// if pre_relu   { v = max(v, 0) }
/// if scale_bias { v = v * scale[o] + bias[o] }   // folded batch norm
/// if residual   { v = v + residual[o] }
/// if post_relu  { v = max(v, 0) }
/// ```
///
/// This covers `conv -> relu -> BN -> add` (QuickNet binary blocks),
/// `conv -> BN -> relu` (stems) and plain `conv -> BN`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Epilogue {
    pub pre_relu: bool,
    pub scale_bias: Option<(Vec<f32>, Vec<f32>)>,
    pub residual: bool,
    pub post_relu: bool,
}

impl Epilogue {
    /// Panics if the scale/bias vectors do not have `channels` entries.
    pub fn check(&self, channels: usize) {
        if let Some((s, b)) = &self.scale_bias {
            assert_eq!(s.len(), channels, "epilogue: scale length != channels");
            assert_eq!(b.len(), channels, "epilogue: bias length != channels");
        }
    }

    /// Apply in place to one pixel's channels. `residual` must be `Some` iff
    /// `self.residual` (checked).
    #[inline]
    pub fn apply(&self, acc: &mut [f32], residual: Option<&[f32]>) {
        assert_eq!(
            self.residual,
            residual.is_some(),
            "epilogue: residual operand mismatch"
        );
        if self.pre_relu {
            acc.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        if let Some((s, b)) = &self.scale_bias {
            let s = &s[..acc.len()];
            let b = &b[..acc.len()];
            for ((v, &sv), &bv) in acc.iter_mut().zip(s).zip(b) {
                *v = *v * sv + bv;
            }
        }
        if let Some(r) = residual {
            let r = &r[..acc.len()];
            acc.iter_mut().zip(r).for_each(|(v, &rv)| *v += rv);
        }
        if self.post_relu {
            acc.iter_mut().for_each(|v| *v = v.max(0.0));
        }
    }
}
