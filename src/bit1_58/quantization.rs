#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryPackType {
    Pack4,
    Pack5,
}

pub fn pack_1_58bit_4pack(weights: &[f32], w_scale: f32) -> Vec<u8> {
    let mut packed = Vec::with_capacity(weights.len().div_ceil(4));
    for chunk in weights.chunks(4) {
        let mut b = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            let scaled = if w_scale > 0.0 { w / w_scale } else { 0.0 };
            let q = scaled.round().clamp(-1.0, 1.0);

            let val = if q < -0.5 {
                0b00
            } else if q > 0.5 {
                0b10
            } else {
                0b01
            };
            b |= val << (i * 2);
        }
        packed.push(b);
    }
    packed
}

pub fn unpack_1_58bit_4pack(packed: &[u8], len: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(len);
    for &b in packed {
        for j in 0..4 {
            if weights.len() == len {
                break;
            }
            let val = (b >> (j * 2)) & 0b11;
            let w = match val {
                0b00 => -1.0,
                0b10 => 1.0,
                _ => 0.0,
            };
            weights.push(w);
        }
    }
    weights
}

pub fn pack_1_58bit_5pack(weights: &[f32], w_scale: f32) -> Vec<u8> {
    let mut packed = Vec::with_capacity(weights.len().div_ceil(5));
    for chunk in weights.chunks(5) {
        let mut b = 0u8;
        let mut multiplier = 1u8;
        for &w in chunk {
            let scaled = if w_scale > 0.0 { w / w_scale } else { 0.0 };
            let q = scaled.round().clamp(-1.0, 1.0);

            let val = if q < -0.5 {
                0
            } else if q > 0.5 {
                2
            } else {
                1
            };
            b += val * multiplier;
            multiplier *= 3;
        }
        packed.push(b);
    }
    packed
}

pub fn unpack_1_58bit_5pack(packed: &[u8], len: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(len);
    for &b in packed {
        let mut current = b;
        for _ in 0..5 {
            if weights.len() == len {
                break;
            }
            let val = current % 3;
            current /= 3;
            let w = match val {
                0 => -1.0,
                2 => 1.0,
                _ => 0.0,
            };
            weights.push(w);
        }
    }
    weights
}

/// Symmetric per-tensor i8 quantisation step for one element: round half to
/// even (matching `vcvtps2dq` under the default MXCSR rounding mode) and clamp
/// to `[-127, 127]`. NaN maps to 0.
#[inline(always)]
fn quantize_one(x: f32, scale: f32) -> i8 {
    let q = (x * scale).round_ties_even();
    // `clamp` would propagate NaN; `max`/`min` map it to -127, so test first.
    if q.is_nan() {
        0
    } else {
        q.clamp(-127.0, 127.0) as i8
    }
}

#[inline(always)]
fn quant_scales(max_abs: f32) -> (f32, f32) {
    if max_abs > 0.0 {
        (127.0 / max_abs, max_abs / 127.0)
    } else {
        (1.0, 1.0)
    }
}

/// Reference scalar implementation of [`quantize_f32_to_i8`].
///
/// Panics if `quantized.len() < activations.len()`.
pub fn quantize_f32_to_i8_scalar(activations: &[f32], quantized: &mut [i8]) -> f32 {
    assert!(
        quantized.len() >= activations.len(),
        "quantize_f32_to_i8: output shorter than input"
    );
    quantize_f32_to_i8_loops(activations, quantized)
}

#[inline(always)]
fn quantize_f32_to_i8_loops(activations: &[f32], quantized: &mut [i8]) -> f32 {
    let mut max_abs: f32 = 0.0;
    for &x in activations {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    let (scale, inv_scale) = quant_scales(max_abs);
    for (q_out, &x) in quantized.iter_mut().zip(activations.iter()) {
        *q_out = quantize_one(x, scale);
    }
    inv_scale
}

/// AVX-512F implementation of [`quantize_f32_to_i8`]. Bit-exact with
/// [`quantize_f32_to_i8_scalar`] for finite inputs.
///
/// # Safety
/// The CPU must support `avx512f`, and `quantized.len() >= activations.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn quantize_f32_to_i8_avx512(activations: &[f32], quantized: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(quantized.len() >= activations.len());
    let mut max_vec = _mm512_setzero_ps();
    let chunks = activations.len() / 16;
    let ptr = activations.as_ptr();

    for i in 0..chunks {
        let v = _mm512_loadu_ps(ptr.add(i * 16));
        // `_mm512_abs_ps` is AVX-512F; `_mm512_and_ps` would need AVX-512DQ
        // (not enabled here, so it would not even inline).
        let abs_v = _mm512_abs_ps(v);
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }

    let mut max_arr = [0f32; 16];
    _mm512_storeu_ps(max_arr.as_mut_ptr(), max_vec);
    let mut max_abs = 0f32;
    for &m in &max_arr {
        if m > max_abs {
            max_abs = m;
        }
    }
    for &x in &activations[chunks * 16..] {
        let abs = x.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    let (scale, inv_scale) = quant_scales(max_abs);

    let v_scale = _mm512_set1_ps(scale);
    let v_max = _mm512_set1_ps(127.0);
    let v_min = _mm512_set1_ps(-127.0);
    let out_ptr = quantized.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm512_loadu_ps(ptr.add(i * 16));
        let scaled = _mm512_mul_ps(v, v_scale);
        let clamped = _mm512_max_ps(v_min, _mm512_min_ps(v_max, scaled));

        let q_i32 = _mm512_cvtps_epi32(clamped);
        let q_i8 = _mm512_cvtepi32_epi8(q_i32);
        _mm_storeu_si128(out_ptr.add(i * 16) as *mut __m128i, q_i8);
    }

    for i in chunks * 16..activations.len() {
        quantized[i] = quantize_one(activations[i], scale);
    }

    inv_scale
}

/// Symmetric absmax quantisation of `activations` into `quantized[..n]`
/// (`n = activations.len()`), returning the inverse scale (`max_abs / 127`).
///
/// Output values are in `[-127, 127]` (never `-128`), which the ternary dot
/// product kernels rely on. Ties round half to even.
///
/// Panics if `quantized.len() < activations.len()`.
pub fn quantize_f32_to_i8(activations: &[f32], quantized: &mut [i8]) -> f32 {
    assert!(
        quantized.len() >= activations.len(),
        "quantize_f32_to_i8: output shorter than input"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::util::cpu_features().avx512f {
            // SAFETY: avx512f detected; output length checked above.
            return unsafe { quantize_f32_to_i8_avx512(activations, quantized) };
        }
    }
    // Plain loops; `round_ties_even` needs SSE4.1 to avoid a libm call.
    crate::util::with_best_simd!(quantize_f32_to_i8_loops(activations, quantized))
}

#[inline(always)]
fn silu_mul_one(g: f32, u: f32, use_silu: bool) -> f32 {
    if use_silu {
        g * (1.0 / (1.0 + (-g).exp())) * u
    } else {
        let r = g.max(0.0);
        r * r * u
    }
}

/// Reference scalar implementation of [`silu_inplace`] (libm `exp`).
///
/// Panics if `gate.len() != up.len()`.
pub fn silu_inplace_scalar(gate: &mut [f32], up: &[f32], use_silu: bool) {
    assert_eq!(
        gate.len(),
        up.len(),
        "silu_inplace: gate/up length mismatch"
    );
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = silu_mul_one(*g, u, use_silu);
    }
}

/// Compute `y = silu(gate) * up` (or `relu(gate)^2 * up` when `!use_silu`) in
/// place, writing the result back into `gate`. The SiLU path uses a vectorized
/// `exp` approximation (Pommier `exp_ps` port, relative error ~1e-5) so the
/// activation no longer serializes through a scalar libm `expf` per element.
///
/// Panics if `gate.len() != up.len()`.
pub fn silu_inplace(gate: &mut [f32], up: &[f32], use_silu: bool) {
    assert_eq!(
        gate.len(),
        up.len(),
        "silu_inplace: gate/up length mismatch"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::util::cpu_features().avx512f {
            // SAFETY: avx512f detected; lengths checked above.
            unsafe { silu_inplace_avx512(gate, up, use_silu) };
            return;
        }
    }
    silu_inplace_scalar(gate, up, use_silu);
}

/// AVX-512F implementation of [`silu_inplace`].
///
/// # Safety
/// The CPU must support `avx512f`, and `up.len() >= gate.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_inplace_avx512(gate: &mut [f32], up: &[f32], use_silu: bool) {
    use std::arch::x86_64::*;
    debug_assert!(up.len() >= gate.len());

    let n = gate.len();
    let gp: *mut f32 = gate.as_mut_ptr();
    let up_ptr: *const f32 = up.as_ptr();

    // exp(x) range constants
    let inv_ln2 = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
    #[allow(clippy::excessive_precision)]
    let exp_hi = _mm512_set1_ps(88.3762626647949);
    #[allow(clippy::excessive_precision)]
    let exp_lo = _mm512_set1_ps(-88.3762626647949);
    let one = _mm512_set1_ps(1.0);
    let zero = _mm512_setzero_ps();

    let mut i = 0;
    while i + 16 <= n {
        let g = _mm512_loadu_ps(gp.add(i) as *const f32);
        let u = _mm512_loadu_ps(up_ptr.add(i));

        // y = silu(g) or relu(g)^2
        let act = if use_silu {
            // e = exp(-g) via range reduction: exp(x) = 2^n * exp(x - n*ln2)
            let x = _mm512_max_ps(_mm512_min_ps(_mm512_sub_ps(zero, g), exp_hi), exp_lo);
            let fx = _mm512_mul_ps(x, inv_ln2);
            let n_f = _mm512_roundscale_ps(fx, 0); // round to nearest integer
            let t = _mm512_sub_ps(x, _mm512_mul_ps(n_f, ln2));
            // 4th-order Taylor of exp(t) (|t| < 0.35 -> error ~1e-5)
            let mut e = _mm512_set1_ps(1.0 / 24.0);
            e = _mm512_fmadd_ps(e, t, _mm512_set1_ps(1.0 / 6.0));
            e = _mm512_fmadd_ps(e, t, _mm512_set1_ps(0.5));
            e = _mm512_fmadd_ps(e, t, one);
            e = _mm512_fmadd_ps(e, t, one);
            // 2^n
            let ni = _mm512_cvtps_epi32(n_f);
            let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_add_epi32(ni, _mm512_set1_epi32(0x7f)),
                23,
            ));
            let exp_x = _mm512_mul_ps(e, pow2n);
            let sig = _mm512_div_ps(one, _mm512_add_ps(one, exp_x));
            _mm512_mul_ps(g, sig)
        } else {
            let r = _mm512_max_ps(g, zero);
            _mm512_mul_ps(r, r)
        };

        let y = _mm512_mul_ps(act, u);
        _mm512_storeu_ps(gp.add(i), y);
        i += 16;
    }

    for j in i..n {
        gate[j] = silu_mul_one(gate[j], up[j], use_silu);
    }
}
