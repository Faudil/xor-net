#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryPackType {
    Pack4,
    Pack5,
}

pub fn pack_1_58bit_4pack(weights: &[f32], w_scale: f32) -> Vec<u8> {
    let mut packed = Vec::with_capacity((weights.len() + 3) / 4);
    for chunk in weights.chunks(4) {
        let mut b = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            let scaled = if w_scale > 0.0 { w / w_scale } else { 0.0 };
            let q = scaled.round().max(-1.0).min(1.0);
            
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
    let mut packed = Vec::with_capacity((weights.len() + 4) / 5);
    for chunk in weights.chunks(5) {
        let mut b = 0u8;
        let mut multiplier = 1u8;
        for &w in chunk {
            let scaled = if w_scale > 0.0 { w / w_scale } else { 0.0 };
            let q = scaled.round().max(-1.0).min(1.0);
            
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn quantize_f32_to_i8_avx512(activations: &[f32], quantized: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    let mut max_vec = _mm512_setzero_ps();
    let chunks = activations.len() / 16;
    let ptr = activations.as_ptr();
    
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

    for i in 0..chunks {
        let v = _mm512_loadu_ps(ptr.add(i * 16));
        let abs_v = _mm512_and_ps(sign_mask, v);
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }
    
    let mut max_arr = [0f32; 16];
    _mm512_storeu_ps(max_arr.as_mut_ptr(), max_vec);
    let mut max_abs = 0f32;
    for &m in &max_arr {
        if m > max_abs { max_abs = m; }
    }
    for i in chunks * 16..activations.len() {
        let abs = activations[i].abs();
        if abs > max_abs { max_abs = abs; }
    }
    
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    let inv_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    
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
        let q = (activations[i] * scale).round();
        let q_clamped = if q > 127.0 { 127.0 } else if q < -127.0 { -127.0 } else { q };
        quantized[i] = q_clamped as i8;
    }
    
    inv_scale
}

use std::sync::atomic::{AtomicU8, Ordering};
static HAS_AVX512F: AtomicU8 = AtomicU8::new(0); // 0: uninitialized, 1: no, 2: yes

#[inline(always)]
fn has_avx512f() -> bool {
    let val = HAS_AVX512F.load(Ordering::Relaxed);
    if val != 0 {
        return val == 2;
    }
    let detected = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    };
    HAS_AVX512F.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

pub fn quantize_f32_to_i8(activations: &[f32], quantized: &mut [i8]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_avx512f() {
            let inv_scale = unsafe { quantize_f32_to_i8_avx512(activations, quantized) };
            return inv_scale;
        }
    }

    let mut max_abs: f32 = 0.0;
    for &x in activations {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    let inv_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    
    for (q_out, &x) in quantized.iter_mut().zip(activations.iter()) {
        let q = (x * scale).round();
        *q_out = q.max(-127.0).min(127.0) as i8;
    }
    
    inv_scale
}
