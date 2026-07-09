#[derive(Debug, Clone, Copy)]
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

/// Absmax quantization for converting f32 activations to i8
/// Returns (quantized_i8, scale_f32)
pub fn quantize_f32_to_i8(activations: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs: f32 = 0.0;
    for &x in activations {
        if x.abs() > max_abs {
            max_abs = x.abs();
        }
    }
    
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    let inv_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    
    let mut quantized = Vec::with_capacity(activations.len());
    for &x in activations {
        let q = (x * scale).round();
        let q_clamped = q.max(-127.0).min(127.0) as i8;
        quantized.push(q_clamped);
    }
    
    (quantized, inv_scale)
}
