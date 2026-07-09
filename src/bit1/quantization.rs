/// Pack 8 f32 values into a single u8 for 1-bit quantization.
/// Values < 0 map to bit 0 (-1), values >= 0 map to bit 1 (+1).
pub fn pack_1bit(weights: &[f32]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((weights.len() + 7) / 8);
    for chunk in weights.chunks(8) {
        let mut b = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            if w >= 0.0 {
                b |= 1 << i;
            }
        }
        packed.push(b);
    }
    packed
}
