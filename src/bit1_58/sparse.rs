//! Lossless sparse ternary weight storage ("XorSparse").
//!
//! BitNet 1.58-bit weights live in `{-1, 0, +1}`. Zeros contribute nothing to a
//! dot product, so a weight row can be stored as *only its non-zero signs* plus
//! a bitmask of which positions are non-zero. This is **lossless**: decoding and
//! expanding the mask reproduces the original packed ternary row exactly, so the
//! engine produces bit-identical logits to the dense `pack4` path.
//!
//! Layout (little-endian everywhere):
//!
//! ```text
//! per row:
//!   num_blocks : u32                       // ceil(in_dim / BLOCK_SIZE)
//!   for each block of BLOCK_SIZE elements:
//!     mask      : u64                       // bit i set  => position i is non-zero
//!     signs     : ceil(nnz/8) bytes         // bit j (scan order) set => weight is -1
//! ```
//!
//! `nnz = popcount(mask)`. With block size 64 the mask is exactly one `u64` and
//! the sign stream is `nnz` single bits. Row byte cost at density `d` is
//! `in_dim/8 * (1 + d)`. Compared to `pack4` (`in_dim/4`) that is a
//! `(1 - (1+d)/2)` saving  ~25% smaller at `d = 0.5`, ~35% at `d = 0.3`.
//!
//! The activation vector is *not* sparse: every block maps to `BLOCK_SIZE`
//! contiguous activations, so the dot product reads the full (quantised) input
//! once and gathers only the non-zero weights. The win is a smaller per-token
//! weight stream, which is the real lever on a bandwidth-bound engine.

pub const BLOCK_SIZE: usize = 64;

/// A sparse ternary tensor.
///
/// `blob`/`row_offsets` keep the on-disk XorSparse layout (set-order signs, so
/// it serialises losslessly and compactly). At load time we *also* expand each
/// block into two lane-order `u64`s  `masks` (non-zero positions) and `signs`
/// (sign of lane *k*, not of the k-th non-zero)  so the per-token AVX-512
/// kernel can build the weight vector with zero scalar work. This decoded form
/// lives only in RAM (≈8 bytes/block extra) and is what `dot_row` actually uses.
///
/// NOTE: the decoded form is the same size per block as `pack4` (16 bytes), so
/// the sparse path is currently performance-neutral versus dense; the on-disk
/// file is still ~18% smaller. Reading the *compact* 12-byte block and expanding
/// signs to lane order with a vectorised gather is the follow-up that turns the
/// on-disk saving into a per-token speedup (see `SPARSE_FORMAT.md`).
#[derive(Clone, Debug)]
pub struct SparseTernary {
    pub blob: Vec<u8>,
    pub row_offsets: Vec<usize>,
    /// One `u64` per block: bit *k* set ⇒ activation lane *k* is non-zero.
    pub masks: Vec<u64>,
    /// One `u64` per block: bit *k* ⇒ the weight at lane *k* is `-1` (else `+1`).
    /// Only meaningful where `masks` bit *k* is set.
    pub signs: Vec<u64>,
    /// Number of 64-element blocks per row (identical for every row).
    pub num_blocks: usize,
}

/// Encode one ternary row (values in `{-1, 0, +1}`) into `buf` in the XorSparse
/// row layout. `in_dim` may be smaller than `BLOCK_SIZE` or not a multiple of it;
/// trailing positions in the final block are simply never set in the mask.
pub fn encode_sparse_row_into(buf: &mut Vec<u8>, vals: &[i8], in_dim: usize) {
    debug_assert!(vals.len() >= in_dim);
    let num_blocks = (in_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buf.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    for b in 0..num_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(in_dim);
        let mut mask: u64 = 0;
        let mut signs: u64 = 0;
        let mut si: u32 = 0;
        for i in start..end {
            let v = vals[i];
            if v != 0 {
                mask |= 1u64 << (i - start);
                if v < 0 {
                    signs |= 1u64 << si;
                }
                si += 1;
            }
        }
        buf.extend_from_slice(&mask.to_le_bytes());
        let sign_bytes = ((si as usize) + 7) / 8;
        let sb = signs.to_le_bytes();
        buf.extend_from_slice(&sb[..sign_bytes]);
    }
}

/// Build a full sparse tensor from a flat `[out_dim * in_dim]` slice of ternary
/// values (each in `{-1, 0, +1}`). Returns `(blob, row_offsets)`.
pub fn encode_sparse_tensor(
    vals: &[i8],
    in_dim: usize,
    out_dim: usize,
) -> (Vec<u8>, Vec<usize>) {
    debug_assert!(vals.len() >= in_dim * out_dim);
    let mut blob = Vec::new();
    let mut row_offsets = Vec::with_capacity(out_dim);
    for r in 0..out_dim {
        row_offsets.push(blob.len());
        encode_sparse_row_into(&mut blob, &vals[r * in_dim..], in_dim);
    }
    (blob, row_offsets)
}

impl SparseTernary {
    /// Reconstruct the per-row byte offsets by walking the row blobs. Cheap
    /// (once, at load) and lets `dot_row` jump straight to a row.
    pub fn build_row_offsets(blob: &[u8], out_dim: usize) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(out_dim);
        let mut p = 0usize;
        for _ in 0..out_dim {
            offsets.push(p);
            let num_blocks =
                u32::from_le_bytes([blob[p], blob[p + 1], blob[p + 2], blob[p + 3]]) as usize;
            p += 4;
            for _ in 0..num_blocks {
                let mask = u64::from_le_bytes([
                    blob[p],
                    blob[p + 1],
                    blob[p + 2],
                    blob[p + 3],
                    blob[p + 4],
                    blob[p + 5],
                    blob[p + 6],
                    blob[p + 7],
                ]);
                p += 8;
                let nnz = mask.count_ones() as usize;
                p += (nnz + 7) / 8;
            }
        }
        offsets
    }

    pub fn from_blob(blob: Vec<u8>, out_dim: usize) -> Self {
        let row_offsets = Self::build_row_offsets(&blob, out_dim);
        let mut masks: Vec<u64> = Vec::new();
        let mut signs: Vec<u64> = Vec::new();
        let mut num_blocks = 0usize;
        let mut p = 0usize;
        for _ in 0..out_dim {
            let nb =
                u32::from_le_bytes([blob[p], blob[p + 1], blob[p + 2], blob[p + 3]]) as usize;
            if num_blocks == 0 {
                num_blocks = nb;
            }
            p += 4;
            for _ in 0..nb {
                let mask = u64::from_le_bytes([
                    blob[p],
                    blob[p + 1],
                    blob[p + 2],
                    blob[p + 3],
                    blob[p + 4],
                    blob[p + 5],
                    blob[p + 6],
                    blob[p + 7],
                ]);
                p += 8;
                let nnz = mask.count_ones() as usize;
                let sign_bytes = (nnz + 7) / 8;
                let mut sb = 0u64;
                for k in 0..sign_bytes {
                    sb |= (blob[p + k] as u64) << (8 * k);
                }
                p += sign_bytes;
                // Expand the set-order sign bits into a lane-order sign mask:
                // bit k of `sgn` is the sign of activation lane k.
                let mut sgn = 0u64;
                let mut si = 0u32;
                let mut m = mask;
                let mut lane = 0u32;
                while m != 0 {
                    if m & 1 != 0 {
                        if (sb >> si) & 1 != 0 {
                            sgn |= 1u64 << lane;
                        }
                        si += 1;
                    }
                    m >>= 1;
                    lane += 1;
                }
                masks.push(mask);
                signs.push(sgn);
            }
        }
        Self {
            blob,
            row_offsets,
            masks,
            signs,
            num_blocks,
        }
    }

    /// Slices of the decoded per-block `masks`/`signs` for output row `r`.
    /// The AVX-512 kernel consumes these directly (lane-order signs).
    #[inline]
    pub fn blocks(&self, r: usize) -> (&[u64], &[u64]) {
        let s = r * self.num_blocks;
        let e = s + self.num_blocks;
        (&self.masks[s..e], &self.signs[s..e])
    }

    /// Decode a single row back into a flat `in_dim` vector of `{-1,0,1}` values.
    /// Used by tests/validation to confirm the format is lossless.
    pub fn decode_row(&self, row: usize, in_dim: usize) -> Vec<i8> {
        let off = self.row_offsets[row];
        let num_blocks =
            u32::from_le_bytes([
                self.blob[off],
                self.blob[off + 1],
                self.blob[off + 2],
                self.blob[off + 3],
            ]) as usize;
        let mut p = off + 4;
        let mut out = vec![0i8; in_dim];
        for b in 0..num_blocks {
            let start = b * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(in_dim);
            let mask = u64::from_le_bytes([
                self.blob[p],
                self.blob[p + 1],
                self.blob[p + 2],
                self.blob[p + 3],
                self.blob[p + 4],
                self.blob[p + 5],
                self.blob[p + 6],
                self.blob[p + 7],
            ]);
            p += 8;
            let nnz = mask.count_ones() as usize;
            let sign_bytes = (nnz + 7) / 8;
            let mut sb = 0u64;
            for k in 0..sign_bytes {
                sb |= (self.blob[p + k] as u64) << (8 * k);
            }
            p += sign_bytes;
            let mut si = 0u32;
            for i in start..end {
                if (mask >> (i - start)) & 1 != 0 {
                    let s = if (sb >> si) & 1 != 0 { -1 } else { 1 };
                    out[i] = s;
                    si += 1;
                }
            }
        }
        out
    }
}
