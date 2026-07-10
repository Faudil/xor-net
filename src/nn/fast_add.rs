use candle_core::{CustomOp2, Error, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

// std::arch imports removed

pub struct FastAddOp;

impl CustomOp2 for FastAddOp {
    fn name(&self) -> &'static str {
        "fast_add"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &Layout,
        s2: &candle_core::CpuStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        let d1 = match s1 {
            candle_core::CpuStorage::F32(slice) => slice,
            _ => return Err(Error::Msg("FastAddOp: input 1 must be f32".into())),
        };
        let d2 = match s2 {
            candle_core::CpuStorage::F32(slice) => slice,
            _ => return Err(Error::Msg("FastAddOp: input 2 must be f32".into())),
        };
        
        let s1_slice = match l1.contiguous_offsets() {
            Some((start, end)) => &d1[start..end],
            None => return Err(Error::Msg("FastAddOp: input 1 not contiguous".into())),
        };
        let s2_slice = match l2.contiguous_offsets() {
            Some((start, end)) => &d2[start..end],
            None => return Err(Error::Msg("FastAddOp: input 2 not contiguous".into())),
        };
        
        if s1_slice.len() != s2_slice.len() {
            return Err(Error::Msg("FastAddOp: shape mismatch".into()));
        }
        
        let mut out_data = crate::tensor::uninit_vec(s1_slice.len());
        
        let chunk_size = 4096; // Good size for parallel
        
        out_data.par_chunks_mut(chunk_size)
            .zip(s1_slice.par_chunks(chunk_size))
            .zip(s2_slice.par_chunks(chunk_size))
            .for_each(|((out_chunk, s1_chunk), s2_chunk): ((&mut [f32], &[f32]), &[f32])| {
                let len = out_chunk.len();
                for j in 0..len {
                    out_chunk[j] = s1_chunk[j] + s2_chunk[j];
                }
            });
            
        Ok((candle_core::CpuStorage::F32(out_data), l1.shape().clone()))
    }
}

pub fn fast_add(t1: &Tensor, t2: &Tensor) -> Result<Tensor> {
    if !t1.is_contiguous() {
        return fast_add(&t1.contiguous()?, t2);
    }
    if !t2.is_contiguous() {
        return fast_add(t1, &t2.contiguous()?);
    }
    t1.apply_op2(t2, FastAddOp)
}
