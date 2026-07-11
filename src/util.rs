use std::sync::atomic::{AtomicUsize, Ordering};

pub const PARALLEL_THRESHOLD: usize = 32768;
pub const CHUNK_SIZE: usize = 4096;

pub fn uninit_vec<T>(size: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(size);
    unsafe {
        v.set_len(size);
    }
    v
}

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn get_num_threads() -> usize {
    let mut val = NUM_THREADS.load(Ordering::Relaxed);
    if val == 0 {
        val = rayon::current_num_threads().max(1);
        NUM_THREADS.store(val, Ordering::Relaxed);
    }
    val
}
