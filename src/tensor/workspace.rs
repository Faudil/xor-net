use std::cell::RefCell;

thread_local! {
    static F32_BUFFER_POOL: RefCell<Vec<Vec<f32>>> = RefCell::new(Vec::with_capacity(32));
}

#[inline(always)]
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    F32_BUFFER_POOL.with(|pool| {
        if let Some(mut buf) = pool.borrow_mut().pop() {
            if buf.capacity() < size {
                buf.reserve(size - buf.capacity());
            }
            unsafe { buf.set_len(size); }
            buf
        } else {
            crate::util::uninit_vec(size)
        }
    })
}

#[inline(always)]
pub fn return_pooled_buffer(buf: Vec<f32>) {
    // Only return buffers that have significant capacity to avoid pooling small vectors
    if buf.capacity() >= 1024 {
        F32_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 128 { // Max 128 buffers per thread
                p.push(buf);
            }
        });
    }
}
