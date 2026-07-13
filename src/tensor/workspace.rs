use std::cell::RefCell;

thread_local! {
    static F32_BUFFER_POOL: RefCell<Vec<Vec<f32>>> = RefCell::new(Vec::with_capacity(32));
}

#[inline(always)]
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    F32_BUFFER_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        if let Some(mut buf) = p.pop() {
            if buf.capacity() >= size {
                unsafe { buf.set_len(size); }
                buf
            } else {
                // Drop buf, fall through to allocate new
                drop(buf);
                vec![0.0f32; size]
            }
        } else {
            vec![0.0f32; size]
        }
    })
}

#[inline(always)]
pub fn return_pooled_buffer(buf: Vec<f32>) {
    if buf.capacity() >= 1024 && buf.capacity() <= 1024 * 1024 * 100 {
        F32_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 64 {
                p.push(buf);
            }
        });
    }
}
