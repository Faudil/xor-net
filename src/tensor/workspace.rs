use std::cell::RefCell;

thread_local! {
    static F32_BUFFER_POOL: RefCell<Vec<Vec<f32>>> = RefCell::new(Vec::with_capacity(32));
    static I8_BUFFER_POOL: RefCell<Vec<Vec<i8>>> = RefCell::new(Vec::with_capacity(32));
}

#[inline(always)]
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    F32_BUFFER_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        if let Some(mut buf) = p.pop() {
            if buf.capacity() < size {
                buf.reserve(size - buf.len());
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
    if buf.capacity() >= 1024 && buf.capacity() <= 1024 * 1024 * 100 {
        F32_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 64 {
                p.push(buf);
            }
        });
    }
}

#[inline(always)]
pub fn get_pooled_buffer_i8(size: usize) -> Vec<i8> {
    I8_BUFFER_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        if let Some(mut buf) = p.pop() {
            if buf.capacity() < size {
                buf.reserve(size - buf.len());
            }
            unsafe { buf.set_len(size); }
            buf
        } else {
            crate::util::uninit_vec(size)
        }
    })
}

#[inline(always)]
pub fn return_pooled_buffer_i8(buf: Vec<i8>) {
    if buf.capacity() >= 1024 && buf.capacity() <= 1024 * 1024 * 100 {
        I8_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 64 {
                p.push(buf);
            }
        });
    }
}
