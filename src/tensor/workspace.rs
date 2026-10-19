//! Thread-local scratch-buffer pools for the decode hot path.
//!
//! Contract: a buffer returned by [`get_pooled_buffer`] /
//! [`get_pooled_buffer_i8`] has exactly the requested length and every element
//! is *initialised*, but its contents are **unspecified** (stale data from a
//! previous user, or zero). Callers must write every element they later read.
//!
//! Soundness: a `Vec` in the pool always has all of its `len` elements
//! initialised (it is an ordinary `Vec`). Reusing it we either `truncate` (no
//! write) or `resize` with zeros (initialises only the newly exposed tail), so
//! no uninitialised memory is ever exposed. In steady-state decode the same
//! sizes recur, so the zero-fill path is rarely taken and costs a short
//! `memset` when it is.

use std::cell::{Cell, RefCell};

thread_local! {
    static F32_BUFFER_POOL: RefCell<Vec<Vec<f32>>> = RefCell::new(Vec::with_capacity(32));
    static I8_BUFFER_POOL: RefCell<Vec<Vec<i8>>> = RefCell::new(Vec::with_capacity(32));
    static POOLING_DISABLED: Cell<u32> = const { Cell::new(0) };
}

/// Run `f` with buffer recycling disabled on this thread: buffers returned
/// meanwhile are freed instead of pooled. Model loaders wrap their work in
/// this so the transient f32 weight tensors (tens of MB each, dropped as
/// `FastTensor`s once packed) are not parked in every loader thread's pool
/// for the life of the process.
pub fn without_pooling<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            POOLING_DISABLED.with(|d| d.set(d.get() - 1));
        }
    }
    POOLING_DISABLED.with(|d| d.set(d.get() + 1));
    let _g = Guard;
    f()
}

#[inline(always)]
fn pooling_enabled() -> bool {
    POOLING_DISABLED.with(|d| d.get() == 0)
}

#[inline(always)]
fn reuse_or_alloc<T: Copy + Default>(popped: Option<Vec<T>>, size: usize) -> Vec<T> {
    match popped {
        Some(mut buf) => {
            if buf.len() >= size {
                buf.truncate(size);
            } else {
                buf.resize(size, T::default());
            }
            buf
        }
        None => vec![T::default(); size],
    }
}

/// Take an `f32` buffer of length `size` from the pool (contents unspecified,
/// see the module docs).
#[inline(always)]
pub fn get_pooled_buffer(size: usize) -> Vec<f32> {
    let popped = F32_BUFFER_POOL.with(|pool| pool.borrow_mut().pop());
    reuse_or_alloc(popped, size)
}

#[inline(always)]
pub fn return_pooled_buffer(buf: Vec<f32>) {
    if buf.capacity() >= 1024 && buf.capacity() <= 1024 * 1024 * 100 && pooling_enabled() {
        F32_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 64 {
                p.push(buf);
            }
        });
    }
}

/// Take an `i8` buffer of length `size` from the pool (contents unspecified,
/// see the module docs).
#[inline(always)]
pub fn get_pooled_buffer_i8(size: usize) -> Vec<i8> {
    let popped = I8_BUFFER_POOL.with(|pool| pool.borrow_mut().pop());
    reuse_or_alloc(popped, size)
}

#[inline(always)]
pub fn return_pooled_buffer_i8(buf: Vec<i8>) {
    if buf.capacity() >= 1024 && buf.capacity() <= 1024 * 1024 * 100 && pooling_enabled() {
        I8_BUFFER_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            if p.len() < 64 {
                p.push(buf);
            }
        });
    }
}
