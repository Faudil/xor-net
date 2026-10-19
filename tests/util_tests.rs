use xor_net::util::{CHUNK_SIZE, PARALLEL_THRESHOLD, get_num_threads, uninit_vec};

#[test]
fn test_uninit_vec_creates_correct_length() {
    let v: Vec<f32> = uninit_vec(100);
    assert_eq!(v.len(), 100);
}

#[test]
fn test_uninit_vec_zero_size() {
    let v: Vec<u8> = uninit_vec(0);
    assert_eq!(v.len(), 0);
}

#[test]
fn test_uninit_vec_large_allocation() {
    let v: Vec<f32> = uninit_vec(1_000_000);
    assert_eq!(v.len(), 1_000_000);
}

#[test]
fn test_parallel_threshold_positive() {
    const { assert!(PARALLEL_THRESHOLD > 0) };
    assert_eq!(PARALLEL_THRESHOLD % 1024, 0);
}

#[test]
fn test_chunk_size_positive() {
    const { assert!(CHUNK_SIZE > 0) };
    assert_eq!(CHUNK_SIZE % 1024, 0);
}

#[test]
fn test_chunk_size_divides_threshold() {
    assert_eq!(PARALLEL_THRESHOLD % CHUNK_SIZE, 0);
}

#[test]
fn test_get_num_threads_returns_positive() {
    let n = get_num_threads();
    assert!(n >= 1);
}

#[test]
fn test_get_num_threads_is_consistent() {
    let a = get_num_threads();
    let b = get_num_threads();
    assert_eq!(a, b);
}

#[test]
fn test_uninit_vec_is_writable() {
    let mut v: Vec<f32> = uninit_vec(10);
    // Now zero-initialised (sound to read before writing).
    assert!(v.iter().all(|&x| x == 0.0));
    for (i, x) in v.iter_mut().enumerate() {
        *x = i as f32;
    }
    for (i, &x) in v.iter().enumerate() {
        assert!((x - i as f32).abs() < 1e-6);
    }
}
