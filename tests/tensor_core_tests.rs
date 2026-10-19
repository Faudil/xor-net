use xor_net::tensor::FastTensor;

#[test]
fn test_new_validates_shape() {
    let t = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn test_new_panics_on_shape_mismatch() {
    FastTensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
}

#[test]
fn test_zeros() {
    let t = FastTensor::zeros(vec![3, 4]);
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(t.elem_count(), 12);
    for &v in t.data.iter() {
        assert!((v - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_zeros_empty_dim() {
    let t = FastTensor::zeros(vec![0]);
    assert_eq!(t.elem_count(), 0);
}

#[test]
fn test_dims_returns_shape() {
    let t = FastTensor::new(vec![1.0, 2.0], vec![1, 2]);
    assert_eq!(t.dims(), &[1, 2]);
}

#[test]
fn test_into_data() {
    let t = FastTensor::new(vec![5.0, 6.0, 7.0], vec![3]);
    let data = t.into_data();
    assert_eq!(data, vec![5.0, 6.0, 7.0]);
}

#[test]
fn test_reshape_valid() {
    let t = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let r = t.reshape(vec![2, 2]).unwrap();
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_reshape_invalid() {
    let t = FastTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(t.reshape(vec![2, 2]).is_err());
}

#[test]
fn test_debug_output() {
    let t = FastTensor::new(vec![1.0, 2.0], vec![2]);
    let debug_str = format!("{:?}", t);
    assert!(debug_str.contains("shape"));
    assert!(debug_str.contains("data_len"));
}

// ───────────── workspace pool (sound reuse) ─────────────

#[test]
fn pooled_buffers_have_requested_length_and_are_reusable() {
    use xor_net::tensor::workspace::*;
    for &(a, b) in &[(2048usize, 4096usize), (4096, 1500), (3000, 3000)] {
        let mut buf = get_pooled_buffer(a);
        assert_eq!(buf.len(), a);
        buf.iter_mut().for_each(|x| *x = 1.5);
        return_pooled_buffer(buf);
        // Growing exposes a zeroed tail; shrinking keeps (stale) initialised data.
        let buf2 = get_pooled_buffer(b);
        assert_eq!(buf2.len(), b);
        assert!(buf2.iter().all(|&x| x == 1.5 || x == 0.0));
        return_pooled_buffer(buf2);

        let mut q = get_pooled_buffer_i8(a);
        assert_eq!(q.len(), a);
        q.iter_mut().for_each(|x| *x = -3);
        return_pooled_buffer_i8(q);
        let q2 = get_pooled_buffer_i8(b);
        assert_eq!(q2.len(), b);
        assert!(q2.iter().all(|&x| x == -3 || x == 0));
        return_pooled_buffer_i8(q2);
    }
}
