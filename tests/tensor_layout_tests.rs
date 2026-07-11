//! Non-regression tests for tensor layout operations: reshape, transpose,
//! embedding, and slice_last_token.

use xor_net::tensor::FastTensor;



#[test]
fn reshape_preserves_data() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = FastTensor::new(data.clone(), vec![2, 3, 4]);
    let r = t.reshape(vec![4, 6]).unwrap();
    assert_eq!(r.shape, vec![4, 6]);
    assert_eq!(r.data, data, "Reshape must not change data order");
}

#[test]
fn transpose_seq_to_heads_roundtrip() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = FastTensor::new(data.clone(), vec![1, 3, 8]); // b=1, seq=3, hidden=8
    let heads = t.transpose_seq_to_heads(2, 4).unwrap();
    assert_eq!(heads.shape, vec![1, 2, 3, 4]);
    let back = heads.transpose_heads_to_seq().unwrap();
    assert_eq!(back.shape, vec![1, 3, 8]);
    assert_eq!(back.data, data, "roundtrip must preserve data");
}

#[test]
fn transpose_seq_to_heads_known_layout() {
    // input [1, 2, 4]: seq=2, hidden=4, 2 heads of dim 2
    // row 0: [h0d0, h0d1, h1d0, h1d1] = [0, 1, 2, 3]
    // row 1: [h0d0, h0d1, h1d0, h1d1] = [4, 5, 6, 7]
    // After transpose → [1, 2, 2, 2]:
    // head 0: [[0,1],[4,5]], head 1: [[2,3],[6,7]]
    let t = FastTensor::new((0..8).map(|i| i as f32).collect(), vec![1, 2, 4]);
    let h = t.transpose_seq_to_heads(2, 2).unwrap();
    assert_eq!(h.data, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn embedding_lookup() {
    let weight = FastTensor::new(
        vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        vec![3, 2], // vocab=3, dim=2
    );
    let ids = vec![2, 0, 1];
    let out = FastTensor::embedding(&ids, &weight).unwrap();
    assert_eq!(out.shape, vec![1, 3, 2]);
    assert_eq!(out.data, vec![30.0, 31.0, 10.0, 11.0, 20.0, 21.0]);
}

#[test]
fn embedding_single_token() {
    let weight = FastTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ids = vec![1];
    let out = FastTensor::embedding(&ids, &weight).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2]);
    assert_eq!(out.data, vec![3.0, 4.0]);
}

#[test]
fn slice_last_token_single_batch() {
    let t = FastTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1, 3, 2], // b=1, seq=3, hidden=2
    );
    let out = t.slice_last_token().unwrap();
    assert_eq!(out.shape, vec![1, 1, 2]);
    assert_eq!(out.data, vec![5.0, 6.0], "Must slice the last token");
}

#[test]
fn slice_last_token_batched() {
    let t = FastTensor::new(vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ], vec![2, 2, 2]);
    let sliced = t.slice_last_token().unwrap();
    assert_eq!(sliced.shape(), &[2, 1, 2]);
    assert_eq!(sliced.data, vec![3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn slice_last_token_single_seq() {
    // seq_len = 1 → slicing last token returns the same data
    let t = FastTensor::new(vec![7.0, 8.0, 9.0], vec![1, 1, 3]);
    let out = t.slice_last_token().unwrap();
    assert_eq!(out.data, vec![7.0, 8.0, 9.0]);
}
