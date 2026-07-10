use std::fs::File;
use std::io::Write;
use xor_net::{SafeTensorLoader, SafeTensorRepo};

#[test]
fn test_loader_and_conversion() -> anyhow::Result<()> {
    let file_path = "tests/dummy_loader_test.safetensors";
    
    // Create a dummy safetensors metadata and data.
    // safetensors structure:
    // 8 bytes: header size N (little endian u64)
    // N bytes: JSON header
    // Remaining: raw tensor bytes
    
    let header_json = r#"{"test.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"model.layers.0.norm.weight":{"dtype":"F32","shape":[3],"data_offsets":[24,36]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    
    let mut file = File::create(&file_path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    
    // Tensor 1 data: 6 float32s: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let t1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut raw_bytes = Vec::new();
    for &val in &t1_data {
        raw_bytes.extend_from_slice(&val.to_le_bytes());
    }
    
    // Tensor 2 data: 3 float32s: [-0.5, 0.0, 0.5]
    let t2_data: Vec<f32> = vec![-0.5, 0.0, 0.5];
    for &val in &t2_data {
        raw_bytes.extend_from_slice(&val.to_le_bytes());
    }
    
    file.write_all(&raw_bytes)?;
    file.sync_all()?;
    
    // Load via SafeTensorRepo
    let repo = SafeTensorRepo::load(&[file_path])?;
    let loader = SafeTensorLoader::new(&repo);
    
    // Check loading test.weight
    let t1 = loader.get(&[2, 3], "test.weight")?;
    assert_eq!(t1.shape, vec![2, 3]);
    assert_eq!(t1.data, t1_data);
    
    // Check loading model.layers.0.norm.weight
    let sub_loader = loader.pp("model.layers.0");
    let t2 = sub_loader.get_vector(3, "norm.weight")?;
    assert_eq!(t2.shape, vec![3]);
    assert_eq!(t2.data, t2_data);
    std::fs::remove_file(file_path)?;
    Ok(())
}
