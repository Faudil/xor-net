use std::time::Instant;
use rayon::prelude::*;

fn main() {
    let size_mb = 1024; // 1 GB
    let size_bytes = size_mb * 1024 * 1024;
    
    println!("Allocating {} MB...", size_mb);
    let mut data = vec![1u8; size_bytes];
    
    // Warm up the CPU and force memory to actually be mapped
    data.iter_mut().for_each(|x| *x = 2);
    
    println!("Measuring single-threaded read bandwidth...");
    let start = Instant::now();
    let mut sum = 0u64;
    for &byte in data.iter() {
        sum += byte as u64;
    }
    let elap = start.elapsed();
    let bw_gbps = (size_bytes as f64 / 1_000_000_000.0) / elap.as_secs_f64();
    println!("Sum: {} (ignore this)", sum);
    println!("Single-threaded bandwidth: {:.2} GB/s", bw_gbps);
    
    println!("Measuring multi-threaded (Rayon) read bandwidth...");
    let start = Instant::now();
    let sum: u64 = data.par_iter().map(|&x| x as u64).sum();
    let elap = start.elapsed();
    let bw_gbps = (size_bytes as f64 / 1_000_000_000.0) / elap.as_secs_f64();
    println!("Sum: {} (ignore this)", sum);
    println!("Multi-threaded bandwidth: {:.2} GB/s", bw_gbps);
}
