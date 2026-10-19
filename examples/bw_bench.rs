use rayon::prelude::*;

fn main() {
    let total: usize = 3_400_000_000 / 4;
    let buf: Vec<u32> = vec![7u32; total];
    let t0 = std::time::Instant::now();
    let n = 3;
    for _ in 0..n {
        let sum: u64 = buf.par_iter().map(|&x| x as u64).sum();
        std::hint::black_box(sum);
    }
    let dt = t0.elapsed().as_secs_f64() / n as f64;
    let gb = buf.len() as f64 * 4.0 / 1e9;
    let ms = dt * 1000.0;
    let gbs = gb / dt;
    println!("sequential read: {gb:.2} GB in {ms:.0} ms = {gbs:.1} GB/s");
}
