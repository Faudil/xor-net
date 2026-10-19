use std::sync::Arc;
use std::time::Instant;
use xor_net::GgufFile;
use xor_net::models::bonsai_qwen35::LazyLinear;

fn main() -> anyhow::Result<()> {
    let gguf = Arc::new(GgufFile::open("/tmp/bonsai-27b-q1_0.gguf")?);
    let lin = LazyLinear::new_bench(&gguf, "blk.0.ffn_gate.weight")?;
    let in_dim = lin.in_dim();
    let out_dim = lin.out_dim();
    let x = vec![0.5f32; in_dim];

    let _ = lin.bench_forward(&x)?; // warmup
    let n = 5;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = lin.bench_forward(&x)?;
    }
    let dt = t0.elapsed().as_secs_f64() / n as f64;
    let bits = (in_dim * out_dim) as f64;
    let bytes = bits / 128.0 * 18.0;
    println!(
        "ffn_gate [{out_dim},{in_dim}]: {:.1} ms  ({:.1} Gelem/s, {:.1} GB/s packed)",
        dt * 1000.0, bits / dt / 1e9, bytes / dt / 1e9
    );
    Ok(())
}
