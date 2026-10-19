//! Weight-free tests of the Bonsai fast models on tiny random networks:
//! `.xq1` save/load is bit-identical, prefill in one call equals feeding the
//! same tokens one by one (and in uneven chunks), and state reset works.

use std::sync::Arc;

use xor_net::bit1::q1_0::{ByteBuf, Q1Weight};
use xor_net::models::q1::format::{self, TensorDigest, XqReader, XqWriter};
use xor_net::models::q1::qwen3::{Qwen3Config, Qwen3Layer, Qwen3Model};
use xor_net::models::q1::qwen35::{
    AttnMixer, GdnMixer, Mixer, Qwen35Config, Qwen35Layer, Qwen35Model,
};
use xor_net::models::q1::{Embedding, Linear, ModelAssets, Yarn};
use xor_net::models::traits::{CausalLM, ModelState};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, center: f32, spread: f32) -> Vec<f32> {
        (0..n).map(|_| center + spread * self.f32()).collect()
    }
    /// Random 1-bit matrix with scales ~ 1/sqrt(cols).
    fn q1(&mut self, rows: usize, cols: usize, block: usize) -> Linear {
        let bits: Vec<u8> = (0..rows * cols / 8).map(|_| self.next() as u8).collect();
        let s = 1.0 / (cols as f32).sqrt();
        let scales: Vec<u8> = (0..rows * cols / block)
            .flat_map(|_| f32_to_f16(s * (0.5 + 0.5 * (self.f32() + 1.0))).to_le_bytes())
            .collect();
        Linear::Q1(
            Q1Weight::new(
                rows,
                cols,
                block,
                ByteBuf::from_vec(bits),
                ByteBuf::from_vec(scales),
            )
            .unwrap(),
        )
    }
}

/// f32 -> f16 for normal, moderate values (test helper).
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "value out of f16 normal range");
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn tiny_qwen3(rng: &mut Rng, block: usize) -> Qwen3Model {
    let cfg = Qwen3Config {
        hidden: 128,
        intermediate: 256,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        eps: 1e-6,
        rope_theta: 10_000.0,
        yarn: Some(Yarn {
            factor: 4.0,
            original_context: 64,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }),
        context_length: 600,
        vocab: 300,
        bos: None,
        eos: vec![1],
    };
    let layers = (0..cfg.layers)
        .map(|_| Qwen3Layer {
            attn_norm: rng.vec(128, 1.0, 0.2),
            qkv: rng.q1((4 + 2 * 2) * 32, 128, block),
            q_norm: rng.vec(32, 1.0, 0.2),
            k_norm: rng.vec(32, 1.0, 0.2),
            o: rng.q1(128, 128, block),
            ffn_norm: rng.vec(128, 1.0, 0.2),
            gate_up: rng.q1(512, 128, block),
            down: rng.q1(128, 256, block),
        })
        .collect();
    let embed = Embedding(rng.q1(300, 128, block));
    Qwen3Model::new(cfg, embed, layers, rng.vec(128, 1.0, 0.2), None).unwrap()
}

fn tiny_qwen35(rng: &mut Rng) -> Qwen35Model {
    let cfg = Qwen35Config {
        hidden: 128,
        intermediate: 256,
        layers: 4,
        recurrent: vec![true, true, true, false],
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        rot_dim: 16,
        eps: 1e-6,
        rope_theta: 10_000.0,
        yarn: None,
        context_length: 600,
        vocab: 300,
        bos: None,
        eos: vec![1],
        conv_kernel: 4,
        k_heads: 2,
        v_heads: 4,
        head_k_dim: 32,
        head_v_dim: 32,
    };
    let (cd, vd) = (cfg.conv_dim(), cfg.value_dim());
    let layers = cfg
        .recurrent
        .clone()
        .into_iter()
        .map(|rec| {
            let mixer = if rec {
                Mixer::Gdn(GdnMixer {
                    qkvz: rng.q1(cd + vd, 128, 128),
                    ba: rng.q1(2 * 4, 128, 32),
                    conv: rng.vec(cd * 4, 0.0, 0.5),
                    dt_bias: rng.vec(4, 0.0, 1.0),
                    ssm_a: rng.vec(4, -1.0, 0.5),
                    norm: rng.vec(32, 1.0, 0.2),
                    out: rng.q1(128, vd, 128),
                })
            } else {
                Mixer::Attn(AttnMixer {
                    qkv: rng.q1(2 * 4 * 32 + 2 * 2 * 32, 128, 128),
                    q_norm: rng.vec(32, 1.0, 0.2),
                    k_norm: rng.vec(32, 1.0, 0.2),
                    o: rng.q1(128, 128, 128),
                })
            };
            Qwen35Layer {
                attn_norm: rng.vec(128, 1.0, 0.2),
                mixer,
                post_norm: rng.vec(128, 1.0, 0.2),
                gate_up: rng.q1(512, 128, 128),
                down: rng.q1(128, 256, 128),
            }
        })
        .collect();
    let embed = Embedding(rng.q1(300, 128, 128));
    let head = Some(rng.q1(300, 128, 128));
    Qwen35Model::new(cfg, embed, layers, rng.vec(128, 1.0, 0.2), head).unwrap()
}

fn tiny_assets() -> ModelAssets {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
        "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        "vocab":{"a":0,"b":1},"merges":[]}}"#;
    ModelAssets {
        tokenizer: tokenizers::Tokenizer::from_bytes(json).unwrap(),
        chat_template: Some("{{ messages[0].content }}".into()),
    }
}

fn tokens(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 300).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Last-position logits after feeding `toks` in the given chunk sizes.
fn run_chunked(m: &dyn CausalLM, toks: &[u32], chunks: &[usize]) -> Vec<f32> {
    let mut st = m.new_state().unwrap();
    let mut pos = 0;
    let mut out = Vec::new();
    for &c in chunks {
        out = m.forward(&toks[pos..pos + c], pos, st.as_mut()).unwrap();
        pos += c;
    }
    assert_eq!(pos, toks.len());
    out
}

fn check_chunking(m: &dyn CausalLM, what: &str) {
    // 300 tokens: crosses the 256-entry initial KV capacity.
    let toks = tokens(300);
    let all = run_chunked(m, &toks, &[300]);
    let ones = run_chunked(m, &toks, &[1; 300]);
    let mixed = run_chunked(m, &toks, &[7, 1, 120, 130, 2, 40]);
    let scale = all.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        all.iter().all(|v| v.is_finite()),
        "{what}: non-finite logits"
    );
    // Activation quantization depends only on the row itself, so batching
    // must not change results beyond float summation order.
    assert!(
        max_diff(&all, &ones) <= 1e-3 * scale,
        "{what}: prefill vs one-by-one {}",
        max_diff(&all, &ones)
    );
    assert!(
        max_diff(&all, &mixed) <= 1e-3 * scale,
        "{what}: prefill vs chunks"
    );
}

#[test]
fn qwen3_prefill_matches_incremental() {
    for block in [32, 128] {
        let m = tiny_qwen3(&mut Rng(1), block);
        check_chunking(&m, &format!("qwen3 block {block}"));
    }
}

#[test]
fn qwen35_prefill_matches_incremental() {
    let m = tiny_qwen35(&mut Rng(2));
    check_chunking(&m, "qwen35");
}

#[test]
fn qwen35_reset_restarts_recurrence() {
    let m = tiny_qwen35(&mut Rng(3));
    let toks = tokens(20);
    let fresh = run_chunked(&m, &toks, &[20]);
    let mut st = m.new_state().unwrap();
    m.forward(&tokens(9), 0, st.as_mut()).unwrap();
    st.reset().unwrap();
    let again = m.forward(&toks, 0, st.as_mut()).unwrap();
    assert_eq!(
        fresh, again,
        "reset must make the state equivalent to a fresh one"
    );
}

#[test]
fn forward_rejects_bad_input() {
    let m = tiny_qwen3(&mut Rng(4), 128);
    let mut st = m.new_state().unwrap();
    assert!(m.forward(&[], 0, st.as_mut()).is_err());
    assert!(
        m.forward(&[300], 0, st.as_mut()).is_err(),
        "token id out of range"
    );
    assert!(m.forward(&[1], 600, st.as_mut()).is_err(), "beyond context");
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("xornet-{}-{name}", std::process::id()))
}

#[test]
fn xq1_roundtrip_is_bit_identical_qwen3() {
    let m = tiny_qwen3(&mut Rng(5), 128);
    let p = tmp_path("qwen3.xq1");
    let mut w = XqWriter::create(&p).unwrap();
    m.write_xq1(&mut w);
    w.finish(
        "qwen3",
        serde_json::to_value(&m.cfg).unwrap(),
        &tiny_assets(),
    )
    .unwrap();
    assert!(format::is_xq1(&p).unwrap());
    let r = XqReader::open(&p).unwrap();
    let back = Qwen3Model::read_xq1(&r).unwrap();
    let (mut a, mut b) = (TensorDigest::default(), TensorDigest::default());
    m.write_xq1(&mut a);
    back.write_xq1(&mut b);
    assert_eq!(a, b);
    assert_eq!(back.cfg, m.cfg);
    let toks = tokens(10);
    assert_eq!(
        run_chunked(&m, &toks, &[10]),
        run_chunked(&back, &toks, &[10])
    );
    let (model, assets) = format::load_with_assets(&p).unwrap();
    assert_eq!(model.vocab_size(), 300);
    assert_eq!(
        assets.chat_template.as_deref(),
        Some("{{ messages[0].content }}")
    );
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn xq1_roundtrip_is_bit_identical_qwen35() {
    let m = tiny_qwen35(&mut Rng(6));
    let p = tmp_path("qwen35.xq1");
    let mut w = XqWriter::create(&p).unwrap();
    m.write_xq1(&mut w);
    w.finish(
        "qwen35",
        serde_json::to_value(&m.cfg).unwrap(),
        &tiny_assets(),
    )
    .unwrap();
    let back = Qwen35Model::read_xq1(&XqReader::open(&p).unwrap()).unwrap();
    let (mut a, mut b) = (TensorDigest::default(), TensorDigest::default());
    m.write_xq1(&mut a);
    back.write_xq1(&mut b);
    assert_eq!(a, b);
    let toks = tokens(10);
    assert_eq!(
        run_chunked(&m, &toks, &[10]),
        run_chunked(&back, &toks, &[10])
    );
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn xq1_rejects_truncated_file() {
    let m = tiny_qwen3(&mut Rng(7), 128);
    let p = tmp_path("trunc.xq1");
    let mut w = XqWriter::create(&p).unwrap();
    m.write_xq1(&mut w);
    w.finish(
        "qwen3",
        serde_json::to_value(&m.cfg).unwrap(),
        &tiny_assets(),
    )
    .unwrap();
    let len = std::fs::metadata(&p).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.set_len(len / 2).unwrap();
    assert!(XqReader::open(&p).is_err());
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn generator_runs_on_tiny_model() {
    let m: Arc<dyn CausalLM> = Arc::new(tiny_qwen3(&mut Rng(8), 128));
    let mut g = xor_net::models::q1::make_generator(m, tiny_assets()).unwrap();
    let out = g
        .generate_from_tokens(
            &[5, 6, 7],
            &xor_net::GenerationParams::default().with_max_tokens(5),
            |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
    assert!(!out.tokens.is_empty() && out.tokens.len() <= 5);
}

/// Feed `chunks` of `toks` into `st` starting at `pos`; last logits.
fn feed(m: &dyn CausalLM, st: &mut dyn ModelState, toks: &[u32], pos: usize) -> Vec<f32> {
    m.forward(toks, pos, st).unwrap()
}

#[test]
fn truncate_and_checkpoint_match_fresh_prefill() {
    let a = tokens(12);
    let b: Vec<u32> = (0..8).map(|i| (i * 11 + 5) % 300).collect();
    let c: Vec<u32> = (0..6).map(|i| (i * 53 + 7) % 300).collect();
    // Fresh: a then c, same chunking as the reused state below.
    let fresh = |m: &dyn CausalLM, prefix: &[u32]| {
        let mut st = m.new_state().unwrap();
        feed(m, st.as_mut(), prefix, 0);
        feed(m, st.as_mut(), &c, prefix.len())
    };

    // Recurrent hybrid: only the checkpoint (or 0 / current length) works.
    let m = tiny_qwen35(&mut Rng(7));
    let mut st = m.new_state().unwrap();
    feed(&m, st.as_mut(), &a, 0);
    st.checkpoint();
    assert_eq!(st.checkpoint_len(), Some(a.len()));
    feed(&m, st.as_mut(), &b, a.len());
    assert!(!st.truncate(5), "recurrent state cannot roll back to 5");
    assert!(!st.truncate(a.len() + 3), "nor past the checkpoint");
    assert!(st.truncate(a.len() + b.len()), "current length is a no-op");
    assert!(
        m.forward(&c, 3, st.as_mut()).is_err(),
        "a position that does not match the state is rejected"
    );
    assert!(st.truncate(a.len()));
    let got = feed(&m, st.as_mut(), &c, a.len());
    assert_eq!(got, fresh(&m, &a), "qwen35: checkpoint restore");
    // The checkpoint survives a restore and can be used again.
    assert!(st.truncate(a.len()));
    assert_eq!(feed(&m, st.as_mut(), &c, a.len()), got);
    st.reset().unwrap();
    assert_eq!(st.checkpoint_len(), None, "reset drops the checkpoint");

    // Dense KV-only model: any prefix.
    let m = tiny_qwen3(&mut Rng(8), 32);
    let mut st = m.new_state().unwrap();
    feed(&m, st.as_mut(), &a, 0);
    feed(&m, st.as_mut(), &b, a.len());
    assert!(st.truncate(7));
    assert_eq!(feed(&m, st.as_mut(), &c, 7), fresh(&m, &a[..7]), "qwen3");
}
