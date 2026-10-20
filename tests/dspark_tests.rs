//! DSpark drafter on a tiny random checkpoint written as a real GGUF file
//! (arch `dspark`, Q4_1 / Q1_0 / BF16 / F32 tensors): loading, context
//! ingestion invariance (chunking, rewind), determinism and draft validity.

use std::io::Write;

use xor_net::models::q1::dspark::{DsparkHead, DsparkModel};
use xor_net::speculative::{DraftModel, DraftSampler};

const H: usize = 128; // hidden (Q1_0 g128 embedding needs a multiple of 128)
const FF: usize = 128;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 32;
const LAYERS: usize = 2;
const VOCAB: usize = 64;
const RANK: usize = 32;
const TAPS: [u32; 2] = [0, 3];

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
}

enum Val {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrU32(Vec<u32>),
}

struct Tensor {
    name: String,
    /// ggml order: `ne0` (columns) first.
    ne: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn write_gguf(path: &std::path::Path, kv: &[(&str, Val)], tensors: &[Tensor]) {
    let mut out = Vec::new();
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        put_str(&mut out, k);
        match v {
            Val::U32(x) => {
                out.extend(4u32.to_le_bytes());
                out.extend(x.to_le_bytes());
            }
            Val::F32(x) => {
                out.extend(6u32.to_le_bytes());
                out.extend(x.to_le_bytes());
            }
            Val::Bool(x) => {
                out.extend(7u32.to_le_bytes());
                out.push(*x as u8);
            }
            Val::Str(s) => {
                out.extend(8u32.to_le_bytes());
                put_str(&mut out, s);
            }
            Val::ArrU32(a) => {
                out.extend(9u32.to_le_bytes());
                out.extend(4u32.to_le_bytes());
                out.extend((a.len() as u64).to_le_bytes());
                for x in a {
                    out.extend(x.to_le_bytes());
                }
            }
        }
    }
    let mut offset = 0u64;
    for t in tensors {
        put_str(&mut out, &t.name);
        out.extend((t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            out.extend(d.to_le_bytes());
        }
        out.extend(t.ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (t.data.len() as u64).next_multiple_of(32);
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for t in tensors {
        out.extend(&t.data);
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
    }
    std::fs::File::create(path)
        .unwrap()
        .write_all(&out)
        .unwrap();
}

fn f32_tensor(name: &str, v: Vec<f32>) -> Tensor {
    Tensor {
        name: name.into(),
        ne: vec![v.len() as u64],
        ty: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn bf16_tensor(rng: &mut Rng, name: &str, rows: usize, cols: usize, scale: f32) -> Tensor {
    Tensor {
        name: name.into(),
        ne: vec![cols as u64, rows as u64],
        ty: 30,
        data: (0..rows * cols)
            .flat_map(|_| {
                half::bf16::from_f32(rng.f32() * scale)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect(),
    }
}

fn q4_1_tensor(rng: &mut Rng, name: &str, rows: usize, cols: usize) -> Tensor {
    let s = 1.0 / (cols as f32).sqrt();
    let mut data = Vec::new();
    for _ in 0..rows * cols / 32 {
        let d = half::f16::from_f32(s * 0.25 * (1.0 + rng.f32().abs()));
        let m = half::f16::from_f32(-s * 2.0);
        data.extend(d.to_bits().to_le_bytes());
        data.extend(m.to_bits().to_le_bytes());
        for _ in 0..16 {
            data.push(rng.next() as u8);
        }
    }
    Tensor {
        name: name.into(),
        ne: vec![cols as u64, rows as u64],
        ty: 3,
        data,
    }
}

/// Upstream Q1_0 (type 41): blocks of 128 = `[d: f16][16 B of signs]`.
fn q1_tensor(rng: &mut Rng, name: &str, rows: usize, cols: usize) -> Tensor {
    let mut data = Vec::new();
    for _ in 0..rows * cols / 128 {
        data.extend(
            half::f16::from_f32(0.1 + 0.05 * rng.f32())
                .to_bits()
                .to_le_bytes(),
        );
        for _ in 0..16 {
            data.push(rng.next() as u8);
        }
    }
    Tensor {
        name: name.into(),
        ne: vec![cols as u64, rows as u64],
        ty: 41,
        data,
    }
}

fn tiny_dspark(path: &std::path::Path) {
    let mut rng = Rng(42);
    let norm = |rng: &mut Rng, n: usize| (0..n).map(|_| 1.0 + 0.1 * rng.f32()).collect::<Vec<_>>();
    let kv = vec![
        ("general.architecture", Val::Str("dspark".into())),
        ("dspark.block_count", Val::U32(LAYERS as u32)),
        ("dspark.context_length", Val::U32(4096)),
        ("dspark.embedding_length", Val::U32(H as u32)),
        ("dspark.feed_forward_length", Val::U32(FF as u32)),
        ("dspark.attention.head_count", Val::U32(HEADS as u32)),
        ("dspark.attention.head_count_kv", Val::U32(KV as u32)),
        ("dspark.attention.key_length", Val::U32(HD as u32)),
        ("dspark.attention.value_length", Val::U32(HD as u32)),
        ("dspark.rope.freq_base", Val::F32(10000.0)),
        ("dspark.attention.layer_norm_rms_epsilon", Val::F32(1e-6)),
        ("dspark.dspark.block_size", Val::U32(4)),
        ("dspark.dspark.mask_token_id", Val::U32(VOCAB as u32 - 1)),
        ("dspark.dspark.target_layers", Val::ArrU32(TAPS.to_vec())),
        ("dspark.dspark.markov_rank", Val::U32(RANK as u32)),
        ("dspark.dspark.log_snr_conditioning", Val::Bool(true)),
        ("dspark.dspark.min_log_snr", Val::F32(-9.0)),
        ("dspark.dspark.max_log_snr", Val::F32(9.0)),
    ];
    let mut t = vec![
        q1_tensor(&mut rng, "token_embd.weight", VOCAB, H),
        q4_1_tensor(&mut rng, "dspark.fc.weight", H, TAPS.len() * H),
        f32_tensor("dspark.hidden_norm.weight", norm(&mut rng, H)),
        f32_tensor("output_norm.weight", norm(&mut rng, H)),
        q4_1_tensor(&mut rng, "output.weight", VOCAB, H),
        bf16_tensor(&mut rng, "dspark.markov_head_a.weight", VOCAB, RANK, 1.0),
        q4_1_tensor(&mut rng, "dspark.markov_head_b.weight", VOCAB, RANK),
        bf16_tensor(&mut rng, "dspark.log_snr_fc1.weight", H, 128, 0.1),
        f32_tensor(
            "dspark.log_snr_fc1.bias",
            (0..H).map(|_| 0.1 * rng.f32()).collect(),
        ),
        bf16_tensor(&mut rng, "dspark.log_snr_fc2.weight", H, H, 0.1),
        f32_tensor(
            "dspark.log_snr_fc2.bias",
            (0..H).map(|_| 0.1 * rng.f32()).collect(),
        ),
    ];
    for i in 0..LAYERS {
        let p = |n: &str| format!("blk.{i}.{n}");
        t.push(f32_tensor(&p("attn_norm.weight"), norm(&mut rng, H)));
        t.push(f32_tensor(&p("ffn_norm.weight"), norm(&mut rng, H)));
        t.push(f32_tensor(&p("attn_q_norm.weight"), norm(&mut rng, HD)));
        t.push(f32_tensor(&p("attn_k_norm.weight"), norm(&mut rng, HD)));
        t.push(q4_1_tensor(&mut rng, &p("attn_q.weight"), HEADS * HD, H));
        t.push(q4_1_tensor(&mut rng, &p("attn_k.weight"), KV * HD, H));
        t.push(q4_1_tensor(&mut rng, &p("attn_v.weight"), KV * HD, H));
        t.push(q4_1_tensor(
            &mut rng,
            &p("attn_output.weight"),
            H,
            HEADS * HD,
        ));
        t.push(q4_1_tensor(&mut rng, &p("ffn_gate.weight"), FF, H));
        t.push(q4_1_tensor(&mut rng, &p("ffn_up.weight"), FF, H));
        t.push(q4_1_tensor(&mut rng, &p("ffn_down.weight"), H, FF));
    }
    write_gguf(path, &kv, &t);
}

struct Greedy;
impl DraftSampler for Greedy {
    fn pick(&mut self, l: &mut [f32], _d: &[u32]) -> anyhow::Result<(u32, Option<Vec<f32>>)> {
        let mut best = 0;
        for (i, &v) in l.iter().enumerate() {
            if v > l[best] {
                best = i;
            }
        }
        Ok((best as u32, None))
    }
}

fn taps(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n * TAPS.len() * H).map(|_| rng.f32() * 2.0).collect()
}

#[test]
fn dspark_tiny_checkpoint_drafts_consistently() {
    let dir = std::env::temp_dir().join(format!("xornet-dspark-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny-dspark.gguf");
    tiny_dspark(&path);
    let m = DsparkModel::load(&path, None, DsparkHead::Own).unwrap();
    assert_eq!(m.config().target_layers, vec![0, 3]);
    assert_eq!(m.max_draft(), 4);
    assert!(
        DsparkModel::load(&path, None, DsparkHead::SharedTarget).is_err(),
        "the shared head needs the target"
    );

    let mut rng = Rng(7);
    let toks: Vec<u32> = (0..24u32).map(|i| (i * 5 + 1) % 60).collect();
    let tp = taps(&mut rng, toks.len());
    let w = TAPS.len() * H;
    let mut ctx = toks[..20].to_vec();
    ctx.push(9); // anchor (sampled, not fed yet)

    // One drafter fed in one call.
    let mut a = m.new_drafter().unwrap();
    assert_eq!(a.taps(), Some(&[0usize, 3][..]));
    a.observe(&toks[..20], 0, Some(&tp[..20 * w])).unwrap();
    let da = a.draft(&ctx, 4, &mut Greedy).unwrap();
    assert_eq!(da.len(), 4);
    for d in &da {
        assert!(
            (d.token as usize) < VOCAB - 1,
            "valid, never the mask token"
        );
        assert!(d.probs.is_none());
    }
    // Deterministic, and a shorter request is a prefix of the block.
    assert_eq!(a.draft(&ctx, 4, &mut Greedy).unwrap(), da);
    assert_eq!(a.draft(&ctx, 2, &mut Greedy).unwrap(), da[..2].to_vec());

    // The drafts depend on the anchor and the context (not a constant).
    let mut seen = std::collections::HashSet::new();
    for anchor in 0..12u32 {
        let mut c2 = toks[..20].to_vec();
        c2.push(anchor);
        let d = a.draft(&c2, 4, &mut Greedy).unwrap();
        seen.insert(d.iter().map(|t| t.token).collect::<Vec<_>>());
    }
    assert!(seen.len() > 1, "drafts ignore the anchor");

    // Same context fed in chunks: bit-identical drafts.
    let mut b = m.new_drafter().unwrap();
    b.observe(&toks[..7], 0, Some(&tp[..7 * w])).unwrap();
    b.observe(&toks[7..20], 7, Some(&tp[7 * w..20 * w]))
        .unwrap();
    assert_eq!(b.draft(&ctx, 4, &mut Greedy).unwrap(), da);

    // More context changes the draft input; rewinding restores it.
    b.observe(&toks[20..24], 20, Some(&tp[20 * w..24 * w]))
        .unwrap();
    let mut longer = toks.clone();
    longer.push(9);
    let _ = b.draft(&longer, 4, &mut Greedy).unwrap();
    b.rewind(&toks[..20]);
    assert_eq!(b.draft(&ctx, 4, &mut Greedy).unwrap(), da);

    // A divergent history drops the rows after the common prefix.
    let mut other = toks[..20].to_vec();
    other[10] = 63;
    b.rewind(&other);
    b.observe(&other[10..20], 10, Some(&tp[10 * w..20 * w]))
        .unwrap();
    assert_eq!(
        b.draft(&ctx, 4, &mut Greedy).unwrap(),
        da,
        "same taps, same rows"
    );

    // Bad tap shapes are rejected.
    assert!(a.observe(&toks[..2], 20, Some(&tp[..w])).is_err());
    assert!(a.observe(&toks[..2], 20, None).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
