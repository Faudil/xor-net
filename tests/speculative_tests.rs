//! Speculative decoding: multi-position verification + rollback on tiny
//! random Bonsai models (bit-exact against plain feeding), drafters, the
//! generator integration (greedy output identical to plain decoding) and a
//! statistical check that speculative *sampling* preserves the target
//! distribution.

use std::ops::ControlFlow;
use std::sync::Arc;

use xor_net::bit1::q1_0::{ByteBuf, Q1Weight};
use xor_net::generation::{GenerationParams, Generator};
use xor_net::models::q1::qwen3::{Qwen3Config, Qwen3Layer, Qwen3Model};
use xor_net::models::q1::qwen35::{
    AttnMixer, GdnMixer, Mixer, Qwen35Config, Qwen35Layer, Qwen35Model, Qwen35State,
};
use xor_net::models::q1::{Embedding, Linear, ModelAssets};
use xor_net::models::traits::{
    CausalLM, ForwardOpts, HiddenTaps, Logits, ModelState, downcast_state,
};
use xor_net::sampler::{Sampler, SamplingParams};
use xor_net::speculative::{
    DraftModel, DraftSampler, DraftToken, Drafter, PromptLookup, SpecConfig, SpecStats,
};

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
    fn q1(&mut self, rows: usize, cols: usize, block: usize) -> Linear {
        let bits: Vec<u8> = (0..rows * cols / 8).map(|_| self.next() as u8).collect();
        let s = 1.0 / (cols as f32).sqrt();
        let scales: Vec<u8> = (0..rows * cols / block)
            .flat_map(|_| half::f16::from_f32(s * (0.5 + 0.5 * (self.f32() + 1.0))).to_le_bytes())
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

const VOCAB: usize = 300;

fn tiny_qwen3(rng: &mut Rng) -> Qwen3Model {
    let cfg = Qwen3Config {
        hidden: 128,
        intermediate: 256,
        layers: 2,
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        eps: 1e-6,
        rope_theta: 10_000.0,
        yarn: None,
        context_length: 600,
        vocab: VOCAB,
        bos: None,
        eos: vec![1],
    };
    let layers = (0..cfg.layers)
        .map(|_| Qwen3Layer {
            attn_norm: rng.vec(128, 1.0, 0.2),
            qkv: rng.q1((4 + 2 * 2) * 32, 128, 32),
            q_norm: rng.vec(32, 1.0, 0.2),
            k_norm: rng.vec(32, 1.0, 0.2),
            o: rng.q1(128, 128, 32),
            ffn_norm: rng.vec(128, 1.0, 0.2),
            gate_up: rng.q1(512, 128, 32),
            down: rng.q1(128, 256, 32),
        })
        .collect();
    let embed = Embedding(rng.q1(VOCAB, 128, 32));
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
        vocab: VOCAB,
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
    let embed = Embedding(rng.q1(VOCAB, 128, 128));
    let head = Some(rng.q1(VOCAB, 128, 128));
    Qwen35Model::new(cfg, embed, layers, rng.vec(128, 1.0, 0.2), head).unwrap()
}

fn tokens(n: usize, mul: u32, add: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * mul + add) % VOCAB as u32)
        .collect()
}

/// Logits of every position when fed one token at a time.
fn sequential_logits(
    m: &dyn CausalLM,
    st: &mut dyn ModelState,
    toks: &[u32],
    pos: usize,
) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, &t) in toks.iter().enumerate() {
        out.extend(m.forward(&[t], pos + i, st).unwrap());
    }
    out
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|l| l.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn recurrent_bits(st: &mut dyn ModelState) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let s = st.as_any_mut().downcast_mut::<Qwen35State>().unwrap();
    let (c, m) = s.recurrent_state();
    (bits(c), bits(m))
}

/// For every verify width and every accepted length: a provisional forward
/// returns the same logits as sequential decoding, and after `commit` the
/// recurrent state is bit-identical to feeding only the accepted tokens
/// (and so are the next logits).
#[test]
fn qwen35_verify_then_commit_is_bit_exact() {
    let m = tiny_qwen35(&mut Rng(11));
    let prompt = tokens(37, 37, 11);
    let draft = tokens(8, 53, 7);
    let next = tokens(3, 17, 3);
    for width in [1usize, 2, 3, 5, 8] {
        let draft = &draft[..width];
        let mut reference = m.new_state().unwrap();
        m.forward(&prompt, 0, reference.as_mut()).unwrap();
        let want = sequential_logits(&m, reference.as_mut(), draft, prompt.len());
        for keep in 0..=width {
            let mut st = m.new_state().unwrap();
            m.forward(&prompt, 0, st.as_mut()).unwrap();
            let before = recurrent_bits(st.as_mut());
            let mut opts = ForwardOpts {
                all_logits: true,
                provisional: true,
                taps: None,
            };
            let got = m
                .forward_with(draft, prompt.len(), st.as_mut(), &mut opts)
                .unwrap();
            assert_eq!(got.len(), width * VOCAB);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "width {width}: verify logits differ from sequential decode"
            );
            assert_eq!(
                recurrent_bits(st.as_mut()),
                before,
                "provisional forward must not touch the recurrent state"
            );
            assert!(
                m.forward(&next, prompt.len() + width, st.as_mut()).is_err(),
                "forward before commit is rejected"
            );
            st.commit(prompt.len() + keep).unwrap();

            let mut fresh = m.new_state().unwrap();
            m.forward(&prompt, 0, fresh.as_mut()).unwrap();
            if keep > 0 {
                sequential_logits(&m, fresh.as_mut(), &draft[..keep], prompt.len());
            }
            assert_eq!(
                recurrent_bits(st.as_mut()),
                recurrent_bits(fresh.as_mut()),
                "width {width} keep {keep}: state after commit"
            );
            let pos = prompt.len() + keep;
            let a = m.forward(&next, pos, st.as_mut()).unwrap();
            let b = m.forward(&next, pos, fresh.as_mut()).unwrap();
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "width {width} keep {keep}: next logits"
            );
        }
    }
}

#[test]
fn qwen35_commit_rejects_out_of_range_and_truncate_resolves() {
    let m = tiny_qwen35(&mut Rng(12));
    let prompt = tokens(9, 37, 11);
    let draft = tokens(4, 53, 7);
    let mut st = m.new_state().unwrap();
    m.forward(&prompt, 0, st.as_mut()).unwrap();
    st.checkpoint();
    let mut opts = ForwardOpts {
        all_logits: true,
        provisional: true,
        taps: None,
    };
    m.forward_with(&draft, 9, st.as_mut(), &mut opts).unwrap();
    assert!(st.commit(9 + 5).is_err(), "past the provisional range");
    // The failed commit leaves the base state, which can continue.
    let a = m.forward(&draft[..2], 9, st.as_mut()).unwrap();
    let mut fresh = m.new_state().unwrap();
    m.forward(&prompt, 0, fresh.as_mut()).unwrap();
    let b = m.forward(&draft[..2], 9, fresh.as_mut()).unwrap();
    assert_eq!(a, b);
    // Snapshot is refused while pending; truncate to the checkpoint works.
    m.forward_with(&draft[..1], 11, st.as_mut(), &mut opts)
        .unwrap();
    assert!(st.snapshot(12).is_none());
    assert!(st.truncate(9), "back to the checkpoint");
    assert_eq!(m.forward(&draft[..2], 9, st.as_mut()).unwrap(), b);
}

#[test]
fn qwen3_verify_matches_sequential_and_taps() {
    let m = tiny_qwen3(&mut Rng(13));
    let prompt = tokens(20, 37, 11);
    let draft = tokens(5, 53, 7);
    let mut reference = m.new_state().unwrap();
    m.forward(&prompt, 0, reference.as_mut()).unwrap();
    let want = sequential_logits(&m, reference.as_mut(), &draft, prompt.len());
    let mut st = m.new_state().unwrap();
    m.forward(&prompt, 0, st.as_mut()).unwrap();
    let mut taps = HiddenTaps {
        layers: vec![1, 0],
        data: Vec::new(),
    };
    let mut opts = ForwardOpts {
        all_logits: true,
        provisional: true,
        taps: Some(&mut taps),
    };
    let got = m
        .forward_with(&draft, prompt.len(), st.as_mut(), &mut opts)
        .unwrap();
    assert_eq!(got, want);
    assert_eq!(taps.data.len(), 5 * 2 * 128);
    st.commit(prompt.len() + 2).unwrap();
    assert!(m.batched_verify());
}

/// Taps report each layer's output: the last layer's tap, normed and
/// projected, reproduces the logits.
#[test]
fn qwen35_taps_are_layer_outputs() {
    let m = tiny_qwen35(&mut Rng(14));
    let toks = tokens(6, 37, 11);
    let mut st = m.new_state().unwrap();
    let mut taps = HiddenTaps {
        layers: vec![3, 0],
        data: Vec::new(),
    };
    let mut opts = ForwardOpts {
        all_logits: true,
        provisional: false,
        taps: Some(&mut taps),
    };
    let logits = m.forward_with(&toks, 0, st.as_mut(), &mut opts).unwrap();
    let h = 128;
    for t in 0..toks.len() {
        let mut x = taps.data[t * 2 * h..t * 2 * h + h].to_vec();
        xor_net::models::q1::rms_norm_rows(&mut x, &m.output_norm, 1e-6);
        let mut out = vec![0.0; VOCAB];
        m.lm_head()
            .forward(&x, 1, &mut Default::default(), &mut out);
        assert_eq!(out, logits[t * VOCAB..(t + 1) * VOCAB], "position {t}");
    }
    // Without taps / all_logits the plain forward is unchanged.
    let mut st2 = m.new_state().unwrap();
    assert_eq!(
        m.forward(&toks, 0, st2.as_mut()).unwrap(),
        logits[5 * VOCAB..].to_vec()
    );
}

fn tiny_assets() -> ModelAssets {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
        "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        "vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;
    ModelAssets {
        tokenizer: tokenizers::Tokenizer::from_bytes(json).unwrap(),
        chat_template: Some("{{ messages[0].content }}".into()),
    }
}

fn generator(m: Arc<dyn CausalLM>) -> Generator {
    xor_net::models::q1::make_generator(m, tiny_assets()).unwrap()
}

// ---------------------------------------------------------------------------
// Drafters used by the tests
// ---------------------------------------------------------------------------

/// Proposes the known greedy continuation, corrupted every few rounds, so
/// verification sees full accepts, partial accepts and immediate rejects.
#[derive(Clone)]
struct Oracle {
    prompt_len: usize,
    truth: Vec<u32>,
    round: usize,
}

impl Drafter for Oracle {
    fn draft(
        &mut self,
        context: &[u32],
        max: usize,
        _s: &mut dyn DraftSampler,
    ) -> anyhow::Result<Vec<DraftToken>> {
        self.round += 1;
        let i = context.len() - self.prompt_len; // index of the next token
        let mut d: Vec<u32> = self.truth.iter().skip(i).take(max).copied().collect();
        if !d.is_empty() && self.round.is_multiple_of(3) {
            let j = self.round % d.len();
            d[j] = (d[j] + 1) % VOCAB as u32;
        }
        if self.round.is_multiple_of(7) {
            d.truncate(1);
        }
        Ok(d.into_iter()
            .map(|token| DraftToken { token, probs: None })
            .collect())
    }
}

struct OracleModel(Oracle);
impl DraftModel for OracleModel {
    fn name(&self) -> &str {
        "oracle"
    }
    fn new_drafter(&self) -> anyhow::Result<Box<dyn Drafter>> {
        Ok(Box::new(self.0.clone()))
    }
}

fn greedy_params(n: usize) -> GenerationParams {
    GenerationParams::default()
        .with_max_tokens(n)
        .with_ignore_eos(true)
        .with_sampling(SamplingParams::greedy())
}

/// Greedy speculative output equals plain greedy output, for both Bonsai
/// architectures, several draft lengths, with and without adaptive k, with
/// an oracle-ish drafter and with prompt lookup.
#[test]
fn greedy_speculative_is_token_identical() {
    let models: Vec<(&str, Arc<dyn CausalLM>)> = vec![
        ("qwen35", Arc::new(tiny_qwen35(&mut Rng(21)))),
        ("qwen3", Arc::new(tiny_qwen3(&mut Rng(22)))),
    ];
    for (name, m) in models {
        for (pi, prompt) in [tokens(5, 37, 11), tokens(23, 7, 3), tokens(1, 1, 9)]
            .into_iter()
            .enumerate()
        {
            let base = generator(m.clone())
                .generate_from_tokens(&prompt, &greedy_params(40), |_| ControlFlow::Continue(()))
                .unwrap();
            assert_eq!(base.tokens.len(), 40);
            for k in [1usize, 3, 5] {
                for adaptive in [false, true] {
                    let oracle: Arc<dyn DraftModel> = Arc::new(OracleModel(Oracle {
                        prompt_len: prompt.len(),
                        truth: base.tokens.clone(),
                        round: 0,
                    }));
                    let lookup: Arc<dyn DraftModel> = Arc::new(PromptLookup::new(1, 3, 8));
                    for drafter in [oracle, lookup] {
                        let what = format!(
                            "{name} prompt {pi} k {k} adaptive {adaptive} {}",
                            drafter.name()
                        );
                        let mut g = generator(m.clone());
                        let params = greedy_params(40).with_speculative(
                            SpecConfig::new(drafter.clone(), k).with_adaptive(adaptive),
                        );
                        let got = g
                            .generate_from_tokens(&prompt, &params, |_| ControlFlow::Continue(()))
                            .unwrap();
                        assert_eq!(got.tokens, base.tokens, "{what}");
                        assert_eq!(got.text, base.text, "{what}");
                        assert_eq!(got.finish_reason, base.finish_reason, "{what}");
                        assert_eq!(got.usage, base.usage, "{what}");
                        let st = got.speculative.unwrap();
                        if drafter.name() == "oracle" {
                            assert!(
                                st.accepted > 0 && st.accepted < st.drafted,
                                "{what}: {st:?}"
                            );
                        }
                        // The generator's state is reusable: a follow-up
                        // request continuing the output matches a fresh run.
                        let mut follow = prompt.clone();
                        follow.extend_from_slice(&got.tokens[..20]);
                        let a = g
                            .generate_from_tokens(&follow, &params, |_| ControlFlow::Continue(()))
                            .unwrap();
                        let b = generator(m.clone())
                            .generate_from_tokens(&follow, &greedy_params(40), |_| {
                                ControlFlow::Continue(())
                            })
                            .unwrap();
                        assert_eq!(a.tokens, b.tokens, "{what}: follow-up");
                        assert!(a.usage.cached_prompt_tokens > 0, "{what}: prefix reused");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mock models for stop handling and sampling statistics
// ---------------------------------------------------------------------------

/// KV-like mock: logits are a hash of (last two tokens, position, token),
/// so they depend on the context and a wrong rollback shows. Uses the
/// trait's default (sequential) multi-position forward.
struct HashLM {
    vocab: usize,
    seed: u64,
    eos: Vec<u32>,
    sharp: f32,
}

struct HashState {
    fed: Vec<u32>,
}

impl ModelState for HashState {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.fed.clear();
        Ok(())
    }
    fn truncate(&mut self, len: usize) -> bool {
        if len > self.fed.len() {
            return false;
        }
        self.fed.truncate(len);
        true
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn hash_logits(seed: u64, ctx: &[u32], vocab: usize, sharp: f32) -> Vec<f32> {
    let a = *ctx.last().unwrap_or(&0) as u64;
    let b = if ctx.len() >= 2 {
        ctx[ctx.len() - 2] as u64
    } else {
        99
    };
    (0..vocab as u64)
        .map(|v| {
            let mut r = Rng(seed ^ (a << 40) ^ (b << 20) ^ ((ctx.len() as u64) << 8) ^ v);
            r.next();
            r.f32() * sharp
        })
        .collect()
}

impl CausalLM for HashLM {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(HashState { fed: Vec::new() }))
    }
    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        let st = downcast_state::<HashState>(state)?;
        anyhow::ensure!(pos == st.fed.len(), "pos {pos} != fed {}", st.fed.len());
        st.fed.extend_from_slice(tokens);
        Ok(hash_logits(self.seed, &st.fed, self.vocab, self.sharp))
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        4096
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
}

/// Model drafter for the mock: its own logits = the target's plus noise,
/// sampled through the request's pipeline (non-trivial `q`).
#[derive(Clone)]
struct NoisyDrafter {
    seed: u64,
    target_seed: u64,
    vocab: usize,
    sharp: f32,
}

impl Drafter for NoisyDrafter {
    fn draft(
        &mut self,
        context: &[u32],
        max: usize,
        s: &mut dyn DraftSampler,
    ) -> anyhow::Result<Vec<DraftToken>> {
        let mut ctx = context.to_vec();
        let mut drafted = Vec::new();
        let mut out = Vec::new();
        for _ in 0..max {
            let t = hash_logits(self.target_seed, &ctx, self.vocab, self.sharp);
            let n = hash_logits(self.seed, &ctx, self.vocab, 1.5);
            let mut l: Vec<f32> = t.iter().zip(&n).map(|(a, b)| a + b).collect();
            let (tok, probs) = s.pick(&mut l, &drafted)?;
            drafted.push(tok);
            ctx.push(tok);
            out.push(DraftToken { token: tok, probs });
        }
        Ok(out)
    }
}

struct NoisyModel(NoisyDrafter);
impl DraftModel for NoisyModel {
    fn name(&self) -> &str {
        "noisy"
    }
    fn new_drafter(&self) -> anyhow::Result<Box<dyn Drafter>> {
        Ok(Box::new(self.0.clone()))
    }
}

fn mock_generator(m: Arc<dyn CausalLM>) -> Generator {
    Generator::new(m, tiny_assets().tokenizer)
}

/// Exact probability of each 3-token continuation under the target with
/// the request's sampling pipeline, from `Sampler::distribution`.
fn exact_joint(m: &HashLM, prompt: &[u32], sp: &SamplingParams) -> Vec<f64> {
    let v = m.vocab;
    let mut s = Sampler::from_params(sp.clone().with_seed(Some(1)));
    let mut joint = vec![0.0f64; v * v * v];
    let mut probs = Vec::new();
    let mut dist = |ctx: &[u32], out: &[u32]| -> Vec<f32> {
        let mut l = hash_logits(m.seed, ctx, v, m.sharp);
        s.distribution(&mut l, prompt, out, &mut probs).unwrap();
        probs.clone()
    };
    let p1 = dist(prompt, &[]);
    for a in 0..v {
        let ctx1: Vec<u32> = prompt.iter().copied().chain([a as u32]).collect();
        let p2 = dist(&ctx1, &[a as u32]);
        for b in 0..v {
            let ctx2: Vec<u32> = ctx1.iter().copied().chain([b as u32]).collect();
            let p3 = dist(&ctx2, &[a as u32, b as u32]);
            for c in 0..v {
                joint[(a * v + b) * v + c] = p1[a] as f64 * p2[b] as f64 * p3[c] as f64;
            }
        }
    }
    joint
}

fn chi2(counts: &[u64], p: &[f64], n: u64) -> (f64, usize) {
    let mut x = 0.0;
    let mut df = 0usize;
    let mut other_c = 0.0;
    let mut other_p = 0.0;
    for (&c, &pi) in counts.iter().zip(p) {
        let e = pi * n as f64;
        if e >= 5.0 {
            x += (c as f64 - e).powi(2) / e;
            df += 1;
        } else {
            assert!(pi > 0.0 || c == 0, "sampled a token of probability 0");
            other_c += c as f64;
            other_p += pi;
        }
    }
    if other_p > 0.0 {
        let e = other_p * n as f64;
        x += (other_c - e).powi(2) / e.max(1e-9);
        df += 1;
    }
    (x, df.saturating_sub(1))
}

/// Speculative sampling reproduces the target's distribution: the joint
/// distribution of the first three tokens matches the exact one (chi-square
/// test), with a stochastic drafter and with a deterministic one, under
/// temperature / top-k / top-p / min-p / penalties.
#[test]
fn speculative_sampling_preserves_distribution() {
    let v = 5usize;
    let m = HashLM {
        vocab: v,
        seed: 7,
        eos: Vec::new(),
        sharp: 2.0,
    };
    let prompt = vec![1u32, 3];
    let configs = [
        SamplingParams::default().with_temperature(1.0),
        SamplingParams::default()
            .with_temperature(0.7)
            .with_top_k(4)
            .with_top_p(0.9),
        SamplingParams::default()
            .with_temperature(1.3)
            .with_min_p(0.05)
            .with_repetition_penalty(1.3)
            .with_presence_penalty(0.4),
    ];
    let model: Arc<dyn CausalLM> = Arc::new(HashLM { ..m });
    let m = HashLM {
        vocab: v,
        seed: 7,
        eos: Vec::new(),
        sharp: 2.0,
    };
    let noisy: Arc<dyn DraftModel> = Arc::new(NoisyModel(NoisyDrafter {
        seed: 99,
        target_seed: 7,
        vocab: v,
        sharp: 2.0,
    }));
    let lookup: Arc<dyn DraftModel> = Arc::new(PromptLookup::new(1, 2, 4));
    const N: u64 = 40_000;
    for (ci, sp) in configs.iter().enumerate() {
        let exact = exact_joint(&m, &prompt, sp);
        assert!((exact.iter().sum::<f64>() - 1.0).abs() < 1e-4);
        for drafter in [Some(noisy.clone()), Some(lookup.clone()), None] {
            let mut g = mock_generator(model.clone());
            let mut counts = vec![0u64; v * v * v];
            let mut stats = SpecStats::default();
            for i in 0..N {
                let mut params = GenerationParams::default()
                    .with_max_tokens(5)
                    .with_sampling(sp.clone().with_seed(Some(1000 + i)));
                if let Some(d) = &drafter {
                    params = params.with_speculative(SpecConfig::new(d.clone(), 3));
                }
                let out = g
                    .generate_from_tokens(&prompt, &params, |_| ControlFlow::Continue(()))
                    .unwrap();
                // The first three of five tokens: drafts of length 3 are
                // verified as a chain from the first round on.
                assert_eq!(out.tokens.len(), 5);
                let t = &out.tokens;
                counts[(t[0] as usize * v + t[1] as usize) * v + t[2] as usize] += 1;
                if let Some(s) = out.speculative {
                    stats.drafted += s.drafted;
                    stats.accepted += s.accepted;
                }
            }
            let (x, df) = chi2(&counts, &exact, N);
            let name = drafter.as_ref().map_or("plain", |d| d.name());
            // mean df, sd sqrt(2 df): allow 5 sd.
            let limit = df as f64 + 5.0 * (2.0 * df as f64).sqrt();
            assert!(
                x < limit,
                "config {ci} drafter {name}: chi2 {x:.1} (df {df}, limit {limit:.1}); {stats:?}"
            );
            if name == "noisy" {
                assert!(
                    stats.accepted > 0 && stats.accepted < stats.drafted,
                    "{stats:?}"
                );
            }
        }
    }
}

/// Stop ids, stop strings, abort and max_tokens inside an accepted draft
/// behave exactly as in plain decoding (same text, tokens, finish reason,
/// usage), and the state stays consistent for the next request.
#[test]
fn stops_inside_drafts_match_plain() {
    let v = 8usize;
    // Sharp logits: greedy runs are deterministic and lookup has repeats.
    let mk = || HashLM {
        vocab: v,
        seed: 5,
        eos: vec![7],
        sharp: 4.0,
    };
    let model: Arc<dyn CausalLM> = Arc::new(mk());
    let prompt: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    let base = mock_generator(model.clone())
        .generate_from_tokens(&prompt, &greedy_params(30).with_ignore_eos(true), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    let truth = base.tokens.clone();
    let oracle: Arc<dyn DraftModel> = Arc::new(OracleModel(Oracle {
        prompt_len: prompt.len(),
        truth: truth.clone(),
        round: 0,
    }));
    let text: String = mock_generator(model.clone()).decode(&truth, true).unwrap();
    let stop_str: String = text.chars().skip(7).take(2).collect();
    let mut cases: Vec<(String, GenerationParams)> = vec![
        ("eos".into(), greedy_params(30).with_ignore_eos(false)),
        (
            "stop id".into(),
            greedy_params(30).with_stop_token_ids([truth[9]]),
        ),
        (
            "stop string".into(),
            greedy_params(30).with_stop([stop_str.clone()]),
        ),
        ("max tokens".into(), greedy_params(13)),
    ];
    cases.push((
        "sampled".into(),
        greedy_params(30).with_sampling(SamplingParams::default().with_seed(Some(3))),
    ));
    for (what, params) in cases {
        for abort_at in [None, Some(5usize)] {
            let run = |spec: bool| {
                let mut g = mock_generator(model.clone());
                let p = if spec {
                    params
                        .clone()
                        .with_speculative(SpecConfig::new(oracle.clone(), 4))
                } else {
                    params.clone()
                };
                let mut n = 0;
                let mut streamed = String::new();
                let out = g
                    .generate_from_tokens(&prompt, &p, |t| {
                        streamed.push_str(t);
                        n += 1;
                        if Some(n) == abort_at {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    })
                    .unwrap();
                assert_eq!(streamed, out.text, "{what}: streamed text");
                // The state holds exactly prompt + what was fed: continuing
                // the conversation matches a fresh generator.
                let mut follow = prompt.clone();
                follow.extend_from_slice(&out.tokens);
                let a = g
                    .generate_from_tokens(&follow, &greedy_params(6), |_| ControlFlow::Continue(()))
                    .unwrap();
                (out, a.tokens)
            };
            let (plain, plain_next) = run(false);
            let (spec, spec_next) = run(true);
            let what = format!("{what} abort {abort_at:?}");
            if !what.starts_with("sampled") {
                assert_eq!(spec.tokens, plain.tokens, "{what}");
                assert_eq!(spec.text, plain.text, "{what}");
                assert_eq!(spec.finish_reason, plain.finish_reason, "{what}");
                assert_eq!(spec.stop_string, plain.stop_string, "{what}");
                assert_eq!(spec.usage, plain.usage, "{what}");
                assert_eq!(spec_next, plain_next, "{what}: follow-up");
            }
        }
    }
}
