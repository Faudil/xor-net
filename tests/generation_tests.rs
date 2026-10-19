//! Generator / detokenizer / chat-template tests against a tiny mock model
//! and in-test tokenizers (no weights, no network).

use std::ops::ControlFlow;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use tokenizers::Tokenizer;
use xor_net::generation::{
    ChatMessage, ChatTemplate, FinishReason, GenerationError, GenerationParams, Generator,
    ReasoningFormat, StreamDecoder, StreamEvent,
};
use xor_net::models::traits::{CausalLM, Logits, ModelState, downcast_state};
use xor_net::sampler::SamplingParams;

// ---------------------------------------------------------------------------
// Mock model: greedy-deterministic "next token = script(fed tokens)".
// ---------------------------------------------------------------------------

struct MockState {
    fed: Vec<u32>,
}

impl ModelState for MockState {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.fed.clear();
        Ok(())
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

type Script = dyn Fn(&[u32]) -> u32 + Send + Sync;

struct MockLM {
    vocab: usize,
    max_seq: usize,
    eos: Vec<u32>,
    bos: Option<u32>,
    script: Box<Script>,
    /// (pos, len) of every forward call.
    calls: Mutex<Vec<(usize, usize)>>,
}

impl MockLM {
    fn new(vocab: usize, script: impl Fn(&[u32]) -> u32 + Send + Sync + 'static) -> Self {
        Self {
            vocab,
            max_seq: 4096,
            eos: vec![1],
            bos: Some(0),
            script: Box::new(script),
            calls: Mutex::new(Vec::new()),
        }
    }
    /// Emits `out` after the prompt (prompt length = first forward length),
    /// then EOS forever.
    fn scripted(vocab: usize, out: Vec<u32>) -> Self {
        let prompt_len = Arc::new(Mutex::new(None::<usize>));
        Self::new(vocab, move |fed| {
            let mut pl = prompt_len.lock().unwrap();
            let p = *pl.get_or_insert(fed.len());
            out.get(fed.len() - p).copied().unwrap_or(1)
        })
    }
}

impl CausalLM for MockLM {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(MockState { fed: Vec::new() }))
    }
    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        let st = downcast_state::<MockState>(state)?;
        anyhow::ensure!(pos == st.fed.len(), "pos {pos} != fed {}", st.fed.len());
        anyhow::ensure!(pos + tokens.len() <= self.max_seq, "overflow");
        self.calls.lock().unwrap().push((pos, tokens.len()));
        st.fed.extend_from_slice(tokens);
        let mut l = vec![0.0f32; self.vocab];
        l[(self.script)(&st.fed) as usize] = 10.0;
        Ok(l)
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn bos_token_id(&self) -> Option<u32> {
        self.bos
    }
}

// ---------------------------------------------------------------------------
// In-test tokenizers
// ---------------------------------------------------------------------------

const SP_WORDS: &[&str] = &["▁Hello", "▁world", "▁User", ":", "▁", "!", "\n"];

/// SentencePiece-style BPE with byte fallback (Llama-2 / BitNet-3B style):
/// ids 0..3 = <s>, </s>, <unk>; 3..259 = <0xXX>; then `SP_WORDS`.
fn sp_tokenizer() -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["<s>", "</s>", "<unk>"].iter().enumerate() {
        vocab.insert(t.to_string(), i.into());
    }
    for b in 0..256u32 {
        vocab.insert(format!("<0x{b:02X}>"), (3 + b).into());
    }
    for (i, w) in SP_WORDS.iter().enumerate() {
        vocab.insert(w.to_string(), (259 + i as u32).into());
    }
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "▁"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        ]},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]},
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>",
                  "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": true, "byte_fallback": true, "vocab": vocab, "merges": []}
    });
    Tokenizer::from_str(&json.to_string()).expect("sp tokenizer")
}

fn sp(word: &str) -> u32 {
    259 + SP_WORDS.iter().position(|w| *w == word).unwrap() as u32
}
fn byte(b: u8) -> u32 {
    3 + b as u32
}

/// GPT-2 `bytes_to_unicode`.
fn byte_level_alphabet() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec![' '; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

/// Byte-level BPE (GPT-2 / Llama-3 style): ids 0..256 = single bytes,
/// 256 = "Ġworld", 257 = "<|eot|>" (special).
fn byte_level_tokenizer() -> Tokenizer {
    let alpha = byte_level_alphabet();
    let mut vocab = serde_json::Map::new();
    for (b, c) in alpha.iter().enumerate() {
        vocab.insert(c.to_string(), b.into());
    }
    vocab.insert("Ġworld".into(), 256.into());
    vocab.insert("<|eot|>".into(), 257.into());
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 257, "content": "<|eot|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": []}
    });
    Tokenizer::from_str(&json.to_string()).expect("byte-level tokenizer")
}

fn greedy(max_tokens: usize) -> GenerationParams {
    GenerationParams::default()
        .with_max_tokens(max_tokens)
        .with_sampling(SamplingParams::greedy())
}

fn collect_stream(
    g: &mut Generator,
    prompt: &str,
    p: &GenerationParams,
) -> (xor_net::Generation, Vec<String>) {
    let mut deltas = Vec::new();
    let out = g
        .generate_stream(prompt, p, |d| {
            deltas.push(d.to_string());
            ControlFlow::Continue(())
        })
        .unwrap();
    (out, deltas)
}

// ---------------------------------------------------------------------------
// Detokenizer
// ---------------------------------------------------------------------------

/// Streams `ids` and checks every delta is complete UTF-8 without U+FFFD
/// and that the concatenation equals the one-shot decode.
fn check_stream(tok: &Tokenizer, ctx: &[u32], ids: &[u32]) -> Vec<String> {
    let mut d = StreamDecoder::new(ctx, true);
    let mut deltas = Vec::new();
    for &id in ids {
        if let Some(s) = d.push(tok, id).unwrap() {
            assert!(!s.contains('\u{FFFD}'), "partial char emitted: {s:?}");
            deltas.push(s);
        }
    }
    if let Some(s) = d.flush(tok).unwrap() {
        deltas.push(s);
    }
    let mut all = ctx.to_vec();
    all.extend_from_slice(ids);
    let full = tok.decode(&all, true).unwrap();
    let pre = tok.decode(ctx, true).unwrap();
    assert_eq!(pre.clone() + &deltas.concat(), full, "deltas {deltas:?}");
    deltas
}

#[test]
fn stream_decoder_sentencepiece_byte_fallback_and_spaces() {
    let tok = sp_tokenizer();
    let smile = "😀".as_bytes();
    let ids: Vec<u32> = [sp("▁Hello"), sp("▁")]
        .into_iter()
        .chain(smile.iter().map(|&b| byte(b)))
        .chain([sp("▁world"), sp("!")])
        .collect();
    // No context: leading space stripped like a one-shot decode.
    let deltas = check_stream(&tok, &[], &ids);
    assert_eq!(deltas[0], "Hello");
    assert!(deltas.contains(&"😀".to_string()), "{deltas:?}");
    // With prompt context, the first token keeps its leading space.
    let deltas = check_stream(&tok, &[sp("▁Hello")], &[sp("▁world"), sp("!")]);
    assert_eq!(deltas, vec![" world", "!"]);
}

#[test]
fn stream_decoder_byte_level() {
    let tok = byte_level_tokenizer();
    let text = "héllo 世界 🎉";
    let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
    let deltas = check_stream(&tok, &[], &ids);
    assert_eq!(deltas.concat(), text);
    // Context + leading-space token.
    let deltas = check_stream(&tok, &[b'a' as u32], &[256, b'!' as u32]);
    assert_eq!(deltas, vec![" world", "!"]);
    // Truncated multi-byte char at the end is flushed lossily, not dropped.
    let mut d = StreamDecoder::new(&[], true);
    assert_eq!(d.push(&tok, 0xE4).unwrap(), None);
    assert_eq!(d.flush(&tok).unwrap().as_deref(), Some("\u{FFFD}"));
}

/// Real tokenizer (downloads ~9 MB tokenizer.json, no weights).
#[test]
#[ignore]
fn stream_decoder_real_llama3_tokenizer() {
    let api = hf_hub::api::sync::Api::new().unwrap();
    let path = api
        .model("microsoft/bitnet-b1.58-2B-4T".into())
        .get("tokenizer.json")
        .unwrap();
    let tok = Tokenizer::from_file(path).unwrap();
    let text = "Héllo wörld! 你好，世界 🎉🎉 naïve café\n\tcode: fn main() {}";
    let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
    let ctx = tok.encode("Say:", true).unwrap().get_ids().to_vec();
    check_stream(&tok, &ctx, &ids);
    check_stream(&tok, &[], &ids);
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

#[test]
fn generate_until_eos() {
    let out_ids = vec![sp("▁Hello"), sp("▁world"), sp("!")];
    let model = Arc::new(MockLM::scripted(300, out_ids.clone()));
    let mut g = Generator::new(model, sp_tokenizer());
    let (out, deltas) = collect_stream(&mut g, "Hello", &greedy(50));
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert_eq!(out.text, " Hello world!");
    assert_eq!(out.text, deltas.concat());
    assert_eq!(out.tokens, out_ids);
    assert_eq!(out.usage.completion_tokens, 4, "EOS counts as sampled");
    assert_eq!(out.usage.total_tokens, out.usage.prompt_tokens + 4);
    assert_eq!(out.timings.decode_tokens, 4);
    // ignore_eos runs to the length limit
    let out = g
        .generate("Hello", &greedy(6).with_ignore_eos(true))
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.tokens.len(), 6);
}

#[test]
fn generate_length_limit_and_stop_token_ids() {
    let model = Arc::new(MockLM::scripted(300, vec![sp("▁Hello"); 100]));
    let mut g = Generator::new(model.clone(), sp_tokenizer());
    let out = g.generate("Hello", &greedy(5)).unwrap();
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.tokens.len(), 5);
    assert_eq!(out.text, " Hello Hello Hello Hello Hello");
    // The last sampled token is never fed: prefill + 4 decode steps.
    assert_eq!(model.calls.lock().unwrap().len(), 5);

    let out = g
        .generate("Hello", &greedy(5).with_stop_token_ids([sp("▁Hello")]))
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert!(out.text.is_empty() && out.tokens.is_empty());
}

#[test]
fn stop_string_split_across_tokens_is_not_emitted() {
    let ids = vec![sp("▁Hello"), sp("\n"), sp("▁User"), sp(":"), sp("▁world")];
    let model = Arc::new(MockLM::scripted(300, ids));
    let mut g = Generator::new(model, sp_tokenizer());
    let p = greedy(50).with_stop(["\n User:"]);
    let (out, deltas) = collect_stream(&mut g, "Hi", &p);
    assert_eq!(out.finish_reason, FinishReason::StopString);
    assert_eq!(out.text, " Hello");
    assert_eq!(deltas.concat(), " Hello");
    assert!(deltas.iter().all(|d| !d.contains('\n')));
    // A partial match that never completes is released at the end.
    let model = Arc::new(MockLM::scripted(300, vec![sp("▁Hello"), sp("\n")]));
    let mut g = Generator::new(model, sp_tokenizer());
    let (out, deltas) = collect_stream(&mut g, "Hi", &p);
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert_eq!(out.text, " Hello\n");
    assert_eq!(deltas.concat(), out.text);
}

#[test]
fn multibyte_output_streams_whole_characters() {
    let mut ids: Vec<u32> = "é😀".bytes().map(byte).collect();
    ids.push(sp("!"));
    let model = Arc::new(MockLM::scripted(300, ids));
    let mut g = Generator::new(model, sp_tokenizer());
    let (out, deltas) = collect_stream(&mut g, "Hi", &greedy(20));
    assert_eq!(out.text, "é😀!");
    assert!(deltas.iter().all(|d| !d.contains('\u{FFFD}')), "{deltas:?}");
}

#[test]
fn abort_from_callback() {
    let model = Arc::new(MockLM::scripted(300, vec![sp("▁Hello"); 100]));
    let mut g = Generator::new(model, sp_tokenizer());
    let mut n = 0;
    let out = g
        .generate_stream("Hi", &greedy(50), |_| {
            n += 1;
            if n == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Aborted);
    assert_eq!(out.tokens.len(), 3);
    assert_eq!(out.text, " Hello Hello Hello");
}

#[test]
fn context_overflow_and_empty_prompt() {
    let mut m = MockLM::scripted(300, vec![]);
    m.max_seq = 8;
    let mut g = Generator::new(Arc::new(m), sp_tokenizer());
    let err = g.generate("Hello world", &greedy(7)).unwrap_err();
    match err.downcast_ref::<GenerationError>() {
        Some(GenerationError::ContextOverflow {
            max_seq_len: 8,
            max_tokens: 7,
            ..
        }) => {}
        other => panic!("unexpected {other:?}"),
    }
    // Empty prompt -> BOS only.
    let out = g
        .generate("", &greedy(3).with_add_special_tokens(false))
        .unwrap();
    assert_eq!(out.usage.prompt_tokens, 1);
    // Without BOS -> typed error.
    let mut m = MockLM::scripted(300, vec![]);
    m.bos = None;
    let mut g = Generator::new(Arc::new(m), sp_tokenizer());
    let err = g.generate_from_tokens(&[], &greedy(3), |_| ControlFlow::Continue(()));
    assert_eq!(
        err.unwrap_err().downcast_ref::<GenerationError>(),
        Some(&GenerationError::EmptyPrompt)
    );
    // max_tokens = 0 does no work.
    let out = g.generate("Hi", &greedy(0)).unwrap();
    assert_eq!(out.finish_reason, FinishReason::Length);
}

#[test]
fn chunked_prefill_and_prefix_caching() {
    let model = Arc::new(MockLM::new(300, |fed: &[u32]| {
        // Echo-ish: depends on the whole history so caching bugs show up.
        if fed.len().is_multiple_of(3) {
            sp("▁Hello")
        } else {
            sp("▁world")
        }
    }));
    let mut g = Generator::new(model.clone(), sp_tokenizer());
    let prompt: Vec<u32> = (0..10).map(|i| byte(b'a' + i)).collect();
    let p = greedy(4).with_prefill_chunk_size(3);
    let a = g
        .generate_from_tokens(&prompt, &p, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(
        *model.calls.lock().unwrap(),
        vec![(0, 3), (3, 3), (6, 3), (9, 1), (10, 1), (11, 1), (12, 1)]
    );
    assert!(a.timings.prefill_tokens == 10);

    // Second turn extends the first: only the new suffix is prefilled.
    model.calls.lock().unwrap().clear();
    let mut turn2 = prompt.clone();
    turn2.extend_from_slice(&a.tokens[..3]); // the 3 tokens that were fed
    turn2.extend_from_slice(&[byte(b'x'), byte(b'y')]);
    let b = g
        .generate_from_tokens(&turn2, &p, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(b.usage.cached_prompt_tokens, 13);
    assert_eq!(model.calls.lock().unwrap()[0], (13, 2));

    // Same result as a cold run.
    let mut cold = Generator::new(model.clone(), sp_tokenizer());
    cold.set_prefix_caching(false);
    let c = cold
        .generate_from_tokens(&turn2, &p, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(b.tokens, c.tokens);
    assert_eq!(c.usage.cached_prompt_tokens, 0);

    // A non-extending prompt resets the state.
    let d = g
        .generate_from_tokens(&prompt[..4], &p, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(d.usage.cached_prompt_tokens, 0);
}

#[test]
fn sampling_is_seeded_and_penalties_apply() {
    // Uniform logits: output depends only on the sampler seed.
    let model = Arc::new(MockLM::new(300, |_| 2));
    let mut g = Generator::new(model, sp_tokenizer());
    let p = GenerationParams::default()
        .with_max_tokens(16)
        .with_ignore_eos(true)
        .with_sampling(
            SamplingParams::default()
                .with_temperature(5.0)
                .with_seed(Some(7)),
        );
    let a = g.generate("Hi", &p).unwrap();
    let b = g.generate("Hi", &p).unwrap();
    assert_eq!(a.tokens, b.tokens);
    let c = g
        .generate(
            "Hi",
            &p.clone()
                .with_sampling(p.sampling.clone().with_seed(Some(8))),
        )
        .unwrap();
    assert_ne!(a.tokens, c.tokens);

    // Presence penalty forces a new token every step under greedy.
    let p = GenerationParams::default()
        .with_max_tokens(5)
        .with_ignore_eos(true)
        .with_sampling(SamplingParams::greedy().with_presence_penalty(100.0));
    let out = g.generate("Hi", &p).unwrap();
    let mut uniq = out.tokens.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), 5, "{:?}", out.tokens);
}

#[test]
fn fork_shares_model_with_own_state() {
    let model = Arc::new(MockLM::scripted(300, vec![sp("▁Hello")]));
    let g = Generator::new(model, sp_tokenizer());
    let mut a = g.fork();
    let mut b = g.fork();
    let ta = std::thread::spawn(move || a.generate("Hi", &greedy(3)).unwrap().text);
    let tb = std::thread::spawn(move || b.generate("Hi", &greedy(3)).unwrap().text);
    assert_eq!(ta.join().unwrap(), " Hello");
    assert_eq!(tb.join().unwrap(), " Hello");
}

// ---------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------

const LLAMA3_TEMPLATE: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

#[test]
fn chat_template_llama3_from_tokenizer_config() {
    let cfg = serde_json::json!({
        "bos_token": "<|begin_of_text|>",
        "eos_token": {"content": "<|eot_id|>", "lstrip": false},
        "chat_template": LLAMA3_TEMPLATE,
    });
    let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
    let s = t
        .render(
            &[ChatMessage::system("Be brief. "), ChatMessage::user("Hi")],
            true,
        )
        .unwrap();
    assert_eq!(
        s,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn chat_template_raise_exception_list_form_and_pycompat() {
    let src = "{% for m in messages %}{% if m.role not in ['user', 'assistant'] %}\
               {{ raise_exception('bad role: ' + m.role) }}{% endif %}\
               [{{ m.role.upper() }}] {{ m.content.strip() }}\n{% endfor %}";
    let cfg = serde_json::json!({
        "chat_template": [
            {"name": "rag", "template": "nope"},
            {"name": "default", "template": src}
        ]
    });
    let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
    assert_eq!(
        t.render(&[ChatMessage::user("  hi  ")], false).unwrap(),
        "[USER] hi\n"
    );
    let err = t.render(&[ChatMessage::system("x")], false).unwrap_err();
    assert!(format!("{err:#}").contains("bad role: system"), "{err:#}");
    assert!(
        ChatTemplate::from_tokenizer_config(&serde_json::json!({}))
            .unwrap()
            .is_none()
    );
}

#[test]
fn from_dir_loads_template_and_eos_and_chats() {
    let dir = std::env::temp_dir().join(format!("xornet-gen-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    sp_tokenizer()
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id": [1, 265]}"#,
    )
    .unwrap();
    let mut m = MockLM::scripted(300, vec![sp("▁world"), sp("\n"), sp("▁Hello")]);
    m.eos = vec![];
    let mut g = Generator::from_dir(Arc::new(m), &dir).unwrap();
    assert_eq!(g.eos_token_ids(), &[1, 265]);
    assert_eq!(
        g.apply_chat_template(&[ChatMessage::user("Hello")], true)
            .unwrap(),
        "<s>Hello"
    );
    // "\n" (id 265) is an EOS from generation_config.json
    let out = g.chat(&[ChatMessage::user("Hello")], &greedy(10)).unwrap();
    assert_eq!(out.text, " world");
    assert_eq!(out.finish_reason, FinishReason::Stop);
    std::fs::remove_dir_all(&dir).ok();

    // No template: the fallback is used.
    let g = Generator::new(Arc::new(MockLM::scripted(300, vec![])), sp_tokenizer());
    let s = g
        .apply_chat_template(&[ChatMessage::system("S"), ChatMessage::user("U")], true)
        .unwrap();
    assert_eq!(s, "<s>System: S\nUser: U\nAssistant:");
}

// ---------------------------------------------------------------------------
// Chat template kwargs and reasoning separation
// ---------------------------------------------------------------------------

/// Qwen3.5-style template: thinking on unless `enable_thinking` is false.
const THINKING_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n\
{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}\
{% endif %}";

/// Byte-level tokenizer plus non-special `<think>` / `</think>` tokens and
/// a special `<|im_end|>` (the EOS).
fn thinking_tokenizer() -> (Tokenizer, u32, u32, u32) {
    use tokenizers::AddedToken;
    let mut tok = byte_level_tokenizer();
    tok.add_tokens([
        AddedToken::from("<think>", false),
        AddedToken::from("</think>", false),
    ])
    .unwrap();
    tok.add_special_tokens([AddedToken::from("<|im_end|>", true)])
        .unwrap();
    let id = |s: &str| tok.token_to_id(s).unwrap();
    let (open, close, end) = (id("<think>"), id("</think>"), id("<|im_end|>"));
    (tok, open, close, end)
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

fn thinking_generator(script: Vec<u32>) -> Generator {
    let (tok, ..) = thinking_tokenizer();
    // Scripted output ends with id 1, the mock EOS.
    let m = MockLM::scripted(tok.get_vocab_size(true), script);
    Generator::new(Arc::new(m), tok)
        .with_chat_template(ChatTemplate::new(THINKING_TEMPLATE, "", "<|im_end|>").unwrap())
}

/// Collect `(text deltas, reasoning deltas)` of a streaming chat.
fn chat_events(
    g: &mut Generator,
    p: &GenerationParams,
) -> (xor_net::Generation, Vec<String>, Vec<String>) {
    let (mut text, mut reasoning) = (Vec::new(), Vec::new());
    let out = g
        .chat_stream_events(&[ChatMessage::user("2+2?")], p, |ev| {
            match ev {
                StreamEvent::Text(s) => text.push(s.to_owned()),
                StreamEvent::Reasoning(s) => reasoning.push(s.to_owned()),
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    (out, text, reasoning)
}

#[test]
fn chat_template_kwargs_enable_thinking() {
    let g = thinking_generator(vec![]);
    let msgs = [ChatMessage::user("hi")];
    let on = g.apply_chat_template(&msgs, true).unwrap();
    assert!(on.ends_with("<|im_start|>assistant\n<think>\n"), "{on:?}");
    let p = GenerationParams::default().with_enable_thinking(false);
    assert_eq!(
        p.chat_template_kwargs["enable_thinking"],
        serde_json::json!(false)
    );
    let off = g
        .apply_chat_template_with_kwargs(&msgs, true, &p.chat_template_kwargs)
        .unwrap();
    assert!(
        off.ends_with("assistant\n<think>\n\n</think>\n\n"),
        "{off:?}"
    );
    // Arbitrary kwargs reach the template; fixed variables can't be overridden.
    let t = ChatTemplate::new(
        "{{ greeting }} {{ messages|length }} {{ add_generation_prompt }}",
        "",
        "",
    )
    .unwrap();
    let kw = GenerationParams::default()
        .with_chat_template_kwarg("greeting", "yo")
        .with_chat_template_kwarg("add_generation_prompt", "hijack")
        .chat_template_kwargs;
    let s = t.render_with_kwargs(&msgs, true, &kw).unwrap();
    assert!(s.starts_with("yo 1 ") && !s.contains("hijack"), "{s:?}");
    let think = ChatTemplate::new(THINKING_TEMPLATE, "", "").unwrap();
    assert!(!t.has_thinking() && think.has_thinking() && think.supports_enable_thinking());
}

#[test]
fn reasoning_separated_when_prompt_opens_think() {
    let (_, _, close, _) = thinking_tokenizer();
    let mut script = bytes("Let me think.\nOK.\n");
    script.push(close);
    script.extend(bytes("\n\nThe answer is 4."));

    // Separate: reasoning split off, no tags anywhere, streams consistent.
    let mut g = thinking_generator(script.clone());
    let p = greedy(100).with_reasoning_format(ReasoningFormat::Separate);
    let (out, text, reasoning) = chat_events(&mut g, &p);
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert_eq!(out.text, "The answer is 4.");
    assert_eq!(out.reasoning.as_deref(), Some("Let me think.\nOK."));
    assert_eq!(text.concat(), out.text);
    assert_eq!(reasoning.concat(), "Let me think.\nOK.");
    assert!(
        text.iter()
            .chain(&reasoning)
            .all(|d| !d.is_empty() && !d.contains("think>"))
    );
    assert!(
        reasoning.len() > 3,
        "reasoning streams incrementally: {reasoning:?}"
    );

    // The old text-only callback sees just the answer.
    let mut g = thinking_generator(script.clone());
    let mut seen = String::new();
    let out = g
        .chat_stream(&[ChatMessage::user("2+2?")], &p, |d| {
            seen.push_str(d);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(seen, "The answer is 4.");
    assert_eq!(out.text, seen);

    // None: raw text including the closing tag, no reasoning.
    let mut g = thinking_generator(script);
    let (out, text, reasoning) = chat_events(&mut g, &greedy(100));
    assert_eq!(out.text, "Let me think.\nOK.\n</think>\n\nThe answer is 4.");
    assert_eq!(out.reasoning, None);
    assert!(reasoning.is_empty());
    assert_eq!(text.concat(), out.text);
}

#[test]
fn reasoning_tags_split_across_byte_tokens() {
    // The model spells the tags byte by byte (no dedicated tokens) and the
    // prompt has thinking disabled (closed block).
    let script = bytes("<think>\nhmm <b>\n</think>\n\nHi <thin");
    let mut g = thinking_generator(script);
    let p = greedy(100)
        .with_reasoning_format(ReasoningFormat::Separate)
        .with_enable_thinking(false);
    let (out, text, reasoning) = chat_events(&mut g, &p);
    assert_eq!(out.reasoning.as_deref(), Some("hmm <b>"));
    // After the first block, tag-like text is ordinary text.
    assert_eq!(out.text, "Hi <thin");
    assert_eq!(text.concat(), out.text);
    assert_eq!(reasoning.concat(), "hmm <b>");

    // A partial open tag that never completes is released at the end.
    let mut g = thinking_generator(bytes("a <thi"));
    let (out, text, _) = chat_events(&mut g, &p);
    assert_eq!(out.text, "a <thi");
    assert_eq!(text, ["a", " ", "<thi"]);

    // No think block at all: everything is text, reasoning None.
    let mut g = thinking_generator(bytes("Plain answer."));
    let (out, _, reasoning) = chat_events(&mut g, &p);
    assert_eq!((out.text.as_str(), out.reasoning), ("Plain answer.", None));
    assert!(reasoning.is_empty());
}

#[test]
fn reasoning_with_stop_string_length_and_abort() {
    let (_, _, close, _) = thinking_tokenizer();
    let mut script = bytes("plan");
    script.push(close);
    script.extend(bytes("Answer. END tail"));
    let p = greedy(100).with_reasoning_format(ReasoningFormat::Separate);

    let mut g = thinking_generator(script.clone());
    let (out, ..) = chat_events(&mut g, &p.clone().with_stop(["END"]));
    assert_eq!(out.finish_reason, FinishReason::StopString);
    assert_eq!(
        (out.text.as_str(), out.reasoning.as_deref()),
        ("Answer. ", Some("plan"))
    );

    // Length limit inside the reasoning: no text, reasoning so far.
    let mut g = thinking_generator(script.clone());
    let (out, text, _) = chat_events(&mut g, &p.clone().with_max_tokens(3));
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(
        (out.text.as_str(), out.reasoning.as_deref()),
        ("", Some("pla"))
    );
    assert!(text.is_empty());

    // Abort from a reasoning event.
    let mut g = thinking_generator(script);
    let mut n = 0;
    let out = g
        .chat_stream_events(&[ChatMessage::user("x")], &p, |ev| {
            assert!(matches!(ev, StreamEvent::Reasoning(_)));
            n += 1;
            if n == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Aborted);
    assert_eq!(out.reasoning.as_deref(), Some("pl"));
}

// ---------------------------------------------------------------------------
// Prefix reuse with rollback (truncate / checkpoint)
// ---------------------------------------------------------------------------

/// State of [`RollLM`]: a running hash of every token fed. In `kv` mode it
/// can roll back to any prefix (like a KV cache); otherwise (like a
/// recurrent layer) only to the current length, 0, or its checkpoint.
struct RollState {
    kv: bool,
    toks: Vec<u32>,
    h: u64,
    ckpt: Option<(usize, u64)>,
}

const ROLL_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn roll(h: u64, t: u32) -> u64 {
    (h ^ t as u64).wrapping_mul(0x100_0000_01b3).rotate_left(17) ^ 0xdead_beef
}

impl ModelState for RollState {
    fn reset(&mut self) -> anyhow::Result<()> {
        self.toks.clear();
        self.h = ROLL_SEED;
        self.ckpt = None;
        Ok(())
    }
    fn truncate(&mut self, len: usize) -> bool {
        if len == self.toks.len() {
            return true;
        }
        if len > self.toks.len() {
            return false;
        }
        if self.kv {
            self.toks.truncate(len);
            self.h = self.toks.iter().fold(ROLL_SEED, |h, &t| roll(h, t));
            return true;
        }
        match self.ckpt {
            Some((n, h)) if n == len => {
                self.toks.truncate(len);
                self.h = h;
                true
            }
            _ => false,
        }
    }
    fn checkpoint(&mut self) {
        if !self.kv {
            self.ckpt = Some((self.toks.len(), self.h));
        }
    }
    fn checkpoint_len(&self) -> Option<usize> {
        self.ckpt.map(|c| c.0)
    }
    fn needs_checkpoint(&self) -> bool {
        !self.kv
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Logits are a pseudo-random function of the whole history, so any
/// rollback bug changes them.
struct RollLM {
    kv: bool,
    vocab: usize,
    /// (pos, len, logits) of every forward call.
    log: Mutex<Vec<(usize, usize, Vec<f32>)>>,
}

impl CausalLM for RollLM {
    fn new_state(&self) -> anyhow::Result<Box<dyn ModelState>> {
        Ok(Box::new(RollState {
            kv: self.kv,
            toks: Vec::new(),
            h: ROLL_SEED,
            ckpt: None,
        }))
    }
    fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut dyn ModelState,
    ) -> anyhow::Result<Logits> {
        let st = downcast_state::<RollState>(state)?;
        anyhow::ensure!(pos == st.toks.len(), "pos {pos} != fed {}", st.toks.len());
        for &t in tokens {
            st.toks.push(t);
            st.h = roll(st.h, t);
        }
        let l: Vec<f32> = (0..self.vocab as u32)
            .map(|i| (roll(st.h, i) >> 40) as f32)
            .collect();
        self.log
            .lock()
            .unwrap()
            .push((pos, tokens.len(), l.clone()));
        Ok(l)
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn max_seq_len(&self) -> usize {
        4096
    }
    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
}

/// Runs `prompt` on `g`; returns the output tokens, the cached count and
/// the logits of the last prefill chunk followed by every decode step.
fn roll_run(
    g: &mut Generator,
    model: &RollLM,
    prompt: &[u32],
    p: &GenerationParams,
) -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    model.log.lock().unwrap().clear();
    let out = g
        .generate_from_tokens(prompt, p, |_| ControlFlow::Continue(()))
        .unwrap();
    let log = model.log.lock().unwrap();
    let first = log
        .iter()
        .position(|(pos, len, _)| pos + len == prompt.len())
        .expect("last prefill chunk");
    let logits = log[first..].iter().map(|(_, _, l)| l.clone()).collect();
    assert_eq!(out.usage.prompt_tokens, prompt.len());
    assert_eq!(
        out.timings.prefill_tokens,
        prompt.len() - out.usage.cached_prompt_tokens
    );
    (out.tokens, out.usage.cached_prompt_tokens, logits)
}

#[test]
fn prefix_reuse_rolls_back_and_matches_fresh_prefill() {
    for kv in [true, false] {
        let model = Arc::new(RollLM {
            kv,
            vocab: 259,
            log: Mutex::new(Vec::new()),
        });
        let p = greedy(6).with_prefill_chunk_size(4);
        let mut g = Generator::new(model.clone(), sp_tokenizer());
        let fresh = |prompt: &[u32]| {
            let mut cold = Generator::new(model.clone(), sp_tokenizer());
            cold.set_prefix_caching(false);
            roll_run(&mut cold, &model, prompt, &p)
        };

        let p1: Vec<u32> = (0..10).map(|i| byte(b'a' + i)).collect();
        let (out1, c1, _) = roll_run(&mut g, &model, &p1, &p);
        assert_eq!(c1, 0);
        assert_eq!(out1.len(), 6);

        // Turn 2 keeps 2 of the generated tokens, then diverges (the chat
        // template re-rendered the reply differently).
        let mut p2 = p1.clone();
        p2.extend_from_slice(&out1[..2]);
        p2.extend_from_slice(&[byte(b'x'), byte(b'y'), byte(b'z')]);
        // Turn 3 diverges inside turn 2's prompt, before its checkpoint.
        let mut p3 = p2[..5].to_vec();
        p3.extend_from_slice(&[byte(b'q'), byte(b'r')]);
        // Turn 4 extends turn 3's prompt (pure extension, no rollback).
        let mut p4 = p3.clone();
        p4.push(byte(b's'));
        // Turn 5 diverges right after turn 4's prompt: the checkpoint.
        let mut p5 = p4.clone();
        p5.extend_from_slice(&[byte(b't'), byte(b'u')]);
        // Turn 6 repeats turn 5 exactly: one token must still be prefilled.
        let p6 = p5.clone();

        let cases: [(&[u32], usize, usize); 5] = [
            // (prompt, cached with a KV state, cached with a recurrent state)
            (&p2, 12, 10),
            (&p3, 5, 0),
            (&p4, p3.len(), p3.len()),
            (&p5, p4.len(), p4.len()),
            (&p6, p6.len() - 1, 0),
        ];
        for (i, (prompt, want_kv, want_rec)) in cases.into_iter().enumerate() {
            let (toks, cached, logits) = roll_run(&mut g, &model, prompt, &p);
            let want = if kv { want_kv } else { want_rec };
            assert_eq!(cached, want, "kv={kv} turn {}", i + 2);
            let (ftoks, fcached, flogits) = fresh(prompt);
            assert_eq!(fcached, 0);
            assert_eq!(toks, ftoks, "kv={kv} turn {}", i + 2);
            assert_eq!(logits, flogits, "kv={kv} turn {}: logits differ", i + 2);
        }
    }
}

#[test]
fn chat_checkpoint_at_history_end_for_recurrent_state() {
    // The generation prompt `<think>\n\n</think>\n\n` is not reproduced when
    // the reply is re-rendered, so the next prompt diverges *before* the end
    // of this one: a recurrent state must checkpoint at the end of the
    // conversation history instead.
    let (tok, ..) = thinking_tokenizer();
    let model = Arc::new(RollLM {
        kv: false,
        vocab: tok.get_vocab_size(true),
        log: Mutex::new(Vec::new()),
    });
    let template = || ChatTemplate::new(THINKING_TEMPLATE, "", "<|im_end|>").unwrap();
    let mut g = Generator::new(model.clone(), tok.clone()).with_chat_template(template());
    let p = greedy(5).with_enable_thinking(false);
    let mut msgs = vec![ChatMessage::user("hello")];
    let mut prev_hist = 0;
    for turn in 0..4 {
        let hist = g
            .apply_chat_template_with_kwargs(&msgs, false, &p.chat_template_kwargs)
            .unwrap();
        let hist_len = g.encode(&hist, false).unwrap().len();
        let out = g.chat(&msgs, &p).unwrap();
        let mut cold = Generator::new(model.clone(), tok.clone()).with_chat_template(template());
        cold.set_prefix_caching(false);
        let fresh = cold.chat(&msgs, &p).unwrap();
        assert_eq!(out.tokens, fresh.tokens, "turn {turn}");
        // Everything up to the previous turn's generation prompt is reused.
        assert_eq!(out.usage.cached_prompt_tokens, prev_hist, "turn {turn}");
        prev_hist = hist_len;
        msgs.push(ChatMessage::assistant(format!("reply {turn}")));
        msgs.push(ChatMessage::user(format!("again {turn}")));
    }
}
