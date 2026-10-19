//! OpenAI / llama.cpp request validation and response bodies.

use serde_json::{Map, Value, json};

use crate::engine::{ChatMessage, FinishReason, GenResult, Prompt, SamplingParams};
use crate::http::Response;

/// Which API surface a request came in on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Chat,
    Completions,
    /// llama.cpp native `/completion`.
    LlamaCpp,
}

/// OpenAI-style error.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn invalid(param: &str, message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            kind: "invalid_request_error",
            param: (!param.is_empty()).then(|| param.to_string()),
            code: None,
        }
    }

    pub fn new(
        status: u16,
        kind: &'static str,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            kind,
            param: None,
            code: Some(code),
        }
    }

    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub fn body(&self) -> Value {
        json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }})
    }

    pub fn response(&self) -> Response {
        Response::json(self.status, &self.body())
    }
}

/// A validated generation request (before tokenization).
#[derive(Debug, Clone)]
pub struct ParsedRequest {
    pub prompt: Prompt,
    /// `None` = use the server default (bounded by the context).
    pub max_tokens: Option<usize>,
    pub sampling: SamplingParams,
    /// Whether the client provided a seed.
    pub seeded: bool,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    /// Extra chat template variables: vLLM-style `chat_template_kwargs`
    /// plus a top-level `enable_thinking` (chat only).
    pub chat_template_kwargs: Map<String, Value>,
}

pub const MAX_STOP_STRINGS: usize = 16;
pub const MAX_STOP_LEN: usize = 256;

fn get_f32(o: &Map<String, Value>, key: &str) -> Result<Option<f32>, ApiError> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => match n.as_f64() {
            Some(f) if f.is_finite() => Ok(Some(f as f32)),
            _ => Err(ApiError::invalid(
                key,
                format!("'{key}' must be a finite number"),
            )),
        },
        Some(_) => Err(ApiError::invalid(key, format!("'{key}' must be a number"))),
    }
}

fn get_i64(o: &Map<String, Value>, key: &str) -> Result<Option<i64>, ApiError> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Ok(Some(i))
            } else if n.as_u64().is_some() {
                Ok(Some(i64::MAX))
            } else {
                Err(ApiError::invalid(
                    key,
                    format!("'{key}' must be an integer"),
                ))
            }
        }
        Some(_) => Err(ApiError::invalid(
            key,
            format!("'{key}' must be an integer"),
        )),
    }
}

fn get_bool(o: &Map<String, Value>, key: &str) -> Result<Option<bool>, ApiError> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ApiError::invalid(key, format!("'{key}' must be a boolean"))),
    }
}

fn in_range(key: &str, v: Option<f32>, lo: f32, hi: f32, lo_open: bool) -> Result<(), ApiError> {
    if let Some(v) = v {
        let ok = if lo_open { v > lo } else { v >= lo } && v <= hi;
        if !ok {
            let l = if lo_open { "(" } else { "[" };
            return Err(ApiError::invalid(
                key,
                format!("'{key}' must be in {l}{lo}, {hi}], got {v}"),
            ));
        }
    }
    Ok(())
}

fn parse_content(v: &Value, idx: usize) -> Result<String, ApiError> {
    let param = format!("messages[{idx}].content");
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match (
                    p.get("type").and_then(Value::as_str),
                    p.get("text").and_then(Value::as_str),
                ) {
                    (Some("text"), Some(t)) => out.push_str(t),
                    _ => {
                        return Err(ApiError::invalid(
                            &param,
                            "only text content parts are supported",
                        ));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::invalid(
            &param,
            "content must be a string or an array of parts",
        )),
    }
}

fn parse_messages(o: &Map<String, Value>) -> Result<Vec<ChatMessage>, ApiError> {
    let arr = match o.get("messages") {
        Some(Value::Array(a)) if !a.is_empty() => a,
        Some(Value::Array(_)) => {
            return Err(ApiError::invalid(
                "messages",
                "'messages' must not be empty",
            ));
        }
        None | Some(Value::Null) => {
            return Err(ApiError::invalid("messages", "'messages' is required"));
        }
        Some(_) => return Err(ApiError::invalid("messages", "'messages' must be an array")),
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let m = m.as_object().ok_or_else(|| {
            ApiError::invalid(&format!("messages[{i}]"), "message must be an object")
        })?;
        let role = match m.get("role") {
            Some(Value::String(r)) if !r.is_empty() => r.clone(),
            _ => {
                return Err(ApiError::invalid(
                    &format!("messages[{i}].role"),
                    "'role' must be a non-empty string",
                ));
            }
        };
        let content = parse_content(m.get("content").unwrap_or(&Value::Null), i)?;
        let reasoning_content = match m.get("reasoning_content") {
            None | Some(Value::Null) => None,
            Some(Value::String(r)) => Some(r.clone()),
            Some(_) => {
                return Err(ApiError::invalid(
                    &format!("messages[{i}].reasoning_content"),
                    "'reasoning_content' must be a string",
                ));
            }
        };
        out.push(ChatMessage {
            role,
            content,
            reasoning_content,
        });
    }
    Ok(out)
}

fn parse_prompt(o: &Map<String, Value>) -> Result<Prompt, ApiError> {
    let token = |x: &Value| -> Option<u32> { x.as_u64().and_then(|v| u32::try_from(v).ok()) };
    match o.get("prompt") {
        Some(Value::String(s)) => Ok(Prompt::Text(s.clone())),
        Some(Value::Array(a)) if a.is_empty() => {
            Err(ApiError::invalid("prompt", "'prompt' must not be empty"))
        }
        Some(Value::Array(a)) => {
            if a.iter().all(Value::is_number) {
                let ids: Option<Vec<u32>> = a.iter().map(token).collect();
                return ids.map(Prompt::Tokens).ok_or_else(|| {
                    ApiError::invalid("prompt", "token ids must be non-negative 32-bit integers")
                });
            }
            if a.len() == 1 {
                match &a[0] {
                    Value::String(s) => return Ok(Prompt::Text(s.clone())),
                    Value::Array(t) if !t.is_empty() => {
                        let ids: Option<Vec<u32>> = t.iter().map(token).collect();
                        return ids.map(Prompt::Tokens).ok_or_else(|| {
                            ApiError::invalid(
                                "prompt",
                                "token ids must be non-negative 32-bit integers",
                            )
                        });
                    }
                    _ => {}
                }
            }
            Err(ApiError::invalid(
                "prompt",
                "batched prompts are not supported; send a single string or token array",
            ))
        }
        None | Some(Value::Null) => Err(ApiError::invalid("prompt", "'prompt' is required")),
        Some(_) => Err(ApiError::invalid(
            "prompt",
            "'prompt' must be a string or token array",
        )),
    }
}

fn parse_stop(o: &Map<String, Value>) -> Result<Vec<String>, ApiError> {
    let v = match o.get("stop") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(v) => v,
    };
    let list: Vec<String> = match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::invalid("stop", "'stop' entries must be strings"))
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(ApiError::invalid(
                "stop",
                "'stop' must be a string or array of strings",
            ));
        }
    };
    if list.len() > MAX_STOP_STRINGS {
        return Err(ApiError::invalid(
            "stop",
            format!("at most {MAX_STOP_STRINGS} stop strings are allowed"),
        ));
    }
    if list.iter().any(|s| s.len() > MAX_STOP_LEN) {
        return Err(ApiError::invalid(
            "stop",
            format!("stop strings must be at most {MAX_STOP_LEN} bytes"),
        ));
    }
    Ok(list.into_iter().filter(|s| !s.is_empty()).collect())
}

/// Validate a JSON body for `api`. Tokenization and context checks happen
/// later (they need the engine).
pub fn parse_request(
    api: Api,
    body: &[u8],
    max_tokens_cap: usize,
) -> Result<ParsedRequest, ApiError> {
    if body.is_empty() {
        return Err(ApiError::invalid("", "request body is empty"));
    }
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid("", format!("invalid JSON body: {e}")))?;
    let o = v
        .as_object()
        .ok_or_else(|| ApiError::invalid("", "request body must be a JSON object"))?;

    let prompt = match api {
        Api::Chat => Prompt::Chat(parse_messages(o)?),
        Api::Completions | Api::LlamaCpp => parse_prompt(o)?,
    };

    if get_i64(o, "n")?.is_some_and(|n| n != 1) {
        return Err(ApiError::invalid("n", "only n=1 is supported"));
    }
    if api == Api::Completions && get_i64(o, "best_of")?.is_some_and(|b| b != 1) {
        return Err(ApiError::invalid("best_of", "only best_of=1 is supported"));
    }

    // max tokens: OpenAI `max_tokens` / `max_completion_tokens`, llama.cpp
    // `n_predict` (-1 = unlimited, i.e. the server default/context).
    let mut max_tokens = None;
    for key in ["max_completion_tokens", "max_tokens", "n_predict"] {
        if let Some(n) = get_i64(o, key)? {
            if key == "n_predict" && n < 0 {
                continue;
            }
            if n < 1 {
                return Err(ApiError::invalid(key, format!("'{key}' must be >= 1")));
            }
            if n as u64 > max_tokens_cap as u64 {
                return Err(ApiError::invalid(
                    key,
                    format!("'{key}' must be <= {max_tokens_cap}"),
                ));
            }
            max_tokens = Some(n as usize);
            break;
        }
    }

    let temperature = get_f32(o, "temperature")?;
    in_range("temperature", temperature, 0.0, 2.0, false)?;
    let top_p = get_f32(o, "top_p")?;
    in_range("top_p", top_p, 0.0, 1.0, true)?;
    let min_p = get_f32(o, "min_p")?;
    in_range("min_p", min_p, 0.0, 1.0, false)?;
    let presence_penalty = get_f32(o, "presence_penalty")?;
    in_range("presence_penalty", presence_penalty, -2.0, 2.0, false)?;
    let frequency_penalty = get_f32(o, "frequency_penalty")?;
    in_range("frequency_penalty", frequency_penalty, -2.0, 2.0, false)?;
    let mut repetition_penalty = get_f32(o, "repetition_penalty")?;
    let mut rp_key = "repetition_penalty";
    if repetition_penalty.is_none() {
        repetition_penalty = get_f32(o, "repeat_penalty")?;
        rp_key = "repeat_penalty";
    }
    in_range(rp_key, repetition_penalty, 0.0, 10.0, true)?;
    let top_k = match get_i64(o, "top_k")? {
        None => 0,
        Some(k) if k < 0 => {
            return Err(ApiError::invalid("top_k", "'top_k' must be >= 0"));
        }
        Some(k) => usize::try_from(k).unwrap_or(usize::MAX),
    };
    let seed = get_i64(o, "seed")?;
    // llama.cpp uses seed = -1 for "random".
    let seed = seed.filter(|&s| !(api == Api::LlamaCpp && s == -1));

    let stream = get_bool(o, "stream")?.unwrap_or(false);
    let include_usage = match o.get("stream_options") {
        None | Some(Value::Null) => false,
        Some(Value::Object(so)) => get_bool(so, "include_usage")
            .map_err(|_| ApiError::invalid("stream_options.include_usage", "must be a boolean"))?
            .unwrap_or(false),
        Some(_) => {
            return Err(ApiError::invalid(
                "stream_options",
                "'stream_options' must be an object",
            ));
        }
    };
    let wants_logprobs = match api {
        Api::Chat => get_bool(o, "logprobs")? == Some(true),
        Api::Completions => get_i64(o, "logprobs")?.is_some_and(|n| n > 0),
        Api::LlamaCpp => false,
    };
    if wants_logprobs {
        return Err(ApiError::invalid("logprobs", "logprobs are not supported"));
    }

    let sampling = SamplingParams {
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_k,
        min_p: min_p.unwrap_or(0.0),
        presence_penalty: presence_penalty.unwrap_or(0.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        seed: seed.map(|s| s as u64).unwrap_or(0),
    };

    let mut chat_template_kwargs = Map::new();
    if api == Api::Chat {
        match o.get("chat_template_kwargs") {
            None | Some(Value::Null) => {}
            Some(Value::Object(kw)) => chat_template_kwargs = kw.clone(),
            Some(_) => {
                return Err(ApiError::invalid(
                    "chat_template_kwargs",
                    "'chat_template_kwargs' must be an object",
                ));
            }
        }
        if let Some(on) = get_bool(o, "enable_thinking")? {
            // An explicit chat_template_kwargs entry wins.
            chat_template_kwargs
                .entry("enable_thinking")
                .or_insert(Value::Bool(on));
        }
    }

    Ok(ParsedRequest {
        prompt,
        max_tokens,
        sampling,
        seeded: seed.is_some(),
        stop: parse_stop(o)?,
        stream,
        include_usage,
        chat_template_kwargs,
    })
}

/// Resolve the effective `max_tokens` against the context window.
pub fn resolve_max_tokens(
    requested: Option<usize>,
    prompt_len: usize,
    ctx: usize,
    default_max: usize,
    api: Api,
) -> Result<usize, ApiError> {
    let param = if api == Api::Chat {
        "messages"
    } else {
        "prompt"
    };
    if prompt_len == 0 {
        return Err(ApiError::invalid(
            param,
            "prompt is empty after tokenization",
        ));
    }
    if prompt_len >= ctx {
        return Err(ApiError::invalid(
            param,
            format!(
                "This model's maximum context length is {ctx} tokens, but the prompt has {prompt_len} tokens."
            ),
        )
        .with_code("context_length_exceeded"));
    }
    let room = ctx - prompt_len;
    match requested {
        Some(n) if n > room => Err(ApiError::invalid(
            "max_tokens",
            format!(
                "This model's maximum context length is {ctx} tokens. However, you requested {} tokens ({prompt_len} in the prompt, {n} for the completion).",
                prompt_len + n
            ),
        )
        .with_code("context_length_exceeded")),
        Some(n) => Ok(n),
        None => Ok(default_max.min(room).max(1)),
    }
}

pub fn finish_reason(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Length => "length",
        _ => "stop",
    }
}

pub fn usage(prompt: usize, completion: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

/// Identity shared by all chunks of one response.
pub struct RespMeta<'a> {
    pub id: &'a str,
    pub created: u64,
    pub model: &'a str,
    pub prompt_tokens: usize,
}

/// llama.cpp-style `timings` object.
fn timings(res: &GenResult) -> Value {
    match &res.timings {
        Some(t) => json!({
            "prompt_n": t.prefill_tokens,
            "prompt_ms": t.prefill_ms(),
            "prompt_per_second": t.prefill_tok_per_s(),
            "predicted_n": t.decode_tokens,
            "predicted_ms": t.decode.as_secs_f64() * 1e3,
            "predicted_per_second": t.decode_tok_per_s(),
        }),
        None => Value::Null,
    }
}

/// Non-streaming response body. `reasoning` is `Some` when reasoning
/// separation is on (chat only): it becomes `message.reasoning_content`
/// (`null` when the model produced none).
pub fn full_body(
    api: Api,
    m: &RespMeta,
    text: &str,
    reasoning: Option<&str>,
    res: &GenResult,
) -> Value {
    let finish = finish_reason(res.finish);
    let mut v = match api {
        Api::Chat => {
            let mut msg = json!({"role": "assistant", "content": text});
            if let Some(r) = reasoning {
                msg["reasoning_content"] = if r.is_empty() {
                    Value::Null
                } else {
                    Value::String(r.to_owned())
                };
            }
            json!({
                "id": m.id,
                "object": "chat.completion",
                "created": m.created,
                "model": m.model,
                "choices": [{
                    "index": 0,
                    "message": msg,
                    "logprobs": Value::Null,
                    "finish_reason": finish,
                }],
                "usage": usage(m.prompt_tokens, res.completion_tokens),
            })
        }
        Api::Completions => json!({
            "id": m.id,
            "object": "text_completion",
            "created": m.created,
            "model": m.model,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": Value::Null,
                "finish_reason": finish,
            }],
            "usage": usage(m.prompt_tokens, res.completion_tokens),
        }),
        Api::LlamaCpp => return llama_final(m, text, res),
    };
    if res.timings.is_some() {
        v["timings"] = timings(res);
    }
    v
}

fn llama_final(m: &RespMeta, text: &str, res: &GenResult) -> Value {
    let stop_type = match res.finish {
        FinishReason::Eos => "eos",
        FinishReason::StopString => "word",
        FinishReason::Length => "limit",
        FinishReason::Cancelled => "none",
    };
    let mut v = json!({
        "content": text,
        "stop": true,
        "model": m.model,
        "tokens_predicted": res.completion_tokens,
        "tokens_evaluated": m.prompt_tokens,
        "stop_type": stop_type,
        "stopping_word": res.stop_word.clone().unwrap_or_default(),
        "truncated": false,
    });
    if res.timings.is_some() {
        v["timings"] = timings(res);
    }
    v
}

/// Chat streaming chunk carrying a reasoning delta
/// (`delta.reasoning_content`, DeepSeek / vLLM convention).
pub fn reasoning_chunk(m: &RespMeta, delta: &str, include_usage: bool) -> Value {
    let mut v = json!({
        "id": m.id,
        "object": "chat.completion.chunk",
        "created": m.created,
        "model": m.model,
        "choices": [{"index": 0, "delta": {"reasoning_content": delta}, "logprobs": Value::Null, "finish_reason": Value::Null}],
    });
    if include_usage {
        v["usage"] = Value::Null;
    }
    v
}

/// Streaming chunk carrying `delta`. `first` adds the assistant role for chat.
pub fn delta_chunk(api: Api, m: &RespMeta, delta: &str, first: bool, include_usage: bool) -> Value {
    let mut v = match api {
        Api::Chat => {
            let d = if first {
                json!({"role": "assistant", "content": delta})
            } else {
                json!({"content": delta})
            };
            json!({
                "id": m.id,
                "object": "chat.completion.chunk",
                "created": m.created,
                "model": m.model,
                "choices": [{"index": 0, "delta": d, "logprobs": Value::Null, "finish_reason": Value::Null}],
            })
        }
        Api::Completions => json!({
            "id": m.id,
            "object": "text_completion",
            "created": m.created,
            "model": m.model,
            "choices": [{"text": delta, "index": 0, "logprobs": Value::Null, "finish_reason": Value::Null}],
        }),
        Api::LlamaCpp => return json!({"content": delta, "stop": false}),
    };
    if include_usage {
        v["usage"] = Value::Null;
    }
    v
}

/// The closing chunks of a stream, in order (without `[DONE]`).
pub fn final_chunks(api: Api, m: &RespMeta, res: &GenResult, include_usage: bool) -> Vec<Value> {
    let finish = finish_reason(res.finish);
    let mut out = Vec::new();
    match api {
        Api::Chat | Api::Completions => {
            let mut last = if api == Api::Chat {
                json!({
                    "id": m.id,
                    "object": "chat.completion.chunk",
                    "created": m.created,
                    "model": m.model,
                    "choices": [{"index": 0, "delta": {}, "logprobs": Value::Null, "finish_reason": finish}],
                })
            } else {
                json!({
                    "id": m.id,
                    "object": "text_completion",
                    "created": m.created,
                    "model": m.model,
                    "choices": [{"text": "", "index": 0, "logprobs": Value::Null, "finish_reason": finish}],
                })
            };
            if include_usage {
                last["usage"] = Value::Null;
                out.push(last);
                let mut u = json!({
                    "id": m.id,
                    "object": if api == Api::Chat { "chat.completion.chunk" } else { "text_completion" },
                    "created": m.created,
                    "model": m.model,
                    "choices": [],
                });
                u["usage"] = usage(m.prompt_tokens, res.completion_tokens);
                out.push(u);
            } else {
                out.push(last);
            }
        }
        Api::LlamaCpp => out.push(llama_final(m, "", res)),
    }
    out
}

/// Whether the stream ends with `data: [DONE]` (OpenAI yes, llama.cpp no).
pub fn sends_done(api: Api) -> bool {
    api != Api::LlamaCpp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(body: &str) -> Result<ParsedRequest, ApiError> {
        parse_request(Api::Chat, body.as_bytes(), 4096)
    }
    fn cmpl(body: &str) -> Result<ParsedRequest, ApiError> {
        parse_request(Api::Completions, body.as_bytes(), 4096)
    }
    fn err_param(r: Result<ParsedRequest, ApiError>) -> Option<String> {
        r.err().and_then(|e| {
            assert_eq!(e.status, 400);
            e.param
        })
    }

    #[test]
    fn valid_chat_with_all_knobs() {
        let r = chat(
            r#"{"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],
            "temperature":0.5,"top_p":0.9,"top_k":40,"min_p":0.05,"seed":42,"stop":["\n\n","END"],
            "presence_penalty":0.5,"frequency_penalty":-0.5,"repetition_penalty":1.1,"max_tokens":16,"n":1,
            "stream":true,"stream_options":{"include_usage":true}}"#,
        )
        .unwrap();
        assert_eq!(
            r.prompt,
            Prompt::Chat(vec![
                ChatMessage::new("system", "s"),
                ChatMessage::new("user", "ab"),
            ])
        );
        let s = &r.sampling;
        assert_eq!(
            (s.temperature, s.top_p, s.top_k, s.min_p),
            (0.5, 0.9, 40, 0.05)
        );
        assert_eq!(
            (
                s.presence_penalty,
                s.frequency_penalty,
                s.repetition_penalty
            ),
            (0.5, -0.5, 1.1)
        );
        assert_eq!(s.seed, 42);
        assert!(r.seeded && r.stream && r.include_usage);
        assert_eq!(r.max_tokens, Some(16));
        assert_eq!(r.stop, vec!["\n\n", "END"]);
    }

    #[test]
    fn range_validation() {
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","temperature":2.5}"#)).as_deref(),
            Some("temperature")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","temperature":-0.1}"#)).as_deref(),
            Some("temperature")
        );
        assert!(cmpl(r#"{"prompt":"a","temperature":0}"#).is_ok());
        assert!(cmpl(r#"{"prompt":"a","temperature":2}"#).is_ok());
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","top_p":0}"#)).as_deref(),
            Some("top_p")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","top_p":1.01}"#)).as_deref(),
            Some("top_p")
        );
        assert!(cmpl(r#"{"prompt":"a","top_p":1}"#).is_ok());
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","top_k":-1}"#)).as_deref(),
            Some("top_k")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","top_k":1.5}"#)).as_deref(),
            Some("top_k")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","max_tokens":0}"#)).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","max_tokens":5000}"#)).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","max_tokens":"5"}"#)).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","n":2}"#)).as_deref(),
            Some("n")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","repetition_penalty":0}"#)).as_deref(),
            Some("repetition_penalty")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","presence_penalty":3}"#)).as_deref(),
            Some("presence_penalty")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","min_p":1.5}"#)).as_deref(),
            Some("min_p")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","stream":"yes"}"#)).as_deref(),
            Some("stream")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","stop":5}"#)).as_deref(),
            Some("stop")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","stop":[1]}"#)).as_deref(),
            Some("stop")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","logprobs":2}"#)).as_deref(),
            Some("logprobs")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":"a","seed":"x"}"#)).as_deref(),
            Some("seed")
        );
    }

    #[test]
    fn body_and_prompt_shapes() {
        assert!(cmpl("").is_err());
        assert!(
            cmpl("{not json")
                .unwrap_err()
                .message
                .contains("invalid JSON")
        );
        assert!(cmpl("[1,2]").is_err());
        assert_eq!(err_param(cmpl("{}")).as_deref(), Some("prompt"));
        assert_eq!(
            err_param(cmpl(r#"{"prompt":["a","b"]}"#)).as_deref(),
            Some("prompt")
        );
        assert_eq!(
            err_param(cmpl(r#"{"prompt":[-1]}"#)).as_deref(),
            Some("prompt")
        );
        assert_eq!(
            cmpl(r#"{"prompt":["a"]}"#).unwrap().prompt,
            Prompt::Text("a".into())
        );
        assert_eq!(
            cmpl(r#"{"prompt":[1,2]}"#).unwrap().prompt,
            Prompt::Tokens(vec![1, 2])
        );
        assert_eq!(
            cmpl(r#"{"prompt":[[3]]}"#).unwrap().prompt,
            Prompt::Tokens(vec![3])
        );
        assert_eq!(err_param(chat("{}")).as_deref(), Some("messages"));
        assert_eq!(
            err_param(chat(r#"{"messages":[]}"#)).as_deref(),
            Some("messages")
        );
        assert_eq!(
            err_param(chat(r#"{"messages":[{"content":"x"}]}"#)).as_deref(),
            Some("messages[0].role")
        );
        assert_eq!(
            err_param(chat(
                r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#
            ))
            .as_deref(),
            Some("messages[0].content")
        );
        assert_eq!(
            cmpl(r#"{"prompt":"a","stop":"x"}"#).unwrap().stop,
            vec!["x"]
        );
    }

    #[test]
    fn llama_cpp_aliases() {
        let r = parse_request(
            Api::LlamaCpp,
            br#"{"prompt":"a","n_predict":-1,"repeat_penalty":1.2,"seed":-1}"#,
            4096,
        )
        .unwrap();
        assert_eq!(r.max_tokens, None);
        assert_eq!(r.sampling.repetition_penalty, 1.2);
        assert!(!r.seeded);
        let r = parse_request(Api::LlamaCpp, br#"{"prompt":"a","n_predict":7}"#, 4096).unwrap();
        assert_eq!(r.max_tokens, Some(7));
    }

    #[test]
    fn context_length_checks() {
        assert_eq!(
            resolve_max_tokens(Some(10), 90, 100, 256, Api::Completions),
            Ok(10)
        );
        let e = resolve_max_tokens(Some(11), 90, 100, 256, Api::Completions).unwrap_err();
        assert_eq!(e.code, Some("context_length_exceeded"));
        let e = resolve_max_tokens(None, 100, 100, 256, Api::Chat).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("messages"));
        assert_eq!(resolve_max_tokens(None, 90, 100, 256, Api::Chat), Ok(10));
        assert_eq!(resolve_max_tokens(None, 10, 100, 5, Api::Chat), Ok(5));
        assert!(resolve_max_tokens(None, 0, 100, 5, Api::Chat).is_err());
    }

    #[test]
    fn chat_template_kwargs_and_enable_thinking() {
        let r = chat(r#"{"messages":[{"role":"user","content":"x"}]}"#).unwrap();
        assert!(r.chat_template_kwargs.is_empty());
        let r = chat(
            r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":false,
            "chat_template_kwargs":{"foo":[1]}}"#,
        )
        .unwrap();
        assert_eq!(
            Value::Object(r.chat_template_kwargs),
            json!({"foo":[1],"enable_thinking":false})
        );
        // chat_template_kwargs wins over the top-level flag.
        let r = chat(
            r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":false,
            "chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        assert_eq!(r.chat_template_kwargs["enable_thinking"], true);
        assert_eq!(
            err_param(chat(
                r#"{"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":[]}"#
            ))
            .as_deref(),
            Some("chat_template_kwargs")
        );
        assert_eq!(
            err_param(chat(
                r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":"no"}"#
            ))
            .as_deref(),
            Some("enable_thinking")
        );
        // History reasoning is passed through to the template.
        let r =
            chat(r#"{"messages":[{"role":"assistant","content":"4","reasoning_content":"2+2"}]}"#)
                .unwrap();
        let Prompt::Chat(m) = r.prompt else { panic!() };
        assert_eq!(m[0].reasoning_content.as_deref(), Some("2+2"));
        assert!(
            chat(r#"{"messages":[{"role":"assistant","content":"4","reasoning_content":1}]}"#)
                .is_err()
        );
        // Ignored on the completions API.
        assert!(
            cmpl(r#"{"prompt":"a","chat_template_kwargs":5}"#)
                .unwrap()
                .chat_template_kwargs
                .is_empty()
        );
    }

    #[test]
    fn reasoning_in_bodies() {
        let m = RespMeta {
            id: "x",
            created: 1,
            model: "m",
            prompt_tokens: 3,
        };
        let res = GenResult {
            finish: FinishReason::Eos,
            completion_tokens: 2,
            stop_word: None,
            timings: None,
        };
        let v = full_body(Api::Chat, &m, "4", Some("2+2"), &res);
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "2+2");
        assert_eq!(v["choices"][0]["message"]["content"], "4");
        let v = full_body(Api::Chat, &m, "4", Some(""), &res);
        assert!(v["choices"][0]["message"]["reasoning_content"].is_null());
        assert!(
            v["choices"][0]["message"]
                .get("reasoning_content")
                .is_some()
        );
        let v = full_body(Api::Chat, &m, "4", None, &res);
        assert!(
            v["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
        assert!(v.get("timings").is_none());
        let c = reasoning_chunk(&m, "hm", false);
        assert_eq!(c["choices"][0]["delta"]["reasoning_content"], "hm");
        assert!(c["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn error_body_shape() {
        let e = ApiError::invalid("top_p", "bad");
        assert_eq!(
            e.body(),
            json!({"error":{"message":"bad","type":"invalid_request_error","param":"top_p","code":null}})
        );
    }

    #[test]
    fn final_stream_chunks() {
        let m = RespMeta {
            id: "x",
            created: 1,
            model: "m",
            prompt_tokens: 3,
        };
        let res = GenResult {
            finish: FinishReason::Length,
            completion_tokens: 2,
            stop_word: None,
            timings: None,
        };
        let c = final_chunks(Api::Chat, &m, &res, true);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["choices"][0]["finish_reason"], "length");
        assert_eq!(c[1]["usage"]["total_tokens"], 5);
        assert_eq!(c[1]["choices"].as_array().map(Vec::len), Some(0));
        let c = final_chunks(Api::Completions, &m, &res, false);
        assert_eq!(c.len(), 1);
        assert!(c[0].get("usage").is_none());
    }
}
