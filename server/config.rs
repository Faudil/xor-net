//! Server configuration: CLI flags, with `XORNET_*` environment variables
//! as fallback, then defaults.

use std::time::Duration;

use xor_net::generation::ReasoningFormat;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model: String,
    pub model_id: Option<String>,
    /// Explicit tokenizer.json (overrides the model's own).
    pub tokenizer: Option<String>,
    pub tokenizer_repo: String,
    /// `None` = auto: separate when the chat template has `<think>`.
    pub reasoning_format: Option<ReasoningFormat>,
    pub host: String,
    pub port: u16,
    pub slots: usize,
    pub threads_per_slot: usize,
    pub max_connections: usize,
    pub max_queue: usize,
    pub max_body_bytes: usize,
    pub max_header_line: usize,
    pub max_headers: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub idle_timeout: Duration,
    /// Max requests per keep-alive connection (1 disables keep-alive).
    pub keep_alive_max: usize,
    pub max_tokens_cap: usize,
    pub default_max_tokens: usize,
    pub cors_origin: Option<String>,
    /// Shared prompt-cache budget in MiB; `None` = auto (a quarter of the
    /// available memory, at most 1 GiB), `Some(0)` disables the cache.
    pub prompt_cache_mb: Option<usize>,
    /// Shortest prefix (tokens) worth a prompt-cache entry.
    pub prompt_cache_min_tokens: usize,
}

impl ServerConfig {
    pub fn defaults() -> Self {
        let (slots, threads_per_slot) = default_parallelism(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
        );
        Self {
            model: "models/bitnet-2b".into(),
            model_id: None,
            tokenizer: None,
            reasoning_format: None,
            tokenizer_repo: "microsoft/bitnet-b1.58-2B-4T".into(),
            host: "127.0.0.1".into(),
            port: 8080,
            slots,
            threads_per_slot,
            max_connections: 256,
            max_queue: 64,
            max_body_bytes: 4 * 1024 * 1024,
            max_header_line: 8 * 1024,
            max_headers: 100,
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(5),
            keep_alive_max: 100,
            max_tokens_cap: 4096,
            default_max_tokens: 512,
            cors_origin: None,
            prompt_cache_mb: None,
            prompt_cache_min_tokens: xor_net::prompt_cache::DEFAULT_MIN_TOKENS,
        }
    }
}

/// Default `(slots, threads_per_slot)` for `logical` CPUs (assumes SMT-2,
/// i.e. `logical / 2` physical cores). Decode is memory-bound, so splitting
/// the cores over a few slots raises aggregate throughput while one slot
/// with every core gives the lowest single-request latency. Measured on a
/// 6-core Zen 5 with Bonsai-1.7B (decode tok/s, aggregate / single request):
/// 1x6 155/155, 2x3 190/115, 3x2 215/90, 6x1 264/50. Two slots keep most of
/// the single-stream speed while serving two clients at once.
pub fn default_parallelism(logical: usize) -> (usize, usize) {
    let physical = (logical / 2).max(1);
    let slots = if physical >= 4 { 2 } else { 1 };
    (slots, (physical / slots).max(1))
}

pub const USAGE: &str = "\
xor-net-server: OpenAI / llama.cpp compatible HTTP server for xor-net

USAGE:
    xor-net-server [OPTIONS]

Each option falls back to the environment variable shown, then the default.

OPTIONS:
    -m, --model <PATH|REPO>      Bonsai .gguf / .xq1 file, safetensors dir (Llama/BitNet)
                                 or HF repo id              [XORNET_MODEL] (models/bitnet-2b)
        --model-id <ID>          id reported by the API     [XORNET_MODEL_ID] (file stem / dir name)
        --tokenizer <PATH>       tokenizer.json to use instead of the model's own
                                                            [XORNET_TOKENIZER] (none)
        --tokenizer-repo <REPO>  HF repo for tokenizer.json if a model dir has none
                                                            [XORNET_TOKENIZER_REPO] (microsoft/bitnet-b1.58-2B-4T)
        --reasoning-format <F>   auto | none | separate: with `separate`, <think> blocks go to
                                 message.reasoning_content / delta.reasoning_content (chat only);
                                 auto = separate iff the chat template has <think>
                                                            [XORNET_REASONING_FORMAT] (auto)
        --host <HOST>            bind host                  [XORNET_SERVER_HOST] (127.0.0.1)
    -p, --port <PORT>            bind port                  [XORNET_SERVER_PORT] (8080)
        --slots <N>              concurrent decode slots    [XORNET_SLOTS] (2 if >= 4 physical cores, else 1)
        --threads-per-slot <N>   rayon threads per slot     [XORNET_THREADS_PER_SLOT] (physical cores / slots)
                                 Decode is memory-bound: 1 slot x all cores = lowest latency,
                                 more slots = more aggregate tok/s (6-core Zen 5, Bonsai-1.7B:
                                 1x6 155, 2x3 190, 3x2 215, 6x1 264 tok/s aggregate).
        --max-connections <N>    open connection cap (503)  [XORNET_MAX_CONNECTIONS] (256)
        --max-queue <N>          queued jobs cap (503)      [XORNET_MAX_QUEUE] (64)
        --max-body-bytes <N>     request body cap (413)     [XORNET_MAX_BODY_BYTES] (4194304)
        --max-header-line <N>    request/header line cap    [XORNET_MAX_HEADER_LINE] (8192)
        --max-headers <N>        header count cap (431)     [XORNET_MAX_HEADERS] (100)
        --read-timeout <SECS>    time to receive a request  [XORNET_READ_TIMEOUT] (30)
        --write-timeout <SECS>   socket write timeout       [XORNET_WRITE_TIMEOUT] (30)
        --idle-timeout <SECS>    keep-alive idle timeout    [XORNET_IDLE_TIMEOUT] (5)
        --keep-alive-max <N>     requests per connection, 1 = no keep-alive
                                                            [XORNET_KEEP_ALIVE_MAX] (100)
        --max-tokens-cap <N>     upper bound for max_tokens [XORNET_MAX_TOKENS_CAP] (4096)
        --default-max-tokens <N> max_tokens when omitted    [XORNET_DEFAULT_MAX_TOKENS] (512)
        --cors-origin <ORIGIN>   enable CORS for this origin (\"*\" for any)
                                                            [XORNET_CORS_ORIGIN] (disabled)
        --prompt-cache-mb <N>    prompt cache shared by all slots: snapshots of shared prompt
                                 prefixes (system prompt, chat history), LRU within N MiB;
                                 0 disables, auto = 1/4 of available RAM capped at 1024
                                                            [XORNET_PROMPT_CACHE_MB] (auto)
        --prompt-cache-min-tokens <N>
                                 shortest prefix worth caching
                                                            [XORNET_PROMPT_CACHE_MIN_TOKENS] (64)
    -h, --help                   print this help
";

pub enum Parsed {
    Run(Box<ServerConfig>),
    Help,
}

struct Opt {
    long: &'static str,
    short: Option<&'static str>,
    env: &'static str,
}

const OPTS: &[Opt] = &[
    Opt {
        long: "model",
        short: Some("-m"),
        env: "XORNET_MODEL",
    },
    Opt {
        long: "model-id",
        short: None,
        env: "XORNET_MODEL_ID",
    },
    Opt {
        long: "tokenizer",
        short: None,
        env: "XORNET_TOKENIZER",
    },
    Opt {
        long: "reasoning-format",
        short: None,
        env: "XORNET_REASONING_FORMAT",
    },
    Opt {
        long: "tokenizer-repo",
        short: None,
        env: "XORNET_TOKENIZER_REPO",
    },
    Opt {
        long: "host",
        short: None,
        env: "XORNET_SERVER_HOST",
    },
    Opt {
        long: "port",
        short: Some("-p"),
        env: "XORNET_SERVER_PORT",
    },
    Opt {
        long: "slots",
        short: None,
        env: "XORNET_SLOTS",
    },
    Opt {
        long: "threads-per-slot",
        short: None,
        env: "XORNET_THREADS_PER_SLOT",
    },
    Opt {
        long: "max-connections",
        short: None,
        env: "XORNET_MAX_CONNECTIONS",
    },
    Opt {
        long: "max-queue",
        short: None,
        env: "XORNET_MAX_QUEUE",
    },
    Opt {
        long: "max-body-bytes",
        short: None,
        env: "XORNET_MAX_BODY_BYTES",
    },
    Opt {
        long: "max-header-line",
        short: None,
        env: "XORNET_MAX_HEADER_LINE",
    },
    Opt {
        long: "max-headers",
        short: None,
        env: "XORNET_MAX_HEADERS",
    },
    Opt {
        long: "read-timeout",
        short: None,
        env: "XORNET_READ_TIMEOUT",
    },
    Opt {
        long: "write-timeout",
        short: None,
        env: "XORNET_WRITE_TIMEOUT",
    },
    Opt {
        long: "idle-timeout",
        short: None,
        env: "XORNET_IDLE_TIMEOUT",
    },
    Opt {
        long: "keep-alive-max",
        short: None,
        env: "XORNET_KEEP_ALIVE_MAX",
    },
    Opt {
        long: "max-tokens-cap",
        short: None,
        env: "XORNET_MAX_TOKENS_CAP",
    },
    Opt {
        long: "default-max-tokens",
        short: None,
        env: "XORNET_DEFAULT_MAX_TOKENS",
    },
    Opt {
        long: "cors-origin",
        short: None,
        env: "XORNET_CORS_ORIGIN",
    },
    Opt {
        long: "prompt-cache-mb",
        short: None,
        env: "XORNET_PROMPT_CACHE_MB",
    },
    Opt {
        long: "prompt-cache-min-tokens",
        short: None,
        env: "XORNET_PROMPT_CACHE_MIN_TOKENS",
    },
];

fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
    v.trim()
        .parse()
        .map_err(|_| format!("invalid value for --{name}: {v:?}"))
}

fn positive(name: &str, v: &str) -> Result<usize, String> {
    let n: usize = num(name, v)?;
    if n == 0 {
        return Err(format!("--{name} must be >= 1"));
    }
    Ok(n)
}

fn secs(name: &str, v: &str) -> Result<Duration, String> {
    let s: f64 = num(name, v)?;
    if !(s.is_finite() && s > 0.0 && s <= 86_400.0) {
        return Err(format!("--{name} must be in (0, 86400] seconds"));
    }
    Ok(Duration::from_secs_f64(s))
}

/// Parse CLI arguments (without argv[0]); `env` looks up fallbacks.
pub fn parse(
    args: impl IntoIterator<Item = String>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<Parsed, String> {
    let mut given: Vec<(&'static str, String)> = Vec::new();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        if arg == "-h" || arg == "--help" {
            return Ok(Parsed::Help);
        }
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let opt = OPTS
            .iter()
            .find(|o| key.strip_prefix("--") == Some(o.long) || o.short == Some(key.as_str()))
            .ok_or_else(|| format!("unknown argument: {arg} (see --help)"))?;
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .ok_or_else(|| format!("missing value for --{}", opt.long))?,
        };
        given.retain(|(k, _)| *k != opt.long);
        given.push((opt.long, value));
    }

    let mut c = ServerConfig::defaults();
    for opt in OPTS {
        let v = match given.iter().find(|(k, _)| *k == opt.long) {
            Some((_, v)) => v.clone(),
            None => match env(opt.env) {
                Some(v) if !v.trim().is_empty() => v,
                _ => continue,
            },
        };
        let n = opt.long;
        match n {
            "model" => c.model = v,
            "model-id" => c.model_id = Some(v),
            "tokenizer" => c.tokenizer = Some(v),
            "reasoning-format" => {
                c.reasoning_format = match v.trim() {
                    "auto" => None,
                    other => Some(other.parse().map_err(|_| {
                        format!("invalid value for --{n}: {v:?} (auto | none | separate)")
                    })?),
                }
            }
            "tokenizer-repo" => c.tokenizer_repo = v,
            "host" => c.host = v,
            "port" => c.port = num(n, &v)?,
            "slots" => c.slots = positive(n, &v)?,
            "threads-per-slot" => c.threads_per_slot = positive(n, &v)?,
            "max-connections" => c.max_connections = positive(n, &v)?,
            "max-queue" => c.max_queue = positive(n, &v)?,
            "max-body-bytes" => c.max_body_bytes = positive(n, &v)?,
            "max-header-line" => c.max_header_line = positive(n, &v)?.max(64),
            "max-headers" => c.max_headers = positive(n, &v)?,
            "read-timeout" => c.read_timeout = secs(n, &v)?,
            "write-timeout" => c.write_timeout = secs(n, &v)?,
            "idle-timeout" => c.idle_timeout = secs(n, &v)?,
            "keep-alive-max" => c.keep_alive_max = positive(n, &v)?,
            "max-tokens-cap" => c.max_tokens_cap = positive(n, &v)?,
            "default-max-tokens" => c.default_max_tokens = positive(n, &v)?,
            "cors-origin" => c.cors_origin = Some(v),
            "prompt-cache-mb" => {
                c.prompt_cache_mb = match v.trim() {
                    "auto" => None,
                    _ => Some(num(n, &v)?),
                }
            }
            "prompt-cache-min-tokens" => c.prompt_cache_min_tokens = positive(n, &v)?,
            _ => {}
        }
    }
    c.default_max_tokens = c.default_max_tokens.min(c.max_tokens_cap);
    Ok(Parsed::Run(Box::new(c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str], env: &[(&str, &str)]) -> Result<ServerConfig, String> {
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        match parse(args.iter().map(|s| s.to_string()), |k| {
            env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())
        })? {
            Parsed::Run(c) => Ok(*c),
            Parsed::Help => Err("help".into()),
        }
    }

    #[test]
    fn defaults_and_precedence() {
        let c = run(&[], &[]).unwrap();
        assert_eq!(c.host, "127.0.0.1");
        assert_eq!(c.port, 8080);
        let c = run(&["--port", "9000"], &[("XORNET_SERVER_PORT", "9001")]).unwrap();
        assert_eq!(c.port, 9000);
        let c = run(
            &[],
            &[("XORNET_SERVER_PORT", "9001"), ("XORNET_SLOTS", "3")],
        )
        .unwrap();
        assert_eq!((c.port, c.slots), (9001, 3));
        let c = run(
            &["-m", "/x/y", "--read-timeout=1.5", "--cors-origin=*"],
            &[],
        )
        .unwrap();
        assert_eq!(c.model, "/x/y");
        assert_eq!(c.read_timeout, Duration::from_millis(1500));
        assert_eq!(c.cors_origin.as_deref(), Some("*"));
    }

    #[test]
    fn errors_and_help() {
        assert_eq!(run(&["--help"], &[]).unwrap_err(), "help");
        assert!(run(&["--port", "x"], &[]).is_err());
        assert!(run(&["--port"], &[]).is_err());
        assert!(run(&["--bogus"], &[]).is_err());
        assert!(run(&["--slots", "0"], &[]).is_err());
        assert!(run(&[], &[("XORNET_SLOTS", "abc")]).is_err());
        assert!(run(&["--read-timeout", "0"], &[]).is_err());
        assert!(run(&["--reasoning-format", "maybe"], &[]).is_err());
    }

    #[test]
    fn default_parallelism_splits_physical_cores() {
        assert_eq!(default_parallelism(12), (2, 3));
        assert_eq!(default_parallelism(16), (2, 4));
        assert_eq!(default_parallelism(4), (1, 2));
        assert_eq!(default_parallelism(1), (1, 1));
    }

    #[test]
    fn reasoning_and_tokenizer_flags() {
        let c = run(&[], &[]).unwrap();
        assert_eq!((c.reasoning_format, c.tokenizer), (None, None));
        let c = run(
            &["--reasoning-format", "none", "--tokenizer", "t.json"],
            &[],
        )
        .unwrap();
        assert_eq!(c.reasoning_format, Some(ReasoningFormat::None));
        assert_eq!(c.tokenizer.as_deref(), Some("t.json"));
        let c = run(&[], &[("XORNET_REASONING_FORMAT", "separate")]).unwrap();
        assert_eq!(c.reasoning_format, Some(ReasoningFormat::Separate));
        let c = run(&["--reasoning-format=auto"], &[]).unwrap();
        assert_eq!(c.reasoning_format, None);
    }

    #[test]
    fn prompt_cache_flags() {
        let c = run(&[], &[]).unwrap();
        assert_eq!((c.prompt_cache_mb, c.prompt_cache_min_tokens), (None, 64));
        let c = run(
            &["--prompt-cache-mb", "0", "--prompt-cache-min-tokens=128"],
            &[],
        )
        .unwrap();
        assert_eq!(
            (c.prompt_cache_mb, c.prompt_cache_min_tokens),
            (Some(0), 128)
        );
        let c = run(&[], &[("XORNET_PROMPT_CACHE_MB", "512")]).unwrap();
        assert_eq!(c.prompt_cache_mb, Some(512));
        let c = run(&["--prompt-cache-mb=auto"], &[]).unwrap();
        assert_eq!(c.prompt_cache_mb, None);
        assert!(run(&["--prompt-cache-mb", "-1"], &[]).is_err());
        assert!(run(&["--prompt-cache-min-tokens", "0"], &[]).is_err());
    }
}
