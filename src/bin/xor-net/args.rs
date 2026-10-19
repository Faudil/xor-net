//! Hand-rolled argument parsing for `xor-net` (no CLI framework: the
//! surface is small and this keeps the dependency tree lean).
//!
//! Accepted forms: `--flag value`, `--flag=value`, `-n value`, `-n5` is NOT
//! accepted (keep it unambiguous), `--` ends option parsing.

use std::path::PathBuf;

pub const DEFAULT_VISION_MODEL: &str = "models/vision/quicknet/quicknet.safetensors";

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Run(RunArgs),
    Chat(ChatArgs),
    Convert(ConvertArgs),
    Bench(BenchArgs),
    Info(InfoArgs),
    Classify(ClassifyArgs),
    Pull(PullArgs),
    Serve(Vec<String>),
    /// Print help (for a subcommand if given).
    Help(Option<String>),
    Version,
}

/// Sampling / stopping options shared by `run` and `chat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GenOpts {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    /// `Some(false)` = `--no-think`, `Some(true)` = `--think`, `None` = the
    /// template default.
    pub think: Option<bool>,
    pub show_thinking: bool,
}

impl Default for GenOpts {
    fn default() -> Self {
        // Qwen3 "non-thinking" recommended sampling (Bonsai is Qwen3-based).
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: None,
            stop: Vec::new(),
            think: None,
            show_thinking: false,
        }
    }
}

/// Model reference + loading options shared by model subcommands.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelOpts {
    pub model: String,
    /// 0 = auto.
    pub threads: usize,
    pub tokenizer: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunArgs {
    pub m: ModelOpts,
    pub sampling: GenOpts,
    /// `None` = read the prompt from stdin.
    pub prompt: Option<String>,
    pub chat: bool,
    pub system: Option<String>,
    pub quiet: bool,
    pub ignore_eos: bool,
    /// Echo the prompt before the completion (raw mode only).
    pub echo: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChatArgs {
    pub m: ModelOpts,
    pub sampling: GenOpts,
    pub system: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConvertArgs {
    pub input: String,
    pub output: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchArgs {
    pub m: ModelOpts,
    pub prompt_len: usize,
    pub gen_tokens: usize,
    pub reps: usize,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            m: ModelOpts::default(),
            prompt_len: 128,
            gen_tokens: 64,
            reps: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct InfoArgs {
    pub model: String,
    pub template: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassifyArgs {
    pub images: Vec<PathBuf>,
    pub model: PathBuf,
    pub k: usize,
    pub threads: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PullArgs {
    pub model: String,
}

/// Cursor over raw arguments that understands `--key=value`.
struct Args {
    items: std::vec::IntoIter<String>,
    /// Value given inline with `--key=value`, consumed by `value()`.
    inline: Option<String>,
    only_positional: bool,
}

enum Tok {
    Flag(String),
    Pos(String),
}

impl Args {
    fn new(v: Vec<String>) -> Self {
        Self {
            items: v.into_iter(),
            inline: None,
            only_positional: false,
        }
    }

    fn next(&mut self) -> Result<Option<Tok>, String> {
        if let Some(v) = self.inline.take() {
            return Err(format!("unexpected value {v:?}"));
        }
        let Some(a) = self.items.next() else {
            return Ok(None);
        };
        if self.only_positional {
            return Ok(Some(Tok::Pos(a)));
        }
        if a == "--" {
            self.only_positional = true;
            return self.next();
        }
        if a.starts_with("--") {
            if let Some((k, v)) = a.split_once('=') {
                self.inline = Some(v.to_string());
                return Ok(Some(Tok::Flag(k.to_string())));
            }
            return Ok(Some(Tok::Flag(a)));
        }
        // "-" alone is a positional (stdin); negative numbers are values.
        if a.len() > 1 && a.starts_with('-') && !a[1..].starts_with(|c: char| c.is_ascii_digit()) {
            return Ok(Some(Tok::Flag(a)));
        }
        Ok(Some(Tok::Pos(a)))
    }

    fn value(&mut self, flag: &str) -> Result<String, String> {
        if let Some(v) = self.inline.take() {
            return Ok(v);
        }
        self.items
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))
    }

    fn parse<T: std::str::FromStr>(&mut self, flag: &str) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        let v = self.value(flag)?;
        v.parse()
            .map_err(|e| format!("invalid value {v:?} for {flag}: {e}"))
    }

    /// For boolean flags: reject `--flag=value`.
    fn no_value(&mut self, flag: &str) -> Result<(), String> {
        match self.inline.take() {
            Some(v) => Err(format!("{flag} takes no value (got {v:?})")),
            None => Ok(()),
        }
    }
}

fn is_help(f: &str) -> bool {
    f == "-h" || f == "--help"
}

/// Shared model flags. Returns `Ok(false)` if `flag` is not one of them.
fn model_flag(a: &mut Args, f: &str, m: &mut ModelOpts) -> Result<bool, String> {
    match f {
        "-t" | "--threads" => m.threads = a.parse(f)?,
        "--tokenizer" => m.tokenizer = Some(a.value(f)?.into()),
        "-m" | "--model" => m.model = a.value(f)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Shared generation flags. Returns `Ok(false)` if `flag` is not one of them.
fn gen_flag(a: &mut Args, f: &str, g: &mut GenOpts) -> Result<bool, String> {
    match f {
        "-n" | "--max-tokens" | "--n-predict" => g.max_tokens = a.parse(f)?,
        "--temp" | "--temperature" => {
            g.temperature = a.parse(f)?;
            if !(g.temperature >= 0.0 && g.temperature.is_finite()) {
                return Err(format!("{f} must be >= 0"));
            }
        }
        "--top-k" => g.top_k = a.parse(f)?,
        "--top-p" => {
            g.top_p = a.parse(f)?;
            if !(g.top_p > 0.0 && g.top_p <= 1.0) {
                return Err(format!("{f} must be in (0, 1]"));
            }
        }
        "--min-p" => {
            g.min_p = a.parse(f)?;
            if !(0.0..=1.0).contains(&g.min_p) {
                return Err(format!("{f} must be in [0, 1]"));
            }
        }
        "--repeat-penalty" => {
            g.repeat_penalty = a.parse(f)?;
            if !(g.repeat_penalty > 0.0 && g.repeat_penalty.is_finite()) {
                return Err(format!("{f} must be > 0"));
            }
        }
        "--seed" => g.seed = Some(a.parse(f)?),
        "--stop" => g.stop.push(a.value(f)?),
        "--greedy" => {
            a.no_value(f)?;
            g.temperature = 0.0;
        }
        "--think" => {
            a.no_value(f)?;
            g.think = Some(true);
        }
        "--no-think" => {
            a.no_value(f)?;
            g.think = Some(false);
        }
        "--show-thinking" => {
            a.no_value(f)?;
            g.show_thinking = true;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn unknown(cmd: &str, f: &str) -> String {
    format!("unknown option {f} for `xor-net {cmd}` (see `xor-net {cmd} --help`)")
}

fn set_model(m: &mut String, cmd: &str, p: String) -> Result<(), String> {
    if m.is_empty() {
        *m = p;
        Ok(())
    } else {
        Err(format!("unexpected argument {p:?} for `xor-net {cmd}`"))
    }
}

fn need_model(m: &str, cmd: &str) -> Result<(), String> {
    if m.is_empty() {
        Err(format!(
            "`xor-net {cmd}` needs a model (path to .gguf/.xq1/model dir or a Hub id)"
        ))
    } else {
        Ok(())
    }
}

fn parse_run(mut a: Args) -> Result<Command, String> {
    let mut r = RunArgs::default();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("run".into()))),
            Tok::Flag(f) => {
                if model_flag(&mut a, &f, &mut r.m)? || gen_flag(&mut a, &f, &mut r.sampling)? {
                    continue;
                }
                match f.as_str() {
                    "-p" | "--prompt" => r.prompt = Some(a.value(&f)?),
                    "-f" | "--file" => {
                        let p = a.value(&f)?;
                        r.prompt = Some(
                            std::fs::read_to_string(&p)
                                .map_err(|e| format!("reading prompt file {p}: {e}"))?,
                        );
                    }
                    "--chat" => {
                        a.no_value(&f)?;
                        r.chat = true;
                    }
                    "-s" | "--system" => {
                        r.system = Some(a.value(&f)?);
                        r.chat = true;
                    }
                    "-q" | "--quiet" => {
                        a.no_value(&f)?;
                        r.quiet = true;
                    }
                    "--ignore-eos" => {
                        a.no_value(&f)?;
                        r.ignore_eos = true;
                    }
                    "--echo" => {
                        a.no_value(&f)?;
                        r.echo = true;
                    }
                    _ => return Err(unknown("run", &f)),
                }
            }
            Tok::Pos(p) if r.m.model.is_empty() => r.m.model = p,
            // `xor-net run model "prompt"` (prompt as 2nd positional).
            Tok::Pos(p) if r.prompt.is_none() => r.prompt = Some(p),
            Tok::Pos(p) => return Err(format!("unexpected argument {p:?} for `xor-net run`")),
        }
    }
    need_model(&r.m.model, "run")?;
    Ok(Command::Run(r))
}

fn parse_chat(mut a: Args) -> Result<Command, String> {
    let mut c = ChatArgs::default();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("chat".into()))),
            Tok::Flag(f) => {
                if model_flag(&mut a, &f, &mut c.m)? || gen_flag(&mut a, &f, &mut c.sampling)? {
                    continue;
                }
                match f.as_str() {
                    "-s" | "--system" => c.system = Some(a.value(&f)?),
                    _ => return Err(unknown("chat", &f)),
                }
            }
            Tok::Pos(p) => set_model(&mut c.m.model, "chat", p)?,
        }
    }
    need_model(&c.m.model, "chat")?;
    Ok(Command::Chat(c))
}

fn parse_convert(mut a: Args) -> Result<Command, String> {
    let mut c = ConvertArgs::default();
    let mut pos = Vec::new();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("convert".into()))),
            Tok::Flag(f) => match f.as_str() {
                "--tokenizer" => c.tokenizer = Some(a.value(&f)?.into()),
                "-o" | "--output" => c.output = Some(a.value(&f)?.into()),
                _ => return Err(unknown("convert", &f)),
            },
            Tok::Pos(p) => pos.push(p),
        }
    }
    let mut pos = pos.into_iter();
    c.input = pos
        .next()
        .ok_or("`xor-net convert` needs an input .gguf (or Hub id)")?;
    if let Some(o) = pos.next() {
        if c.output.is_some() {
            return Err("output given twice".into());
        }
        c.output = Some(o.into());
    }
    if let Some(p) = pos.next() {
        return Err(format!("unexpected argument {p:?} for `xor-net convert`"));
    }
    Ok(Command::Convert(c))
}

fn parse_bench(mut a: Args) -> Result<Command, String> {
    let mut b = BenchArgs::default();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("bench".into()))),
            Tok::Flag(f) => {
                if model_flag(&mut a, &f, &mut b.m)? {
                    continue;
                }
                match f.as_str() {
                    "-p" | "--prompt-len" => b.prompt_len = a.parse(&f)?,
                    "-n" | "--gen" => b.gen_tokens = a.parse(&f)?,
                    "-r" | "--reps" => b.reps = a.parse(&f)?,
                    _ => return Err(unknown("bench", &f)),
                }
            }
            Tok::Pos(p) => set_model(&mut b.m.model, "bench", p)?,
        }
    }
    need_model(&b.m.model, "bench")?;
    if b.prompt_len == 0 || b.reps == 0 {
        return Err("--prompt-len and --reps must be >= 1".into());
    }
    Ok(Command::Bench(b))
}

fn parse_info(mut a: Args) -> Result<Command, String> {
    let mut i = InfoArgs::default();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("info".into()))),
            Tok::Flag(f) => match f.as_str() {
                "--template" => {
                    a.no_value(&f)?;
                    i.template = true;
                }
                _ => return Err(unknown("info", &f)),
            },
            Tok::Pos(p) => set_model(&mut i.model, "info", p)?,
        }
    }
    need_model(&i.model, "info")?;
    Ok(Command::Info(i))
}

fn parse_classify(mut a: Args, env_model: Option<String>) -> Result<Command, String> {
    let mut c = ClassifyArgs {
        images: Vec::new(),
        model: env_model
            .unwrap_or_else(|| DEFAULT_VISION_MODEL.into())
            .into(),
        k: 5,
        threads: 0,
    };
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("classify".into()))),
            Tok::Flag(f) => match f.as_str() {
                "-m" | "--model" => c.model = a.value(&f)?.into(),
                "-k" | "--top-k" => c.k = a.parse(&f)?,
                "-t" | "--threads" => c.threads = a.parse(&f)?,
                _ => return Err(unknown("classify", &f)),
            },
            Tok::Pos(p) => c.images.push(p.into()),
        }
    }
    if c.images.is_empty() {
        return Err("`xor-net classify` needs at least one image".into());
    }
    if c.k == 0 {
        return Err("-k must be >= 1".into());
    }
    Ok(Command::Classify(c))
}

fn parse_pull(mut a: Args) -> Result<Command, String> {
    let mut p = PullArgs::default();
    while let Some(t) = a.next()? {
        match t {
            Tok::Flag(f) if is_help(&f) => return Ok(Command::Help(Some("pull".into()))),
            Tok::Flag(f) => return Err(unknown("pull", &f)),
            Tok::Pos(x) => set_model(&mut p.model, "pull", x)?,
        }
    }
    need_model(&p.model, "pull")?;
    Ok(Command::Pull(p))
}

/// Parse `argv[1..]`. `env_vision_model` is `$XORNET_VISION_MODEL`.
pub fn parse(argv: Vec<String>, env_vision_model: Option<String>) -> Result<Command, String> {
    let mut it = argv.into_iter();
    let Some(cmd) = it.next() else {
        return Ok(Command::Help(None));
    };
    let rest = Args::new(it.collect());
    match cmd.as_str() {
        "run" | "generate" => parse_run(rest),
        "chat" => parse_chat(rest),
        "convert" => parse_convert(rest),
        "bench" => parse_bench(rest),
        "info" | "show" => parse_info(rest),
        "classify" => parse_classify(rest, env_vision_model),
        "pull" | "download" => parse_pull(rest),
        "serve" | "server" => Ok(Command::Serve(rest.items.collect())),
        "help" => Ok(Command::Help(rest.items.as_slice().first().cloned())),
        "-h" | "--help" => Ok(Command::Help(None)),
        "-V" | "--version" | "version" => Ok(Command::Version),
        other => Err(format!(
            "unknown command {other:?} (expected run, chat, convert, bench, info, classify, pull or serve)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &[&str]) -> Result<Command, String> {
        parse(s.iter().map(|s| s.to_string()).collect(), None)
    }

    fn run(s: &[&str]) -> RunArgs {
        match p(s).unwrap() {
            Command::Run(r) => r,
            c => panic!("{c:?}"),
        }
    }

    #[test]
    fn run_full() {
        let r = run(&[
            "run",
            "m.xq1",
            "-p",
            "hello",
            "-n",
            "32",
            "--temp=0.5",
            "--top-k",
            "40",
            "--top-p",
            "0.9",
            "--seed",
            "7",
            "--threads",
            "4",
            "--chat",
            "--no-think",
            "--stop",
            "\n\n",
            "--stop",
            "END",
            "--quiet",
        ]);
        assert_eq!(r.m.model, "m.xq1");
        assert_eq!(r.m.threads, 4);
        assert_eq!(r.prompt.as_deref(), Some("hello"));
        assert_eq!(r.sampling.max_tokens, 32);
        assert_eq!(r.sampling.temperature, 0.5);
        assert_eq!(r.sampling.top_k, 40);
        assert_eq!(r.sampling.top_p, 0.9);
        assert_eq!(r.sampling.seed, Some(7));
        assert_eq!(r.sampling.think, Some(false));
        assert_eq!(r.sampling.stop, vec!["\n\n".to_string(), "END".to_string()]);
        assert!(r.chat && r.quiet);
    }

    #[test]
    fn run_defaults_and_positional_prompt() {
        let r = run(&["run", "prism-ml/Bonsai-1.7B-gguf", "Tell me a joke"]);
        assert_eq!(r.m.model, "prism-ml/Bonsai-1.7B-gguf");
        assert_eq!(r.prompt.as_deref(), Some("Tell me a joke"));
        assert_eq!(r.sampling, GenOpts::default());
        assert!(!r.chat);
        // No prompt: read stdin.
        assert_eq!(run(&["run", "m.gguf"]).prompt, None);
        // --system implies chat; --greedy sets temp 0.
        let r = run(&[
            "run", "m.gguf", "--system", "be brief", "--greedy", "-p", "x",
        ]);
        assert!(r.chat);
        assert_eq!(r.sampling.temperature, 0.0);
    }

    #[test]
    fn run_errors() {
        assert!(p(&["run"]).unwrap_err().contains("needs a model"));
        assert!(
            p(&["run", "m", "-n"])
                .unwrap_err()
                .contains("needs a value")
        );
        assert!(
            p(&["run", "m", "-n", "x"])
                .unwrap_err()
                .contains("invalid value")
        );
        assert!(
            p(&["run", "m", "--bogus"])
                .unwrap_err()
                .contains("unknown option")
        );
        assert!(
            p(&["run", "m", "--temp", "-1"])
                .unwrap_err()
                .contains(">= 0")
        );
        assert!(p(&["run", "m", "--top-p", "0"]).is_err());
        assert!(
            p(&["run", "m", "--chat=yes"])
                .unwrap_err()
                .contains("takes no value")
        );
        assert!(
            p(&["run", "m", "a", "b"])
                .unwrap_err()
                .contains("unexpected")
        );
    }

    #[test]
    fn double_dash_makes_prompt_positional() {
        let r = run(&["run", "m", "--", "--not-a-flag"]);
        assert_eq!(r.prompt.as_deref(), Some("--not-a-flag"));
    }

    #[test]
    fn chat_args() {
        match p(&[
            "chat",
            "m.gguf",
            "--system",
            "You are terse.",
            "--show-thinking",
            "--think",
        ])
        .unwrap()
        {
            Command::Chat(c) => {
                assert_eq!(c.m.model, "m.gguf");
                assert_eq!(c.system.as_deref(), Some("You are terse."));
                assert!(c.sampling.show_thinking);
                assert_eq!(c.sampling.think, Some(true));
            }
            c => panic!("{c:?}"),
        }
        assert!(p(&["chat"]).is_err());
        assert!(p(&["chat", "a", "b"]).is_err());
    }

    #[test]
    fn convert_args() {
        assert_eq!(
            p(&["convert", "in.gguf", "out.xq1"]).unwrap(),
            Command::Convert(ConvertArgs {
                input: "in.gguf".into(),
                output: Some("out.xq1".into()),
                tokenizer: None,
            })
        );
        match p(&["convert", "in.gguf", "--tokenizer", "t.json"]).unwrap() {
            Command::Convert(c) => {
                assert_eq!(c.output, None);
                assert_eq!(c.tokenizer, Some("t.json".into()));
            }
            c => panic!("{c:?}"),
        }
        assert!(p(&["convert"]).is_err());
        assert!(p(&["convert", "a", "b", "c"]).is_err());
        assert!(p(&["convert", "a", "b", "-o", "c"]).is_err());
    }

    #[test]
    fn bench_info_classify_pull() {
        match p(&[
            "bench",
            "m",
            "-n",
            "16",
            "-p",
            "64",
            "--threads",
            "6",
            "-r",
            "3",
        ])
        .unwrap()
        {
            Command::Bench(b) => {
                assert_eq!(
                    (b.gen_tokens, b.prompt_len, b.m.threads, b.reps),
                    (16, 64, 6, 3)
                );
            }
            c => panic!("{c:?}"),
        }
        assert_eq!(
            p(&["info", "m.xq1", "--template"]).unwrap(),
            Command::Info(InfoArgs {
                model: "m.xq1".into(),
                template: true
            })
        );
        match p(&["classify", "a.jpg", "b.png", "-k", "3"]).unwrap() {
            Command::Classify(c) => {
                assert_eq!(c.images.len(), 2);
                assert_eq!(c.k, 3);
                assert_eq!(c.model, PathBuf::from(DEFAULT_VISION_MODEL));
            }
            c => panic!("{c:?}"),
        }
        match parse(vec!["classify".into(), "a.jpg".into()], Some("q.st".into())).unwrap() {
            Command::Classify(c) => assert_eq!(c.model, PathBuf::from("q.st")),
            c => panic!("{c:?}"),
        }
        assert!(p(&["classify"]).is_err());
        assert!(p(&["classify", "a.jpg", "-k", "0"]).is_err());
        assert_eq!(
            p(&["pull", "prism-ml/Bonsai-8B-gguf"]).unwrap(),
            Command::Pull(PullArgs {
                model: "prism-ml/Bonsai-8B-gguf".into()
            })
        );
    }

    #[test]
    fn help_and_dispatch() {
        assert_eq!(p(&[]).unwrap(), Command::Help(None));
        assert_eq!(p(&["--help"]).unwrap(), Command::Help(None));
        assert_eq!(
            p(&["help", "chat"]).unwrap(),
            Command::Help(Some("chat".into()))
        );
        for c in [
            "run", "chat", "convert", "bench", "info", "classify", "pull",
        ] {
            assert_eq!(
                p(&[c, "--help"]).unwrap(),
                Command::Help(Some(c.into())),
                "{c}"
            );
            assert_eq!(p(&[c, "-h"]).unwrap(), Command::Help(Some(c.into())), "{c}");
        }
        assert_eq!(p(&["--version"]).unwrap(), Command::Version);
        assert_eq!(
            p(&["serve", "--port", "9000"]).unwrap(),
            Command::Serve(vec!["--port".into(), "9000".into()])
        );
        assert!(p(&["frobnicate"]).unwrap_err().contains("unknown command"));
    }
}
