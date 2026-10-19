//! `xor-net run` and `xor-net chat`.

use std::io::{BufRead, IsTerminal, Write};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use xor_net::models::source::{self, LoadOptions};
use xor_net::{
    ChatMessage, FinishReason, Generation, GenerationError, GenerationParams, Generator,
    ReasoningFormat, SamplingParams, StreamEvent,
};

use crate::args::{ChatArgs, GenOpts, ModelOpts, RunArgs};

pub const DIM: &str = "\x1b[2m";
pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";

/// Peak resident set size in MB (Linux only).
pub fn peak_rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

pub fn fmt_rss() -> String {
    peak_rss_mb().map_or_else(|| "n/a".into(), |m| format!("{m:.0} MB"))
}

pub fn init_threads(n: usize) -> anyhow::Result<()> {
    xor_net::init_threads(n).map_err(|e| anyhow::anyhow!("thread pool: {e}"))
}

/// Resolve and load a model, reporting progress on stderr.
pub fn load(m: &ModelOpts, quiet: bool) -> anyhow::Result<(Generator, f64)> {
    init_threads(m.threads)?;
    let resolved = source::resolve(&m.model)?;
    let t0 = Instant::now();
    let opts = LoadOptions {
        tokenizer: m.tokenizer.clone(),
        ..LoadOptions::default()
    };
    let g = source::load_resolved(&resolved, &opts)
        .with_context(|| format!("loading {}", resolved.path.display()))?;
    let secs = t0.elapsed().as_secs_f64();
    if !quiet {
        eprintln!(
            "{}loaded {} in {secs:.2}s ({} threads){}",
            dim_if(std::io::stderr().is_terminal()),
            resolved.path.display(),
            rayon::current_num_threads(),
            reset_if(std::io::stderr().is_terminal()),
        );
    }
    Ok((g, secs))
}

fn dim_if(on: bool) -> &'static str {
    if on { DIM } else { "" }
}
fn reset_if(on: bool) -> &'static str {
    if on { RESET } else { "" }
}

pub fn params(g: &Generator, o: &GenOpts, warn: bool) -> GenerationParams {
    let sampling = if o.temperature == 0.0 {
        SamplingParams::greedy()
    } else {
        SamplingParams::default()
            .with_temperature(o.temperature)
            .with_top_k(o.top_k)
            .with_top_p(o.top_p)
            .with_min_p(o.min_p)
    }
    .with_repetition_penalty(o.repeat_penalty)
    .with_seed(o.seed);
    let mut p = GenerationParams::default()
        .with_max_tokens(o.max_tokens)
        .with_sampling(sampling)
        .with_stop(o.stop.clone())
        .with_reasoning_format(ReasoningFormat::Separate);
    if let Some(on) = o.think {
        let supported = g
            .chat_template()
            .is_some_and(|t| t.supports_enable_thinking());
        if warn && !supported {
            eprintln!(
                "note: this model's chat template ignores enable_thinking; --{} has no effect",
                if on { "think" } else { "no-think" }
            );
        }
        p = p.with_enable_thinking(on);
    }
    p
}

pub fn stats_line(out: &Generation, load_s: Option<f64>) -> String {
    let mut s = format!(
        "{} prompt ({} cached) + {} completion tokens | prefill {:.1} tok/s | decode {:.1} tok/s | finish {}",
        out.usage.prompt_tokens,
        out.usage.cached_prompt_tokens,
        out.usage.completion_tokens,
        out.timings.prefill_tok_per_s(),
        out.timings.decode_tok_per_s(),
        match out.finish_reason {
            FinishReason::Stop => "eos",
            FinishReason::Length => "length",
            FinishReason::StopString => "stop-string",
            FinishReason::Aborted => "interrupted",
        }
    );
    if let Some(l) = load_s {
        s += &format!(" | load {l:.2}s");
    }
    s += &format!(" | peak RSS {}", fmt_rss());
    s
}

/// Streams events to the terminal: answer text to stdout, reasoning dimmed
/// (to stdout in chat, stderr in `run` so piped output stays clean) or
/// replaced by a transient "thinking" marker.
struct Printer {
    show_thinking: bool,
    reasoning_to_stderr: bool,
    color_out: bool,
    color_err: bool,
    in_reasoning: bool,
    marker_shown: bool,
    last_char: Option<char>,
    broken: bool,
}

impl Printer {
    fn new(show_thinking: bool, reasoning_to_stderr: bool) -> Self {
        Self {
            show_thinking,
            reasoning_to_stderr,
            color_out: std::io::stdout().is_terminal(),
            color_err: std::io::stderr().is_terminal(),
            in_reasoning: false,
            marker_shown: false,
            last_char: None,
            broken: false,
        }
    }

    fn out(&mut self, s: &str) {
        let mut o = std::io::stdout().lock();
        if o.write_all(s.as_bytes()).and_then(|_| o.flush()).is_err() {
            self.broken = true; // e.g. `| head`
        }
    }

    fn event(&mut self, ev: StreamEvent<'_>) -> ControlFlow<()> {
        match ev {
            StreamEvent::Reasoning(r) => {
                if self.show_thinking {
                    if self.reasoning_to_stderr {
                        let c = self.color_err;
                        eprint!("{}{r}{}", dim_if(c), reset_if(c));
                    } else {
                        let c = self.color_out;
                        self.out(&format!("{}{r}{}", dim_if(c), reset_if(c)));
                    }
                    self.in_reasoning = true;
                } else if !self.marker_shown && self.color_out && !self.reasoning_to_stderr {
                    self.out(&format!("{DIM}(thinking...){RESET}"));
                    self.marker_shown = true;
                }
            }
            StreamEvent::Text(t) => {
                if self.marker_shown {
                    self.out("\r\x1b[2K");
                    self.marker_shown = false;
                }
                if self.in_reasoning {
                    self.in_reasoning = false;
                    if self.reasoning_to_stderr {
                        eprintln!();
                    } else {
                        self.out("\n");
                    }
                }
                let t = if self.last_char.is_none() {
                    t.trim_start_matches('\n')
                } else {
                    t
                };
                if let Some(c) = t.chars().last() {
                    self.last_char = Some(c);
                }
                self.out(t);
            }
        }
        if self.broken {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    /// End the output with exactly one newline.
    fn finish(&mut self) {
        if self.marker_shown {
            self.out("\r\x1b[2K");
            self.marker_shown = false;
        }
        if self.last_char.is_some_and(|c| c != '\n') || self.in_reasoning {
            self.out("\n");
        }
    }
}

pub fn run(a: RunArgs) -> anyhow::Result<()> {
    let prompt = match a.prompt.clone() {
        Some(p) => p,
        None => {
            if std::io::stdin().is_terminal() {
                eprintln!("(reading the prompt from stdin; end with Ctrl-D)");
            }
            let mut s = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)?;
            s
        }
    };
    let (mut g, load_s) = load(&a.m, a.quiet)?;
    let p = params(&g, &a.sampling, !a.quiet).with_ignore_eos(a.ignore_eos);
    let mut printer = Printer::new(a.sampling.show_thinking, true);
    let out = if a.chat {
        let mut msgs = Vec::new();
        if let Some(s) = &a.system {
            msgs.push(ChatMessage::system(s.clone()));
        }
        msgs.push(ChatMessage::user(prompt));
        g.chat_stream_events(&msgs, &p, |ev| printer.event(ev))?
    } else {
        if a.echo {
            printer.out(&prompt);
            printer.last_char = prompt.chars().last();
        }
        g.generate_stream_events(&prompt, &p, |ev| printer.event(ev))?
    };
    printer.finish();
    if !a.quiet {
        let c = std::io::stderr().is_terminal();
        eprintln!(
            "{}{}{}",
            dim_if(c),
            stats_line(&out, Some(load_s)),
            reset_if(c)
        );
    }
    Ok(())
}

// --- chat --------------------------------------------------------------------

static GENERATING: AtomicBool = AtomicBool::new(false);
static CANCEL: AtomicBool = AtomicBool::new(false);
static LAST_IDLE_INT_MS: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn install_ctrlc() {
    let res = ctrlc::set_handler(|| {
        if GENERATING.load(Ordering::SeqCst) {
            CANCEL.store(true, Ordering::SeqCst);
            return;
        }
        let now = now_ms();
        let last = LAST_IDLE_INT_MS.swap(now, Ordering::SeqCst);
        if now.saturating_sub(last) < 2000 {
            eprintln!();
            std::process::exit(130);
        }
        eprint!("\n(press Ctrl-C again to quit, or type /exit)\n> ");
        let _ = std::io::stderr().flush();
    });
    if let Err(e) = res {
        eprintln!("warning: could not install Ctrl-C handler ({e}); Ctrl-C will exit");
    }
}

const CHAT_HELP: &str = "\
  /reset                 forget the conversation
  /system <text>         set the system message (and reset)
  /think on|off          toggle enable_thinking in the chat template
  /show-thinking on|off  show or hide reasoning
  /exit, /quit           leave (or Ctrl-D)";

enum Slash {
    Continue,
    Exit,
}

fn on_off(arg: &str) -> Option<bool> {
    match arg.trim() {
        "on" | "true" | "1" | "yes" => Some(true),
        "off" | "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn base_messages(system: &Option<String>) -> Vec<ChatMessage> {
    system
        .iter()
        .map(|s| ChatMessage::system(s.clone()))
        .collect()
}

fn slash(
    line: &str,
    msgs: &mut Vec<ChatMessage>,
    system: &mut Option<String>,
    sampling: &mut GenOpts,
    g: &Generator,
) -> Slash {
    let (cmd, arg) = line.split_once(' ').unwrap_or((line, ""));
    match cmd {
        "/exit" | "/quit" | "/q" => return Slash::Exit,
        "/reset" | "/clear" => {
            *msgs = base_messages(system);
            eprintln!("(conversation cleared)");
        }
        "/system" => {
            *system = (!arg.trim().is_empty()).then(|| arg.trim().to_string());
            *msgs = base_messages(system);
            eprintln!("(system message set; conversation cleared)");
        }
        "/think" => match on_off(arg) {
            Some(on) => {
                sampling.think = Some(on);
                if !g
                    .chat_template()
                    .is_some_and(|t| t.supports_enable_thinking())
                {
                    eprintln!("(note: this model's template ignores enable_thinking)");
                }
                eprintln!("(thinking {})", if on { "on" } else { "off" });
            }
            None => eprintln!("usage: /think on|off"),
        },
        "/show-thinking" => match on_off(arg) {
            Some(on) => {
                sampling.show_thinking = on;
                eprintln!("(reasoning {})", if on { "shown" } else { "hidden" });
            }
            None => eprintln!("usage: /show-thinking on|off"),
        },
        "/help" | "/?" => eprintln!("{CHAT_HELP}"),
        _ => eprintln!("unknown command {cmd} (try /help)"),
    }
    Slash::Continue
}

pub fn chat(a: ChatArgs) -> anyhow::Result<()> {
    let (mut g, _) = load(&a.m, false)?;
    let ChatArgs {
        mut sampling,
        mut system,
        ..
    } = a;
    if g.chat_template().is_none() {
        eprintln!("note: the model has no chat template; using a generic fallback");
    }
    // Validate --think once (prints a note if unsupported).
    let _ = params(&g, &sampling, true);
    install_ctrlc();
    let tty_in = std::io::stdin().is_terminal();
    let color = std::io::stdout().is_terminal();
    if tty_in {
        eprintln!("Chatting with xor-net. /help for commands, Ctrl-C stops a reply, Ctrl-D quits.");
    }
    let mut msgs = base_messages(&system);
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        if tty_in {
            print!("{}> {}", if color { BOLD } else { "" }, reset_if(color));
            std::io::stdout().flush()?;
        }
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            if tty_in {
                println!();
            }
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            match slash(input, &mut msgs, &mut system, &mut sampling, &g) {
                Slash::Exit => break,
                Slash::Continue => continue,
            }
        }
        msgs.push(ChatMessage::user(input));
        let p = params(&g, &sampling, false);
        let mut printer = Printer::new(sampling.show_thinking, false);
        CANCEL.store(false, Ordering::SeqCst);
        GENERATING.store(true, Ordering::SeqCst);
        let res = g.chat_stream_events(&msgs, &p, |ev| {
            if CANCEL.load(Ordering::SeqCst) {
                return ControlFlow::Break(());
            }
            printer.event(ev)
        });
        GENERATING.store(false, Ordering::SeqCst);
        printer.finish();
        match res {
            Ok(out) => {
                if out.finish_reason == FinishReason::Aborted {
                    eprintln!("{}[interrupted]{}", dim_if(color), reset_if(color));
                }
                msgs.push(ChatMessage::assistant(out.text.clone()));
                if tty_in {
                    eprintln!(
                        "{}{}{}",
                        dim_if(color),
                        stats_line(&out, None),
                        reset_if(color)
                    );
                }
            }
            Err(e) => {
                msgs.pop();
                if let Some(GenerationError::ContextOverflow { .. }) =
                    e.downcast_ref::<GenerationError>()
                {
                    eprintln!("error: {e} — use /reset to start a new conversation");
                } else {
                    eprintln!("error: {e:#}");
                }
            }
        }
    }
    Ok(())
}
