//! End-to-end tests of the `xor-net` binary.
//!
//! ```text
//! cargo test --release --features cli --test cli_e2e                 # fast, no model
//! BONSAI_XQ1=models/bonsai/Bonsai-1.7B.xq1 \
//!   cargo test --release --features cli --test cli_e2e -- --ignored  # real model
//! ```
#![cfg(feature = "cli")]

use std::process::{Command, Output};

fn xor_net(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xor-net"))
        .args(args)
        .output()
        .expect("spawning xor-net")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn help_for_every_subcommand() {
    let o = xor_net(&["--help"]);
    assert!(o.status.success());
    for c in [
        "run", "chat", "convert", "bench", "info", "classify", "pull", "serve",
    ] {
        assert!(stdout(&o).contains(c), "top-level help lists {c}");
        let o = xor_net(&[c, "--help"]);
        assert!(o.status.success(), "{c} --help");
        assert!(stdout(&o).contains("USAGE"), "{c} --help prints usage");
    }
}

#[test]
fn usage_errors_exit_2() {
    for args in [
        &["bogus"][..],
        &["run"][..],
        &["run", "m.gguf", "--temp"][..],
        &["run", "m.gguf", "--nope"][..],
        &["classify"][..],
    ] {
        let o = xor_net(args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", stderr(&o));
        assert!(stderr(&o).starts_with("error:"), "{args:?}");
    }
}

#[test]
fn missing_model_is_a_clean_error() {
    let o = xor_net(&["info", "no/such/dir/model.gguf"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stderr(&o).contains("does not exist"), "{}", stderr(&o));
}

fn bonsai() -> Option<String> {
    let p = std::env::var("BONSAI_XQ1").ok()?;
    std::path::Path::new(&p).is_file().then_some(p)
}

#[test]
#[ignore = "needs BONSAI_XQ1=<path to a Bonsai .xq1/.gguf>"]
fn run_generates_and_reports_stats() {
    let Some(model) = bonsai() else {
        eprintln!("BONSAI_XQ1 not set; skipping");
        return;
    };
    let o = xor_net(&[
        "run",
        &model,
        "-p",
        "The capital of France is",
        "-n",
        "8",
        "--greedy",
        "--threads",
        "4",
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("Paris"), "stdout: {}", stdout(&o));
    let err = stderr(&o);
    assert!(err.contains("decode") && err.contains("tok/s"), "{err}");

    // --quiet: nothing but the completion; deterministic under greedy.
    let q = xor_net(&[
        "run",
        &model,
        "The capital of France is",
        "-n",
        "8",
        "--greedy",
        "-q",
    ]);
    assert!(q.status.success());
    assert_eq!(stdout(&q), stdout(&o));
    assert!(!stderr(&q).contains("tok/s"));

    // Chat mode with a stop string.
    let c = xor_net(&[
        "run",
        &model,
        "--chat",
        "-p",
        "Count from 1 to 10, comma separated.",
        "-n",
        "48",
        "--greedy",
        "--stop",
        "5",
        "-q",
    ]);
    assert!(c.status.success(), "{}", stderr(&c));
    let text = stdout(&c);
    assert!(text.contains('4') && !text.contains('5'), "{text}");

    let i = xor_net(&["info", &model]);
    assert!(i.status.success());
    assert!(stdout(&i).contains("qwen3") && stdout(&i).contains("chat template"));
}
