//! `xor-net`: command-line front end for the xor-net 1-bit / 1.58-bit CPU
//! runtime. Run `xor-net --help`.

mod args;
mod generate;
mod tools;

use args::Command;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "\
xor-net: fast CPU inference for 1-bit and 1.58-bit models

USAGE:
    xor-net <COMMAND> [ARGS]

COMMANDS:
    run       Generate text from a prompt (streams to stdout)
    chat      Interactive multi-turn chat
    convert   Convert a Bonsai Q1_0 .gguf to the lossless, mmap-able .xq1 format
    bench     Measure prefill / decode throughput and peak memory
    info      Show architecture, size, quantization and chat template of a model
    classify  Classify images with a binary CNN (QuickNet)
    pull      Download a model from the Hugging Face Hub and print its path
    serve     Start the OpenAI-compatible HTTP server (xor-net-server)
    help      Show help for a command

MODELS:
    A model is a path (.gguf, .xq1, or a Llama/BitNet safetensors directory)
    or a Hugging Face repo id such as `prism-ml/Bonsai-1.7B-gguf` (the Q1_0
    .gguf is downloaded into the HF cache). Use `owner/repo:file.gguf` to pick
    a specific file.

EXAMPLES:
    xor-net run prism-ml/Bonsai-1.7B-gguf -p \"The capital of France is\" -n 32
    xor-net chat prism-ml/Bonsai-8B-gguf
    xor-net convert Bonsai-8B-Q1_0.gguf Bonsai-8B.xq1
    xor-net bench Bonsai-8B.xq1 --threads 6
    xor-net classify cat.jpg -m quicknet.safetensors

Run `xor-net <COMMAND> --help` for the options of a command.
";

const GEN_FLAGS: &str = "\
GENERATION OPTIONS:
    -n, --max-tokens <N>     Maximum tokens to generate            [512]
        --temp <T>           Temperature; 0 = greedy               [0.7]
        --top-k <K>          Top-k sampling; 0 = off               [20]
        --top-p <P>          Nucleus sampling                      [0.8]
        --min-p <P>          Min-p sampling; 0 = off               [0]
        --repeat-penalty <R> Repetition penalty; 1 = off           [1.0]
        --greedy             Same as --temp 0
        --seed <N>           Sampling seed (default: random)
        --stop <STR>         Stop string (repeatable)
        --think / --no-think Turn reasoning on/off in the chat template
                             (`enable_thinking`; only templates that read it,
                             e.g. Bonsai 27B; Bonsai 1.7B-8B never think)
        --show-thinking      Print reasoning (dimmed) instead of hiding it
        --draft <D>          Lossless speculative decoding: `lookup` (n-gram
                             prompt lookup, any model) or a DSpark drafter
                             .gguf (Bonsai 27B: Bonsai-27B-dspark-Q4_1.gguf)
        --draft-k <N>        Draft tokens per round   [dspark: 4, lookup: 4]
        --draft-head <H>     DSpark head: shared (the target's, faster) or
                             own (the drafter GGUF's Q4_1 head)    [shared]

MODEL OPTIONS:
    -t, --threads <N>        Worker threads (default: auto, max 8)
        --tokenizer <FILE>   Override the tokenizer.json
";

fn command_help(cmd: &str) -> Option<String> {
    let s = match cmd {
        "run" | "generate" => format!(
            "\
xor-net run: generate text from a prompt

USAGE:
    xor-net run <MODEL> [-p PROMPT | PROMPT] [OPTIONS]

    The completion streams to stdout; a stats line (tokens, prefill/decode
    tok/s, peak memory) goes to stderr. With no prompt, the prompt is read
    from stdin.

OPTIONS:
    -p, --prompt <TEXT>      Prompt text
    -f, --file <FILE>        Read the prompt from a file
        --chat               Wrap the prompt in the model's chat template
    -s, --system <TEXT>      System message (implies --chat)
        --echo               Print the prompt before the completion (raw mode)
        --ignore-eos         Keep generating past end-of-sequence
    -q, --quiet              No stats line on stderr

{GEN_FLAGS}"
        ),
        "chat" => format!(
            "\
xor-net chat: interactive multi-turn chat

USAGE:
    xor-net chat <MODEL> [OPTIONS]

    Type a message and press Enter. Ctrl-C stops the current reply without
    leaving the chat (press it twice at the prompt, or Ctrl-D, to quit).

IN-CHAT COMMANDS:
    /reset               Forget the conversation (keeps the system message)
    /system <TEXT>       Set the system message and reset
    /think on|off        Toggle `enable_thinking`
    /show-thinking on|off  Show or hide reasoning
    /help                List commands
    /exit, /quit         Leave

OPTIONS:
    -s, --system <TEXT>      System message

{GEN_FLAGS}"
        ),
        "convert" => "\
xor-net convert: convert a Bonsai Q1_0 GGUF to .xq1

USAGE:
    xor-net convert <IN.gguf | HUB_ID> [OUT.xq1] [--tokenizer tokenizer.json]

    .xq1 stores the same sign bits and f16 scales as the GGUF (lossless; the
    output is reloaded and verified tensor by tensor) but pre-split and
    pre-fused, with the tokenizer and chat template embedded. Loading is a
    zero-copy mmap: near-instant start-up and page-cache sharing between
    processes. OUT defaults to IN with the extension replaced by .xq1.

OPTIONS:
    -o, --output <FILE>      Output path
        --tokenizer <FILE>   Embed this tokenizer.json instead of the GGUF's
"
        .to_string(),
        "bench" => "\
xor-net bench: prefill / decode throughput

USAGE:
    xor-net bench <MODEL> [OPTIONS]

    Prefills P synthetic tokens in one call, then decodes G tokens one at a
    time (greedy argmax). Reports the best of R repetitions, load time and
    peak RSS.

OPTIONS:
    -p, --prompt-len <P>     Prompt tokens                    [128]
    -n, --gen <G>            Generated tokens                 [64]
    -r, --reps <R>           Repetitions                      [2]
    -t, --threads <N>        Worker threads (default: auto, max 8)
"
        .to_string(),
        "info" | "show" => "\
xor-net info: describe a model without running it

USAGE:
    xor-net info <MODEL> [--template]

OPTIONS:
        --template           Also print the chat template source
"
        .to_string(),
        "classify" => format!(
            "\
xor-net classify: ImageNet classification with a binary CNN

USAGE:
    xor-net classify <IMAGE>... [OPTIONS]

    Images are JPEG or PNG. The model is an `xornet-vision-v1` safetensors
    file (e.g. QuickNet, converted by scripts/vision/convert_larq_zoo.py).

OPTIONS:
    -m, --model <FILE>       Model file  [$XORNET_VISION_MODEL or {}]
    -k, --top-k <K>          Number of classes to print       [5]
    -t, --threads <N>        Worker threads (default: all cores)
",
            args::DEFAULT_VISION_MODEL
        ),
        "pull" | "download" => "\
xor-net pull: download a model from the Hugging Face Hub

USAGE:
    xor-net pull <OWNER/REPO[:FILE]>

    Downloads the repo's Q1_0 .gguf (or the safetensors + tokenizer of a
    Llama/BitNet repo) into the HF cache ($HF_HOME) and prints the local path.
"
        .to_string(),
        "serve" | "server" => "\
xor-net serve: start the HTTP server

USAGE:
    xor-net serve [SERVER OPTIONS]

    Runs `xor-net-server` (built from the same crate) with the given options;
    see `xor-net-server --help`.
"
        .to_string(),
        _ => return None,
    };
    Some(s)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match args::parse(argv, std::env::var("XORNET_VISION_MODEL").ok()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let res = match cmd {
        Command::Help(None) => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Help(Some(c)) => match command_help(&c) {
            Some(h) => {
                print!("{h}");
                Ok(())
            }
            None => {
                eprintln!("error: no help for unknown command {c:?}\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Command::Version => {
            println!("xor-net {VERSION}");
            Ok(())
        }
        Command::Run(a) => generate::run(a),
        Command::Chat(a) => generate::chat(a),
        Command::Convert(a) => tools::convert(a),
        Command::Bench(a) => tools::bench(a),
        Command::Info(a) => tools::info(a),
        Command::Classify(a) => tools::classify(a),
        Command::Pull(a) => tools::pull(a),
        Command::Serve(a) => tools::serve(a),
    };
    if let Err(e) = res {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
