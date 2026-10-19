#!/usr/bin/env bash
#
# bench_all.sh  xor-net comprehensive benchmark matrix driver.
#
# Builds `bench_runner` once, then loops over the full configuration matrix,
# re-invoking the compiled binary once per combo. A fresh process per combo is
# REQUIRED because `init_threads` installs a GLOBAL rayon pool (can only be set
# once per process).
#
# Every run appends one JSON line to bench_logs/<timestamp>.jsonl; a CSV
# summary is produced at the end (scripts/jsonl_to_csv.py).
#
# Environment overrides (comma- or space-separated):
#   MODELS     local dir(s) and/or HF repo ids   (default: models/bitnet-2b)
#   QUANTS     none bit1 bit1_58_pack4 bit1_58_pack5
#   LMHEADS    f32 f16 int8 int4 ternary
#   THREADS    1 2 3 4 6 8 12
#   MODES      single concurrent
#   STREAMS    1 2 4 6 12             (concurrent only)
#   POOL_THREADS 0 1 2                (concurrent only; 0 = shared global pool)
#   TOKENS     tokens per stream       (default 128)
#   PROMPT     prompt text
#   TOK_REPO   HF repo for the tokenizer when the model dir has none
#   SPARSE     1 to also sweep XORNET_WEIGHT_FMT=sparse (needs model.sparse)
#   BENCH_DIR  output dir              (default bench_logs)
#
# CLI: --quick  reduced matrix (fast validation)
#      --dry-run  print the matrix and exit
#      --model X --quant X --mode X  restrict a single axis (repeatable)

set -o pipefail

cd "$(dirname "$0")/.." || exit 1

# ---- defaults ----------------------------------------------------------------
MODELS="${MODELS:-models/bitnet-2b}"
QUANTS="${QUANTS:-none bit1 bit1_58_pack4 bit1_58_pack5}"
LMHEADS="${LMHEADS:-f32 f16 int8 int4 ternary}"
THREADS="${THREADS:-1 2 3 4 6 8 12}"
MODES="${MODES:-single concurrent}"
STREAMS="${STREAMS:-1 2 4 6 12}"
POOL_THREADS="${POOL_THREADS:-0 1 2}"
TOKENS="${TOKENS:-128}"
PROMPT="${PROMPT:-The history of the Roman Empire is}"
TOK_REPO="${TOK_REPO:-microsoft/bitnet-b1.58-2B-4T}"
SPARSE="${SPARSE:-0}"
BENCH_DIR="${BENCH_DIR:-bench_logs}"

QUICK=0
DRYRUN=0
FILTER_MODEL=()
FILTER_QUANT=()
FILTER_MODE=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --model) FILTER_MODEL+=("$2"); shift 2 ;;
    --quant) FILTER_QUANT+=("$2"); shift 2 ;;
    --mode)  FILTER_MODE+=("$2"); shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  THREADS="4 8"
  MODES="single"
  STREAMS="2"
  POOL_THREADS="0 1"
  TOKENS="${TOKENS:-64}"
fi

# ---- build -------------------------------------------------------------------
if [[ "$DRYRUN" -eq 0 ]]; then
  echo "[bench_all] building bench_runner (release)..." >&2
  cargo build --release --example bench_runner || exit 1
fi

BIN="./target/release/examples/bench_runner"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUSTC="$(rustc --version 2>/dev/null || echo unknown)"
FEATURES="${FEATURES:-}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$BENCH_DIR/$TS.jsonl"
mkdir -p "$BENCH_DIR"

RUN_IDX=0
count=0

in_filter() {
  local wanted="$1"; shift
  [[ $# -eq 0 ]] && return 0
  local v
  for v in "$@"; do [[ "$v" == "$wanted" ]] && return 0; done
  return 1
}

run_one() {
  local model="$1" quant="$2" lmhead="$3" fmt="$4"

  for threads in $THREADS; do
    for mode in $MODES; do
      in_filter "$mode" "${FILTER_MODE[@]}" || continue

      if [[ "$mode" == "single" ]]; then
        launch "$model" "$quant" "$lmhead" "$fmt" "$threads" single 1 0
      else
        for streams in $STREAMS; do
          for pool in $POOL_THREADS; do
            launch "$model" "$quant" "$lmhead" "$fmt" "$threads" concurrent "$streams" "$pool"
          done
        done
      fi
    done
  done
}

launch() {
  local model="$1" quant="$2" lmhead="$3" fmt="$4" threads="$5" mode="$6" streams="$7" pool="$8"
  RUN_IDX=$((RUN_IDX+1))
  local run_id
  run_id="$(printf '%s-%05d' "$TS" "$RUN_IDX")"
  count=$((count+1))

  if [[ "$DRYRUN" -eq 1 ]]; then
    echo "[dry] $run_id model=$model quant=$quant lmhead=$lmhead fmt=$fmt threads=$threads mode=$mode streams=$streams pool=$pool"
    return
  fi

  echo "[bench_all] ($count) $run_id quant=$quant lmhead=$lmhead fmt=$fmt threads=$threads mode=$mode streams=$streams pool=$pool" >&2
  XORNET_MODEL="$model" \
  XORNET_QUANT="$quant" \
  XORNET_LMHEAD="$lmhead" \
  XORNET_WEIGHT_FMT="$fmt" \
  XORNET_THREADS="$threads" \
  XORNET_MODE="$mode" \
  XORNET_STREAMS="$streams" \
  XORNET_POOL_THREADS="$pool" \
  XORNET_TOKENS="$TOKENS" \
  XORNET_PROMPT="$PROMPT" \
  XORNET_TOKENIZER_REPO="$TOK_REPO" \
  XORNET_LOG="$LOG" \
  XORNET_RUN_ID="$run_id" \
  XORNET_GIT_SHA="$GIT_SHA" \
  XORNET_RUSTC="$RUSTC" \
  XORNET_FEATURES="$FEATURES" \
  "$BIN" 2>/dev/null
}

# ---- loop --------------------------------------------------------------------
while IFS= read -r model; do
  [[ -z "$model" ]] && continue
  in_filter "$model" "${FILTER_MODEL[@]}" || continue

  model_is_dir=0
  [[ -d "$model" ]] && model_is_dir=1

  for quant in $QUANTS; do
    in_filter "$quant" "${FILTER_QUANT[@]}" || continue

    for lmhead in $LMHEADS; do
      # baseline F32 ignores LM-head precision
      if [[ "$quant" == "none" && "$lmhead" != "f32" ]]; then continue; fi

      run_one "$model" "$quant" "$lmhead" "dense"

      # sparse weight format only applies to the BitNet 1.58 path
      if [[ "$SPARSE" == "1" && "$quant" == bit1_58_* ]]; then
        sparse_ok=1
        if [[ "$model_is_dir" -eq 1 && ! -f "$model/model.sparse" ]]; then
          sparse_ok=0
        fi
        if [[ "$sparse_ok" -eq 1 ]]; then
          run_one "$model" "$quant" "$lmhead" "sparse"
        else
          echo "[bench_all] skip sparse: $model/model.sparse missing" >&2
        fi
      fi
    done
  done
done < <(printf '%s\n' $MODELS)

if [[ "$DRYRUN" -eq 1 ]]; then
  echo "[bench_all] dry-run: $count combos (build skipped)" >&2
  exit 0
fi

echo "[bench_all] done: $count runs -> $LOG" >&2
echo "[bench_all] writing CSV summary..." >&2
python3 scripts/jsonl_to_csv.py "$LOG" "$LOG.csv" && echo "[bench_all] CSV -> $LOG.csv" >&2
