#!/bin/bash
# Batch vs slot scheduler throughput of xor-net-server.
#
#   cargo build --release --bin xor-net-server
#   scripts/bench_scheduler.sh models/bonsai/Bonsai-27B.xq1 [MAX_TOKENS=64] [REPS=3] ["1 2 4 6 8"]
#
# For each stream count N, starts the server in batch mode and in slot mode
# (default 2 slots x physical/2 threads) back to back, so timing noise hits
# both alike, and runs N concurrent greedy streaming requests (counting
# prompts: every stream produces exactly MAX_TOKENS tokens). Prints aggregate
# tok/s, mean per-stream decode tok/s and time to first token (best of REPS
# by aggregate). Prompt cache off. Each server runs under
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice`.
set -u
MODEL=$1
MAXT=${2:-64}
REPS=${3:-3}
NS=${4:-"1 2 4 6 8"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/target/release/xor-net-server
PORT=${PORT:-18200}
LOG=$(mktemp)

run() { # N, server args...
  local n=$1; shift
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 nice \
    "$BIN" -m "$MODEL" --port "$PORT" --prompt-cache-mb 0 "$@" >"$LOG" 2>&1 &
  local pid=$!
  for _ in $(seq 1 1200); do
    curl -s "127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 $pid 2>/dev/null; then cat "$LOG"; return; fi
    sleep 0.5
  done
  printf '%-6s ' "$SCHED"
  python3 -I "$ROOT/scripts/serve_load.py" "$PORT" "$n" "$MAXT" "$REPS"
  kill $pid 2>/dev/null
  wait $pid 2>/dev/null
  PORT=$((PORT + 1))
}

for N in $NS; do
  SCHED=batch run "$N" --scheduler batch --max-batch $((N > 8 ? N : 8))
  SCHED=slots run "$N" --scheduler slots
done
rm -f "$LOG"
