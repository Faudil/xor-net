#!/usr/bin/env python3
"""Convert a bench_runner JSONL log into a flat CSV summary.

Usage: jsonl_to_csv.py <input.jsonl> [output.csv]

All runs share a union of keys (sorted for stability). Array-valued fields
(e.g. per_stream_tps) are joined with ';'. Null becomes the empty string.
"""
import csv
import json
import sys
from collections import OrderedDict


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: jsonl_to_csv.py <input.jsonl> [output.csv]", file=sys.stderr)
        return 2

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path + ".csv"

    rows = []
    keys = OrderedDict()
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip bad line: {e}", file=sys.stderr)
                continue
            rows.append(obj)
            for k in obj:
                keys[k] = True

    # Put the most useful columns first.
    preferred = [
        "run_id", "timestamp", "model", "quant", "lm_head", "weight_fmt",
        "threads", "mode", "streams", "pool_threads",
        "overall_tps", "decode_tps", "prefill_tps", "aggregate_tps",
        "avg_per_stream_tps", "tokens_generated", "total_gen",
        "prefill_ms", "decode_ms", "wall_ms",
        "blocks_ms", "lm_head_ms", "attn_ms", "mlp_ms", "norm_ms", "silu_ms",
        "load_ms", "tokens_requested", "eos",
        "weight_bytes_per_token", "rss_kb", "vmhwm_kb",
        "git_sha", "rustc", "features", "cpu_model",
        "target_features", "prompt", "lm_head_requested", "output_tail",
    ]
    header = [k for k in preferred if k in keys]
    header += [k for k in keys if k not in header]

    def cell(v):
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return ";".join(str(x) for x in v)
        return v

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([cell(r.get(k)) for k in header])

    print(f"wrote {len(rows)} rows x {len(header)} cols -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
