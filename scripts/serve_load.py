#!/usr/bin/env python3
"""Concurrent streaming load against xor-net-server.

usage: serve_load.py PORT N MAX_TOKENS [REPS [STAGGER_SECONDS]]

N clients each stream one /v1/completions request (temperature 0) at the
same time; reports aggregate tok/s (all completion tokens / wall time),
mean per-stream decode tok/s (tokens after the first / time after the first
token) and time to first token (mean / max). Best (max aggregate) of REPS.
"""
import http.client
import json
import sys
import threading
import time

# Counting continues far beyond max_tokens (no early EOS), so every stream
# produces exactly max_tokens tokens.
PROMPTS = [", ".join(str(k) for k in range(s, s + 8)) + "," for s in
           (1, 101, 2001, 37, 512, 9000, 75, 333)]


def one(port, i, max_tokens, out, delay=0.0):
    time.sleep(delay)
    body = json.dumps({
        "prompt": PROMPTS[i % len(PROMPTS)],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    n_delta = 0
    usage = None
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                d = line[6:]
                if d == b"[DONE]":
                    continue
                v = json.loads(d)
                if v.get("usage"):
                    usage = v["usage"]
                ch = v.get("choices") or []
                if ch and ch[0].get("text"):
                    if first is None:
                        first = time.time()
                    n_delta += 1
    t1 = time.time()
    toks = usage["completion_tokens"] if usage else n_delta
    out[i] = (t0, first or t1, t1, toks)


def run(port, n, max_tokens, stagger=0.0):
    out = [None] * n
    th = [threading.Thread(target=one, args=(port, i, max_tokens, out, i * stagger)) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    start = min(o[0] for o in out)
    end = max(o[2] for o in out)
    total = sum(o[3] for o in out)
    agg = total / (end - start)
    per = [(o[3] - 1) / (o[2] - o[1]) for o in out if o[2] > o[1] and o[3] > 1]
    ttft = [o[1] - o[0] for o in out]
    return agg, sum(per) / len(per), sum(ttft) / n, max(ttft), total


def main():
    port, n, max_tokens = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    stagger = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    # Warm-up (page cache, allocations).
    run(port, 1, 8)
    best = None
    for _ in range(reps):
        r = run(port, n, max_tokens, stagger)
        if best is None or r[0] > best[0]:
            best = r
    agg, per, ttft, ttft_max, total = best
    if stagger:
        print(f"N={n} staggered {stagger}s: TTFT mean {ttft*1e3:.0f} ms max {ttft_max*1e3:.0f} ms")
        return
    print(f"N={n} aggregate {agg:.1f} tok/s  per-stream {per:.1f} tok/s  "
          f"TTFT mean {ttft*1e3:.0f} ms max {ttft_max*1e3:.0f} ms  ({total} tokens)")


if __name__ == "__main__":
    main()
