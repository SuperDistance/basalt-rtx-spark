"""Measure what happens to running chats when a long new prompt arrives, on any OpenAI-compatible server.

    python benchmarks/join_bench.py --endpoint http://127.0.0.1:8080/v1 --model flash-next --depths 4000 16000 32000 --out join.json

Each trial starts three streaming chats ("incumbents"), waits until each has produced output, then sends a cold
prompt of the target depth (a fresh nonce leads it, so no engine can reuse a cached prefix). It records when every
streamed update arrived and reports:

- joiner TTFT: from sending the long prompt to its first streamed text
- incumbent stall: the longest gap between two updates of a running chat while the long prompt was being read
- incumbent rate during the read, against its rate before the read (output updates per second, and tokens when the
  server reports usage)

A frozen engine shows a stall about as long as the joiner's TTFT; an engine that reads alongside decode keeps the
stall near one decode window.
"""
import argparse
import http.client
import json
import statistics
import threading
import time
import uuid
from urllib.parse import urlparse

INCUMBENT_PROMPTS = [
    "Write a detailed, step-by-step explanation of how a hash map handles collisions, with a Python example.",
    "Explain how TCP congestion control works, covering slow start, congestion avoidance and fast recovery.",
    "Describe how a compiler turns source code into machine code, stage by stage, with a small example.",
]
FILLER = ("Section {i}. The quarterly maintenance log for station {i} records that pump {a} was inspected, valve {b} "
          "was cleaned, and sensor {c} reported a reading of {d} units before calibration. ")


def long_prompt(depth_tokens: int) -> str:
    """About depth_tokens tokens of varied text, led by a nonce so no prefix cache can match it."""
    parts, i = [f"Reference id {uuid.uuid4()}.\n"], 0
    while sum(len(p) for p in parts) < depth_tokens * 4.1:     # ~4.1 characters per token for this text
        parts.append(FILLER.format(i=i, a=(i * 7) % 97, b=(i * 13) % 89, c=(i * 31) % 83, d=(i * 17) % 1000))
        i += 1
    parts.append("\nIn one sentence, which station has the highest reading?")
    return "".join(parts)


def stream(base, model, prompt, max_tokens, rec, started=None):
    """Stream one chat completion, appending (arrival time, text) for each update to rec['ticks']."""
    url = urlparse(base.rstrip("/") + "/chat/completions")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": max_tokens, "temperature": 0,
            # thinking off on Basalt, llama.cpp and vLLM
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=3600)
    rec["sent"] = time.perf_counter()
    if started is not None:
        started.set()
    conn.request("POST", url.path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        rec["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
        return
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            rec["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if text:
                rec["ticks"].append(time.perf_counter())


def trial(base, model, depth, incumbent_tokens, warm_updates):
    incs = [{"ticks": []} for _ in INCUMBENT_PROMPTS]
    threads = [threading.Thread(target=stream, args=(base, model, p, incumbent_tokens, r), daemon=True)
               for p, r in zip(INCUMBENT_PROMPTS, incs)]
    for t in threads:
        t.start()
    deadline = time.perf_counter() + 600
    while any(len(r["ticks"]) < warm_updates for r in incs):
        if time.perf_counter() > deadline or any("error" in r for r in incs):
            raise RuntimeError(f"incumbents did not start: {[r.get('error') for r in incs]}")
        time.sleep(0.05)
    joiner = {"ticks": []}
    stream(base, model, long_prompt(depth), 16, joiner)
    for t in threads:
        t.join()
    if "error" in joiner or not joiner["ticks"]:
        raise RuntimeError(f"joiner failed: {joiner.get('error')}")
    start, first = joiner["sent"], joiner["ticks"][0]
    out = {"target_depth": depth, "joiner_prompt_tokens": (joiner.get("usage") or {}).get("prompt_tokens"),
           "joiner_ttft_s": round(first - start, 3), "incumbents": []}
    for r in incs:
        tk = r["ticks"]
        gaps = [b - a for a, b in zip(tk, tk[1:]) if b > start and a < first]
        before = [t for t in tk if t < start]
        during = [t for t in tk if start <= t < first]
        rate_before = (len(before) - 1) / (before[-1] - before[0]) if len(before) > 2 else None
        out["incumbents"].append({
            "updates_before": len(before), "updates_during_read": len(during),
            "max_stall_s": round(max(gaps), 3) if gaps else None,
            "updates_per_s_before": round(rate_before, 1) if rate_before else None,
            "updates_per_s_during": round(len(during) / (first - start), 1) if first > start else None,
            "still_running_after_join": tk[-1] > first if tk else False,
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens")})
    out["max_stall_s"] = max((i["max_stall_s"] or 0) for i in out["incumbents"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="flash-next")
    ap.add_argument("--depths", type=int, nargs="+", default=[4000, 16000, 32000])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--incumbent-tokens", type=int, default=3000, help="long enough to outlast the joiner's read")
    ap.add_argument("--warm-updates", type=int, default=16)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trials = []
    for depth in a.depths:
        for rep in range(a.repeats):
            r = trial(a.endpoint, a.model, depth, a.incumbent_tokens, a.warm_updates)
            r["repeat"] = rep + 1
            trials.append(r)
            print(f"depth {depth:>6} rep {rep + 1}: joiner TTFT {r['joiner_ttft_s']:6.2f} s, "
                  f"longest incumbent stall {r['max_stall_s']:6.2f} s, prompt tokens {r['joiner_prompt_tokens']}",
                  flush=True)
    summary = {}
    for depth in a.depths:
        rs = [t for t in trials if t["target_depth"] == depth]
        summary[depth] = {"joiner_ttft_s": statistics.median(t["joiner_ttft_s"] for t in rs),
                          "max_stall_s": statistics.median(t["max_stall_s"] for t in rs)}
    json.dump({"label": a.label, "endpoint": a.endpoint, "summary": summary, "trials": trials},
              open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
