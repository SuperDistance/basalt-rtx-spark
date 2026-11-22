#!/usr/bin/env python3
"""Replay the same concurrent Qwen serving workload on a local Basalt endpoint.

Start the server separately at 127.0.0.1 and give its port. A local forward to
a server on another Spark also works, with network latency then in the results.
"""

import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import random
import statistics
from threading import Barrier
import time


def api_request(port, method, route, body=None, timeout=8):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, route, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.read()
        if response.status != 200:
            raise RuntimeError(f"{route} returned HTTP {response.status}: {result[:600]!r}")
        return json.loads(result)
    finally:
        connection.close()


def make_prompt(seed, entries=24):
    rng = random.Random(seed)
    categories = ["lanterns", "radios", "labels", "cables", "chargers", "hammers", "sensors"]
    places = ["Albany", "Tacoma", "Fresno", "Bend", "Dover", "Tucson", "Savannah"]
    logs = [
        f"{rng.randrange(100000, 999999)}: {rng.choice(places)} received {rng.randrange(4, 888)} "
        f"{rng.choice(categories)}. The shipping team flagged {rng.randrange(1, 9)} cartons for review."
        for _ in range(entries)
    ]
    return "\n".join(logs) + (
        "\nWrite a detailed operational handover as numbered plain-text entries. For each entry in order, "
        "give the shipment identifier, summarize what happened, then explain two follow-up actions with "
        "specific details. Cover every entry and write at least 800 words."
    )


def stream_request(port, alias, seed, max_tokens, start_barrier=None, prompt_entries=24):
    payload = json.dumps({
        "model": alias, "messages": [{"role": "user", "content": make_prompt(seed, prompt_entries)}],
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    if start_barrier:
        start_barrier.wait(timeout=20)
    started = time.perf_counter()
    first_token = None
    usage = None
    timings = None
    finish = None
    try:
        connection.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(1500)!r}")
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                if first_token is None and choice.get("delta", {}).get("content"):
                    first_token = time.perf_counter()
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
        ended = time.perf_counter()
    finally:
        connection.close()
    if first_token is None or not usage or not usage.get("completion_tokens") or finish not in {"stop", "length"}:
        raise RuntimeError(f"Incomplete stream: finish={finish}, usage={usage}")
    return {
        "seed": seed, "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
        "ttft_s": first_token - started, "latency_s": ended - started, "finish": finish,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "started": started, "ended": ended, "timings": timings,
    }


def one_wave(port, alias, seeds, max_tokens, prompt_entries=24):
    barrier = Barrier(len(seeds))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        jobs = [pool.submit(stream_request, port, alias, seed, max_tokens, barrier, prompt_entries) for seed in seeds]
        return [job.result() for job in jobs]


def summarize(concurrency, waves):
    requests = [request for wave in waves for request in wave]
    wall_time = sum(max(r["ended"] for r in w) - min(r["started"] for r in w) for w in waves)
    ttft = [r["ttft_s"] for r in requests]
    latency = [r["latency_s"] for r in requests]
    timed = [r["timings"] for r in requests if r.get("timings")]
    prefill = [t["prompt_per_second"] for t in timed if t.get("prompt_per_second") is not None]
    decode = [t["predicted_per_second"] for t in timed if t.get("predicted_per_second") is not None]
    return {
        "concurrency": concurrency, "requests": len(requests),
        "output_tps": sum(r["output_tokens"] for r in requests) / wall_time,
        "median_ttft_s": statistics.median(ttft),
        "p95_ttft_s": statistics.quantiles(ttft, n=20, method="inclusive")[18],
        "max_ttft_s": max(ttft),
        "median_latency_s": statistics.median(latency),
        "p95_latency_s": statistics.quantiles(latency, n=20, method="inclusive")[18],
        "median_cached_tokens": statistics.median(r["cached_tokens"] for r in requests),
        "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in requests),
        "median_output_tokens": statistics.median(r["output_tokens"] for r in requests),
        "prefill_timing_samples": len(prefill),
        "decode_timing_samples": len(decode),
        "median_prefill_tps": statistics.median(prefill) if prefill else None,
        "median_decode_tps": statistics.median(decode) if decode else None,
        "requests_detail": requests,
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", nargs="+", default=[1, 2, 3, 4], type=int)
    parser.add_argument("--max-tokens", default=160, type=int)
    parser.add_argument("--prompt-entries", default=24, type=int)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.max_tokens < 1 or args.prompt_entries < 1:
        parser.error("concurrency, max-tokens and prompt-entries must be positive")
    alias = args.model or api_request(args.port, "GET", "/v1/models")["data"][0]["id"]
    if not alias:
        raise RuntimeError("Expected a nonempty served model name")
    one_wave(args.port, alias, [91], 32, args.prompt_entries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {"model": alias, "max_tokens": args.max_tokens, "prompt_entries": args.prompt_entries, "results": []}
    for count in args.concurrency:
        waves = [
            one_wave(args.port, alias, [100000 * count + wave * 100 + i for i in range(count)],
                     args.max_tokens, args.prompt_entries)
            for wave in range(max(3, 6 // count))
        ]
        summary = summarize(count, waves)
        results["results"].append(summary)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "requests_detail"}), flush=True)


if __name__ == "__main__":
    main()
