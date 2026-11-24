"""Measure a cold Flash-Next prompt arriving while three requests decode on the native server."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.engine.process import ProcessEngine


def generate(engine, prompt, limit, cancel, ready=None):
    started = time.perf_counter()
    ticks = []
    ids = []
    for token in engine.generate_batched(prompt, limit, {"temperature": 0, "seed": 17}, cancel):
        if type(token) is int:
            ticks.append(time.perf_counter())
            ids.append(token)
            if ready is not None and len(ids) == 16:
                ready.set()
    if not ticks:
        raise RuntimeError("Request ended without output")
    return started, ticks, hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def measure(engine, short, long):
    cancel = threading.Event()
    epoch_offset = time.time() - time.perf_counter()
    pending = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        try:
            for _ in range(3):
                ready = threading.Event()
                job = pool.submit(generate, engine, short, 768, cancel, ready)
                pending.append(job)
                deadline = time.monotonic() + 180
                while not ready.wait(0.25):
                    if job.done():
                        job.result()
                        raise RuntimeError("An incumbent ended before its 16th output token")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("An incumbent did not reach its 16th output token")
            start, join, digest = generate(engine, long, 32, cancel)
            ongoing = []
            for job in pending:
                _, ticks, token_hash = job.result()
                gaps = [b - a for a, b in zip(ticks, ticks[1:]) if a < join[0] and b > start]
                ongoing.append({
                    "output_tokens": len(ticks), "output_sha256": token_hash,
                    "tokens_before_join": sum(t < start for t in ticks),
                    "tokens_during_join_prefill": sum(start <= t < join[0] for t in ticks),
                    "active_past_join_ttft": ticks[-1] > join[0],
                    "max_gap_during_join_s": max(gaps, default=None),
                })
            return {"prompt_tokens": len(long), "output_tokens": len(join), "output_sha256": digest,
                    "join_submit_unix_s": start + epoch_offset, "join_first_token_unix_s": join[0] + epoch_offset,
                    "join_ttft_s": join[0] - start, "ongoing": ongoing}
        finally:
            cancel.set()


def native_activity(log, prompt_tokens):
    reads = re.findall(r"overlapped prompt read: (\d+) windows .*? beside (\d+) chunk jobs", log)
    drafts = re.findall(r"basalt batch spec: .*?drafts (\d+) accepted of (\d+)", log)
    return {
        "cold_read_verified": bool(re.search(
            rf"prompt {prompt_tokens} tokens = 0 reused \+ {prompt_tokens} read\b", log)),
        "overlapped_windows": sum(int(w) for w, _ in reads),
        "chunk_jobs": sum(int(c) for _, c in reads),
        "batched_drafts_accepted": sum(int(a) for a, _ in drafts),
        "batched_drafts_offered": sum(int(o) for _, o in drafts),
        "checkpoint_reuse_observed": bool(re.search(r"prompt \d+ tokens = [1-9]\d* reused", log)),
        "kv_growth_observed": "basalt serve: elastic KV:" in log,
        "separate_prompt_scratch_observed": "outside the full expert cache" in log,
    }


def short_during_prefill(engine, short, long):
    cancel = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(generate, engine, long, 32, cancel)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                progress = engine.progress
                if progress and 1024 <= progress[0] < len(long) - 8192:
                    break
                if job.done():
                    job.result()
                    raise RuntimeError("Long prompt finished before the short arrival")
                time.sleep(0.025)
            else:
                raise TimeoutError("Did not observe the long prompt being read")
            start, reply, digest = generate(engine, short, 16, cancel)
            long_start, long_reply, long_digest = job.result()
            return {"prompt_tokens": len(long), "read_at_arrival": progress[0],
                    "short_ttft_s": reply[0] - start, "long_ttft_s": long_reply[0] - long_start,
                    "short_output_tokens": len(reply), "long_output_tokens": len(long_reply),
                    "short_sha256": digest, "long_sha256": long_digest,
                    "short_finished_before_long_ttft": reply[-1] < long_reply[0]}
        finally:
            cancel.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory for this arm")
    parser.add_argument("--scenario", choices=("cold-join", "short-during-prefill"), default="cold-join")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER, help="Additional native options after --")
    args = parser.parse_args()
    extra = args.engine_args[1:] if args.engine_args[:1] == ["--"] else args.engine_args
    native = ["--model", str(args.model.resolve()), "--gpu", "0", "--batch", "4", "--max-context", "262144",
              "--kv", "int8", "--spec", "8", *extra]
    fixtures = ROOT / "data/bench"
    short = list(map(int, (fixtures / "struct.ids").read_text().strip().split(",")))
    head, body, tail = [list(map(int, line.split(","))) for line in (fixtures / "prefill.ids").read_text().splitlines()]
    interrupt = args.scenario == "short-during-prefill"
    long = head + body * (3 if interrupt else 1) + tail
    args.out.mkdir(parents=True, exist_ok=False)
    log = args.out.resolve() / "engine.log"
    binary = args.engine.resolve()
    digest = hashlib.sha256()
    with binary.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    (args.out / "launch.json").write_text(json.dumps({"engine": str(binary), "sha256": digest.hexdigest(),
                                                     "scenario": args.scenario, "args": native}, indent=2), encoding="utf-8")
    engine = ProcessEngine(str(binary), native, cwd=str(binary.parent), log=str(log))
    try:
        result = short_during_prefill(engine, short, long) if interrupt else measure(engine, short, long)
    finally:
        engine.close()
    lines = log.read_text(encoding="utf-8", errors="replace")
    if interrupt:
        yielded = re.search(rf"the prompt read gives way at (\d+) of {len(long)} tokens", lines)
        at = int(yielded[1]) if yielded else 0
        result["yield_at"] = at
        result["initial_read_cold"] = bool(at and re.search(
            rf"prompt {len(long)} tokens = 0 reused \+ {at} of {len(long)} read\b", lines))
        result["resumed_from_yield"] = bool(at and re.search(
            rf"prompt {len(long)} tokens = {at} reused \+ {len(long) - at} read\b", lines))
        result["passed"] = (result["initial_read_cold"] and result["resumed_from_yield"]
                            and result["short_finished_before_long_ttft"])
    else:
        result.update(native_activity(lines, result["prompt_tokens"]))
        result["passed"] = bool(result["cold_read_verified"] and result["overlapped_windows"] > 0
                                and result["chunk_jobs"] > 0 and result["output_tokens"] == 32
                                and all(r["tokens_before_join"] >= 16 and r["tokens_during_join_prefill"] > 0
                                        and r["active_past_join_ttft"] and r["output_tokens"] == 768
                                        for r in result["ongoing"]))
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
