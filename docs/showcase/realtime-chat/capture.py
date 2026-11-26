"""Record one streamed chat completion with the arrival time of every server update.

    python capture.py http://127.0.0.1:8080/v1 stream.json

Writes the request, each update's arrival time (seconds after the request was sent) and text, and the server's
token usage. render.py replays it at the recorded speed.
"""
import http.client
import json
import sys
import time
from urllib.parse import urlparse

PROMPT = ("Explain in a short paragraph how a Hohmann transfer moves a spacecraft between two circular orbits, then "
          "write a Python function that returns both burn delta-v values and the transfer time.")


def main() -> int:
    base, out = sys.argv[1].rstrip("/"), sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else PROMPT
    url = urlparse(base + "/chat/completions")
    body = {"model": "flash-next", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 1200, "reasoning_effort": "none",
            "temperature": 0.7, "top_p": 0.95, "top_k": 20}
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", url.path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        sys.exit(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    events, usage = [], None
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            text = (choice.get("delta") or {}).get("content") or ""
            if text:
                events.append({"t": round(time.perf_counter() - t0, 4), "text": text})
    rec = {"endpoint": base, "prompt": prompt, "request": {k: v for k, v in body.items() if k != "messages"},
           "events": events, "usage": usage}
    if usage and events:
        decode = events[-1]["t"] - events[0]["t"]
        rec["summary"] = {"first_token_s": events[0]["t"], "total_s": events[-1]["t"],
                          "completion_tokens": usage.get("completion_tokens"),
                          "decode_tok_s": round((usage["completion_tokens"] - 1) / decode, 1) if decode > 0 else None}
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(rec.get("summary")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
