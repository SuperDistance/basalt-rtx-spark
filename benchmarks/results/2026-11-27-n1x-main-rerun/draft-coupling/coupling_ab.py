"""Drawn (coupled) MTP drafts vs the head's top draft, on one server: a request with a fixed seed takes the head's best
draft and keeps it only where the row's own draw equals it (Radiance's draft-coupling off); without a seed the head's
draft is drawn from its law and kept by min(1, p/q) (coupling on). Alternating, at temperatures 0.7 and 1.0."""
import json, os, re, subprocess, sys, time, urllib.request
RT = os.path.join(os.environ["LOCALAPPDATA"], "RTX-Spark-Benchmark")
TREE = os.path.join(RT, sys.argv[1] if len(sys.argv) > 1 else "basalt-n1x-live-main-29a1b77")
PY = os.path.join(RT, "basalt-python", "Scripts", "python.exe")
S = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ)
cuda = os.path.join(RT, "cuda-13.5", "bin")
env["PATH"] = cuda + ";" + os.path.join(cuda, "arm64") + ";" + env["PATH"]
logp = os.path.join(S, "coupling-server.log")
log = open(logp, "w")
p = subprocess.Popen([os.path.join(TREE, "bin", "basalt.exe"), "serve", os.path.join(RT, "models", "Qwen3.8-FN-UD-IQ3_XXS.basalt"),
                      "--python", PY, "--host", "127.0.0.1", "--port", "8090", "--alias", "flash-next"],
                     cwd=TREE, env=env, stdout=log, stderr=subprocess.STDOUT)


def get(path, body=None, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:8090" + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def done_lines():
    return [l for l in open(logp, errors="replace") if " done " in l and " dl=" in l]


prompts = ["Write a Python function that parses an INI file into nested dicts, with tests.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain how a B-tree insertion splits nodes, with a worked example.",
           "Give a JSON schema for a library catalog with books, authors and loans.",
           "Plan a three-day trip to Kyoto for a family with two kids.",
           "Prove that there are infinitely many primes, then explain the proof to a child.",
           "Refactor this into idiomatic Rust: a loop that sums the even squares of a vector of i64.",
           "Summarize the causes and consequences of the 1929 stock market crash in five paragraphs."]
rows = []
try:
    t0 = time.time()
    while time.time() - t0 < 400:
        try:
            if get("/v1/status", timeout=3).get("loaded") is True:
                break
        except Exception:
            pass
        time.sleep(3)
    print("ready after", round(time.time() - t0), "s", flush=True)
    get("/v1/chat/completions", {"model": "flash-next", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32})
    for rep in range(2):
        for T in (0.7, 1.0):
            for i, pr in enumerate(prompts):
                order = ("drawn", "top") if (i + rep) % 2 == 0 else ("top", "drawn")
                for mode in order:
                    body = {"model": "flash-next", "messages": [{"role": "user", "content": pr}], "max_tokens": 600,
                            "temperature": T, "top_p": 0.95, "top_k": 20}
                    if mode == "top":
                        body["seed"] = 1000 * rep + 10 * i + int(T * 10)
                    n0 = len(done_lines())
                    get("/v1/chat/completions", body)
                    for _ in range(50):
                        dl = done_lines()
                        if len(dl) > n0:
                            break
                        time.sleep(0.1)
                    rows.append({"rep": rep, "T": T, "i": i, "mode": mode, "line": dl[-1].strip()})
                    print(T, i, mode, dl[-1].strip()[-160:], flush=True)
finally:
    json.dump(rows, open(os.path.join(S, "rows.json"), "w"), indent=1)
    p.terminate()
    time.sleep(10)
    p.kill()
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'basalt-n1x-live-main-29a1b77|basalt-n1x-dev-build' -and $_.Name -match 'basalt|python' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"])
    print("stopped", flush=True)
