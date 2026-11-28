"""The M5 Max table's test on this laptop: greedy, thinking off, 200 generated tokens, a short question and the
lighthouse-keeper story (Strata docs/MAC.md conditions; story prompt from Strata tools/bench/bench_vs_llama.py).
Five rounds, prompts alternating; decode tok/s from the server's done lines."""
import json, os, re, statistics as st, subprocess, sys, time, urllib.request
RT = os.path.join(os.environ["LOCALAPPDATA"], "RTX-Spark-Benchmark")
TREE = os.path.join(RT, sys.argv[1] if len(sys.argv) > 1 else "basalt-n1x-live-main-29a1b77")
PY = os.path.join(RT, "basalt-python", "Scripts", "python.exe")
S = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ)
cuda = os.path.join(RT, "cuda-13.5", "bin")
env["PATH"] = cuda + ";" + os.path.join(cuda, "arm64") + ";" + env["PATH"]
logp = os.path.join(S, "server.log")
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


prompts = {"question": "What is a hash table, and how does it handle collisions?",
           "story": "Write a short story about a lighthouse keeper."}
res = {k: [] for k in prompts}
try:
    t0 = time.time()
    while time.time() - t0 < 400:
        try:
            if get("/v1/status", timeout=3).get("loaded") is True:
                break
        except Exception:
            pass
        time.sleep(3)
    get("/v1/chat/completions", {"model": "flash-next", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16,
                                 "chat_template_kwargs": {"enable_thinking": False}})
    for rnd in range(5):
        for name in (("question", "story") if rnd % 2 == 0 else ("story", "question")):
            n0 = len(done_lines())
            r = get("/v1/chat/completions", {"model": "flash-next", "messages": [{"role": "user", "content": prompts[name]}],
                                             "max_tokens": 200, "temperature": 0,
                                             "chat_template_kwargs": {"enable_thinking": False}})
            for _ in range(50):
                dl = done_lines()
                if len(dl) > n0:
                    break
                time.sleep(0.1)
            m = re.search(r"tg=(\d+) ([\d.]+)t/s acc=(\d+)% dl=([\d.]+)", dl[-1])
            res[name].append({"tg": int(m[1]), "tps": float(m[2]), "acc": int(m[3]), "dl": float(m[4]),
                              "text": r["choices"][0]["message"].get("content", "")[:120]})
            print(name, dl[-1].strip()[-110:], flush=True)
    for name, v in res.items():
        print(f"{name}: median {st.median(x['tps'] for x in v):.1f} tok/s over {[x['tg'] for x in v]} tokens, "
              f"dl {st.mean(x['dl'] for x in v):.2f}, all texts identical: {len({x['text'] for x in v}) == 1}")
finally:
    json.dump(res, open(os.path.join(S, "results.json"), "w"), indent=1)
    p.terminate()
    time.sleep(10)
    p.kill()
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'basalt-n1x-live-main-29a1b77|basalt-n1x-dev-build' -and $_.Name -match 'basalt|python' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"])
