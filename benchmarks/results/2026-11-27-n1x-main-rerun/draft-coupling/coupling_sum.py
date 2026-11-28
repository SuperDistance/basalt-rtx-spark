"""Summary of coupling_ab.py: tokens per window (token-weighted) and decode tok/s, drawn vs top draft, per temperature."""
import json, os, re, statistics as st
S = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(S, "rows.json")))
agg = {}
for r in rows:
    m = re.search(r"tg=(\d+) ([\d.]+)t/s acc=(\d+)% dl=([\d.]+)", r["line"])
    if not m:
        continue
    tg, tps, acc, dl = int(m[1]), float(m[2]), int(m[3]), float(m[4])
    a = agg.setdefault((r["T"], r["mode"]), {"tg": 0, "win": 0.0, "tps": [], "acc": []})
    a["tg"] += tg; a["win"] += tg / dl; a["tps"].append(tps); a["acc"].append(acc)
for T in sorted({k[0] for k in agg}):
    out = {}
    for mode in ("top", "drawn"):
        a = agg.get((T, mode))
        if a:
            out[mode] = (a["tg"] / a["win"], st.median(a["tps"]), st.mean(a["acc"]), len(a["tps"]))
    line = f"T={T}: " + "  ".join(f"{m}: dl {v[0]:.3f}, median {v[1]:.1f} tok/s, acc {v[2]:.0f}% (n={v[3]})" for m, v in out.items())
    if "top" in out and "drawn" in out:
        line += f"   drawn/top dl {100 * (out['drawn'][0] / out['top'][0] - 1):+.1f}%, tok/s {100 * (out['drawn'][1] / out['top'][1] - 1):+.1f}%"
    print(line)
