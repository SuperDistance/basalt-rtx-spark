#!/usr/bin/env python3
"""Charts and a table from BetterBench result files, for comparing engines or builds on one machine.

    python benchmarks/plot.py --out benchmarks/img --prefix paired A.json "Label A" B.json "Label B" [...]

Writes <prefix>-overview.svg (weighted single-stream decode), <prefix>-decode.svg (single-stream decode by scenario), <prefix>-concurrency.svg (aggregate output at each
concurrency) and <prefix>-prefill.svg (prefill by prompt depth) into --out, and prints a markdown table of the
per-scenario decode medians and the weighted median. Standard library only. The first file is drawn as the subject;
compare only runs made in the same power state (benchmarks/README.md).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from xml.sax.saxutils import escape

COLORS = ("#76B900", "#7A8794", "#3B82C4", "#E0A030", "#9B59B6")
SCENARIOS = ("chat", "code", "file_edit", "json", "math", "prose", "reasoning", "summarization")


def load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    ss = d.get("single_stream", {})
    decode = {c: statistics.median(r["decode_tps"] for r in runs if r.get("ok") and r.get("decode_tps"))
              for c, runs in ss.items() if any(r.get("ok") and r.get("decode_tps") for r in runs)}
    weights = d.get("config", {}).get("weights", {})
    wsum = sum(w for c, w in weights.items() if c in decode)
    weighted = sum(decode[c] * w for c, w in weights.items() if c in decode) / wsum if wsum else None
    conc = {int(x["level"]): float(x["aggregate_tps"]) for x in d.get("concurrency", []) if x.get("aggregate_tps")}
    prefill = {int(x["target_depth"]): statistics.median(x["pp_tps"]) for x in d.get("prefill", []) if x.get("pp_tps")}
    return {"decode": decode, "weighted": weighted, "conc": conc, "prefill": prefill}


def bars(title: str, unit: str, groups: list[str], series: list[tuple[str, list[float | None]]], path: Path) -> None:
    """A grouped bar chart as SVG: one group per x label, one bar per series, the value above each bar."""
    W, H, L, R, T, B = 920, 420, 64, 20, 64, 70
    pw, ph = W - L - R, H - T - B
    top = max((v for _, vals in series for v in vals if v is not None), default=1.0)
    step = 10.0 ** math.floor(math.log10(max(top, 1.0)))   # a round step: 1, 2 or 5 times a power of ten
    step = step / 5 if top / step < 2 else step / 2 if top / step < 5 else step
    ymax = (int(top * 1.1 / step) + 1) * step
    gw = pw / len(groups)
    bw = min(34.0, gw * 0.8 / len(series))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L}" y="28" font-size="18" font-weight="600" fill="#1f2328">{escape(title)}</text>']
    for i in range(6):   # gridlines and y labels
        v = ymax * i / 5
        y = T + ph - ph * i / 5
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#57606a">{v:,.0f}</text>')
    out.append(f'<text x="16" y="{T + ph / 2:.0f}" font-size="12" fill="#57606a" transform="rotate(-90 16 {T + ph / 2:.0f})" '
               f'text-anchor="middle">{escape(unit)}</text>')
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + (gw - bw * len(series)) / 2
        for si, (_, vals) in enumerate(series):
            v = vals[gi]
            if v is None:
                continue
            h = ph * v / ymax
            x = x0 + si * bw
            y = T + ph - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{h:.1f}" rx="2" fill="{COLORS[si % len(COLORS)]}"/>')
            out.append(f'<text x="{x + (bw - 3) / 2:.1f}" y="{y - 4:.1f}" font-size="10" text-anchor="middle" fill="#1f2328">'
                       f'{v:,.0f}</text>')
        out.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 18}" font-size="12" text-anchor="middle" fill="#1f2328">'
                   f'{escape(g)}</text>')
    lx = L
    for si, (name, _) in enumerate(series):   # legend
        out.append(f'<rect x="{lx}" y="{H - 26}" width="12" height="12" rx="2" fill="{COLORS[si % len(COLORS)]}"/>')
        out.append(f'<text x="{lx + 18}" y="{H - 16}" font-size="12" fill="#1f2328">{escape(name)}</text>')
        lx += 30 + 7 * len(name)
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def overview(runs: list[tuple[str, dict]], path: Path) -> None:
    """Label the weighted decode scores directly so the opening chart fits a narrow screen."""
    scores = [(name, run["weighted"]) for name, run in runs if run["weighted"] is not None]
    if not scores:
        return
    row_height = 49
    height = 66 + row_height * len(scores)
    width = 720
    # Reserve room for longer engine labels as well as their scores.
    names = [label.rsplit(" (", 1)[0] for label, _ in scores]
    bar_x = max(210, 28 + 10 * max(map(len, names)))
    width = max(width, bar_x + 350)
    span = width - bar_x - 108
    maximum = max(max(score for _, score in scores), 1)
    desc = ", ".join(f"{name}: {score:.2f} tok/s" for name, score in scores)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'role="img" aria-labelledby="title desc" font-family="Segoe UI, Arial, sans-serif">',
           '<title id="title">Weighted single-stream decode</title>',
           f'<desc id="desc">{escape(desc)}</desc>',
           f'<rect width="{width}" height="{height}" rx="12" fill="#101e26"/>',
           '<text x="22" y="33" font-size="23" font-weight="700" fill="#f0f6f5">Single-stream decode</text>',
           f'<text x="{width - 22}" y="32" text-anchor="end" font-size="16" fill="#b7c9d0">output tokens/s</text>']
    for i, (label, score) in enumerate(scores):
        y = 57 + row_height * i
        name, bracket, pack = label.rpartition(" (")
        if not bracket or not pack.endswith(")"):
            name, pack = label, ""
        else:
            pack = pack[:-1]
        color = "#a6ed60" if i == 0 else "#9eb7c7"
        out.append(f'<text x="22" y="{y + 11}" font-size="17" font-weight="600" fill="#f0f6f5">{escape(name)}</text>')
        if pack:
            out.append(f'<text x="22" y="{y + 28}" font-size="13" fill="#b7c9d0">{escape(pack)}</text>')
        out.append(f'<rect x="{bar_x}" y="{y + 3}" width="{span}" height="14" rx="4" fill="#243742"/>')
        out.append(f'<rect x="{bar_x}" y="{y + 3}" width="{span * score / maximum:.2f}" height="14" rx="4" fill="{color}"/>')
        out.append(f'<text x="{width - 22}" y="{y + 20}" text-anchor="end" font-size="23" font-weight="700" '
                   f'fill="{color}">{score:.2f}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="compare")
    ap.add_argument("pairs", nargs="+", help="result.json label, repeated")
    a = ap.parse_args()
    if len(a.pairs) % 2:
        sys.exit("plot: give each result file a label")
    runs = [(a.pairs[i + 1], load(Path(a.pairs[i]))) for i in range(0, len(a.pairs), 2)]
    a.out.mkdir(parents=True, exist_ok=True)
    overview(runs, a.out / f"{a.prefix}-overview.svg")
    scen = [c for c in SCENARIOS if any(c in r["decode"] for _, r in runs)]
    bars("Single-stream decode by scenario (median)", "tok/s", scen,
         [(n, [r["decode"].get(c) for c in scen]) for n, r in runs], a.out / f"{a.prefix}-decode.svg")
    levels = sorted({lv for _, r in runs for lv in r["conc"]})
    bars("Aggregate output by concurrency", "tok/s", [f"{lv} stream{'s' if lv > 1 else ''}" for lv in levels],
         [(n, [r["conc"].get(lv) for lv in levels]) for n, r in runs], a.out / f"{a.prefix}-concurrency.svg")
    depths = sorted({dp for _, r in runs for dp in r["prefill"]})
    bars("Prefill by prompt depth (median)", "tok/s", [f"{dp // 1000}K" for dp in depths],
         [(n, [r["prefill"].get(dp) for dp in depths]) for n, r in runs], a.out / f"{a.prefix}-prefill.svg")
    print("| Scenario | " + " | ".join(n for n, _ in runs) + " |")
    print("| --- |" + " ---: |" * len(runs))
    for c in scen:
        print(f"| {c} | " + " | ".join(f"{r['decode'][c]:.1f}" if c in r["decode"] else "" for _, r in runs) + " |")
    print("| **weighted** | " + " | ".join(f"**{r['weighted']:.1f}**" if r["weighted"] else "" for _, r in runs) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
