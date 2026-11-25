#!/usr/bin/env python3
"""Charts and a table from BetterBench result files, for comparing engines or builds on one machine.

    python benchmarks/plot.py --out benchmarks/img --prefix paired A.json "Label A" B.json "Label B" [...]

Writes <prefix>-overview.svg (weighted decode and median cold prefill at the 2K target), <prefix>-decode.svg (single-stream decode by scenario), <prefix>-concurrency.svg (aggregate output at each
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

COLORS = ("#76B900", "#008675", "#7A8794", "#3B82C4", "#9B59B6")
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
    legend = []
    lx, row = L, 0
    for name, _ in series:
        label_width = 30 + 7 * len(name)
        if lx > L and lx + label_width > W - R:
            lx, row = L, row + 1
        legend.append((lx, row))
        lx += label_width
    legend_top = H - 26
    H += 22 * row
    B += 22 * row
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
    for si, (name, _) in enumerate(series):
        lx, row = legend[si]
        ly = legend_top + 22 * row
        out.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{COLORS[si % len(COLORS)]}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly + 10}" font-size="12" fill="#1f2328">{escape(name)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def overview(runs: list[tuple[str, dict]], path: Path) -> None:
    """Compare output generation and cold prompt reading with a separate scale for each rate."""
    rows = [(label, run) for label, run in runs if run["weighted"] is not None]
    if not rows:
        return
    metrics = [("Decode · 1 stream", "weighted output tok/s", [run["weighted"] for _, run in rows], ".2f")]
    prefill = [run["prefill"].get(2000) for _, run in rows]
    if any(rate is not None for rate in prefill):
        metrics.append(("Cold prefill", "input tok/s · 2K target", prefill, ",.0f"))

    title = "BetterBench · decode and cold prefill" if len(metrics) == 2 else "BetterBench · single-stream decode"
    names = [label.rsplit(" (", 1)[0] if label.endswith(")") else label for label, _ in rows]
    packs = [label[len(name):].strip(" ()") for (label, _), name in zip(rows, names)]
    # In comparisons, the first engine may have several measured weight packs.
    peer_start = next((i for i, name in enumerate(names) if name != names[0]), len(rows))
    grouped = peer_start > 1 and all(packs[:peer_start])
    label_width = max(224, 36 + 8 * max(map(len, names)))
    col_width, row_height = 166, 47
    width = label_width + col_width * len(metrics) + 12
    row_y = [(122 if grouped else 100) + row_height * i + (26 if grouped and i >= peer_start else 0)
             for i in range(len(rows))]
    height = row_y[-1] + 36
    desc = ". ".join(f"{heading}: " + ", ".join(
        f"{label} {rate:{fmt}} tok/s" for (label, _), rate in zip(rows, rates) if rate is not None
    ) for heading, _, rates, fmt in metrics)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           'role="img" aria-labelledby="title desc" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<title id="title">{title}</title>', f'<desc id="desc">{escape(desc)}</desc>',
           f'<rect width="{width}" height="{height}" rx="12" fill="#101e26"/>',
           f'<text x="22" y="32" font-size="23" font-weight="700" fill="#f0f6f5">{title}</text>']
    if grouped:
        bottom = row_y[peer_start - 1] + 25
        out.extend([
            f'<rect x="10" y="84" width="{width - 20}" height="{bottom - 84}" rx="7" fill="#1b382f"/>',
            f'<rect x="10" y="84" width="4" height="{bottom - 84}" rx="2" fill="#9bd568"/>',
            f'<text x="23" y="105" font-size="14" font-weight="700" fill="#b9ee92">THIS PROJECT · {escape(names[0].upper())}</text>',
        ])
        if peer_start < len(rows):
            out.append(f'<text x="23" y="{row_y[peer_start] - 18}" font-size="14" font-weight="700" fill="#bfccd1">REFERENCE ENGINES</text>')
    for col, (heading, unit, rates, fmt) in enumerate(metrics):
        x = label_width + col * col_width
        out.append(f'<text x="{x}" y="59" font-size="16" font-weight="600" fill="#f0f6f5">{escape(heading)}</text>')
        out.append(f'<text x="{x}" y="76" font-size="12" fill="#b7c9d0">{escape(unit)}</text>')
        maximum = max((rate for rate in rates if rate is not None), default=1) or 1
        for i, rate in enumerate(rates):
            if rate is None:
                continue
            y = row_y[i]
            color = "#a6ed60" if i == 0 else "#7fdbca" if names[i] == names[0] else "#a8c1d0"
            span = col_width - 19
            out.append(f'<text x="{x}" y="{y + 8}" font-size="21" font-weight="700" fill="{color}">{rate:{fmt}}</text>')
            out.append(f'<rect x="{x}" y="{y + 14}" width="{span}" height="6" rx="3" fill="#293b46"/>')
            out.append(f'<rect x="{x}" y="{y + 14}" width="{span * rate / maximum:.2f}" height="6" rx="3" fill="{color}"/>')
    for i, (name, pack) in enumerate(zip(names, packs)):
        y = row_y[i]
        if grouped and i < peer_start:
            out.append(f'<text x="28" y="{y + 9}" font-size="18" font-weight="600" fill="#f0f6f5">{escape(pack)}</text>')
        else:
            out.append(f'<text x="22" y="{y + 6}" font-size="16" font-weight="600" fill="#f0f6f5">{escape(name)}</text>')
            if pack:
                out.append(f'<text x="22" y="{y + 23}" font-size="13" fill="#b7c9d0">{escape(pack)}</text>')
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
    bars("Cold prefill by target prompt depth (median)", "input tok/s", [f"{dp // 1000}K" for dp in depths],
         [(n, [r["prefill"].get(dp) for dp in depths]) for n, r in runs], a.out / f"{a.prefix}-prefill.svg")
    print("| Scenario | " + " | ".join(n for n, _ in runs) + " |")
    print("| --- |" + " ---: |" * len(runs))
    for c in scen:
        print(f"| {c} | " + " | ".join(f"{r['decode'][c]:.1f}" if c in r["decode"] else "" for _, r in runs) + " |")
    print("| **weighted** | " + " | ".join(f"**{r['weighted']:.1f}**" if r["weighted"] else "" for _, r in runs) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
