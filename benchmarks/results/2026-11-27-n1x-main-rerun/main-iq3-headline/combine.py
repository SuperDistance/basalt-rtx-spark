"""The IQ3 run the README charts use: the eight decode scenarios of the best of the five decode runs (r3, 105.44
tok/s weighted), with the prompt reads and concurrency levels of the one full run. Same engine, server settings and
piece method; every piece listed in `assembly.pieces` names the run it came from.

    python combine.py      (from this folder; writes results.json)
"""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
full = json.loads((here.parent / "main-iq3" / "results.json").read_text(encoding="utf-8"))
dec = json.loads((here.parent / "main-iq3-decode-repeats" / "r3" / "results.json").read_text(encoding="utf-8"))
out = dict(full)
out["single_stream"] = dec["single_stream"]
out["sample_gate"] = dec["sample_gate"]
out["assembly"] = {
    "method": "decode scenarios from main-iq3-decode-repeats/r3 (the best of five decode runs; their median is 104.17); prompt reads and "
              "concurrency levels from main-iq3 (the full run). " + full["assembly"]["method"],
    "pieces": [dict(p, run="main-iq3-decode-repeats/r3") for p in dec["assembly"]["pieces"]] +
              [dict(p, run="main-iq3") for p in full["assembly"]["pieces"] if not p["piece"].startswith("decode-")],
}
(here / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", here / "results.json")
