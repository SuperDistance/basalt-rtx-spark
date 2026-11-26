"""Do two engine builds produce the same greedy tokens? Does a request's output depend on batching or its slot?

    python3 tools/greedy_identity.py REF_BIN NEW_BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]
    python3 tools/greedy_identity.py --slots BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]
    python3 tools/greedy_identity.py --slots-dirty BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]

Each binary runs as `--serve` with BASE plus any flags after `--` (which replace BASE values). The bench fixtures'
prompts (conv, struct, prose, code: comma-separated token ids per line) run at temperature 0 and every token is compared.
Adaptive swaps and the PCIe share are off so placement cannot drift; pin the main card's tier too (`-- --expert-cache
N`) when the builds' VRAM use differs, since auto-sized tiers change rounding. A second card's tier holds what fits:
check that its expert count ("peer GPU 1: N experts" in both logs) matches.

--slots: one build, one engine with batch slots. Each prompt runs alone as a solo request, then alone in every slot, then
all prompts together with the prompt-to-slot order rotated through every slot; each run is compared token by token with
the solo one. There are as many prompts as slots, so every rotation runs all four slots at once; drafting is off by
default (`-- --spec 8` checks the production drafting window). A slot reads its prompt through the batched prompt
path, so BASE_SLOTS sets `--short-read 0` and the solo request reads it that way too (a short read through the decode windows is a different computation, see
docs/precision/qwen38-flash-next.md#prefill).

--slots-dirty: the same runs as --slots in the production setting: drafting on (`--spec 8`), the prompt and
conversation caches on, so later runs reuse what earlier ones left (a request restored from a cache, or decoded in a
slot another request used, must still match its cold solo run). Every "together" run must have all four slots
decoding at once: a slot that finished before the last one was admitted fails the run.

Prints "GREEDY IDENTICAL" (exit 0) or "GREEDY DIFFERS" with the first difference (exit 1); exit 2 when an engine fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPTS = ("conv", "struct", "prose", "code")
# flags after `--` replace these
BASE = ["--spec", "7", "--max-context", "8192"]
PINNED = ["--adapt-swaps", "0", "--peer-adapt-swaps", "0"]
# --slots: drafting and the prompt and conversation caches off, so every run starts from the prompt alone
SLOTS = 4
BASE_SLOTS = ["--batch", str(SLOTS), "--spec", "0", "--max-context", "8192", "--prompt-cache", "0",
              "--conversation-cache-mib", "0", "--short-read", "0"]
# --slots-dirty: drafting and both caches on, as served
BASE_SLOTS_DIRTY = ["--batch", str(SLOTS), "--spec", "8", "--max-context", "8192", "--short-read", "0"]


def set_args(args: list[str], extra: list[str]) -> list[str]:
    """`args` with each flag of `extra` replacing the same flag (and its value) in `args`, or added."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        value = [extra[i + 1]] if has_value else []
        if flag in out:
            j = out.index(flag)
            old_has_value = j + 1 < len(out) and not out[j + 1].startswith("--")
            out[j:j + 1 + old_has_value] = [flag, *value]
        else:
            out += [flag, *value]
        i += 1 + has_value
    return out


def run(label: str, engine_bin: str, args: list[str], prompts: list[str], tokens: int, logs: Path) -> list[list[int]] | None:
    """One engine session over the prompts: the generated ids of each, or None when the engine failed."""
    err = open(logs / f"{label}.err", "w")
    proc = subprocess.Popen([engine_bin, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                            text=True, bufsize=1)
    out: list[list[int]] = []
    try:
        for line in proc.stdout:
            if line.startswith("READY"):
                break
        else:
            return None
        for ids in prompts:
            proc.stdin.write(f"GEN {tokens} temperature=0 {ids}\n")
            proc.stdin.flush()
            got: list[int] = []
            for line in proc.stdout:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE"):
                    break
                elif line.startswith("ERR"):
                    print(f"{label}: {line.strip()}", file=sys.stderr)
                    return None
            else:
                return None
            out.append(got)
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=120)
        return out
    except (BrokenPipeError, subprocess.TimeoutExpired):
        return None
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        err.close()


class Engine:
    """One `--serve` engine: solo requests (GEN) and batch-slot requests (BGEN), read line by line."""

    def __init__(self, label: str, engine_bin: str, args: list[str], logs: Path):
        self.err = open(logs / f"{label}.err", "w")
        self.proc = subprocess.Popen([engine_bin, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, text=True, bufsize=1)

    def line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise EOFError("the engine ended")
        if line.startswith("ERR"):
            raise EOFError(line.strip())
        return line

    def ready(self) -> None:
        while not self.line().startswith("READY"):
            pass

    def send(self, text: str) -> None:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def gen(self, ids: str, tokens: int) -> list[int]:
        self.send(f"GEN {tokens} temperature=0 {ids}")
        got: list[int] = []
        while not (line := self.line()).startswith("DONE"):
            if line.startswith("T "):
                got.append(int(line.split()[1]))
        return got

    def batch(self, placement: dict[int, str], tokens: int) -> dict[int, list[int]]:
        """The prompts of `placement` ({slot: ids}) admitted in that order and decoded together until each is done."""
        got: dict[int, list[int]] = {b: [] for b in placement}
        done: set[int] = set()

        def take(line: str, admitting: int) -> None:
            f = line.split()
            if f[0] == "T":   # the admission's first token, when the slot's read samples it
                got[admitting].append(int(f[1]))
            elif f[0] == "BT":
                got[int(f[1])] += [int(x) for x in f[2:]]
            elif f[0] == "BDONE":
                done.add(int(f[1]))

        for b, ids in placement.items():
            self.send(f"BGEN {b} {tokens} temperature=0 {ids}")
            while not (line := self.line()).startswith("BADM"):
                take(line, b)
            if line.split()[2] == "0":   # finished in its read
                done.add(b)
        self.overlapped = not done   # every slot still decoding when the last one was admitted
        while done != set(placement):
            take(self.line(), -1)
        return got

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("QUIT")
                self.proc.wait(timeout=120)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        self.err.close()


def first_diff(ref: list[int], new: list[int]) -> int | None:
    if ref == new:
        return None
    return next((k for k, (x, y) in enumerate(zip(ref, new)) if x != y), min(len(ref), len(new)))


def slots_check(engine_bin: str, args: list[str], names: tuple[str, ...], prompts: list[str], tokens: int,
                logs: Path) -> int:
    """--slots: every prompt solo, alone in each slot, and all together under each rotation of the slot order."""
    if len(prompts) != SLOTS:
        print(f"greedy_identity: --slots needs {SLOTS} prompts to fill every slot, got {len(prompts)}", file=sys.stderr)
        return 2
    e = Engine("slots", engine_bin, args, logs)
    runs: list[tuple[str, int, list[int]]] = []   # (run, prompt index, ids)
    not_together = 0
    try:
        e.ready()
        solo = [e.gen(p, tokens) for p in prompts]
        for b in range(SLOTS):
            for i, p in enumerate(prompts):
                runs.append((f"alone in slot {b}", i, e.batch({b: p}, tokens)[b]))
        for r in range(SLOTS):
            place = {(i + r) % SLOTS: i for i in range(len(prompts))}
            got = e.batch({b: prompts[i] for b, i in place.items()}, tokens)
            order = " ".join(f"{names[i]}@{b}" for b, i in place.items())
            if not e.overlapped:
                not_together += 1
                print(f"NOT TOGETHER: a slot finished before the last admission ({order})")
            for b, i in place.items():
                runs.append((f"together ({order})", i, got[b]))
    except (EOFError, BrokenPipeError, OSError) as x:
        tail = (logs / "slots.err").read_text().strip().splitlines()[-5:]
        print(f"greedy_identity: the engine failed: {x} (logs: {logs})", *tail, sep="\n  ", file=sys.stderr)
        return 2
    finally:
        e.close()
    print(f"solo: {sum(map(len, solo))} tokens; {len(runs)} batch runs; {SLOTS - not_together} of {SLOTS} rotations "
          f"had all {SLOTS} slots decoding at once")
    bad = 0
    for run_name, i, ids in runs:
        k = first_diff(solo[i], ids)
        if k is not None:
            bad += 1
            print(f"GREEDY DIFFERS: {names[i]} {run_name}, token {k}: solo {solo[i][k:k + 5]} batch {ids[k:k + 5]}")
    if bad:
        print(f"{bad} of {len(runs)} batch runs differ from solo (logs: {logs})")
        return 1
    if not_together:
        print(f"greedy_identity: {not_together} rotations did not keep all {SLOTS} slots busy; raise --tokens")
        return 1
    print("GREEDY IDENTICAL")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--slots", metavar="BIN", help="check one build's batch slots against its solo requests")
    ap.add_argument("--slots-dirty", metavar="BIN", help="--slots with drafting and both caches on")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parent / "data" / "bench"))
    ap.add_argument("--tokens", type=int, default=300)
    a = ap.parse_args(argv)
    slots_bin = a.slots or a.slots_dirty
    if a.slots and a.slots_dirty:
        ap.error("give --slots or --slots-dirty, not both")
    if not ((slots_bin is not None and a.ref is None) or (slots_bin is None and a.ref is not None and a.new is not None)):
        ap.error("give REF_BIN NEW_BIN, or --slots BIN, or --slots-dirty BIN")

    prompts = []
    for name in PROMPTS:
        path = Path(a.fixtures) / f"{name}.ids"
        if not path.is_file():
            print(f"greedy_identity: {path} is missing", file=sys.stderr)
            return 2
        prompts.append(path.read_text().strip())
    logs = Path(tempfile.mkdtemp(prefix="greedy-identity-"))
    if slots_bin is not None:
        base = BASE_SLOTS_DIRTY if a.slots_dirty else BASE_SLOTS
        return slots_check(slots_bin, set_args(["--model", a.model, *base], PINNED + extra), PROMPTS, prompts,
                           a.tokens, logs)
    args = set_args(["--model", a.model, *BASE], PINNED + extra)
    results = {}
    for label, engine_bin in (("ref", a.ref), ("new", a.new)):
        r = run(label, engine_bin, args, prompts, a.tokens, logs)
        if r is None:
            tail = (logs / f"{label}.err").read_text().strip().splitlines()[-5:]
            print(f"greedy_identity: the {label} engine failed (logs: {logs})", *tail, sep="\n  ", file=sys.stderr)
            return 2
        print(f"{label}: {sum(map(len, r))} tokens, {len(r)} requests")
        results[label] = r

    for name, ref, new in zip(PROMPTS, results["ref"], results["new"]):
        if ref != new:
            i = next((k for k, (x, y) in enumerate(zip(ref, new)) if x != y), min(len(ref), len(new)))
            print(f"GREEDY DIFFERS: {name}, token {i}: ref {ref[i:i + 5]} new {new[i:i + 5]} (logs: {logs})")
            return 1
    print("GREEDY IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
