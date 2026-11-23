#!/usr/bin/env python3
r"""Packs a built tree into the release archive: basalt-<tag>-<platform>.tar.gz (linux-*) or .zip (windows-*).

Run from a tree that has been built (the executables are in bin/). The archive holds one top-level directory with the
executables, the server and the Python files it imports, the expert profiles and bench fixtures, the requirements,
the licences and the docs. No Python environment is packed, and no CUDA libraries unless `--cuda` names a toolkit:
then the cuBLAS DLLs the engine loads go into bin/ beside it (Windows looks there first) with NVIDIA's licence in
licenses/. windows-arm64 also gets setup.cmd, start.cmd and a basalt_config.json from tools/release/windows-arm64/.

    python tools/package_release.py --tag v0.3.1 --platform linux-x64 [--root DIR] [--out DIR]
    python tools/package_release.py --tag v0.4.1-rtx-spark.1 --platform windows-arm64 --cuda C:\cuda-13.5

Prints the archive's path.
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

EXECUTABLES = ("basalt", "basalt-bench", "basalt-probe", "basalt-vision")
TOOLS = ("tools/tokenizer.py", "tools/model_file.py", "tools/migrations/config_v1_to_v2.py")
FILES = ("requirements.txt", "LICENSE", "THIRD_PARTY_LICENSES", "README.md")
TREES = ("serve", "data/profiles", "data/bench", "docs")
SKIP_DIRS = {"__pycache__", "tests", "img", "community", "assets"}   # docs/img, docs/community, docs/assets are web-only
SKIP_SUFFIXES = (".pyc",)
CUDA_DLLS = ("cublas64_13.dll", "cublasLt64_13.dll")   # what basalt.exe loads (cudart is static); in the toolkit's bin\arm64 or bin
ARM64_EXTRAS = ("setup.cmd", "start.cmd", "basalt_config.json")   # tools/release/windows-arm64/, at the top directory


def cuda_members(cuda: Path) -> list[tuple[Path, str, bool]]:
    out: list[tuple[Path, str, bool]] = []
    for name in CUDA_DLLS:
        src = next((p for p in (cuda / "bin" / "arm64" / name, cuda / "bin" / name) if p.is_file()), None)
        if src is None:
            sys.exit(f"package_release: {name} is not in {cuda}\\bin\\arm64 or {cuda}\\bin")
        out.append((src, f"bin/{name}", False))
    if not (cuda / "LICENSE").is_file():
        sys.exit(f"package_release: {cuda / 'LICENSE'} (NVIDIA's licence for the redistributed DLLs) is missing")
    out.append((cuda / "LICENSE", "licenses/NVIDIA-CUDA-EULA.txt", False))
    return out


def members(root: Path, windows: bool) -> list[tuple[Path, str, bool]]:
    """(source, archive path inside the top directory, executable) for every file, in a fixed order."""
    out: list[tuple[Path, str, bool]] = []
    for name in EXECUTABLES:
        rel = f"bin/{name}{'.exe' if windows else ''}"
        if not (root / rel).is_file():
            sys.exit(f"package_release: {root / rel} is missing; build first")
        out.append((root / rel, rel, True))
    for rel in FILES + TOOLS:
        if not (root / rel).is_file():
            sys.exit(f"package_release: {root / rel} is missing")
        out.append((root / rel, rel, False))
    for tree in TREES:
        base = root / tree
        if not base.is_dir():
            sys.exit(f"package_release: {base} is missing")
        for p in sorted(base.rglob("*")):
            rel_parts = p.relative_to(root).parts
            if not p.is_file() or p.suffix in SKIP_SUFFIXES:
                continue
            # the skip list applies below the tree's own directory, not to the tree itself
            if any(part in SKIP_DIRS for part in p.relative_to(base).parts[:-1]):
                continue
            out.append((p, "/".join(rel_parts), False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, help="the release tag, v<major>.<minor>.<patch>")
    ap.add_argument("--platform", required=True, choices=("linux-x64", "windows-x64", "windows-arm64"))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="the built tree")
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--cuda", type=Path, help="a CUDA 13 toolkit whose cuBLAS DLLs go into bin/ (Windows only)")
    a = ap.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?", a.tag):   # a semver pre-release may hold hyphens
        sys.exit(f"package_release: tag {a.tag!r} is not of the form v0.3.1")
    cmake_text = (a.root / "CMakeLists.txt").read_text(encoding="utf-8")
    version = re.search(r"project\(basalt VERSION (\S+)", cmake_text)
    if not version or a.tag[1:].split("-")[0] != version.group(1):
        sys.exit(f"package_release: tag {a.tag} does not match the CMakeLists version {version and version.group(1)}")
    windows = a.platform.startswith("windows")
    top = f"basalt-{a.tag}-{a.platform}"
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / (top + (".zip" if windows else ".tar.gz"))
    files = members(a.root.resolve(), windows)
    if a.cuda is not None:
        if not windows:
            sys.exit("package_release: --cuda packs Windows DLLs; a Linux archive names the toolkit's libraries instead")
        files += cuda_members(a.cuda.resolve())
    if a.platform == "windows-arm64":
        extras = a.root.resolve() / "tools" / "release" / "windows-arm64"
        for name in ARM64_EXTRAS:
            if not (extras / name).is_file():
                sys.exit(f"package_release: {extras / name} is missing")
            files.append((extras / name, name, False))
    if windows:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for src, rel, _ in files:
                z.write(src, f"{top}/{rel}")
    else:
        with tarfile.open(path, "w:gz") as t:
            for src, rel, exe in files:
                info = t.gettarinfo(str(src), f"{top}/{rel}")
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mode = 0o755 if exe else 0o644
                with open(src, "rb") as f:
                    t.addfile(info, f)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
