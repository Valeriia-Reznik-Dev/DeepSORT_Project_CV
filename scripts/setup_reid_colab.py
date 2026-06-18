#!/usr/bin/env python3
"""Install ReID deps for Colab: torchreid + fast-reid + sklearn."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTREID_DIR = ROOT / "third_party" / "fast_reid"
FASTREID_TAG = "v1.3.0"
PIP_TIMEOUT_S = 300


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def verify_import(module: str) -> None:
    run([sys.executable, "-c", f"import {module}"])


def clone_fastreid() -> None:
    if FASTREID_DIR.is_dir():
        print(f"OK (exists): {FASTREID_DIR}")
        return
    run([
        "git", "clone", "--depth", "1", "--branch", FASTREID_TAG,
        "https://github.com/JDAI-CV/fast-reid.git",
        str(FASTREID_DIR),
    ])


def main() -> None:
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--timeout", str(PIP_TIMEOUT_S),
        "scikit-learn",
        "tabulate",
        "termcolor",
        "yacs",
    ])

    # torchreid from official repo (pip package name: torchreid)
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--timeout", str(PIP_TIMEOUT_S),
        "git+https://github.com/KaiyangZhou/deep-person-reid.git",
    ])
    verify_import("torchreid")

    clone_fastreid()
    run([
        sys.executable, "-m", "pip", "install", "-q", "-e",
        str(FASTREID_DIR),
    ])
    verify_import("fastreid")

    print("\nReID setup OK: torchreid + fastreid")


if __name__ == "__main__":
    main()
