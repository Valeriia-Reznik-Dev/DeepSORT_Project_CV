#!/usr/bin/env python3
"""Install ReID deps for Colab: torchreid + fast-reid + sklearn."""
from __future__ import annotations

import shutil
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


def _fastreid_ready() -> bool:
    return (FASTREID_DIR / "fastreid" / "__init__.py").is_file()


def clone_fastreid() -> None:
    if _fastreid_ready():
        print(f"OK (exists): {FASTREID_DIR}")
        return
    if FASTREID_DIR.is_dir():
        print(f"Removing incomplete fast-reid clone: {FASTREID_DIR}")
        shutil.rmtree(FASTREID_DIR)
    run([
        "git", "clone", "--depth", "1", "--branch", FASTREID_TAG,
        "https://github.com/JDAI-CV/fast-reid.git",
        str(FASTREID_DIR),
    ])
    if not _fastreid_ready():
        raise RuntimeError(f"fast-reid clone incomplete: {FASTREID_DIR}")


def verify_fastreid() -> None:
    cmd = (
        "import sys; "
        f"sys.path.insert(0, {str(FASTREID_DIR)!r}); "
        "import fastreid; "
        "print('fastreid OK')"
    )
    run([sys.executable, "-c", cmd])


def main() -> None:
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--timeout", str(PIP_TIMEOUT_S),
        "scikit-learn",
        "tabulate",
        "termcolor",
        "yacs",
        "prettytable",
        "easydict",
    ])

    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--timeout", str(PIP_TIMEOUT_S),
        "git+https://github.com/KaiyangZhou/deep-person-reid.git",
    ])
    verify_import("torchreid")

    clone_fastreid()
    verify_fastreid()

    print("\nReID setup OK: torchreid + fastreid (sys.path, no pip install)")


if __name__ == "__main__":
    main()
