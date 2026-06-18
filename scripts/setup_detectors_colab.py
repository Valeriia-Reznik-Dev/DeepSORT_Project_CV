#!/usr/bin/env python3
"""Install YOLO + NanoDet + MMDet for Colab (with import checks)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANO_DIR = ROOT / "third_party" / "nanodet"
NANO_TAG = "v1.0.0-alpha-1"  # matches nanodet-plus-m_416 checkpoint release


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def install_yolo() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "scikit-learn"])


def install_nanodet() -> None:
    if not NANO_DIR.is_dir():
        NANO_DIR.parent.mkdir(parents=True, exist_ok=True)
        run([
            "git",
            "clone",
            "-b",
            NANO_TAG,
            "--depth",
            "1",
            "https://github.com/RangiLyu/nanodet.git",
            str(NANO_DIR),
        ])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(NANO_DIR / "requirements.txt")])
    # pip install git+... fails: setup.py imports nanodet before install
    run([sys.executable, "setup.py", "develop"], cwd=NANO_DIR)
    import nanodet  # noqa: F401

    print(f"nanodet OK ({NANO_TAG})")


def install_mmdet() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "setuptools==75.8.2",
        "openmim",
        "mmengine",
        "mmdet",
    ])
    import torch

    torch_v = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    cu = torch.version.cuda or ""
    cu_tag = f"cu{cu.replace('.', '')[:3]}" if cu else "cpu"
    url = f"https://download.openmmlab.com/mmcv/dist/{cu_tag}/torch{torch_v}/index.html"
    run([sys.executable, "-m", "pip", "install", "-q", "mmcv==2.2.0", "-f", url])
    import mmdet  # noqa: F401

    print(f"mmdet OK (mmcv wheel: {cu_tag}/torch{torch_v})")


def main() -> None:
    install_yolo()
    install_nanodet()
    install_mmdet()
    print("Detector deps ready.")


if __name__ == "__main__":
    main()
