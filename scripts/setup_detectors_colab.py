#!/usr/bin/env python3
"""Install YOLO + NanoDet + MMDet for Colab (with import checks)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANO_DIR = ROOT / "third_party" / "nanodet"
NANO_TAG = "v1.0.0-alpha-1"  # matches nanodet-plus-m_416 checkpoint release
NANO_COLLATE = NANO_DIR / "nanodet" / "data" / "collate.py"
TORCH_SIX_PATCH = "string_classes = (str, bytes)"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def verify_import(module: str) -> None:
    """Import in a fresh subprocess (editable installs are not always visible in-process)."""
    run([sys.executable, "-c", f"import {module}"])


def patch_nanodet_for_pytorch2() -> None:
    """NanoDet v1.0.0-alpha-1 uses torch._six, removed in PyTorch 2.0+."""
    if not NANO_COLLATE.is_file():
        raise FileNotFoundError(f"Missing NanoDet collate module: {NANO_COLLATE}")
    text = NANO_COLLATE.read_text()
    if "from torch._six import string_classes" in text:
        text = text.replace(
            "from torch._six import string_classes",
            TORCH_SIX_PATCH,
        )
        NANO_COLLATE.write_text(text)
        print("Patched nanodet/data/collate.py for PyTorch 2.x")


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
    patch_nanodet_for_pytorch2()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(NANO_DIR / "requirements.txt")])
    # setup.py imports nanodet before install; run editable install from repo root
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(NANO_DIR)])
    verify_import("nanodet")

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
    verify_import("mmdet")

    print(f"mmdet OK (mmcv wheel: {cu_tag}/torch{torch_v})")


def main() -> None:
    install_yolo()
    install_nanodet()
    install_mmdet()
    print("Detector deps ready.")


if __name__ == "__main__":
    main()
