#!/usr/bin/env python3
"""Download NanoDet + MMDet weights for detector eval (Colab / local)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NANODET_CONFIG_URL = (
    "https://raw.githubusercontent.com/RangiLyu/nanodet/main/config/nanodet-plus-m_416.yml"
)
NANODET_CKPT_URL = (
    "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/"
    "nanodet-plus-m_416_checkpoint.ckpt"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"OK (exists): {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_nanodet(root: Path) -> None:
    out = root / "resources" / "models" / "nanodet"
    _download(NANODET_CONFIG_URL, out / "nanodet-plus-m_416.yml")
    _download(NANODET_CKPT_URL, out / "nanodet-plus-m_416_checkpoint.ckpt")


def download_mmdet(root: Path) -> None:
    out = root / "resources" / "models" / "mmdet"
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / "rtmdet_tiny_8xb32-300e_coco.py"
    ckpt = out / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    if cfg.is_file() and ckpt.is_file():
        print(f"OK (exists): {cfg.name}, {ckpt.name}")
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "setuptools==75.8.2", "openmim"],
        check=True,
    )
    subprocess.run(
        [
            "mim",
            "download",
            "mmdet",
            "--config",
            "rtmdet_tiny_8xb32-300e_coco",
            "--dest",
            str(out),
        ],
        check=True,
        cwd=root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download detector model files")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--skip-mmdet",
        action="store_true",
        help="Skip MMDet download (mim); NanoDet only",
    )
    args = parser.parse_args()
    root = Path(args.root)

    download_nanodet(root)
    if not args.skip_mmdet:
        download_mmdet(root)

    print("Done. Weights under resources/models/")


if __name__ == "__main__":
    main()
