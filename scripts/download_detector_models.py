#!/usr/bin/env python3
"""Download NanoDet + MMDet weights for detector eval (Colab / local)."""
from __future__ import annotations

import argparse
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
MMDET_CONFIG_TAG = "v3.3.0"
MMDET_CONFIG_BASE = (
    f"https://raw.githubusercontent.com/open-mmlab/mmdetection/{MMDET_CONFIG_TAG}/configs"
)
MMDET_CONFIG_FILES = [
    "rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
    "rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "rtmdet/rtmdet_l_8xb32-300e_coco.py",
    "rtmdet/rtmdet_tta.py",
    "_base_/default_runtime.py",
    "_base_/schedules/schedule_1x.py",
    "_base_/datasets/coco_detection.py",
]
MMDET_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/"
    "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
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
    for rel in MMDET_CONFIG_FILES:
        _download(f"{MMDET_CONFIG_BASE}/{rel}", out / rel)
    _download(
        MMDET_CKPT_URL,
        out / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
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
        help="Skip MMDet weight download; NanoDet only",
    )
    args = parser.parse_args()
    root = Path(args.root)

    download_nanodet(root)
    if not args.skip_mmdet:
        download_mmdet(root)

    print("Done. Weights under resources/models/")


if __name__ == "__main__":
    main()
