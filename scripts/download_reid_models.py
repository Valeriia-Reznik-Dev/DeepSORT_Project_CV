#!/usr/bin/env python3
"""Download ReID model weights (fast-reid + torchreid)."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FASTREID_WEIGHTS = {
    "market_bot_R50.pth": (
        "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth"
    ),
    "market_bot_R50-ibn.pth": (
        "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth"
    ),
}

# torchreid model zoo — ResNet50 trained on Market1501.
TORCHREID_WEIGHTS = {
    "market1501_resnet50.pth": "1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV",
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"OK (exists): {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _download_gdrive(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"OK (exists): {dest}")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading Google Drive {file_id} -> {dest}")
    try:
        import gdown

        gdown.download(url, str(dest), quiet=False)
    except ImportError:
        _download_url(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ReID model weights")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    root = Path(args.root)

    fastreid_out = root / "resources" / "models" / "fastreid"
    for filename, url in FASTREID_WEIGHTS.items():
        _download_url(url, fastreid_out / filename)

    torchreid_out = root / "resources" / "models" / "torchreid"
    for filename, file_id in TORCHREID_WEIGHTS.items():
        _download_gdrive(file_id, torchreid_out / filename)

    print(
        "Done. osnet (torchreid) downloads weights on first run; "
        "resnet50 uses resources/models/torchreid/market1501_resnet50.pth; "
        "resnet50_ibn/fastreid use fast-reid checkpoints."
    )


if __name__ == "__main__":
    main()
