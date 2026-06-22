#!/usr/bin/env python3
"""Download fast-reid weights."""
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


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"OK (exists): {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ReID model weights")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "resources" / "models" / "fastreid"

    for filename, url in FASTREID_WEIGHTS.items():
        _download(url, out / filename)

    print(
        "Done. torchreid (osnet) downloads ReID weights on first run; "
        "resnet50_ibn uses fast-reid market_bot_R50-ibn.pth."
    )


if __name__ == "__main__":
    main()
