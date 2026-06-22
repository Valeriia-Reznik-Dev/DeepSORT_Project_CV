#!/usr/bin/env python3
"""Download ReID model weights (fast-reid + torchreid)."""
from __future__ import annotations

import argparse
import subprocess
import sys
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


def _ensure_gdown():
    try:
        import gdown
    except ImportError:
        print("Installing gdown ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=True,
        )
        import gdown
    return gdown


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"OK (exists): {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _looks_like_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    head = path.read_bytes()[:256].lstrip()
    return not (
        head.startswith(b"<!DOCTYPE")
        or head.startswith(b"<!doctype")
        or head.startswith(b"<html")
    )


def _validate_torchreid_checkpoint(path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from reid.torchreid_ext import _load_torchreid_checkpoint

    _load_torchreid_checkpoint(str(path))


def _download_gdrive(file_id: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        if not force and _looks_like_checkpoint(dest):
            try:
                _validate_torchreid_checkpoint(dest)
                print(f"OK (exists): {dest}")
                return
            except Exception as exc:
                print(f"Existing checkpoint invalid ({exc}); re-downloading ...")
        dest.unlink(missing_ok=True)

    gdown = _ensure_gdown()
    print(f"Downloading Google Drive {file_id} -> {dest}")
    gdown.download(id=file_id, output=str(dest), quiet=False)

    if not _looks_like_checkpoint(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed or returned HTML instead of weights: {dest}. "
            "Try again with --force or check Google Drive access."
        )
    _validate_torchreid_checkpoint(dest)
    print(f"Verified checkpoint: {dest} ({dest.stat().st_size // 1_000_000} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ReID model weights")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download torchreid checkpoints even if present.",
    )
    args = parser.parse_args()
    root = Path(args.root)

    fastreid_out = root / "resources" / "models" / "fastreid"
    for filename, url in FASTREID_WEIGHTS.items():
        _download_url(url, fastreid_out / filename)

    torchreid_out = root / "resources" / "models" / "torchreid"
    for filename, file_id in TORCHREID_WEIGHTS.items():
        _download_gdrive(file_id, torchreid_out / filename, force=args.force)

    print(
        "Done. osnet (torchreid) downloads weights on first run; "
        "resnet50 uses resources/models/torchreid/market1501_resnet50.pth; "
        "resnet50_ibn/fastreid use fast-reid checkpoints."
    )


if __name__ == "__main__":
    main()
