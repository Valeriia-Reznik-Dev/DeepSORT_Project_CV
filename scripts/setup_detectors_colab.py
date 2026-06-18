#!/usr/bin/env python3
"""Install YOLO + NanoDet + MMDet for Colab (with import checks)."""
from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANO_DIR = ROOT / "third_party" / "nanodet"
NANO_TAG = "v1.0.0-alpha-1"  # matches nanodet-plus-m_416 checkpoint release
NANO_COLLATE = NANO_DIR / "nanodet" / "data" / "collate.py"
TORCH_SIX_PATCH = "string_classes = (str, bytes)"
MMCV_VERSION = "2.2.0"
MMCV_TORCH = "2.4.0"
PIP_TIMEOUT_S = 300


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


def _has_gpu_runtime() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return "NVIDIA" in out
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _mmcv_wheel_index(cu_tag: str, torch_tag: str) -> str | None:
    url = f"https://download.openmmlab.com/mmcv/dist/{cu_tag}/torch{torch_tag}/index.html"
    try:
        html = urllib.request.urlopen(url, timeout=15).read().decode()
    except (urllib.error.URLError, TimeoutError):
        return None
    if f"mmcv-{MMCV_VERSION}" in html and ".whl" in html:
        return url
    return None


def _ensure_torch_for_mmcv() -> tuple[str, str]:
    """Pin PyTorch to a version with prebuilt mmcv wheels (avoids 20+ min source builds)."""
    import torch

    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    cu_tag = "cu121" if _has_gpu_runtime() else "cpu"
    if torch_mm == MMCV_TORCH and _mmcv_wheel_index(cu_tag, MMCV_TORCH):
        print(f"PyTorch {torch.__version__} OK for mmcv ({cu_tag}/torch{MMCV_TORCH})")
        return cu_tag, MMCV_TORCH

    if cu_tag == "cpu":
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--timeout",
            str(PIP_TIMEOUT_S),
            f"torch=={MMCV_TORCH}",
            "torchvision==0.19.0",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ])
    else:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--timeout",
            str(PIP_TIMEOUT_S),
            f"torch=={MMCV_TORCH}",
            "torchvision==0.19.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ])

    import importlib

    importlib.invalidate_caches()
    import torch as torch_mod

    print(f"Pinned PyTorch {torch_mod.__version__} for mmcv ({cu_tag}/torch{MMCV_TORCH})")
    return cu_tag, MMCV_TORCH


def install_mmdet() -> None:
    cu_tag, torch_tag = _ensure_torch_for_mmcv()
    mmcv_index = _mmcv_wheel_index(cu_tag, torch_tag)
    if not mmcv_index:
        raise RuntimeError(
            f"No prebuilt mmcv=={MMCV_VERSION} wheel for {cu_tag}/torch{torch_tag}. "
            "Use a GPU Colab runtime or report this combo."
        )

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "setuptools==75.8.2",
        "openmim",
        "mmengine",
    ])
    print(f"Installing mmcv from prebuilt wheel ({cu_tag}/torch{torch_tag}) ...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        f"mmcv=={MMCV_VERSION}",
        "-f",
        mmcv_index,
        "--only-binary",
        "mmcv",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "mmdet",
    ])
    verify_import("mmdet")

    print(f"mmdet OK (mmcv wheel: {cu_tag}/torch{torch_tag})")


def main() -> None:
    install_yolo()
    install_nanodet()
    install_mmdet()
    print("Detector deps ready.")


if __name__ == "__main__":
    main()
