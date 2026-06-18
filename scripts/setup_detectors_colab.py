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
MMDET_MMCV_MAX_VERSION = "2.3.0"  # mmdet 3.3.0 excludes mmcv==2.2.0; patch assert
PIP_TIMEOUT_S = 300


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def verify_import(module: str) -> None:
    """Import in a fresh subprocess (editable installs are not always visible in-process)."""
    run([sys.executable, "-c", f"import {module}"])


def _torch_version() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()


def _torch_base_version(version: str) -> str:
    return version.split("+")[0]


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
        pass
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)",
            ],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
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


def _pin_torch(cu_tag: str) -> None:
    index = (
        "https://download.pytorch.org/whl/cu121"
        if cu_tag != "cpu"
        else "https://download.pytorch.org/whl/cpu"
    )
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "--force-reinstall",
        f"torch=={MMCV_TORCH}",
        "torchvision==0.19.0",
        "--index-url",
        index,
    ])


def _ensure_torch_for_mmcv() -> tuple[str, str]:
    """Pin PyTorch to a version with prebuilt mmcv wheels (avoids 20+ min source builds)."""
    cu_tag = "cu121" if _has_gpu_runtime() else "cpu"
    torch_ver = _torch_version()

    if _torch_base_version(torch_ver) != MMCV_TORCH or not _mmcv_wheel_index(cu_tag, MMCV_TORCH):
        print(f"Colab PyTorch {torch_ver} has no mmcv wheel; pinning torch=={MMCV_TORCH} ...")
        _pin_torch(cu_tag)
        torch_ver = _torch_version()

    if _torch_base_version(torch_ver) != MMCV_TORCH:
        raise RuntimeError(
            f"Failed to pin PyTorch to {MMCV_TORCH}, still have {torch_ver}. "
            "Restart runtime and rerun setup."
        )
    if not _mmcv_wheel_index(cu_tag, MMCV_TORCH):
        raise RuntimeError(
            f"No prebuilt mmcv=={MMCV_VERSION} wheel for {cu_tag}/torch{MMCV_TORCH}."
        )

    print(f"PyTorch {torch_ver} ready for mmcv ({cu_tag}/torch{MMCV_TORCH})")
    return cu_tag, MMCV_TORCH


def _patch_mmdet_mmcv_check() -> None:
    """mmdet 3.3.0 rejects mmcv==2.2.0; loosen the hard-coded upper bound."""
    import site

    old = "mmcv_maximum_version = '2.2.0'"
    new = f"mmcv_maximum_version = '{MMDET_MMCV_MAX_VERSION}'"
    for root in site.getsitepackages():
        path = Path(root) / "mmdet" / "__init__.py"
        if not path.is_file():
            continue
        text = path.read_text()
        if old in text:
            path.write_text(text.replace(old, new))
            print(f"Patched {path} for mmcv=={MMCV_VERSION}")
            return
        if new in text:
            print(f"Already patched: {path}")
            return
    raise RuntimeError("mmdet __init__.py not found for mmcv version patch")


def install_mmdet() -> None:
    cu_tag, torch_tag = _ensure_torch_for_mmcv()
    mmcv_index = _mmcv_wheel_index(cu_tag, torch_tag)
    assert mmcv_index is not None

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
    print(f"Installing mmcv=={MMCV_VERSION} from prebuilt wheel ({cu_tag}/torch{torch_tag}) ...")
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
        "mmdet==3.3.0",
    ])
    _patch_mmdet_mmcv_check()
    verify_import("mmdet")

    print(f"mmdet OK (mmcv=={MMCV_VERSION}, {cu_tag}/torch{torch_tag})")


def main() -> None:
    install_yolo()
    install_nanodet()
    install_mmdet()
    print("Detector deps ready.")


if __name__ == "__main__":
    main()
