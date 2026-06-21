#!/usr/bin/env python3
"""Install YOLO, NanoDet, MMDet for Colab."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANO_DIR = ROOT / "third_party" / "nanodet"
NANO_TAG = "v1.0.0-alpha-1"  # matches nanodet-plus-m_416 checkpoint release
NANO_COLLATE = NANO_DIR / "nanodet" / "data" / "collate.py"
NANO_LOGGER = NANO_DIR / "nanodet" / "util" / "logger.py"
NANO_ONE_STAGE = NANO_DIR / "nanodet" / "model" / "arch" / "one_stage_detector.py"
TORCH_SIX_PATCH = "string_classes = (str, bytes)"
CUDA_SYNC = "torch.cuda.synchronize()"
CUDA_SYNC_GUARDED = "torch.cuda.synchronize() if torch.cuda.is_available() else None"
LIGHTNING_IMPORTS_OLD = (
    "from pytorch_lightning.loggers import LightningLoggerBase\n"
    "from pytorch_lightning.loggers.base import rank_zero_experiment\n"
    "from pytorch_lightning.utilities import rank_zero_only\n"
    "from pytorch_lightning.utilities.cloud_io import get_filesystem"
)
LIGHTNING_IMPORTS_NEW = (
    "try:\n"
    "    from pytorch_lightning.loggers import Logger as LightningLoggerBase\n"
    "except ImportError:\n"
    "    from pytorch_lightning.loggers import LightningLoggerBase\n"
    "\n"
    "try:\n"
    "    from pytorch_lightning.loggers.logger import rank_zero_experiment\n"
    "except ImportError:\n"
    "    from pytorch_lightning.loggers.base import rank_zero_experiment\n"
    "\n"
    "try:\n"
    "    from pytorch_lightning.utilities.rank_zero import rank_zero_only\n"
    "except ImportError:\n"
    "    from pytorch_lightning.utilities import rank_zero_only\n"
    "\n"
    "try:\n"
    "    from pytorch_lightning.utilities.cloud_io import get_filesystem\n"
    "except ImportError:\n"
    "    from lightning_fabric.utilities.cloud_io import get_filesystem"
)
MMCV_VERSION = "2.2.0"
MMCV_TORCH = "2.4.0"
MMDET_MMCV_MAX_VERSION = "2.3.0"  # mmdet 3.3.0 excludes mmcv==2.2.0; patch assert
PIP_TIMEOUT_S = 300


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def verify_import(module: str) -> None:
    run([sys.executable, "-c", f"import {module}"])


def _torch_version() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()


def _torch_base_version(version: str) -> str:
    return version.split("+")[0]


def patch_nanodet_for_pytorch2() -> None:
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


def patch_nanodet_for_lightning() -> None:
    if not NANO_LOGGER.is_file():
        raise FileNotFoundError(f"Missing NanoDet logger module: {NANO_LOGGER}")
    text = NANO_LOGGER.read_text()
    if LIGHTNING_IMPORTS_OLD in text:
        text = text.replace(LIGHTNING_IMPORTS_OLD, LIGHTNING_IMPORTS_NEW)
        NANO_LOGGER.write_text(text)
        print("Patched nanodet/util/logger.py for pytorch-lightning 2.x")


def patch_nanodet_for_cpu_inference() -> None:
    if not NANO_ONE_STAGE.is_file():
        raise FileNotFoundError(f"Missing NanoDet detector arch: {NANO_ONE_STAGE}")
    text = NANO_ONE_STAGE.read_text()
    if CUDA_SYNC in text and CUDA_SYNC_GUARDED not in text:
        text = text.replace(CUDA_SYNC, CUDA_SYNC_GUARDED)
        NANO_ONE_STAGE.write_text(text)
        print("Patched nanodet/model/arch/one_stage_detector.py for CPU inference")


def install_yolo() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "scikit-learn"])


def _nanodet_ready() -> bool:
    return (NANO_DIR / "nanodet" / "data" / "batch_process.py").is_file()


def install_nanodet() -> None:
    if NANO_DIR.is_dir() and not _nanodet_ready():
        print(f"Removing incomplete NanoDet clone: {NANO_DIR}")
        shutil.rmtree(NANO_DIR)
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
    patch_nanodet_for_lightning()
    patch_nanodet_for_cpu_inference()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(NANO_DIR / "requirements.txt")])
    # setup.py imports nanodet before install; run editable install from repo root
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(NANO_DIR)])
    verify_import("nanodet")
    run([
        sys.executable,
        "-c",
        "from nanodet.data.batch_process import stack_batch_img; "
        "from nanodet.model.arch import build_model; "
        "from nanodet.util import Logger, load_model_weight",
    ])

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


def repair_mmcv_torch() -> None:
    cu_tag, torch_tag = _ensure_torch_for_mmcv()
    mmcv_index = _mmcv_wheel_index(cu_tag, torch_tag)
    if mmcv_index is None:
        raise RuntimeError(
            f"No prebuilt mmcv=={MMCV_VERSION} wheel for {cu_tag}/torch{torch_tag}."
        )

    print(f"Reinstalling mmcv=={MMCV_VERSION} for torch {torch_tag} ({cu_tag}) ...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "--force-reinstall",
        f"mmcv=={MMCV_VERSION}",
        "-f",
        mmcv_index,
        "--only-binary",
        "mmcv",
    ])
    _patch_mmdet_mmcv_check()
    verify_import("mmcv")
    try:
        verify_import("mmdet")
        print("mmcv/mmdet import OK after repair.")
    except subprocess.CalledProcessError:
        print("mmcv OK; mmdet not installed (skip mmdet verify).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install detector deps for Colab")
    parser.add_argument(
        "--repair-mmcv-only",
        action="store_true",
        help="Only re-pin torch and reinstall mmcv (after ReID setup broke MMDet).",
    )
    parser.add_argument(
        "--nanodet-only",
        action="store_true",
        help="Only clone/patch/install NanoDet (fixes missing nanodet.data).",
    )
    args = parser.parse_args()

    if args.repair_mmcv_only:
        repair_mmcv_torch()
        return

    if args.nanodet_only:
        install_nanodet()
        return

    install_yolo()
    install_nanodet()
    install_mmdet()
    print("Detector deps ready.")


if __name__ == "__main__":
    main()
