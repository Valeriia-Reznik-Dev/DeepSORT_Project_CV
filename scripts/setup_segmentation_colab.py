#!/usr/bin/env python3
"""Install detectron2 and SMP for Colab."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMP_CKPT = ROOT / "resources" / "models" / "smp" / "deeplabv3plus_cityscapes.pth"
PIP_TIMEOUT_S = 600
DETECTRON2_GIT = "git+https://github.com/facebookresearch/detectron2.git"

# Legacy OpenMMLab URLs (often 404); kept as first attempt only.
SMP_CKPT_URLS = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/"
    "deeplabv3_r50-d8_512x1024_80k_cityscapes/"
    "deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_114645-9daa3b74.pth",
)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def _py_info() -> tuple[str, str, str]:
    code = (
        "import sys, torch; "
        "v = torch.__version__; "
        "base = '.'.join(v.split('+')[0].split('.')[:2]); "
        "cuda = 'cpu'; "
        "if '+cu' in v: cuda = 'cu' + v.split('+cu')[1].split('.')[0]; "
        "elif torch.cuda.is_available() and torch.version.cuda: "
        "cuda = 'cu' + torch.version.cuda.replace('.', '')[:3]; "
        "print(f'{sys.version_info.major}.{sys.version_info.minor}|{cuda}|{base}')"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    py, cuda, torch_base = out.split("|")
    return py, cuda, torch_base


def _wheel_indices(cuda: str, torch_base: str) -> list[str]:
    indices = [
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch{torch_base}/index.html",
    ]
    if cuda.startswith("cu"):
        alt = cuda[:5] if len(cuda) > 5 else cuda  # cu124 -> cu124, cu1210 bug guard
        for tag in (cuda, alt, "cu121", "cu124", "cu118"):
            indices.append(
                f"https://dl.fbaipublicfiles.com/detectron2/wheels/{tag}/torch{torch_base}/index.html"
            )
    indices.append(
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch{torch_base}/index.html"
    )
    seen: set[str] = set()
    out: list[str] = []
    for url in indices:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _verify_detectron2() -> None:
    run([sys.executable, "-c", "import detectron2; print('detectron2 OK')"])


def _install_detectron2_wheels(indices: list[str]) -> bool:
    for index in indices:
        print(f"Trying detectron2 wheel: {index}")
        proc = run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--timeout",
                str(PIP_TIMEOUT_S),
                "detectron2",
                "-f",
                index,
            ],
            check=False,
        )
        if proc.returncode == 0:
            return True
    return False


def _install_detectron2_source() -> None:
    print("Building detectron2 from source (needed on Colab Py3.12 / torch 2.4+) ...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "ninja",
        "opencv-python-headless",
        "pycocotools",
    ])
    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "2")
    print(">", sys.executable, "-m pip install", DETECTRON2_GIT)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--timeout",
            str(PIP_TIMEOUT_S),
            "--no-build-isolation",
            DETECTRON2_GIT,
        ],
        env=env,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "detectron2 source build failed. "
            "Use --detector yolo_seg (works without detectron2) or restart runtime and retry."
        )


def install_detectron2() -> None:
    py, cuda, torch_base = _py_info()
    print(f"Python {py}, torch {torch_base}, cuda tag {cuda}")
    if _install_detectron2_wheels(_wheel_indices(cuda, torch_base)):
        _verify_detectron2()
        return
    print("No prebuilt detectron2 wheel for this Python/torch/CUDA combo.")
    _install_detectron2_source()
    _verify_detectron2()


def install_smp() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--timeout",
        str(PIP_TIMEOUT_S),
        "segmentation-models-pytorch",
    ])
    run([
        sys.executable,
        "-c",
        "import segmentation_models_pytorch as smp; print('smp OK', smp.__version__)",
    ])


def _download(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.is_file() and dest.stat().st_size > 1_000_000
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  skip {url}: {exc}")
        return False


def _bootstrap_smp_from_torchvision(dest: Path) -> None:
    import torch
    import torchvision.models.segmentation as seg

    print("Bootstrapping SMP weights from torchvision DeepLabV3-ResNet50 (COCO/VOC) ...")
    weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT
    model = seg.deeplabv3_resnet50(weights=weights)
    torch.save({"format": "torchvision_deeplabv3_r50", "state_dict": model.state_dict()}, dest)
    print(f"Saved {dest} ({dest.stat().st_size // 1024} KiB)")


def download_smp_weights() -> None:
    SMP_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if SMP_CKPT.is_file() and SMP_CKPT.stat().st_size > 1_000_000:
        print(f"SMP weights already present: {SMP_CKPT}")
        return

    for url in SMP_CKPT_URLS:
        print(f"Downloading {url}")
        if _download(url, SMP_CKPT):
            print("Download complete.")
            return

    _bootstrap_smp_from_torchvision(SMP_CKPT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install segmentation deps for Colab")
    parser.add_argument("--detectron2-only", action="store_true")
    parser.add_argument("--smp-only", action="store_true")
    parser.add_argument(
        "--download-smp-weights",
        action="store_true",
        help="Download/bootstrap weights for smp_seg.",
    )
    parser.add_argument(
        "--skip-detectron2",
        action="store_true",
        help="Install SMP only (yolo_seg does not need detectron2).",
    )
    args = parser.parse_args()

    if args.download_smp_weights:
        download_smp_weights()
        return

    if args.detectron2_only:
        install_detectron2()
        return
    if args.smp_only:
        install_smp()
        return

    if not args.skip_detectron2:
        try:
            install_detectron2()
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"WARN: detectron2 unavailable ({exc}). Use --detector yolo_seg.")

    install_smp()
    print("Segmentation deps ready.")
    print("Optional: python scripts/setup_segmentation_colab.py --download-smp-weights")


if __name__ == "__main__":
    main()
