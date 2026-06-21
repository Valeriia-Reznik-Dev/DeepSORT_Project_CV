#!/usr/bin/env python3
"""Install detectron2 and SMP for Colab."""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMP_CKPT = ROOT / "resources" / "models" / "smp" / "deeplabv3plus_cityscapes.pth"
SMP_CKPT_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/"
    "deeplabv3_r50-d8_512x1024_80k_cityscapes/"
    "deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_114645-9daa3b74.pth"
)
PIP_TIMEOUT_S = 300


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _torch_tag() -> tuple[str, str]:
    version = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()
    base = version.split("+")[0]
    cuda = "cpu"
    if "+cu" in version:
        cuda = "cu" + version.split("+cu")[1].split(".")[0]
    elif subprocess.call(
        [sys.executable, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0:
        cuda = subprocess.check_output(
            [sys.executable, "-c", "import torch; print('cu'+str(torch.version.cuda).replace('.','')[:3])"],
            text=True,
        ).strip()
    return cuda, base


def install_detectron2() -> None:
    cu_tag, torch_tag = _torch_tag()
    wheel_index = (
        f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cu_tag}/torch{torch_tag}/index.html"
    )
    print(f"Installing detectron2 from {wheel_index} ...")
    try:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--timeout",
            str(PIP_TIMEOUT_S),
            "detectron2",
            "-f",
            wheel_index,
        ])
    except subprocess.CalledProcessError:
        print("Prebuilt detectron2 wheel failed; trying pip detectron2 ...")
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--timeout",
            str(PIP_TIMEOUT_S),
            "detectron2",
        ])
    run([sys.executable, "-c", "import detectron2; print('detectron2 OK')"])


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
    run([sys.executable, "-c", "import segmentation_models_pytorch as smp; print('smp OK', smp.__version__)"])


def download_smp_weights() -> None:
    SMP_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if SMP_CKPT.is_file():
        print(f"SMP weights already present: {SMP_CKPT}")
        return
    print(f"Downloading SMP/Cityscapes weights -> {SMP_CKPT}")
    urllib.request.urlretrieve(SMP_CKPT_URL, SMP_CKPT)
    print("Download complete (mmseg format; smp loader uses strict=False).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install segmentation deps for Colab")
    parser.add_argument("--detectron2-only", action="store_true")
    parser.add_argument("--smp-only", action="store_true")
    parser.add_argument(
        "--download-smp-weights",
        action="store_true",
        help="Download Cityscapes DeepLabV3 weights for smp_seg (ResNet-50 mmseg ckpt).",
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

    install_detectron2()
    install_smp()
    print("Segmentation deps ready (detectron2 + smp).")
    print("Optional: python scripts/setup_segmentation_colab.py --download-smp-weights")


if __name__ == "__main__":
    main()
