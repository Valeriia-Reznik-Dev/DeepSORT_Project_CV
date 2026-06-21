#!/usr/bin/env python3
"""Evaluate detectors (P/R/F1 vs GT)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors.base import create_detector  # noqa: E402
from eval.detector_metrics import (  # noqa: E402
    evaluate_detector,
    print_detector_summary,
    save_detector_report,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_device(det_cfg: dict, device: str | None) -> None:
    import torch

    for det in det_cfg["detectors"].values():
        if device:
            det["device"] = device
        elif det.get("device") in (None, "auto") and torch.cuda.is_available():
            det["device"] = "cuda:0"
        elif det.get("device") in (None, "auto"):
            det["device"] = "cpu"


def build_jobs(project_cfg: dict) -> list[tuple[str, str, str, bool]]:
    paths = project_cfg["paths"]
    jobs = []
    for name in project_cfg["sequences"]["mot15"]:
        jobs.append((
            name,
            os.path.join(paths["mot15_dir"], name),
            os.path.join(paths["mot15_dir"], name, "gt", "gt.txt"),
            False,
        ))
    for name in project_cfg["sequences"]["mot16"]:
        jobs.append((
            name,
            os.path.join(paths["mot16_dir"], name),
            os.path.join(paths["mot16_dir"], name, "gt", "gt.txt"),
            True,
        ))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detectors vs MOT GT")
    parser.add_argument(
        "--project-config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "detectors.yaml"),
    )
    parser.add_argument(
        "--detector",
        nargs="+",
        default=["yolo", "nanodet", "mmdet"],
        choices=["yolo", "nanodet", "mmdet"],
        help="Detectors to evaluate (default: all three).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for all detectors, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per sequence (overrides config).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    det_cfg = load_yaml(args.config)
    _apply_device(det_cfg, args.device)
    eval_cfg = det_cfg["detector_eval"]
    max_frames = args.max_frames if args.max_frames is not None else eval_cfg.get("max_frames")
    jobs = build_jobs(project_cfg)
    output_dir = Path(eval_cfg["output_dir"])
    iou_threshold = eval_cfg.get("iou_threshold", 0.5)

    reports = []
    for name in args.detector:
        print(f"\nLoading detector: {name}")
        detector = create_detector(name, det_cfg["detectors"][name])
        print(f"Evaluating {name} on {len(jobs)} sequences ...")
        reports.append(
            evaluate_detector(
                detector,
                name,
                jobs,
                iou_threshold=iou_threshold,
                max_frames=max_frames,
            )
        )

    paths = save_detector_report(reports, output_dir)
    for report in reports:
        report["summary_paths"] = paths
    print_detector_summary(reports)
    print(f"\nSaved: {paths}")


if __name__ == "__main__":
    main()
