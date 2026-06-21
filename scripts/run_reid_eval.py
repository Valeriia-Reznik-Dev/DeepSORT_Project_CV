#!/usr/bin/env python3
"""Evaluate ReID on GT crops."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.reid_metrics import (  # noqa: E402
    evaluate_reid,
    print_reid_summary,
    save_reid_report,
)
from reid.base import create_reid_extractor  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_device(reid_cfg: dict, device: str | None) -> None:
    import torch

    for model in reid_cfg["reid_models"].values():
        if device:
            model["device"] = device
        elif model.get("device") in (None, "auto") and torch.cuda.is_available():
            model["device"] = "cuda:0"
        elif model.get("device") in (None, "auto"):
            model["device"] = "cpu"


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
    parser = argparse.ArgumentParser(description="Evaluate ReID on GT pedestrian crops")
    parser.add_argument(
        "--project-config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "reid.yaml"),
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["osnet", "resnet50_ibn", "fastreid"],
        choices=["osnet", "resnet50_ibn", "fastreid"],
        help="ReID models to evaluate (default: all three).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for all models, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per sequence.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max GT crops per sequence (stratified subsample).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    reid_cfg = load_yaml(args.config)
    _apply_device(reid_cfg, args.device)
    eval_cfg = reid_cfg["reid_eval"]
    max_frames = args.max_frames if args.max_frames is not None else eval_cfg.get("max_frames")
    max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples")
    seed = eval_cfg.get("seed", 0)
    jobs = build_jobs(project_cfg)
    output_dir = Path(eval_cfg["output_dir"])

    reports = []
    for name in args.model:
        print(f"\nLoading ReID model: {name}")
        extractor = create_reid_extractor(name, reid_cfg["reid_models"][name])
        print(f"Evaluating {name} on {len(jobs)} sequences ...")
        reports.append(
            evaluate_reid(
                extractor,
                name,
                jobs,
                max_frames=max_frames,
                max_samples=max_samples,
                seed=seed,
            )
        )

    paths = save_reid_report(reports, output_dir)
    for report in reports:
        report["summary_paths"] = paths
    print_reid_summary(reports)
    print(f"\nSaved: {paths}")


if __name__ == "__main__":
    main()
