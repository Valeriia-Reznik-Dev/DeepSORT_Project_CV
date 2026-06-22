#!/usr/bin/env python3
"""Evaluate identity DB on GT crops."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.identity_metrics import (  # noqa: E402
    evaluate_identity,
    print_identity_summary,
    save_identity_report,
)
from identity.manager import IdentityParams  # noqa: E402
from reid.base import create_reid_extractor  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(cfg: dict, device: str | None) -> dict:
    import torch

    cfg = dict(cfg)
    if device:
        cfg["device"] = device
    elif cfg.get("device") in (None, "auto"):
        cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    return cfg


def build_jobs(project_cfg: dict) -> list[tuple[str, str, str, bool]]:
    paths = project_cfg["paths"]
    jobs: list[tuple[str, str, str, bool]] = []
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
    parser = argparse.ArgumentParser(description="Evaluate standalone identity DB")
    parser.add_argument(
        "--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml")
    )
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument(
        "--identity-config", default=os.path.join(ROOT, "configs", "identity.yaml")
    )
    parser.add_argument(
        "--reid", default="osnet", choices=["osnet", "resnet50_ibn", "fastreid"]
    )
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "identity"))
    parser.add_argument("--max-frames", type=int, default=None)
    identity_defaults = load_yaml(
        os.path.join(ROOT, "configs", "identity.yaml")
    ).get("identity", {})
    # Identity DB parameters (defaults from configs/identity.yaml).
    parser.add_argument(
        "--radius",
        type=float,
        default=identity_defaults.get("radius", 0.4),
    )
    parser.add_argument("--k", type=int, default=identity_defaults.get("k", 1))
    parser.add_argument(
        "--representation",
        default=identity_defaults.get("representation", "centroid"),
        choices=["centroid", "knn"],
    )
    parser.add_argument("--window", type=int, default=identity_defaults.get("window", 30))
    parser.add_argument(
        "--conflict-policy",
        default=identity_defaults.get("conflict_policy", "distance"),
        choices=["distance", "none"],
    )
    parser.add_argument(
        "--max-per-identity",
        type=int,
        default=identity_defaults.get("max_per_identity", 50),
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    reid_cfg = load_yaml(args.reid_config)["reid_models"][args.reid]
    reid_cfg = _resolve_device(reid_cfg, args.device)

    params = IdentityParams(
        radius=args.radius,
        k=args.k,
        representation=args.representation,
        window=args.window,
        conflict_policy=args.conflict_policy,
        max_per_identity=args.max_per_identity,
    )

    print(f"ReID: {args.reid} | params: {params}")
    extractor = create_reid_extractor(args.reid, reid_cfg)
    jobs = build_jobs(project_cfg)

    report = evaluate_identity(
        extractor, args.reid, jobs, params=params, max_frames=args.max_frames
    )
    paths = save_identity_report(report, Path(args.output_dir))
    print_identity_summary(report)
    print(f"\nSaved: {paths}")


if __name__ == "__main__":
    main()
