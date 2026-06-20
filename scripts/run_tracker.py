#!/usr/bin/env python3
"""Run the modern DeepSORT (detector + ReID + core) on the MOT test sequences.

Writes MOTChallenge results per sequence, then evaluate with:
    python scripts/run_eval.py --tracker-name <name> --results-dir <out>
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors.base import create_detector  # noqa: E402
from reid.base import create_reid_extractor  # noqa: E402
from tracking.pipeline import TrackerParams, track_sequence  # noqa: E402


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


def build_jobs(project_cfg: dict) -> list[tuple[str, str]]:
    paths = project_cfg["paths"]
    jobs: list[tuple[str, str]] = []
    for name in project_cfg["sequences"]["mot15"]:
        jobs.append((name, os.path.join(paths["mot15_dir"], name)))
    for name in project_cfg["sequences"]["mot16"]:
        jobs.append((name, os.path.join(paths["mot16_dir"], name)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modern DeepSORT tracker")
    parser.add_argument(
        "--detector", default="yolo", choices=["yolo", "nanodet", "mmdet", "yolo_seg"]
    )
    parser.add_argument(
        "--mask-background",
        action="store_true",
        help="Use instance masks (segmentation detector) to remove background "
        "before ReID. Requires a segmentation detector, e.g. --detector yolo_seg.",
    )
    parser.add_argument(
        "--reid", default="osnet", choices=["osnet", "resnet50_ibn", "fastreid"]
    )
    parser.add_argument("--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml"))
    parser.add_argument("--detectors-config", default=os.path.join(ROOT, "configs", "detectors.yaml"))
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--tracker-name", default=None, help="Label (default: <detector>_<reid>)")
    parser.add_argument("--output-root", default=os.path.join(ROOT, "results", "tracking"))
    parser.add_argument("--max-frames", type=int, default=None)
    # Tracker params (defaults from project config 'tracker' section).
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-detection-height", type=int, default=None)
    parser.add_argument("--nms-max-overlap", type=float, default=None)
    parser.add_argument("--max-cosine-distance", type=float, default=None)
    parser.add_argument("--nn-budget", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    det_cfg = load_yaml(args.detectors_config)["detectors"][args.detector]
    reid_cfg = load_yaml(args.reid_config)["reid_models"][args.reid]
    det_cfg = _resolve_device(det_cfg, args.device)
    reid_cfg = _resolve_device(reid_cfg, args.device)

    tdefaults = project_cfg.get("tracker", {})

    def pick(value, key, default):
        if value is not None:
            return value
        return tdefaults.get(key, default)

    params = TrackerParams(
        min_confidence=pick(args.min_confidence, "min_confidence", 0.3),
        min_detection_height=pick(args.min_detection_height, "min_detection_height", 0),
        nms_max_overlap=pick(args.nms_max_overlap, "nms_max_overlap", 1.0),
        max_cosine_distance=pick(args.max_cosine_distance, "max_cosine_distance", 0.2),
        nn_budget=pick(args.nn_budget, "nn_budget", 100),
    )

    default_name = f"{args.detector}_{args.reid}"
    if args.mask_background:
        default_name += "_seg"
    tracker_name = args.tracker_name or default_name
    output_dir = os.path.join(args.output_root, tracker_name)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Detector: {args.detector} | ReID: {args.reid} | "
        f"mask_background: {args.mask_background} | tracker_name: {tracker_name}"
    )
    print(f"Params: {params}")
    detector = create_detector(args.detector, det_cfg)
    reid = create_reid_extractor(args.reid, reid_cfg)

    jobs = build_jobs(project_cfg)
    stats: dict[str, dict[str, float]] = {}
    for seq_name, seq_dir in jobs:
        if not os.path.isdir(seq_dir):
            print(f"SKIP (missing): {seq_dir}")
            continue
        out_file = os.path.join(output_dir, f"{seq_name}.txt")
        print(f"\nTracking {seq_name} ...")
        stats[seq_name] = track_sequence(
            detector,
            reid,
            seq_dir,
            out_file,
            params=params,
            max_frames=args.max_frames,
            mask_background=args.mask_background,
        )
        s = stats[seq_name]
        print(
            f"  {seq_name}: {s['fps']:.2f} FPS "
            f"(det {s['det_ms']:.1f}ms | reid {s['reid_ms']:.1f}ms | track {s['track_ms']:.1f}ms)"
        )

    if stats:
        mean_fps = sum(s["fps"] for s in stats.values()) / len(stats)
        print("\n=== Tracking FPS summary ===")
        print(f"{'Sequence':<20} {'FPS':>8} {'det ms':>8} {'reid ms':>8} {'track ms':>9}")
        print("-" * 56)
        for seq, s in sorted(stats.items()):
            print(
                f"{seq:<20} {s['fps']:8.2f} {s['det_ms']:8.1f} "
                f"{s['reid_ms']:8.1f} {s['track_ms']:9.1f}"
            )
        print("-" * 56)
        print(f"{'MEAN':<20} {mean_fps:8.2f}")

    print(f"\nResults: {output_dir}")
    print(f"Evaluate: python scripts/run_eval.py --tracker-name {tracker_name} --results-dir {output_dir}")


if __name__ == "__main__":
    main()
