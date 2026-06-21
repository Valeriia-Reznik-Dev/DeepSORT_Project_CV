#!/usr/bin/env python3
"""Run modern DeepSORT on MOT sequences."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from detectors.base import create_detector  # noqa: E402
from eval.reid_metrics import load_mot_gt_with_ids  # noqa: E402
from identity.manager import IdentityManager, IdentityParams  # noqa: E402
from reid.base import create_reid_extractor  # noqa: E402
from tracking.params import load_params_config, params_for  # noqa: E402
from tracking.pipeline import track_sequence  # noqa: E402

DETECTOR_CHOICES = ["yolo", "nanodet", "mmdet", "yolo_seg", "detectron2_seg", "smp_seg"]
REID_CHOICES = ["osnet", "resnet50_ibn", "fastreid"]


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


def _load_gt_detections(seq_dir: str, *, is_mot16: bool) -> dict[int, np.ndarray]:
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    by_frame_entries = load_mot_gt_with_ids(gt_path, is_mot16=is_mot16)
    out: dict[int, np.ndarray] = {}
    for frame_idx, entries in by_frame_entries.items():
        out[frame_idx] = np.stack([e[0] for e in entries]) if entries else np.zeros((0, 4))
    return out


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
    parser.add_argument("--detector", default="yolo", choices=DETECTOR_CHOICES)
    parser.add_argument(
        "--mask-background",
        action="store_true",
        help="Use instance masks (segmentation detector) to remove background "
        "before ReID. Requires a segmentation detector, e.g. --detector yolo_seg.",
    )
    parser.add_argument(
        "--gt-detections",
        action="store_true",
        help="Feed GT boxes instead of the detector (perfect-detector / 'disabled "
        "SORT' REID evaluation): isolates appearance/association from detector errors.",
    )
    parser.add_argument("--reid", default="osnet", choices=REID_CHOICES)
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Enable standalone body-ReID identity DB (lookup, vote, conflict resolution).",
    )
    parser.add_argument(
        "--identity-config",
        default=os.path.join(ROOT, "configs", "identity.yaml"),
        help="Identity DB parameters (radius, window, etc.).",
    )
    parser.add_argument(
        "--identity-reid",
        default=None,
        choices=REID_CHOICES,
        help="Separate ReID model for the identity gallery (default: reuse tracker ReID).",
    )
    parser.add_argument("--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml"))
    parser.add_argument("--detectors-config", default=os.path.join(ROOT, "configs", "detectors.yaml"))
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--tracker-name", default=None, help="Label (default: <detector>_<reid>)")
    parser.add_argument("--output-root", default=os.path.join(ROOT, "results", "tracking"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--params-config",
        default=None,
        help="YAML with per-video tracker params (e.g. configs/tracker_params.yaml). "
        "CLI param flags below override it globally.",
    )
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

    if args.params_config:
        params_cfg = load_params_config(args.params_config)
    else:
        params_cfg = {"default": project_cfg.get("tracker", {}), "per_video": {}}

    cli_overrides = {
        "min_confidence": args.min_confidence,
        "min_detection_height": args.min_detection_height,
        "nms_max_overlap": args.nms_max_overlap,
        "max_cosine_distance": args.max_cosine_distance,
        "nn_budget": args.nn_budget,
    }
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    default_name = f"{args.detector}_{args.reid}"
    if args.mask_background:
        default_name += "_seg"
    if args.identity:
        default_name += "_id"
    if args.gt_detections:
        default_name = f"gtdet_{args.reid}"
        if args.identity:
            default_name += "_id"
    tracker_name = args.tracker_name or default_name
    output_dir = os.path.join(args.output_root, tracker_name)
    os.makedirs(output_dir, exist_ok=True)

    identity_cfg = load_yaml(args.identity_config).get("identity", {}) if args.identity else {}
    identity_manager = None
    identity_reid = None
    if args.identity:
        identity_manager = IdentityManager(
            IdentityParams(
                radius=float(identity_cfg.get("radius", 0.3)),
                k=int(identity_cfg.get("k", 1)),
                representation=str(identity_cfg.get("representation", "centroid")),
                window=int(identity_cfg.get("window", 30)),
                conflict_policy=str(identity_cfg.get("conflict_policy", "distance")),
                max_per_identity=int(identity_cfg.get("max_per_identity", 50)),
            )
        )
        id_reid_name = args.identity_reid or identity_cfg.get("reid_model")
        if id_reid_name:
            id_reid_cfg = _resolve_device(
                load_yaml(args.reid_config)["reid_models"][id_reid_name], args.device
            )
            identity_reid = create_reid_extractor(id_reid_name, id_reid_cfg)

    print(
        f"Detector: {'GT-boxes' if args.gt_detections else args.detector} | "
        f"ReID: {args.reid} | mask_background: {args.mask_background} | "
        f"identity: {args.identity} | tracker_name: {tracker_name}"
    )
    if cli_overrides:
        print(f"CLI param overrides (all videos): {cli_overrides}")
    # In GT-detections mode the detector is unused (skip loading it).
    detector = None if args.gt_detections else create_detector(args.detector, det_cfg)
    reid = create_reid_extractor(args.reid, reid_cfg)

    mot16_set = set(project_cfg["sequences"]["mot16"])
    jobs = build_jobs(project_cfg)
    stats: dict[str, dict[str, float]] = {}
    for seq_name, seq_dir in jobs:
        if not os.path.isdir(seq_dir):
            print(f"SKIP (missing): {seq_dir}")
            continue
        params = params_for(params_cfg, seq_name, cli_overrides)
        gt_dets = None
        if args.gt_detections:
            gt_dets = _load_gt_detections(seq_dir, is_mot16=seq_name in mot16_set)
        out_file = os.path.join(output_dir, f"{seq_name}.txt")
        id_out = os.path.join(output_dir, f"{seq_name}_identity.csv") if args.identity else None
        print(f"\nTracking {seq_name} ... params={params}")
        stats[seq_name] = track_sequence(
            detector,
            reid,
            seq_dir,
            out_file,
            params=params,
            max_frames=args.max_frames,
            mask_background=args.mask_background,
            gt_detections=gt_dets,
            identity_manager=identity_manager,
            identity_reid=identity_reid,
            identity_output=id_out,
        )
        s = stats[seq_name]
        line = (
            f"  {seq_name}: {s['fps']:.2f} FPS "
            f"(det {s['det_ms']:.1f}ms | reid {s['reid_ms']:.1f}ms | track {s['track_ms']:.1f}ms"
        )
        if "identity_ms" in s:
            line += f" | identity {s['identity_ms']:.1f}ms | gallery {int(s['num_identities'])}"
        line += ")"
        print(line)

    if stats:
        mean_fps = sum(s["fps"] for s in stats.values()) / len(stats)
        print("\n=== Tracking FPS summary ===")
        print(f"{'Sequence':<20} {'FPS':>8} {'det ms':>8} {'reid ms':>8} {'track ms':>9}", end="")
        if args.identity:
            print(f" {'id ms':>8}", end="")
        print()
        print("-" * (56 + (8 if args.identity else 0)))
        for seq, s in sorted(stats.items()):
            row = (
                f"{seq:<20} {s['fps']:8.2f} {s['det_ms']:8.1f} "
                f"{s['reid_ms']:8.1f} {s['track_ms']:9.1f}"
            )
            if args.identity:
                row += f" {s.get('identity_ms', 0.0):8.1f}"
            print(row)
        print("-" * 56)
        print(f"{'MEAN':<20} {mean_fps:8.2f}")

    print(f"\nResults: {output_dir}")
    print(f"Evaluate: python scripts/run_eval.py --tracker-name {tracker_name} --results-dir {output_dir}")


if __name__ == "__main__":
    main()
