#!/usr/bin/env python3
"""Sweep one tracker parameter (HOTA + FPS)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors.base import create_detector  # noqa: E402
from eval import eval_plan_from_config, run_eval_plan  # noqa: E402
from reid.base import create_reid_extractor  # noqa: E402
from tracking.params import params_for  # noqa: E402
from tracking.pipeline import track_sequence  # noqa: E402

INT_PARAMS = {"nn_budget", "min_detection_height"}
SWEEPABLE = (
    "max_cosine_distance",
    "min_confidence",
    "nms_max_overlap",
    "nn_budget",
    "min_detection_height",
)


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


def _eval_hota(project_cfg: dict, tracker_name: str, results_dir: str) -> dict[str, float]:
    plan = eval_plan_from_config(project_cfg, tracker_name=tracker_name, results_dir=results_dir)
    report = run_eval_plan(plan, use_symlinks=False)
    data = report["trackers"][tracker_name]
    scores = {seq: s["HOTA"] for seq, s in data["sequences"].items()}
    scores["MEAN"] = data["HOTA_mean"]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep one tracker parameter")
    parser.add_argument("--param", required=True, choices=SWEEPABLE)
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--detector", default="yolo", choices=["yolo", "nanodet", "mmdet", "yolo_seg"])
    parser.add_argument("--reid", default="osnet", choices=["osnet", "resnet50_ibn", "fastreid"])
    parser.add_argument("--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml"))
    parser.add_argument("--detectors-config", default=os.path.join(ROOT, "configs", "detectors.yaml"))
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=os.path.join(ROOT, "results", "param_sweep"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    det_cfg = _resolve_device(load_yaml(args.detectors_config)["detectors"][args.detector], args.device)
    reid_cfg = _resolve_device(load_yaml(args.reid_config)["reid_models"][args.reid], args.device)

    base_cfg = {"default": project_cfg.get("tracker", {}), "per_video": {}}
    seq_order = list(project_cfg["sequences"]["mot15"]) + list(project_cfg["sequences"]["mot16"])

    print(f"Loading {args.detector} + {args.reid} ...")
    detector = create_detector(args.detector, det_cfg)
    reid = create_reid_extractor(args.reid, reid_cfg)
    jobs = build_jobs(project_cfg)

    rows: list[dict] = []
    for raw_value in args.values:
        value = int(raw_value) if args.param in INT_PARAMS else float(raw_value)
        tracker_name = f"{args.detector}_{args.reid}__{args.param}_{value}"
        out_dir = os.path.join(args.output_root, tracker_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {args.param} = {value} ===")

        fps_values: list[float] = []
        for seq_name, seq_dir in jobs:
            if not os.path.isdir(seq_dir):
                print(f"  SKIP (missing): {seq_dir}")
                continue
            params = replace(params_for(base_cfg, seq_name, None), **{args.param: value})
            out_file = os.path.join(out_dir, f"{seq_name}.txt")
            stats = track_sequence(
                detector, reid, seq_dir, out_file, params=params, max_frames=args.max_frames
            )
            fps_values.append(stats["fps"])
            print(f"  {seq_name}: {stats['fps']:.2f} FPS")

        mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
        hota = _eval_hota(project_cfg, tracker_name, out_dir)
        rows.append({"value": value, "hota": hota, "mean_fps": mean_fps})

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print(f"PARAMETER EVOLUTION: {args.param}  (detector={args.detector}, reid={args.reid})")
    print("=" * 100)
    header = f"{args.param:<22}" + "".join(f"{s[:10]:>11}" for s in seq_order) + f"{'MEAN':>9}{'FPS':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{str(r['value']):<22}" + "".join(f"{r['hota'].get(s, 0.0):>11.2f}" for s in seq_order)
        line += f"{r['hota'].get('MEAN', 0.0):>9.2f}{r['mean_fps']:>8.2f}"
        print(line)
    print("-" * len(header))
    best = max(rows, key=lambda r: r["hota"].get("MEAN", 0.0))
    print(f"BEST {args.param} by mean HOTA: {best['value']} (HOTA {best['hota']['MEAN']:.2f}, {best['mean_fps']:.2f} FPS)")

    # ---- Save ----
    os.makedirs(args.output_root, exist_ok=True)
    csv_path = os.path.join(args.output_root, f"sweep_{args.param}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [args.param] + seq_order + ["MEAN", "mean_fps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                args.param: r["value"],
                **{s: round(r["hota"].get(s, 0.0), 2) for s in seq_order},
                "MEAN": round(r["hota"].get("MEAN", 0.0), 2),
                "mean_fps": round(r["mean_fps"], 2),
            })
    json_path = os.path.join(args.output_root, f"sweep_{args.param}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"param": args.param, "detector": args.detector, "reid": args.reid, "rows": rows}, f, indent=2)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
