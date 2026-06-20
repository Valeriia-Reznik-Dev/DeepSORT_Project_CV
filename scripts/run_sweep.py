#!/usr/bin/env python3
"""Sweep detector x ReID combos: run tracker + HOTA, compare with baseline.

Produces one summary table (per-video HOTA, mean HOTA, mean FPS, whether the
combo beats the baseline on EVERY video) and saves it to CSV/JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors.base import create_detector  # noqa: E402
from eval import eval_plan_from_config, run_eval_plan  # noqa: E402
from reid.base import create_reid_extractor  # noqa: E402
from tracking.pipeline import TrackerParams, track_sequence  # noqa: E402

DEFAULT_COMBOS = ["yolo:osnet", "yolo:resnet50_ibn", "nanodet:osnet", "mmdet:osnet"]


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
    """Return {seq: HOTA, ..., 'MEAN': mean} for a results folder."""
    plan = eval_plan_from_config(project_cfg, tracker_name=tracker_name, results_dir=results_dir)
    report = run_eval_plan(plan, use_symlinks=False)
    data = report["trackers"][tracker_name]
    scores = {seq: s["HOTA"] for seq, s in data["sequences"].items()}
    scores["MEAN"] = data["HOTA_mean"]
    return scores


def run_combo(
    detector_name: str,
    reid_name: str,
    project_cfg: dict,
    det_cfg_all: dict,
    reid_cfg_all: dict,
    params: TrackerParams,
    device: str | None,
    output_root: str,
    max_frames: int | None,
) -> tuple[dict[str, float], float]:
    name = f"{detector_name}_{reid_name}"
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    det_cfg = _resolve_device(det_cfg_all["detectors"][detector_name], device)
    reid_cfg = _resolve_device(reid_cfg_all["reid_models"][reid_name], device)
    detector = create_detector(detector_name, det_cfg)
    reid = create_reid_extractor(reid_name, reid_cfg)

    fps_values: list[float] = []
    for seq_name, seq_dir in build_jobs(project_cfg):
        if not os.path.isdir(seq_dir):
            print(f"  SKIP (missing): {seq_dir}")
            continue
        out_file = os.path.join(out_dir, f"{seq_name}.txt")
        stats = track_sequence(
            detector, reid, seq_dir, out_file, params=params, max_frames=max_frames
        )
        fps_values.append(stats["fps"])
        print(f"  {seq_name}: {stats['fps']:.2f} FPS")

    mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    hota = _eval_hota(project_cfg, name, out_dir)
    return hota, mean_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep detector x ReID combos")
    parser.add_argument("--combos", nargs="+", default=DEFAULT_COMBOS, help="detector:reid pairs")
    parser.add_argument("--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml"))
    parser.add_argument("--detectors-config", default=os.path.join(ROOT, "configs", "detectors.yaml"))
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=os.path.join(ROOT, "results", "tracking"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-cosine-distance", type=float, default=None)
    parser.add_argument("--nn-budget", type=int, default=None)
    parser.add_argument(
        "--baseline-dir",
        default=os.path.join(ROOT, "results", "baseline", "original"),
        help="Baseline (original DeepSORT) results to compare against.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    det_cfg_all = load_yaml(args.detectors_config)
    reid_cfg_all = load_yaml(args.reid_config)
    tdefaults = project_cfg.get("tracker", {})

    params = TrackerParams(
        min_confidence=args.min_confidence if args.min_confidence is not None else tdefaults.get("min_confidence", 0.3),
        min_detection_height=tdefaults.get("min_detection_height", 0),
        nms_max_overlap=tdefaults.get("nms_max_overlap", 1.0),
        max_cosine_distance=args.max_cosine_distance if args.max_cosine_distance is not None else tdefaults.get("max_cosine_distance", 0.2),
        nn_budget=args.nn_budget if args.nn_budget is not None else tdefaults.get("nn_budget", 100),
    )

    seq_order = list(project_cfg["sequences"]["mot15"]) + list(project_cfg["sequences"]["mot16"])

    baseline = None
    if os.path.isdir(args.baseline_dir) and any(f.endswith(".txt") for f in os.listdir(args.baseline_dir)):
        print("Evaluating baseline (original) ...")
        baseline = _eval_hota(project_cfg, "original", args.baseline_dir)

    rows: list[dict] = []
    for combo in args.combos:
        detector_name, reid_name = combo.split(":")
        print(f"\n=== Combo: {detector_name} + {reid_name} ===")
        hota, mean_fps = run_combo(
            detector_name, reid_name, project_cfg, det_cfg_all, reid_cfg_all,
            params, args.device, args.output_root, args.max_frames,
        )
        beats_all = None
        if baseline is not None:
            beats_all = all(hota.get(s, 0.0) > baseline.get(s, 0.0) for s in seq_order)
        rows.append({
            "combo": f"{detector_name}+{reid_name}",
            "hota": hota,
            "mean_fps": mean_fps,
            "beats_all": beats_all,
        })

    # ---- Summary table ----
    print("\n" + "=" * 100)
    print("SWEEP SUMMARY (HOTA per video, mean HOTA, mean FPS)")
    print("=" * 100)
    header = f"{'Combo':<22}" + "".join(f"{s[:10]:>11}" for s in seq_order) + f"{'MEAN':>9}{'FPS':>8}{'>base':>7}"
    print(header)
    print("-" * len(header))
    if baseline is not None:
        base_line = f"{'original (baseline)':<22}" + "".join(f"{baseline.get(s, 0.0):>11.2f}" for s in seq_order)
        base_line += f"{baseline.get('MEAN', 0.0):>9.2f}{'-':>8}{'-':>7}"
        print(base_line)
    best_combo, best_mean = None, -1.0
    for r in rows:
        line = f"{r['combo']:<22}" + "".join(f"{r['hota'].get(s, 0.0):>11.2f}" for s in seq_order)
        flag = "yes" if r["beats_all"] else ("no" if r["beats_all"] is not None else "?")
        line += f"{r['hota'].get('MEAN', 0.0):>9.2f}{r['mean_fps']:>8.2f}{flag:>7}"
        print(line)
        if r["hota"].get("MEAN", 0.0) > best_mean:
            best_mean, best_combo = r["hota"]["MEAN"], r["combo"]
    print("-" * len(header))
    print(f"BEST by mean HOTA: {best_combo} ({best_mean:.2f})")
    rt = [r for r in rows if r["mean_fps"] >= 5.0 and r["beats_all"]]
    if rt:
        best_rt = max(rt, key=lambda r: r["hota"]["MEAN"])
        print(
            f"BEST real-time (>=5 FPS) beating baseline on every video: "
            f"{best_rt['combo']} (HOTA {best_rt['hota']['MEAN']:.2f}, {best_rt['mean_fps']:.2f} FPS)"
        )

    # ---- Save ----
    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["combo"] + seq_order + ["MEAN", "mean_fps", "beats_all"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if baseline is not None:
            writer.writerow({"combo": "original", **{s: round(baseline.get(s, 0.0), 2) for s in seq_order},
                             "MEAN": round(baseline.get("MEAN", 0.0), 2), "mean_fps": "", "beats_all": ""})
        for r in rows:
            writer.writerow({"combo": r["combo"], **{s: round(r["hota"].get(s, 0.0), 2) for s in seq_order},
                             "MEAN": round(r["hota"].get("MEAN", 0.0), 2),
                             "mean_fps": round(r["mean_fps"], 2), "beats_all": r["beats_all"]})
    json_path = os.path.join(out_dir, "sweep_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "combos": rows}, f, indent=2)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
