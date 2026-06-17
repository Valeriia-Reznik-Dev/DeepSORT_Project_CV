#!/usr/bin/env python3
"""Run original DeepSORT baseline on all 6 evaluation sequences."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_sort_app  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_sequence(sequence_dir, detection_file, output_file, tracker_cfg):
    if not os.path.isdir(sequence_dir):
        print(f"SKIP (missing sequence): {sequence_dir}")
        return False
    if not os.path.isfile(detection_file):
        print(f"SKIP (missing detections): {detection_file}")
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Running {os.path.basename(sequence_dir)}")
    deep_sort_app.run(
        sequence_dir=sequence_dir,
        detection_file=detection_file,
        output_file=output_file,
        min_confidence=tracker_cfg["min_confidence"],
        nms_max_overlap=tracker_cfg["nms_max_overlap"],
        min_detection_height=tracker_cfg["min_detection_height"],
        max_cosine_distance=tracker_cfg["max_cosine_distance"],
        nn_budget=tracker_cfg["nn_budget"],
        display=False,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Run original DeepSORT baseline")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    cfg = load_config(args.config)
    tracker_cfg = cfg["tracker"]
    paths = cfg["paths"]
    results_dir = paths["results_dir"]

    jobs = []
    for name in cfg["sequences"]["mot15"]:
        jobs.append((
            os.path.join(paths["mot15_dir"], name),
            os.path.join(paths["detections_mot15"], f"{name}.npy"),
            os.path.join(results_dir, f"{name}.txt"),
        ))
    for name in cfg["sequences"]["mot16"]:
        jobs.append((
            os.path.join(paths["mot16_dir"], name),
            os.path.join(paths["detections_mot16"], f"{name}.npy"),
            os.path.join(results_dir, f"{name}.txt"),
        ))

    ok, skipped = 0, 0
    for sequence_dir, detection_file, output_file in jobs:
        if run_sequence(sequence_dir, detection_file, output_file, tracker_cfg):
            ok += 1
        else:
            skipped += 1

    print(f"Done: {ok} sequences, {skipped} skipped.")
    if skipped:
        print("Check paths in configs/baseline_original.yaml and generated .npy files.")


if __name__ == "__main__":
    main()
