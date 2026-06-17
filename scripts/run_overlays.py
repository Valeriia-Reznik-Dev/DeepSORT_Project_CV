#!/usr/bin/env python3
"""Render baseline tracking overlays for available sequences."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_videos  # noqa: E402
import show_results  # noqa: E402


def run_for_mot_dir(mot_dir, result_dir, output_dir, convert_h264, update_ms):
    if not os.path.isdir(mot_dir):
        print(f"SKIP overlays (missing): {mot_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Rendering overlays from {mot_dir}")
    for sequence_txt in os.listdir(result_dir):
        if not sequence_txt.endswith(".txt"):
            continue
        sequence = os.path.splitext(sequence_txt)[0]
        sequence_dir = os.path.join(mot_dir, sequence)
        if not os.path.isdir(sequence_dir):
            continue
        result_file = os.path.join(result_dir, sequence_txt)
        video_filename = os.path.join(output_dir, f"{sequence}.avi")
        print(f"Saving {sequence} -> {video_filename}")
        show_results.run(
            sequence_dir, result_file, False, None, update_ms, video_filename)

    if not convert_h264:
        return
    for sequence_txt in os.listdir(result_dir):
        sequence = os.path.splitext(sequence_txt)[0]
        filename_in = os.path.join(output_dir, f"{sequence}.avi")
        filename_out = os.path.join(output_dir, f"{sequence}.mp4")
        if os.path.isfile(filename_in):
            generate_videos.convert(filename_in, filename_out)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline overlay videos")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
    )
    parser.add_argument("--convert_h264", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    run_for_mot_dir(
        paths["mot15_dir"],
        paths["results_dir"],
        paths["overlays_dir"],
        args.convert_h264,
        None,
    )
    run_for_mot_dir(
        paths["mot16_dir"],
        paths["results_dir"],
        paths["overlays_dir"],
        args.convert_h264,
        None,
    )


if __name__ == "__main__":
    main()
