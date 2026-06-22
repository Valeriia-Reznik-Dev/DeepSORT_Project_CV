#!/usr/bin/env python3
"""Render tracking overlay videos (headless, no GUI)."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_videos  # noqa: E402
from eval.overlay_render import render_overlays_for_mot_dir  # noqa: E402


def run_for_mot_dir(mot_dir, result_dir, output_dir, convert_h264, fourcc):
    written = render_overlays_for_mot_dir(
        mot_dir, result_dir, output_dir, fourcc=fourcc
    )
    if not convert_h264:
        return written

    for avi_path in written:
        mp4_path = avi_path.with_suffix(".mp4")
        generate_videos.convert(str(avi_path), str(mp4_path))
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate tracking overlay videos")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Folder with <seq>.txt results (default: paths.results_dir = baseline).",
    )
    parser.add_argument(
        "--overlays-dir",
        default=None,
        help="Output folder for overlay videos (default: paths.overlays_dir).",
    )
    parser.add_argument("--convert_h264", action="store_true")
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="OpenCV VideoWriter fourcc (default: MJPG).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    results_dir = args.results_dir or paths["results_dir"]
    overlays_dir = args.overlays_dir or paths["overlays_dir"]
    run_for_mot_dir(
        paths["mot15_dir"],
        results_dir,
        overlays_dir,
        args.convert_h264,
        args.fourcc,
    )
    run_for_mot_dir(
        paths["mot16_dir"],
        results_dir,
        overlays_dir,
        args.convert_h264,
        args.fourcc,
    )


if __name__ == "__main__":
    main()
