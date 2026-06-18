#!/usr/bin/env python3
"""Run TrackEval HOTA evaluation on tracking results."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval import eval_plan_from_config, load_yaml, print_summary, run_eval_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracking results with TrackEval")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "baseline_original.yaml"),
        help="Shared project config (sequences, paths, eval settings).",
    )
    parser.add_argument(
        "--tracker-name",
        default="original",
        help="Tracker label in reports (default: original).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Folder with <seq>.txt MOT results (default: paths.results_dir from config).",
    )
    parser.add_argument(
        "--copy-instead-of-symlink",
        action="store_true",
        help="Copy tracker txt files instead of symlinking (needed on some filesystems).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    plan = eval_plan_from_config(
        load_yaml(args.config),
        tracker_name=args.tracker_name,
        results_dir=args.results_dir,
    )
    report = run_eval_plan(plan, use_symlinks=not args.copy_instead_of_symlink)
    print_summary(report)


if __name__ == "__main__":
    main()
