#!/usr/bin/env python3
"""Sweep one identity-DB parameter (FM / ARI / V)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.identity_metrics import evaluate_identity  # noqa: E402
from identity.manager import IdentityParams  # noqa: E402
from reid.base import REID_MODEL_NAMES, create_reid_extractor  # noqa: E402

INT_PARAMS = {"k", "window", "max_per_identity"}
FLOAT_PARAMS = {"radius"}
STR_PARAMS = {"representation", "conflict_policy"}
SWEEPABLE = sorted(INT_PARAMS | FLOAT_PARAMS | STR_PARAMS)


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


def build_jobs(project_cfg: dict) -> list[tuple[str, Path, Path, bool]]:
    paths = project_cfg["paths"]
    jobs: list[tuple[str, Path, Path, bool]] = []
    for name in project_cfg["sequences"]["mot15"]:
        jobs.append((
            name,
            Path(paths["mot15_dir"]) / name,
            Path(paths["mot15_dir"]) / name / "gt" / "gt.txt",
            False,
        ))
    for name in project_cfg["sequences"]["mot16"]:
        jobs.append((
            name,
            Path(paths["mot16_dir"]) / name,
            Path(paths["mot16_dir"]) / name / "gt" / "gt.txt",
            True,
        ))
    return jobs


def _cast(param: str, value: str):
    if param in INT_PARAMS:
        return int(float(value))
    if param in FLOAT_PARAMS:
        return float(value)
    return value


def _aggregate(report: dict, labels: str) -> dict[str, float]:
    seqs = report["sequences"].values()
    fm = float(np.mean([s[labels]["fowlkes_mallows"] for s in seqs]))
    ari = float(np.mean([s[labels]["adjusted_rand"] for s in seqs]))
    vmeasure = float(np.mean([s[labels]["v_measure"] for s in seqs]))
    pred_ids = float(np.mean([s[labels]["n_pred_ids"] for s in seqs]))
    true_ids = float(np.mean([s[labels]["n_true_ids"] for s in seqs]))
    return {"fm": fm, "ari": ari, "v": vmeasure, "pred_ids": pred_ids, "true_ids": true_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep one identity-DB parameter")
    parser.add_argument("--param", required=True, choices=SWEEPABLE)
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--reid", default="osnet", choices=list(REID_MODEL_NAMES))
    parser.add_argument("--project-config", default=os.path.join(ROOT, "configs", "baseline_original.yaml"))
    parser.add_argument("--reid-config", default=os.path.join(ROOT, "configs", "reid.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=os.path.join(ROOT, "results", "identity_sweep"))
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    project_cfg = load_yaml(args.project_config)
    reid_cfg = _resolve_device(load_yaml(args.reid_config)["reid_models"][args.reid], args.device)

    print(f"Loading ReID: {args.reid} ...")
    extractor = create_reid_extractor(args.reid, reid_cfg)
    jobs = build_jobs(project_cfg)
    base = IdentityParams()

    rows: list[dict] = []
    for raw_value in args.values:
        value = _cast(args.param, raw_value)
        params = replace(base, **{args.param: value})
        print(f"\n=== {args.param} = {value} ===")
        report = evaluate_identity(
            extractor, args.reid, jobs, params=params, max_frames=args.max_frames
        )
        rows.append({
            "value": value,
            "resolved": _aggregate(report, "resolved"),
            "db_raw": _aggregate(report, "db_raw"),
        })

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print(f"IDENTITY-DB PARAMETER EVOLUTION: {args.param}  (reid={args.reid})")
    print("=" * 92)
    print(
        f"{args.param:<16} {'res_FM':>8} {'res_ARI':>8} {'res_V':>8} "
        f"{'raw_FM':>8} {'predID':>8} {'trueID':>8}"
    )
    print("-" * 72)
    for r in rows:
        rr, raw = r["resolved"], r["db_raw"]
        print(
            f"{str(r['value']):<16} {rr['fm']:8.3f} {rr['ari']:8.3f} {rr['v']:8.3f} "
            f"{raw['fm']:8.3f} {rr['pred_ids']:8.1f} {rr['true_ids']:8.1f}"
        )
    print("-" * 72)
    best = max(rows, key=lambda r: r["resolved"]["fm"])
    print(f"BEST {args.param} by resolved FM: {best['value']} (FM {best['resolved']['fm']:.3f})")

    # ---- Save ----
    os.makedirs(args.output_root, exist_ok=True)
    csv_path = os.path.join(args.output_root, f"sweep_{args.param}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            args.param, "resolved_fm", "resolved_ari", "resolved_v",
            "db_raw_fm", "pred_ids", "true_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                args.param: r["value"],
                "resolved_fm": round(r["resolved"]["fm"], 3),
                "resolved_ari": round(r["resolved"]["ari"], 3),
                "resolved_v": round(r["resolved"]["v"], 3),
                "db_raw_fm": round(r["db_raw"]["fm"], 3),
                "pred_ids": round(r["resolved"]["pred_ids"], 1),
                "true_ids": round(r["resolved"]["true_ids"], 1),
            })
    json_path = os.path.join(args.output_root, f"sweep_{args.param}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"param": args.param, "reid": args.reid, "rows": rows}, f, indent=2)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
