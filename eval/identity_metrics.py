"""Standalone identity-DB evaluation on MOT GT crops (label-based metrics).

Feeds GT pedestrian crops frame-by-frame into the online IdentityManager (GT
track_id plays the role of the tracker's track), then compares the assigned
identities to the GT ids with clustering metrics. Two label sets are reported:
``db_raw`` (per-crop DB identity) and ``resolved`` (after window vote + conflict).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)

from eval.detector_metrics import _sequence_frames
from eval.reid_metrics import load_mot_gt_with_ids
from identity.manager import IdentityManager, IdentityParams
from reid.base import ReIDExtractor

_ZERO = {
    "fowlkes_mallows": 0.0,
    "adjusted_rand": 0.0,
    "v_measure": 0.0,
    "homogeneity": 0.0,
    "completeness": 0.0,
    "n_true_ids": 0.0,
    "n_pred_ids": 0.0,
    "n_samples": 0.0,
}


def _label_metrics(true: list[int], pred: list[int]) -> dict[str, float]:
    true_arr = np.asarray(true)
    pred_arr = np.asarray(pred)
    if len(true_arr) < 2 or len(np.unique(true_arr)) < 2:
        out = dict(_ZERO)
        out["n_samples"] = float(len(true_arr))
        out["n_true_ids"] = float(len(np.unique(true_arr))) if len(true_arr) else 0.0
        out["n_pred_ids"] = float(len(np.unique(pred_arr))) if len(pred_arr) else 0.0
        return out

    homo, comp, vmeasure = homogeneity_completeness_v_measure(true_arr, pred_arr)
    return {
        "fowlkes_mallows": float(fowlkes_mallows_score(true_arr, pred_arr)),
        "adjusted_rand": float(adjusted_rand_score(true_arr, pred_arr)),
        "v_measure": float(vmeasure),
        "homogeneity": float(homo),
        "completeness": float(comp),
        "n_true_ids": float(len(np.unique(true_arr))),
        "n_pred_ids": float(len(np.unique(pred_arr))),
        "n_samples": float(len(true_arr)),
    }


def evaluate_identity_on_sequence(
    extractor: ReIDExtractor,
    sequence_dir: str | Path,
    gt_path: str | Path,
    *,
    is_mot16: bool,
    params: IdentityParams,
    max_frames: int | None = None,
) -> dict[str, dict[str, float]]:
    gt_by_frame = load_mot_gt_with_ids(gt_path, is_mot16=is_mot16)
    frames = _sequence_frames(Path(sequence_dir))
    if max_frames is not None:
        frames = frames[:max_frames]

    manager = IdentityManager(params)
    true: list[int] = []
    pred_raw: list[int] = []
    pred_res: list[int] = []

    for frame_idx, img_path in frames:
        entries = gt_by_frame.get(frame_idx)
        if not entries:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        boxes = np.stack([e[0] for e in entries])
        ids = [e[1] for e in entries]
        feats = extractor.extract(frame, boxes)
        if feats.shape[0] != len(ids):
            continue

        detections = [(ids[j], feats[j]) for j in range(len(ids))]
        resolved, raw = manager.update(frame_idx, detections)
        for tid in ids:
            true.append(tid)
            pred_raw.append(raw[tid])
            pred_res.append(resolved[tid])

    return {
        "db_raw": _label_metrics(true, pred_raw),
        "resolved": _label_metrics(true, pred_res),
    }


def evaluate_identity(
    extractor: ReIDExtractor,
    model_name: str,
    jobs: list[tuple[str, Path, Path, bool]],
    *,
    params: IdentityParams,
    max_frames: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model": model_name,
        "params": vars(params),
        "sequences": {},
    }
    fm_values: list[float] = []
    for seq_name, sequence_dir, gt_path, is_mot16 in jobs:
        print(f"  {seq_name} ...")
        scores = evaluate_identity_on_sequence(
            extractor,
            sequence_dir,
            gt_path,
            is_mot16=is_mot16,
            params=params,
            max_frames=max_frames,
        )
        report["sequences"][seq_name] = scores
        fm_values.append(scores["resolved"]["fowlkes_mallows"])
    report["resolved_fm_mean"] = float(np.mean(fm_values)) if fm_values else 0.0
    return report


def save_identity_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    fieldnames = [
        "model",
        "sequence",
        "labels",
        "fowlkes_mallows",
        "adjusted_rand",
        "v_measure",
        "homogeneity",
        "completeness",
        "n_true_ids",
        "n_pred_ids",
        "n_samples",
    ]
    rows: list[dict[str, Any]] = []
    for seq, label_sets in report["sequences"].items():
        for labels_name, scores in label_sets.items():
            rows.append(
                {
                    "model": report["model"],
                    "sequence": seq,
                    "labels": labels_name,
                    **scores,
                }
            )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"csv": str(summary_csv), "json": str(summary_json)}


def print_identity_summary(report: dict[str, Any]) -> None:
    print("\n=== Identity DB summary (GT crops) ===")
    print(f"Model: {report['model']} | params: {report['params']}")
    print(
        f"{'Sequence':<18} {'labels':<8} {'FM':>7} {'ARI':>7} {'V':>7} "
        f"{'trueID':>7} {'predID':>7} {'N':>6}"
    )
    print("-" * 70)
    for seq, label_sets in sorted(report["sequences"].items()):
        for labels_name, s in label_sets.items():
            print(
                f"{seq:<18} {labels_name:<8} {s['fowlkes_mallows']:7.3f} "
                f"{s['adjusted_rand']:7.3f} {s['v_measure']:7.3f} "
                f"{int(s['n_true_ids']):7d} {int(s['n_pred_ids']):7d} "
                f"{int(s['n_samples']):6d}"
            )
    print("-" * 70)
    print(f"MEAN resolved FM ({len(report['sequences'])} videos): {report['resolved_fm_mean']:.3f}")
