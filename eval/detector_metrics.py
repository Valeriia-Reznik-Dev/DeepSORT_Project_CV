"""Detector quality metrics: Precision / Recall / F1 vs MOT GT."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from deep_sort.iou_matching import iou
from detectors.base import Detector

MOT16_PEDESTRIAN_CLASS = 1


def load_mot_gt(gt_path: str | Path, *, is_mot16: bool) -> dict[int, list[np.ndarray]]:
    """Load GT boxes per frame (1-indexed). Returns frame -> list of tlwh."""
    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame = int(float(parts[0]))
        marked = float(parts[6])
        if marked == 0:
            continue
        if is_mot16:
            if int(float(parts[7])) != MOT16_PEDESTRIAN_CLASS:
                continue
            if float(parts[8]) <= 0:
                continue
        x, y, w, h = map(float, parts[2:6])
        by_frame[frame].append(np.array([x, y, w, h], dtype=np.float64))
    return dict(by_frame)


def _match_frame(
    detections: list[np.ndarray],
    ground_truth: list[np.ndarray],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Return TP, FP, FN for one frame via IoU + Hungarian assignment."""
    n_det, n_gt = len(detections), len(ground_truth)
    if n_det == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_det, 0

    det_arr = np.asarray(detections)
    gt_arr = np.asarray(ground_truth)
    iou_matrix = np.zeros((n_gt, n_det), dtype=np.float64)
    for i, gt_box in enumerate(gt_arr):
        iou_matrix[i] = iou(gt_box, det_arr)

    cost = 1.0 - iou_matrix
    cost[iou_matrix < iou_threshold] = 1e6
    gt_idx, det_idx = linear_sum_assignment(cost)
    tp = int(np.sum(iou_matrix[gt_idx, det_idx] >= iou_threshold))
    fp = n_det - tp
    fn = n_gt - tp
    return tp, fp, fn


def _sequence_frames(sequence_dir: Path) -> list[tuple[int, Path]]:
    img_dir = sequence_dir / "img1"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing img1: {img_dir}")
    frames: list[tuple[int, Path]] = []
    for path in sorted(img_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            frame_idx = int(path.stem)
            frames.append((frame_idx, path))
    return frames


def evaluate_detector_on_sequence(
    detector: Detector,
    sequence_dir: str | Path,
    gt_path: str | Path,
    *,
    is_mot16: bool,
    iou_threshold: float = 0.5,
    max_frames: int | None = None,
) -> dict[str, float]:
    """Run detector on sequence frames and compute aggregate P/R/F1."""
    sequence_dir = Path(sequence_dir)
    gt_by_frame = load_mot_gt(gt_path, is_mot16=is_mot16)
    frames = _sequence_frames(sequence_dir)
    if max_frames is not None:
        frames = frames[:max_frames]

    tp_total = fp_total = fn_total = 0
    for frame_idx, img_path in frames:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        dets = detector.detect(frame)
        det_boxes = [d.tlwh for d in dets]
        gt_boxes = gt_by_frame.get(frame_idx, [])
        tp, fp, fn = _match_frame(det_boxes, gt_boxes, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "frames": len(frames),
    }


def evaluate_detector(
    detector: Detector,
    detector_name: str,
    jobs: list[tuple[str, Path, Path, bool]],
    *,
    iou_threshold: float = 0.5,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Evaluate one detector on multiple sequences."""
    report: dict[str, Any] = {"detector": detector_name, "sequences": {}, "iou_threshold": iou_threshold}
    f1_values: list[float] = []

    for seq_name, sequence_dir, gt_path, is_mot16 in jobs:
        print(f"  {seq_name} ...")
        scores = evaluate_detector_on_sequence(
            detector,
            sequence_dir,
            gt_path,
            is_mot16=is_mot16,
            iou_threshold=iou_threshold,
            max_frames=max_frames,
        )
        report["sequences"][seq_name] = scores
        f1_values.append(scores["f1"])

    report["f1_mean"] = float(np.mean(f1_values)) if f1_values else 0.0
    return report


def save_detector_report(reports: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    for report in reports:
        name = report["detector"]
        for seq, scores in report["sequences"].items():
            rows.append({"detector": name, "sequence": seq, **scores})
        rows.append(
            {
                "detector": name,
                "sequence": "MEAN",
                "precision": "",
                "recall": "",
                "f1": report["f1_mean"],
                "tp": "",
                "fp": "",
                "fn": "",
                "frames": "",
            }
        )

    fieldnames = ["detector", "sequence", "precision", "recall", "f1", "tp", "fp", "fn", "frames"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return {"csv": str(summary_csv), "json": str(summary_json)}


def print_detector_summary(reports: list[dict[str, Any]]) -> None:
    print("\n=== Detector F1 summary ===")
    for report in reports:
        print(f"\nDetector: {report['detector']}")
        print(f"{'Sequence':<20} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 48)
        for seq, scores in sorted(report["sequences"].items()):
            print(
                f"{seq:<20} {scores['precision']:8.3f} "
                f"{scores['recall']:8.3f} {scores['f1']:8.3f}"
            )
        print("-" * 48)
        n = len(report["sequences"])
        print(f"{'MEAN (' + str(n) + ' videos)':<20} {'':>8} {'':>8} {report['f1_mean']:8.3f}")
