"""ReID clustering metrics on GT crops."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    fowlkes_mallows_score,
    silhouette_score,
)

from eval.detector_metrics import _sequence_frames
from reid.base import ReIDExtractor

MOT16_PEDESTRIAN_CLASS = 1


def load_mot_gt_with_ids(
    gt_path: str | Path,
    *,
    is_mot16: bool,
) -> dict[int, list[tuple[np.ndarray, int]]]:
    gt_path = Path(gt_path)
    by_frame: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)

    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        marked = float(parts[6])
        if marked == 0:
            continue
        if is_mot16:
            if int(float(parts[7])) != MOT16_PEDESTRIAN_CLASS:
                continue
            if float(parts[8]) <= 0:
                continue
        x, y, w, h = map(float, parts[2:6])
        by_frame[frame].append((np.array([x, y, w, h], dtype=np.float64), track_id))

    return dict(by_frame)


def _subsample_stratified(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) <= max_samples:
        return features, labels

    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    per_id = max(1, max_samples // len(unique))
    keep_idx: list[int] = []

    for label in unique:
        idx = np.where(labels == label)[0]
        if len(idx) <= per_id:
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=per_id, replace=False).tolist())

    if len(keep_idx) > max_samples:
        keep_idx = rng.choice(np.array(keep_idx), size=max_samples, replace=False).tolist()

    keep_idx = np.array(sorted(keep_idx), dtype=np.int64)
    return features[keep_idx], labels[keep_idx]


def _clustering_metrics(
    features: np.ndarray,
    gt_labels: np.ndarray,
) -> dict[str, float]:
    n_samples = len(gt_labels)
    n_identities = len(np.unique(gt_labels))

    if n_samples < 2 or n_identities < 2:
        return {
            "silhouette": 0.0,
            "calinski_harabasz": 0.0,
            "fowlkes_mallows": 0.0,
            "n_samples": float(n_samples),
            "n_identities": float(n_identities),
        }

    sil = float(silhouette_score(features, gt_labels, metric="cosine"))
    ch = float(calinski_harabasz_score(features, gt_labels))

    kmeans = KMeans(n_clusters=n_identities, n_init=10, random_state=0)
    pred_labels = kmeans.fit_predict(features)
    fm = float(fowlkes_mallows_score(gt_labels, pred_labels))

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "fowlkes_mallows": fm,
        "n_samples": float(n_samples),
        "n_identities": float(n_identities),
    }


def collect_gt_features(
    extractor: ReIDExtractor,
    sequence_dir: str | Path,
    gt_path: str | Path,
    *,
    is_mot16: bool,
    max_frames: int | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_dir = Path(sequence_dir)
    gt_by_frame = load_mot_gt_with_ids(gt_path, is_mot16=is_mot16)
    frames = _sequence_frames(sequence_dir)
    if max_frames is not None:
        frames = frames[:max_frames]

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[int] = []

    for frame_idx, img_path in frames:
        entries = gt_by_frame.get(frame_idx)
        if not entries:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        boxes = np.stack([e[0] for e in entries])
        track_ids = [e[1] for e in entries]
        feats = extractor.extract(frame, boxes)
        if feats.shape[0] != len(track_ids):
            continue
        feature_chunks.append(feats)
        label_chunks.extend(track_ids)

    if not feature_chunks:
        dim = extractor.feature_dim
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    features = np.vstack(feature_chunks)
    labels = np.asarray(label_chunks, dtype=np.int64)

    if max_samples is not None:
        features, labels = _subsample_stratified(features, labels, max_samples, seed)

    return features, labels


def evaluate_reid_on_sequence(
    extractor: ReIDExtractor,
    sequence_dir: str | Path,
    gt_path: str | Path,
    *,
    is_mot16: bool,
    max_frames: int | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    features, labels = collect_gt_features(
        extractor,
        sequence_dir,
        gt_path,
        is_mot16=is_mot16,
        max_frames=max_frames,
        max_samples=max_samples,
        seed=seed,
    )
    return _clustering_metrics(features, labels)


def evaluate_reid(
    extractor: ReIDExtractor,
    model_name: str,
    jobs: list[tuple[str, Path, Path, bool]],
    *,
    max_frames: int | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    report: dict[str, Any] = {"model": model_name, "sequences": {}}
    fm_values: list[float] = []

    for seq_name, sequence_dir, gt_path, is_mot16 in jobs:
        print(f"  {seq_name} ...")
        scores = evaluate_reid_on_sequence(
            extractor,
            sequence_dir,
            gt_path,
            is_mot16=is_mot16,
            max_frames=max_frames,
            max_samples=max_samples,
            seed=seed,
        )
        report["sequences"][seq_name] = scores
        fm_values.append(scores["fowlkes_mallows"])

    report["fowlkes_mallows_mean"] = float(np.mean(fm_values)) if fm_values else 0.0
    return report


def save_reid_report(reports: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    for report in reports:
        name = report["model"]
        for seq, scores in report["sequences"].items():
            rows.append({"model": name, "sequence": seq, **scores})
        rows.append(
            {
                "model": name,
                "sequence": "MEAN",
                "silhouette": "",
                "calinski_harabasz": "",
                "fowlkes_mallows": report["fowlkes_mallows_mean"],
                "n_samples": "",
                "n_identities": "",
            }
        )

    fieldnames = [
        "model",
        "sequence",
        "silhouette",
        "calinski_harabasz",
        "fowlkes_mallows",
        "n_samples",
        "n_identities",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return {"csv": str(summary_csv), "json": str(summary_json)}


def print_reid_summary(reports: list[dict[str, Any]]) -> None:
    print("\n=== ReID clustering summary (GT crops) ===")
    for report in reports:
        print(f"\nModel: {report['model']}")
        print(
            f"{'Sequence':<20} {'Silh':>8} {'CH':>10} {'FM':>8} "
            f"{'N':>6} {'IDs':>5}"
        )
        print("-" * 62)
        for seq, scores in sorted(report["sequences"].items()):
            print(
                f"{seq:<20} {scores['silhouette']:8.3f} "
                f"{scores['calinski_harabasz']:10.1f} "
                f"{scores['fowlkes_mallows']:8.3f} "
                f"{int(scores['n_samples']):6d} "
                f"{int(scores['n_identities']):5d}"
            )
        print("-" * 62)
        n = len(report["sequences"])
        print(
            f"{'MEAN FM (' + str(n) + ' videos)':<20} {'':>8} {'':>10} "
            f"{report['fowlkes_mallows_mean']:8.3f}"
        )
