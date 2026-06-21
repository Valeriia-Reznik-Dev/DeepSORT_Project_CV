"""Live detections + ReID through DeepSORT core."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors.base import DetectionResult, Detector
from eval.detector_metrics import _sequence_frames
from identity.manager import IdentityManager
from reid.base import ReIDExtractor
from segmentation.base import Segmenter, apply_background_removal


@dataclass(frozen=True)
class TrackerParams:
    min_confidence: float = 0.3
    min_detection_height: int = 0
    nms_max_overlap: float = 1.0
    max_cosine_distance: float = 0.2
    nn_budget: int | None = 100


def _write_mot(output_file: Path, results: list[list[float]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for frame_idx, track_id, x, y, w, h in results:
            f.write(
                "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n"
                % (frame_idx, track_id, x, y, w, h)
            )


def _write_identity_sidecar(
    output_file: Path,
    rows: list[tuple[int, int, int, int]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "identity_raw", "identity_resolved"])
        writer.writerows(rows)


def _active_track_descriptors(
    tracker: Tracker,
    reid_frame: np.ndarray,
    *,
    identity_reid: ReIDExtractor | None,
) -> list[tuple[int, np.ndarray]]:
    entries: list[tuple[int, np.ndarray]] = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update != 0:
            continue
        if identity_reid is not None:
            box = track.to_tlwh().reshape(1, 4)
            desc = identity_reid.extract(reid_frame, box)[0]
        elif track.features:
            desc = track.features[-1]
        else:
            continue
        entries.append((track.track_id, np.asarray(desc, dtype=np.float32)))
    return entries


def track_sequence(
    detector: Detector,
    reid: ReIDExtractor,
    sequence_dir: str | Path,
    output_file: str | Path,
    *,
    params: TrackerParams | None = None,
    max_frames: int | None = None,
    mask_background: bool = False,
    gt_detections: dict[int, np.ndarray] | None = None,
    identity_manager: IdentityManager | None = None,
    identity_reid: ReIDExtractor | None = None,
    identity_output: str | Path | None = None,
) -> dict[str, float]:
    params = params or TrackerParams()
    use_gt = gt_detections is not None
    use_masks = mask_background and isinstance(detector, Segmenter) and not use_gt
    sequence_dir = Path(sequence_dir)
    frames = _sequence_frames(sequence_dir)
    if max_frames is not None:
        frames = frames[:max_frames]

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", params.max_cosine_distance, params.nn_budget
    )
    tracker = Tracker(metric)
    results: list[list[float]] = []
    identity_rows: list[tuple[int, int, int, int]] = []

    det_time = reid_time = track_time = identity_time = pipe_time = 0.0
    timed_frames = 0

    for i, (frame_idx, img_path) in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        reid_frame = frame
        t0 = time.perf_counter()
        if use_gt:
            gt_boxes = gt_detections.get(frame_idx, np.zeros((0, 4)))
            raw = [DetectionResult(np.asarray(b, dtype=np.float64), 1.0) for b in gt_boxes]
        elif use_masks:
            segs = detector.segment(frame)
            raw = [DetectionResult(s.tlwh, s.confidence) for s in segs]
            reid_frame = apply_background_removal(frame, [s.mask for s in segs])
        else:
            raw = detector.detect(frame)
        t1 = time.perf_counter()

        dets = [
            d
            for d in raw
            if d.confidence >= params.min_confidence
            and d.tlwh[3] >= params.min_detection_height
        ]
        boxes = (
            np.array([d.tlwh for d in dets], dtype=np.float64)
            if dets
            else np.zeros((0, 4), dtype=np.float64)
        )

        t2 = time.perf_counter()
        features = reid.extract(reid_frame, boxes)
        t3 = time.perf_counter()

        detections = [
            Detection(dets[j].tlwh, dets[j].confidence, features[j])
            for j in range(len(dets))
        ]
        if detections:
            box_arr = np.array([d.tlwh for d in detections])
            score_arr = np.array([d.confidence for d in detections])
            keep = preprocessing.non_max_suppression(
                box_arr, params.nms_max_overlap, score_arr
            )
            detections = [detections[k] for k in keep]

        tracker.predict()
        tracker.update(detections)
        t4 = time.perf_counter()

        t5 = t4
        if identity_manager is not None:
            id_entries = _active_track_descriptors(
                tracker, reid_frame, identity_reid=identity_reid
            )
            resolved, raw_ids = identity_manager.update(frame_idx, id_entries)
            for track_id, _ in id_entries:
                identity_rows.append(
                    (
                        frame_idx,
                        track_id,
                        raw_ids[track_id],
                        resolved[track_id],
                    )
                )
            t5 = time.perf_counter()

        if i > 0:
            det_time += t1 - t0
            reid_time += t3 - t2
            track_time += t4 - t3
            identity_time += t5 - t4
            pipe_time += (t1 - t0) + (t3 - t2) + (t4 - t3) + (t5 - t4)
            timed_frames += 1

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            results.append([frame_idx, track.track_id, x, y, w, h])

    _write_mot(Path(output_file), results)
    if identity_manager is not None and identity_output is not None:
        _write_identity_sidecar(Path(identity_output), identity_rows)

    def _fps(t: float) -> float:
        return timed_frames / t if t > 0 else 0.0

    def _ms(t: float) -> float:
        return 1000.0 * t / timed_frames if timed_frames else 0.0

    stats = {
        "frames": float(len(frames)),
        "fps": _fps(pipe_time),
        "det_ms": _ms(det_time),
        "reid_ms": _ms(reid_time),
        "track_ms": _ms(track_time),
    }
    if identity_manager is not None:
        stats["identity_ms"] = _ms(identity_time)
        stats["num_identities"] = float(identity_manager.db.num_identities)
    return stats
