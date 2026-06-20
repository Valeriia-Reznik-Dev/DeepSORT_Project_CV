"""Run detector + ReID through the original DeepSORT core on a MOT sequence.

This replaces the precomputed ``det.txt`` + ``.npy`` inputs of the upstream
``deep_sort_app`` with live detections and appearance descriptors, while keeping
the original Kalman filter / matching cascade / Tracker untouched.
"""
from __future__ import annotations

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
from reid.base import ReIDExtractor
from segmentation.base import Segmenter, apply_background_removal


@dataclass(frozen=True)
class TrackerParams:
    """DeepSORT tracker parameters (per-video tunable)."""

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


def track_sequence(
    detector: Detector,
    reid: ReIDExtractor,
    sequence_dir: str | Path,
    output_file: str | Path,
    *,
    params: TrackerParams | None = None,
    max_frames: int | None = None,
    mask_background: bool = False,
) -> dict[str, float]:
    """Track one MOT sequence and write MOTChallenge results.

    Returns timing stats (FPS and per-stage ms), measured end-to-end excluding
    the first (warmup) frame and image I/O for the FPS figure.

    If ``mask_background`` is True and ``detector`` is a :class:`Segmenter`,
    instance masks are used to zero out background pixels before ReID feature
    extraction (cleaner appearance descriptors).
    """
    params = params or TrackerParams()
    use_masks = mask_background and isinstance(detector, Segmenter)
    sequence_dir = Path(sequence_dir)
    frames = _sequence_frames(sequence_dir)
    if max_frames is not None:
        frames = frames[:max_frames]

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", params.max_cosine_distance, params.nn_budget
    )
    tracker = Tracker(metric)
    results: list[list[float]] = []

    det_time = reid_time = track_time = pipe_time = 0.0
    timed_frames = 0

    for i, (frame_idx, img_path) in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        reid_frame = frame
        t0 = time.perf_counter()
        if use_masks:
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

        if i > 0:  # skip warmup frame (lazy CUDA/model init)
            det_time += t1 - t0
            reid_time += t3 - t2
            track_time += t4 - t3
            pipe_time += (t1 - t0) + (t3 - t2) + (t4 - t3)
            timed_frames += 1

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            results.append([frame_idx, track.track_id, x, y, w, h])

    _write_mot(Path(output_file), results)

    def _fps(t: float) -> float:
        return timed_frames / t if t > 0 else 0.0

    def _ms(t: float) -> float:
        return 1000.0 * t / timed_frames if timed_frames else 0.0

    return {
        "frames": float(len(frames)),
        "fps": _fps(pipe_time),
        "det_ms": _ms(det_time),
        "reid_ms": _ms(reid_time),
        "track_ms": _ms(track_time),
    }
