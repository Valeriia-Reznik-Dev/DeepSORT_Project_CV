"""Detector interface and bbox helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    tlwh: np.ndarray
    confidence: float


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        raise NotImplementedError


def xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def tlwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def create_detector(name: str, cfg: dict) -> Detector:
    key = name.lower()
    if key == "yolo":
        from detectors.yolo import YoloDetector

        return YoloDetector(**cfg)
    if key == "nanodet":
        from detectors.nanodet import NanoDetDetector

        return NanoDetDetector(**cfg)
    if key == "mmdet":
        from detectors.mmdet import MMDetDetector

        return MMDetDetector(**cfg)
    if key in ("yolo_seg", "detectron2_seg", "smp_seg"):
        from segmentation.base import create_segmenter

        return create_segmenter(key, cfg)
    raise ValueError(
        f"Unknown detector: {name}. Choose yolo, nanodet, mmdet, "
        "yolo_seg, detectron2_seg, or smp_seg."
    )
