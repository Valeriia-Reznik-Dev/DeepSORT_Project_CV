"""Segmenter interface and mask helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegResult:
    tlwh: np.ndarray
    confidence: float
    mask: np.ndarray  # bool array of shape (H, W) aligned to the frame


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray) -> list[SegResult]:
        raise NotImplementedError


def union_mask(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        out |= m
    return out


def apply_background_removal(
    frame: np.ndarray, masks: list[np.ndarray]
) -> np.ndarray:
    if not masks:
        return frame
    keep = union_mask(masks, frame.shape[:2])
    out = frame.copy()
    out[~keep] = 0
    return out


def create_segmenter(name: str, cfg: dict) -> Segmenter:
    key = name.lower()
    if key == "yolo_seg":
        from segmentation.yolo_seg import YoloSegDetector

        return YoloSegDetector(**cfg)
    if key == "detectron2_seg":
        from segmentation.detectron2_seg import Detectron2SegDetector

        return Detectron2SegDetector(**cfg)
    if key == "smp_seg":
        from segmentation.smp_seg import SmpSegDetector

        return SmpSegDetector(**cfg)
    raise ValueError(f"Unknown segmenter: {name}. Choose yolo_seg, detectron2_seg, or smp_seg.")
