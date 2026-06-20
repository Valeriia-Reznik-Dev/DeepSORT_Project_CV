"""Segmenter interface and mask helpers.

A ``Segmenter`` produces person boxes together with a full-frame boolean
instance mask. Masks are used to remove background clutter from ReID crops
(cleaner appearance descriptors). Every concrete segmenter here also implements
:class:`detectors.base.Detector`, so it is selectable wherever a detector is.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegResult:
    """Person instance: tlwh box, confidence and a full-frame boolean mask."""

    tlwh: np.ndarray
    confidence: float
    mask: np.ndarray  # bool array of shape (H, W) aligned to the frame


class Segmenter(ABC):
    """Abstract person segmenter: frame (BGR) -> list of SegResult."""

    @abstractmethod
    def segment(self, frame: np.ndarray) -> list[SegResult]:
        raise NotImplementedError


def union_mask(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Boolean OR of all instance masks (foreground = any person)."""
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        out |= m
    return out


def apply_background_removal(
    frame: np.ndarray, masks: list[np.ndarray]
) -> np.ndarray:
    """Return a copy of ``frame`` with all non-person pixels set to zero.

    Using the union of person masks keeps every person (so box crops still
    contain their subject) while stripping background, which yields cleaner
    ReID descriptors. If no masks are given, the frame is returned unchanged.
    """
    if not masks:
        return frame
    keep = union_mask(masks, frame.shape[:2])
    out = frame.copy()
    out[~keep] = 0
    return out


def create_segmenter(name: str, cfg: dict) -> Segmenter:
    """Factory: yolo_seg."""
    key = name.lower()
    if key == "yolo_seg":
        from segmentation.yolo_seg import YoloSegDetector

        return YoloSegDetector(**cfg)
    raise ValueError(f"Unknown segmenter: {name}. Choose yolo_seg.")
