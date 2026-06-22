"""ReID interface and factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Person ReID standard input size (height, width).
REID_PATCH_SHAPE = (256, 128)

REID_MODEL_NAMES = ("osnet", "resnet50", "resnet50_ibn", "fastreid")
REID_SOURCES = {
    "osnet": "torchreid",
    "resnet50": "torchreid",
    "resnet50_ibn": "fast-reid",
    "fastreid": "fast-reid",
}


class ReIDExtractor(ABC):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract(self, frame: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def extract_image_patch(
    image: np.ndarray,
    bbox: np.ndarray,
    patch_shape: tuple[int, int] = REID_PATCH_SHAPE,
) -> np.ndarray | None:
    bbox = np.asarray(bbox, dtype=np.float64).copy()
    target_aspect = float(patch_shape[1]) / patch_shape[0]
    new_width = target_aspect * bbox[3]
    bbox[0] -= (new_width - bbox[2]) / 2
    bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None

    sx, sy, ex, ey = bbox
    patch = image[sy:ey, sx:ex]
    import cv2

    return cv2.resize(patch, (patch_shape[1], patch_shape[0]))


def crop_person_patches(
    frame: np.ndarray,
    boxes_tlwh: np.ndarray,
    *,
    patch_shape: tuple[int, int] = REID_PATCH_SHAPE,
) -> list[np.ndarray]:
    boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64)
    if boxes_tlwh.size == 0:
        return []
    patches: list[np.ndarray] = []
    for box in boxes_tlwh:
        patch = extract_image_patch(frame, box, patch_shape)
        if patch is not None:
            patches.append(patch)
    return patches


def crop_person_patches_with_mask(
    frame: np.ndarray,
    boxes_tlwh: np.ndarray,
    *,
    patch_shape: tuple[int, int] = REID_PATCH_SHAPE,
) -> tuple[list[np.ndarray], np.ndarray]:
    boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64)
    mask = np.zeros(len(boxes_tlwh), dtype=bool)
    patches: list[np.ndarray] = []
    for i, box in enumerate(boxes_tlwh):
        patch = extract_image_patch(frame, box, patch_shape)
        if patch is not None:
            patches.append(patch)
            mask[i] = True
    return patches, mask


def scatter_features(
    encoded: np.ndarray,
    mask: np.ndarray,
    feature_dim: int,
) -> np.ndarray:
    full = np.zeros((len(mask), feature_dim), dtype=np.float32)
    if encoded.size:
        full[mask] = encoded
    return full


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features.reshape(0, 0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def create_reid_extractor(name: str, cfg: dict) -> ReIDExtractor:
    key = name.lower()
    cfg = dict(cfg)
    if key in ("osnet", "resnet50"):
        from reid.torchreid_ext import TorchReIDExtractor

        defaults = {
            "osnet": "osnet_x1_0",
            "resnet50": "resnet50",
        }
        model_name = cfg.pop("model_name", defaults[key])
        return TorchReIDExtractor(model_name=model_name, **cfg)
    if key == "resnet50_ibn":
        from reid.fastreid_ext import FastReIDExtractor

        default_config = ROOT / "third_party" / "fast_reid" / "configs" / "Market1501" / "bagtricks_R50-ibn.yml"
        default_checkpoint = ROOT / "resources" / "models" / "fastreid" / "market_bot_R50-ibn.pth"
        config = cfg.pop("config", str(default_config))
        checkpoint = cfg.pop("checkpoint", str(default_checkpoint))
        return FastReIDExtractor(config=config, checkpoint=checkpoint, **cfg)
    if key == "fastreid":
        from reid.fastreid_ext import FastReIDExtractor

        return FastReIDExtractor(**cfg)
    raise ValueError(
        f"Unknown ReID model: {name}. Choose one of: {', '.join(REID_MODEL_NAMES)}."
    )
