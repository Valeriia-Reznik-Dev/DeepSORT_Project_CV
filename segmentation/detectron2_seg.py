"""Mask R-CNN segmenter (detectron2)."""
from __future__ import annotations

import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh
from segmentation.base import SegResult, Segmenter

COCO_PERSON_CLASS = 0


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    import cv2

    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


class Detectron2SegDetector(Detector, Segmenter):
    def __init__(
        self,
        config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        weights: str | None = None,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise ImportError(
                "detectron2 is not installed. Run: python scripts/setup_segmentation_colab.py"
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))
        cfg.MODEL.WEIGHTS = weights or model_zoo.get_checkpoint_url(config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.DEVICE = device if device != "auto" else ("cuda" if _cuda_available() else "cpu")
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(cfg)
        self.conf_threshold = conf_threshold

    def _predict(self, frame: np.ndarray):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.predictor(rgb)["instances"]

    def _to_segs(self, frame: np.ndarray, instances) -> list[SegResult]:
        if instances is None or len(instances) == 0:
            return []

        h, w = frame.shape[:2]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy()
        else:
            masks = None

        out: list[SegResult] = []
        for i in range(len(scores)):
            if int(classes[i]) != COCO_PERSON_CLASS:
                continue
            if float(scores[i]) < self.conf_threshold:
                continue
            x1, y1, x2, y2 = boxes[i]
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)
            if masks is not None:
                mask = _resize_mask(masks[i], h, w)
            else:
                mask = np.zeros((h, w), dtype=bool)
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
                mask[yi1:yi2, xi1:xi2] = True
            out.append(SegResult(tlwh=tlwh, confidence=float(scores[i]), mask=mask))
        return out

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        instances = self._predict(frame)
        return [
            DetectionResult(tlwh=s.tlwh, confidence=s.confidence)
            for s in self._to_segs(frame, instances)
        ]

    def segment(self, frame: np.ndarray) -> list[SegResult]:
        return self._to_segs(frame, self._predict(frame))


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
