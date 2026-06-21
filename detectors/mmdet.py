"""RTMDet / MMDetection detector."""
from __future__ import annotations

import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh

COCO_PERSON_CLASS = 0


class MMDetDetector(Detector):
    def __init__(
        self,
        config: str,
        checkpoint: str,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        from mmdet.apis import init_detector

        self.conf_threshold = conf_threshold
        self.model = init_detector(config, checkpoint, device=device)

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        from mmdet.apis import inference_detector

        result = inference_detector(self.model, frame)
        pred = result.pred_instances
        keep = (pred.labels == COCO_PERSON_CLASS) & (pred.scores >= self.conf_threshold)
        if not keep.any():
            return []

        boxes = pred.bboxes[keep].cpu().numpy()
        scores = pred.scores[keep].cpu().numpy()
        tlwh = xyxy_to_tlwh(boxes)
        return [
            DetectionResult(tlwh=tlwh[i], confidence=float(scores[i]))
            for i in range(len(scores))
        ]
