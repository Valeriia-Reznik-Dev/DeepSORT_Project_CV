"""YOLOv8 detector."""
from __future__ import annotations

import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh

COCO_PERSON_CLASS = 0


class YoloDetector(Detector):
    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS],
            device=self.device,
            verbose=False,
        )[0]
        if results.boxes is None or len(results.boxes) == 0:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        tlwh = xyxy_to_tlwh(boxes)
        return [
            DetectionResult(tlwh=tlwh[i], confidence=float(confs[i]))
            for i in range(len(confs))
        ]
