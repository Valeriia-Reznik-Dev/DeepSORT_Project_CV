"""YOLOv8-seg detector."""
from __future__ import annotations

import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh
from segmentation.base import SegResult, Segmenter

COCO_PERSON_CLASS = 0


class YoloSegDetector(Detector, Segmenter):
    def __init__(
        self,
        model: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf_threshold = conf_threshold
        self.device = device

    def _predict(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=[COCO_PERSON_CLASS],
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self._predict(frame)
        if results.boxes is None or len(results.boxes) == 0:
            return []
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        tlwh = xyxy_to_tlwh(boxes)
        return [
            DetectionResult(tlwh=tlwh[i], confidence=float(confs[i]))
            for i in range(len(confs))
        ]

    def segment(self, frame: np.ndarray) -> list[SegResult]:
        results = self._predict(frame)
        if results.boxes is None or len(results.boxes) == 0:
            return []
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        tlwh = xyxy_to_tlwh(boxes)
        h, w = frame.shape[:2]

        if results.masks is not None:
            mask_data = results.masks.data.cpu().numpy().astype(bool)
        else:
            mask_data = np.ones((len(confs), h, w), dtype=bool)

        out: list[SegResult] = []
        for i in range(len(confs)):
            mask = mask_data[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            out.append(SegResult(tlwh=tlwh[i], confidence=float(confs[i]), mask=mask))
        return out
