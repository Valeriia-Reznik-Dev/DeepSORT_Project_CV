"""NanoDet-Plus detector (RangiLyu/nanodet)."""
from __future__ import annotations

import cv2
import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh

COCO_PERSON_CLASS = 0


class NanoDetDetector(Detector):
    """Requires: pip install git+https://github.com/RangiLyu/nanodet.git"""

    def __init__(
        self,
        config: str,
        checkpoint: str,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        from nanodet.util import cfg, load_config
        from nanodet.model.arch import build_model
        from nanodet.util import load_model_weight
        import torch

        self.conf_threshold = conf_threshold
        self.device = torch.device(device)
        load_config(cfg, config)
        self.model = build_model(cfg.model)
        checkpoint = load_model_weight(self.model, checkpoint, device=self.device)
        self.model = self.model.to(self.device).eval()
        self.input_size = cfg.data.train.input_size
        self.class_names = cfg.class_names

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        from nanodet.data.transform import Pipeline

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        meta = {
            "raw_img": img,
            "img": img,
            "img_info": {
                "height": img.shape[0],
                "width": img.shape[1],
                "id": 0,
            },
        }
        meta = Pipeline(None, self.input_size, keep_ratio=True)(meta)
        meta["img"] = meta["img"].transpose(2, 0, 1)

        import torch

        with torch.no_grad():
            results = self.model.inference(meta)[0]

        detections: list[DetectionResult] = []
        person_label = self.class_names[COCO_PERSON_CLASS]
        for box in results.get(person_label, []):
            if box[-1] < self.conf_threshold:
                continue
            x1, y1, x2, y2, score = box
            tlwh = xyxy_to_tlwh(np.array([[x1, y1, x2, y2]], dtype=np.float64))[0]
            detections.append(DetectionResult(tlwh=tlwh, confidence=float(score)))
        return detections
