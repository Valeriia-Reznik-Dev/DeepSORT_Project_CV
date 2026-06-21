"""NanoDet-Plus detector."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh

COCO_PERSON_CLASS = 0
NANO_DIR = Path(__file__).resolve().parents[1] / "third_party" / "nanodet"


def _ensure_nanodet_on_path() -> None:
    if not (NANO_DIR / "nanodet" / "data").is_dir():
        raise FileNotFoundError(
            f"NanoDet not found at {NANO_DIR}. "
            "Run: python scripts/setup_detectors_colab.py --nanodet-only"
        )
    root = str(NANO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


class NanoDetDetector(Detector):
    def __init__(
        self,
        config: str,
        checkpoint: str,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        _ensure_nanodet_on_path()
        import torch
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate
        from nanodet.data.transform import Pipeline
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg, load_config, load_model_weight

        self.conf_threshold = conf_threshold
        self.device = device
        load_config(cfg, config)
        self.cfg = cfg
        logger = Logger(0, use_tensorboard=False)
        model = build_model(cfg.model)
        ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.input_size = cfg.data.val.input_size
        self.class_names = cfg.class_names
        self._collate = naive_collate
        self._stack = stack_batch_img

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        import torch

        img = frame
        h, w = img.shape[:2]
        meta = {
            "img_info": {"id": 0, "file_name": None, "height": h, "width": w},
            "raw_img": img,
            "img": img,
        }
        meta = self.pipeline(None, meta, self.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = self._collate([meta])
        meta["img"] = self._stack(meta["img"], divisible=32)


        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            results = self.model.inference(meta)

        dets = results[0]
        detections: list[DetectionResult] = []
        for box in dets.get(COCO_PERSON_CLASS, []):
            if box[-1] < self.conf_threshold:
                continue
            x1, y1, x2, y2, score = box
            tlwh = xyxy_to_tlwh(np.array([[x1, y1, x2, y2]], dtype=np.float64))[0]
            detections.append(DetectionResult(tlwh=tlwh, confidence=float(score)))
        return detections
