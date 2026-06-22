"""DeepLabV3+ segmenter (segmentation_models.pytorch)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from detectors.base import DetectionResult, Detector, xyxy_to_tlwh
from segmentation.base import SegResult, Segmenter

COCO_PERSON_CLASS = 1


def _load_smp_checkpoint(model, raw: dict) -> None:
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise ValueError("Unexpected SMP checkpoint format")
    mapped: dict[str, torch.Tensor] = {}
    model_keys = set(model.state_dict().keys())
    for key, value in state.items():
        candidates = [key]
        if key.startswith("backbone."):
            candidates.append("encoder." + key[len("backbone.") :])
        if key.startswith("decode_head."):
            candidates.append("segmentation_head." + key[len("decode_head.") :])
        for cand in candidates:
            if cand in model_keys:
                mapped[cand] = value
                break
    model.load_state_dict(mapped, strict=False)
    print(f"SMP: loaded {len(mapped)} tensors from mmseg-style checkpoint.")


class SmpSegDetector(Detector, Segmenter):
    def __init__(
        self,
        arch: str = "DeepLabV3Plus",
        encoder: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        classes: int = 19,
        person_class: int = 11,
        checkpoint: str | None = None,
        conf_threshold: float = 0.5,
        min_area: int = 400,
        device: str = "cpu",
    ):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.conf_threshold = float(conf_threshold)
        self.min_area = int(min_area)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._backend = "smp"
        self.person_class = int(person_class)

        ckpt_path = Path(checkpoint) if checkpoint else None
        tv_payload = _read_tv_payload(ckpt_path) if ckpt_path and ckpt_path.is_file() else None

        if tv_payload is not None:
            import torchvision.models.segmentation as seg

            self._backend = "torchvision"
            self.person_class = COCO_PERSON_CLASS
            self.model = seg.deeplabv3_resnet50(weights=None)
            self.model.load_state_dict(tv_payload)
            self.model.to(self.device).eval()
            return

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation-models-pytorch is not installed. "
                "Run: python scripts/setup_segmentation_colab.py"
            ) from exc

        if not hasattr(smp, arch):
            raise ValueError(f"Unknown SMP arch: {arch}")
        model_cls = getattr(smp, arch)
        self.model = model_cls(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
        )
        if ckpt_path is not None and ckpt_path.is_file():
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _load_smp_checkpoint(self.model, payload)
        elif ckpt_path is not None:
            print(
                f"SMP checkpoint not found ({ckpt_path}); using encoder_weights={encoder_weights!r}. "
                "Run: python scripts/setup_segmentation_colab.py --download-smp-weights"
            )
        self.model.to(self.device).eval()

    def _predict_logits(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self._backend == "torchvision":
                logits = self.model(tensor)["out"][0].cpu().numpy()
            else:
                logits = self.model(tensor)[0].cpu().numpy()
        return logits

    def _person_mask(self, logits: np.ndarray, height: int, width: int) -> np.ndarray:
        if logits.ndim == 3:
            probs = self._softmax(logits)
            person = probs[self.person_class]
        else:
            person = 1.0 / (1.0 + np.exp(-logits[0]))
        if person.shape != (height, width):
            person = cv2.resize(person, (width, height), interpolation=cv2.INTER_LINEAR)
        return person >= self.conf_threshold

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-12)

    def _components_to_segs(self, person_mask: np.ndarray) -> list[SegResult]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            person_mask.astype(np.uint8), connectivity=8
        )
        out: list[SegResult] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            mask = labels == label
            tlwh = np.array([x, y, w, h], dtype=np.float64)
            out.append(SegResult(tlwh=tlwh, confidence=1.0, mask=mask))
        return out

    def segment(self, frame: np.ndarray) -> list[SegResult]:
        h, w = frame.shape[:2]
        logits = self._predict_logits(frame)
        person_mask = self._person_mask(logits, h, w)
        return self._components_to_segs(person_mask)

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        return [
            DetectionResult(tlwh=s.tlwh, confidence=s.confidence)
            for s in self.segment(frame)
        ]


def _read_tv_payload(path: Path) -> dict | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == "torchvision_deeplabv3_r50":
        return payload["state_dict"]
    return None
