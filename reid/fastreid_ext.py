"""fast-reid adapter (JDAI-CV/fast-reid) — second model source."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reid.base import ReIDExtractor, crop_person_patches, l2_normalize

ROOT = Path(__file__).resolve().parents[1]
FASTREID_DIR = ROOT / "third_party" / "fast_reid"


def _ensure_fastreid_on_path() -> None:
    if not (FASTREID_DIR / "fastreid").is_dir():
        raise FileNotFoundError(
            f"fast-reid not found at {FASTREID_DIR}. "
            "Run: python scripts/setup_reid_colab.py"
        )
    if str(FASTREID_DIR) not in sys.path:
        sys.path.insert(0, str(FASTREID_DIR))


def _model_features(model: torch.nn.Module, tensors: torch.Tensor) -> torch.Tensor:
    """Run fast-reid Baseline in eval mode; input is Bx3xHxW RGB float (0-255)."""
    with torch.no_grad():
        outputs = model(tensors)
    if isinstance(outputs, dict):
        if "features" in outputs:
            return outputs["features"]
        return next(v for v in outputs.values() if isinstance(v, torch.Tensor))
    return outputs


class FastReIDExtractor(ReIDExtractor):
    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cpu",
        batch_size: int = 32,
        **_: object,
    ):
        _ensure_fastreid_on_path()
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        self.device = torch.device(device)
        self.batch_size = batch_size

        cfg = get_cfg()
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.MODEL.DEVICE = str(self.device)
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()

        self.model = build_model(cfg)
        self.model.to(self.device)
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()

        self.height = cfg.INPUT.SIZE_TEST[0]
        self.width = cfg.INPUT.SIZE_TEST[1]

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.height, self.width, device=self.device)
            out = _model_features(self.model, dummy)
            self._feature_dim = int(out.shape[1])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _preprocess(self, patch_bgr: np.ndarray) -> torch.Tensor:
        import cv2

        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height))
        return torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    def _encode_patches(self, patches: list[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        feats: list[np.ndarray] = []
        for start in range(0, len(patches), self.batch_size):
            batch = patches[start : start + self.batch_size]
            tensors = torch.cat([self._preprocess(p) for p in batch], dim=0)
            with torch.no_grad():
                emb = _model_features(self.model, tensors)
                emb = F.normalize(emb, p=2, dim=1)
            feats.append(emb.cpu().numpy())
        return np.vstack(feats).astype(np.float32)

    def extract(self, frame: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64)
        if boxes_tlwh.size == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        patches = crop_person_patches(frame, boxes_tlwh, patch_shape=(self.height, self.width))
        return l2_normalize(self._encode_patches(patches))
