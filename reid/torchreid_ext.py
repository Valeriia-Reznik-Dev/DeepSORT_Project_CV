"""torchreid adapter (OSNet, ResNet50-IBN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from reid.base import (
    REID_PATCH_SHAPE,
    ReIDExtractor,
    crop_person_patches_with_mask,
    scatter_features,
)


class TorchReIDExtractor(ReIDExtractor):
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        batch_size: int = 32,
        **_: object,
    ):
        import torchreid

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = model_name

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            loss="softmax",
            pretrained=True,
        )
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, REID_PATCH_SHAPE[0], REID_PATCH_SHAPE[1], device=self.device)
            out = self.model(dummy)
            self._feature_dim = int(out.shape[1])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(REID_PATCH_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _encode_patches(self, patches: list[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        feats: list[np.ndarray] = []
        for start in range(0, len(patches), self.batch_size):
            batch_patches = patches[start : start + self.batch_size]
            tensors = torch.stack(
                [self.transform(patch[:, :, ::-1]) for patch in batch_patches]
            ).to(self.device)
            with torch.no_grad():
                emb = self.model(tensors)
                emb = F.normalize(emb, p=2, dim=1)
            feats.append(emb.cpu().numpy())

        return np.vstack(feats).astype(np.float32)

    def extract(self, frame: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        boxes_tlwh = np.asarray(boxes_tlwh, dtype=np.float64)
        if boxes_tlwh.size == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        patches, mask = crop_person_patches_with_mask(frame, boxes_tlwh)
        encoded = self._encode_patches(patches)  # already L2-normalized
        return scatter_features(encoded, mask, self.feature_dim)
