"""torchreid models (OSNet, ResNet50-IBN) — KaiyangZhou/deep-person-reid."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from reid.base import REID_PATCH_SHAPE, ReIDExtractor, crop_person_patches, l2_normalize


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

        patches = crop_person_patches(frame, boxes_tlwh)
        if len(patches) != len(boxes_tlwh):
            # Keep row alignment: invalid crops -> zero vector (rare at GT boxes).
            aligned: list[np.ndarray | None] = []
            for box in boxes_tlwh:
                from reid.base import extract_image_patch

                aligned.append(extract_image_patch(frame, box))
            patches = [p for p in aligned if p is not None]
            features = self._encode_patches(patches)
            if len(patches) == len(boxes_tlwh):
                return l2_normalize(features)

            full = np.zeros((len(boxes_tlwh), self.feature_dim), dtype=np.float32)
            j = 0
            for i, patch in enumerate(aligned):
                if patch is None:
                    continue
                full[i] = features[j]
                j += 1
            return l2_normalize(full)

        return l2_normalize(self._encode_patches(patches))
