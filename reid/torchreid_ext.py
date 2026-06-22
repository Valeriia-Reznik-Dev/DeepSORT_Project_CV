"""torchreid adapter (OSNet, ResNet50)."""
from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path

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


def _strip_classification_head(model: torch.nn.Module) -> None:
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        model.classifier = torch.nn.Identity()
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        model.fc = torch.nn.Identity()


def _looks_like_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    head = path.read_bytes()[:256].lstrip()
    return not (
        head.startswith(b"<!DOCTYPE")
        or head.startswith(b"<!doctype")
        or head.startswith(b"<html")
    )


def _load_torchreid_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not _looks_like_checkpoint(path):
        raise ValueError(
            f"Checkpoint file looks invalid (too small or HTML): {checkpoint_path}. "
            "Re-download with: python scripts/download_reid_models.py --force"
        )

    load_errors: list[str] = []
    ckpt = None
    for loader in (_torch_load_default, _torch_load_latin1):
        try:
            ckpt = loader(path)
            break
        except Exception as exc:  # noqa: BLE001 - try fallbacks
            load_errors.append(str(exc))

    if ckpt is None:
        joined = "; ".join(load_errors)
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {joined}")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")
    return state_dict


def _torch_load_default(path: Path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _torch_load_latin1(path: Path):
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            pickle_load_args={"encoding": "latin1"},
        )
    except TypeError:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")


def _load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    state_dict = _load_torchreid_checkpoint(checkpoint_path)
    model_state = model.state_dict()
    new_state_dict = OrderedDict()
    matched: list[str] = []
    discarded: list[str] = []

    for key, value in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        if name in model_state and model_state[name].shape == value.shape:
            new_state_dict[name] = value
            matched.append(name)
        else:
            discarded.append(name)

    model_state.update(new_state_dict)
    model.load_state_dict(model_state)
    print(
        f"Loaded torchreid weights from {checkpoint_path} "
        f"({len(matched)} matched layers, {len(discarded)} discarded)"
    )


def _build_torchreid_model(model_name: str, checkpoint: str | None):
    import torchreid
    from torchreid.utils import check_isfile

    is_osnet = model_name.startswith("osnet")
    num_classes = 1 if is_osnet else 751
    pretrained = not (checkpoint and check_isfile(checkpoint))

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=num_classes,
        loss="softmax",
        pretrained=pretrained,
    )
    _strip_classification_head(model)

    if checkpoint and check_isfile(checkpoint):
        _load_pretrained_weights(model, checkpoint)

    return model


class TorchReIDExtractor(ReIDExtractor):
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        batch_size: int = 32,
        checkpoint: str | None = None,
        **_: object,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = model_name

        self.model = _build_torchreid_model(model_name, checkpoint)
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
