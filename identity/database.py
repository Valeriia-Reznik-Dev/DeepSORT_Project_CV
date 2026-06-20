"""Online identity gallery for the standalone body-ReID DB.

Grows a gallery of identities and, for each incoming descriptor, decides whether
it matches a known identity (cosine distance <= radius) or starts a new one.
Descriptors are assumed L2-normalized (the ReID extractors guarantee this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


@dataclass
class GalleryEntry:
    identity_id: int
    centroid: np.ndarray
    count: int
    descriptors: list[np.ndarray] = field(default_factory=list)


class IdentityDatabase:
    """Growing gallery; assigns a known or new identity by cosine distance.

    Parameters
    ----------
    radius : float
        New-identity threshold: nearest distance above it -> create new identity.
    k : int
        Neighbours used for the identity vote in ``knn`` representation.
    representation : str
        ``"centroid"`` keeps one running-mean vector per identity (fast, capped
        memory); ``"knn"`` keeps up to ``max_per_identity`` descriptors and votes.
    max_per_identity : int
        Cap on stored descriptors per identity in ``knn`` mode (memory guard).
    """

    def __init__(
        self,
        *,
        radius: float = 0.3,
        k: int = 1,
        representation: str = "centroid",
        max_per_identity: int = 50,
    ):
        if representation not in ("centroid", "knn"):
            raise ValueError("representation must be 'centroid' or 'knn'")
        self.radius = float(radius)
        self.k = int(k)
        self.representation = representation
        self.max_per_identity = int(max_per_identity)
        self.entries: list[GalleryEntry] = []
        self._next_id = 0

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_identities(self) -> int:
        return len(self.entries)

    def _new_entry(self, desc: np.ndarray) -> GalleryEntry:
        desc = desc.astype(np.float32)
        entry = GalleryEntry(
            identity_id=self._next_id,
            centroid=_l2(desc.copy()),
            count=1,
            descriptors=[desc],
        )
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def _candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Stacked vectors and their identity ids for distance computation."""
        if self.representation == "centroid":
            mat = np.stack([e.centroid for e in self.entries])
            labels = np.array([e.identity_id for e in self.entries])
            return mat, labels
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for entry in self.entries:
            for desc in entry.descriptors:
                rows.append(desc)
                labels.append(entry.identity_id)
        return np.stack(rows), np.array(labels)

    def _entry_by_id(self, identity_id: int) -> GalleryEntry:
        for entry in self.entries:
            if entry.identity_id == identity_id:
                return entry
        raise KeyError(identity_id)

    def _update(self, entry: GalleryEntry, desc: np.ndarray) -> None:
        desc = desc.astype(np.float32)
        entry.count += 1
        running = entry.centroid * (entry.count - 1) + _l2(desc)
        entry.centroid = _l2(running)
        if self.representation == "knn":
            entry.descriptors.append(desc)
            if len(entry.descriptors) > self.max_per_identity:
                entry.descriptors.pop(0)

    def create_identity(self, desc: np.ndarray) -> int:
        """Force a brand-new identity (used by conflict resolution)."""
        return self._new_entry(np.asarray(desc, dtype=np.float32)).identity_id

    def assign(self, desc: np.ndarray) -> tuple[int, float, bool]:
        """Match ``desc`` to a known identity or create a new one.

        Returns ``(identity_id, nearest_distance, is_new)``.
        """
        desc = np.asarray(desc, dtype=np.float32)
        query = _l2(desc)
        if not self.entries:
            entry = self._new_entry(desc)
            return entry.identity_id, 1.0, True

        mat, labels = self._candidates()
        dists = 1.0 - mat @ query  # cosine distance (vectors L2-normalized)
        order = np.argsort(dists)[: max(1, self.k)]
        nearest = float(dists[order[0]])

        if nearest > self.radius:
            entry = self._new_entry(desc)
            return entry.identity_id, nearest, True

        knn_labels = labels[order]
        vals, counts = np.unique(knn_labels, return_counts=True)
        identity_id = int(vals[np.argmax(counts)])
        self._update(self._entry_by_id(identity_id), desc)
        return identity_id, nearest, False

    def distance_to(self, identity_id: int, desc: np.ndarray) -> float:
        """Cosine distance from ``desc`` to an identity centroid."""
        query = _l2(np.asarray(desc, dtype=np.float32))
        try:
            entry = self._entry_by_id(identity_id)
        except KeyError:
            return 1.0
        return float(1.0 - entry.centroid @ query)
