"""Identity assignment pipeline: DB lookup -> window vote -> conflict resolution.

Wraps :class:`IdentityDatabase` with the per-track logic from the spec:
1. each detection gets a DB identity (known or new);
2. the identity is appended to the track's identity history;
3. the track's final identity is the majority vote over the last ``window`` frames;
4. if two active tracks resolve to the same identity, the farther one (by distance
   to the identity centroid) is split off into a fresh identity.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from identity.database import IdentityDatabase


@dataclass
class IdentityParams:
    radius: float = 0.3
    k: int = 1
    representation: str = "centroid"
    window: int = 30
    conflict_policy: str = "distance"  # "distance" | "none"
    max_per_identity: int = 50


class IdentityManager:
    def __init__(self, params: IdentityParams | None = None):
        self.params = params or IdentityParams()
        self.db = IdentityDatabase(
            radius=self.params.radius,
            k=self.params.k,
            representation=self.params.representation,
            max_per_identity=self.params.max_per_identity,
        )
        self.history: dict[int, deque[tuple[int, int]]] = {}
        self.last_desc: dict[int, np.ndarray] = {}

    def _vote(self, track_id: int) -> int:
        ids = [identity for _, identity in self.history[track_id]]
        return Counter(ids).most_common(1)[0][0]

    def update(
        self, frame_idx: int, detections: list[tuple[int, np.ndarray]]
    ) -> tuple[dict[int, int], dict[int, int]]:
        """Process one frame.

        Parameters
        ----------
        frame_idx : int
            Current frame index (used as the identity timestamp).
        detections : list[(track_id, descriptor)]
            One entry per active track in this frame.

        Returns
        -------
        (resolved, raw)
            ``raw[track_id]`` is the per-detection DB identity; ``resolved[track_id]``
            is the window-voted, conflict-resolved identity.
        """
        raw: dict[int, int] = {}
        dist: dict[int, float] = {}
        for track_id, desc in detections:
            identity, nearest, _ = self.db.assign(desc)
            raw[track_id] = identity
            dist[track_id] = nearest
            self.last_desc[track_id] = np.asarray(desc, dtype=np.float32)
            hist = self.history.setdefault(track_id, deque(maxlen=self.params.window))
            hist.append((frame_idx, identity))

        resolved = {tid: self._vote(tid) for tid in raw}
        if self.params.conflict_policy != "none":
            resolved = self._resolve_conflicts(resolved, dist)
        return resolved, raw

    def _resolve_conflicts(
        self, resolved: dict[int, int], dist: dict[int, float]
    ) -> dict[int, int]:
        by_identity: dict[int, list[int]] = {}
        for track_id, identity in resolved.items():
            by_identity.setdefault(identity, []).append(track_id)

        for tracks in by_identity.values():
            if len(tracks) < 2:
                continue
            winner = min(tracks, key=lambda t: dist.get(t, 1.0))
            for track_id in tracks:
                if track_id == winner:
                    continue
                new_id = self.db.create_identity(self.last_desc[track_id])
                resolved[track_id] = new_id
                hist = self.history.get(track_id)
                if hist:
                    frame_last, _ = hist[-1]
                    hist[-1] = (frame_last, new_id)
        return resolved
