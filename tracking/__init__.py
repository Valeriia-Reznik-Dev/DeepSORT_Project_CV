"""Live tracking pipeline (detector + ReID + DeepSORT core)."""

from tracking.pipeline import TrackerParams, track_sequence

__all__ = ["TrackerParams", "track_sequence"]
