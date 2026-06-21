"""Per-video tracker params from YAML."""
from __future__ import annotations

import yaml

from tracking.pipeline import TrackerParams

PARAM_FIELDS = (
    "min_confidence",
    "min_detection_height",
    "nms_max_overlap",
    "max_cosine_distance",
    "nn_budget",
)


def load_params_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    if override:
        out.update({k: v for k, v in override.items() if k in PARAM_FIELDS})
    return out


def params_for(
    cfg: dict | None,
    seq_name: str,
    cli_overrides: dict | None = None,
) -> TrackerParams:
    cfg = cfg or {}
    default = cfg.get("default", {}) or {}
    per_video = (cfg.get("per_video") or {}).get(seq_name, {}) or {}
    merged = _merge(_merge(default, per_video), cli_overrides)
    base = TrackerParams()
    fields = {f: merged.get(f, getattr(base, f)) for f in PARAM_FIELDS}
    return TrackerParams(**fields)
