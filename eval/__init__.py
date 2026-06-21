"""Evaluation helpers (lazy TrackEval import)."""

from typing import Any

__all__ = ["eval_plan_from_config", "load_yaml", "print_summary", "run_eval_plan"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from eval import trackeval_wrap

        return getattr(trackeval_wrap, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
