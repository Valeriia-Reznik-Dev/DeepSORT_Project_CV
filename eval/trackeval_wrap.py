"""TrackEval wrapper: MOT HOTA evaluation from baseline_original.yaml."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

DEFAULT_METRICS = ("HOTA", "CLEAR", "Identity")


@dataclass(frozen=True)
class _BenchmarkSpec:
    name: str
    gt_folder: Path
    sequences: tuple[str, ...]
    do_preproc: bool


@dataclass(frozen=True)
class _TrackerSpec:
    name: str
    results_dir: Path


@dataclass(frozen=True)
class _EvalPlan:
    metrics: tuple[str, ...]
    output_dir: Path
    trackers: tuple[_TrackerSpec, ...]
    benchmarks: tuple[_BenchmarkSpec, ...]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_plan_from_config(
    cfg: dict[str, Any],
    *,
    tracker_name: str = "original",
    results_dir: str | Path | None = None,
) -> _EvalPlan:
    paths = cfg["paths"]
    sequences = cfg["sequences"]
    eval_cfg = cfg.get("eval", {})

    return _EvalPlan(
        metrics=tuple(eval_cfg.get("metrics", DEFAULT_METRICS)),
        output_dir=Path(eval_cfg.get("output_dir", "results/eval")),
        trackers=(
            _TrackerSpec(
                name=tracker_name,
                results_dir=Path(results_dir or paths["results_dir"]),
            ),
        ),
        benchmarks=(
            _BenchmarkSpec(
                "MOT15",
                Path(paths["mot15_dir"]),
                tuple(sequences["mot15"]),
                True,  # official TrackEval default; no effect on our MOT15 GT (verified)
            ),
            _BenchmarkSpec(
                "MOT16",
                Path(paths["mot16_dir"]),
                tuple(sequences["mot16"]),
                True,  # required: distractor classes + ignore zones
            ),
        ),
    )


@contextmanager
def _tracker_staging(
    tracker_name: str,
    results_dir: Path,
    *,
    use_symlinks: bool = True,
) -> Iterator[Path]:
    tmp_root = Path(tempfile.mkdtemp(prefix="trackeval_"))
    dest = tmp_root / tracker_name / "data"
    dest.mkdir(parents=True)
    try:
        for txt in sorted(results_dir.glob("*.txt")):
            target = dest / txt.name
            if use_symlinks:
                target.symlink_to(txt.resolve())
            else:
                shutil.copy2(txt, target)
        yield tmp_root
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_benchmark(
    benchmark: _BenchmarkSpec,
    *,
    tracker_name: str,
    trackers_folder: Path,
    output_folder: Path,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "DISPLAY_LESS_PROGRESS": True,
            "PLOT_CURVES": False,
            "PRINT_CONFIG": False,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(benchmark.gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "OUTPUT_FOLDER": str(output_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "BENCHMARK": benchmark.name,
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": benchmark.do_preproc,
            "SEQ_INFO": {seq: None for seq in benchmark.sequences},
            "PRINT_CONFIG": False,
        }
    )

    metrics_config = {"METRICS": list(metrics), "THRESHOLD": 0.5}
    metric_objects = [getattr(trackeval.metrics, name)(metrics_config) for name in metrics]
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    raw_results, _ = evaluator.evaluate([dataset], metric_objects)
    return raw_results


def _hota_scalar(hota_block: dict[str, Any]) -> float:
    return float(np.mean(hota_block["HOTA"]) * 100.0)


def _sequence_scores(
    raw: dict[str, Any],
    tracker_name: str,
    sequences: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    tracker_res = raw["MotChallenge2DBox"][tracker_name]
    scores: dict[str, dict[str, float]] = {}
    for seq in sequences:
        cls_data = tracker_res[seq]["pedestrian"]
        hota = cls_data["HOTA"]
        clear = cls_data["CLEAR"]
        ident = cls_data["Identity"]
        scores[seq] = {
            "HOTA": _hota_scalar(hota),
            "MOTA": float(clear["MOTA"] * 100.0),
            "IDF1": float(ident["IDF1"] * 100.0),
            "DetA": float(np.mean(hota["DetA"]) * 100.0),
            "AssA": float(np.mean(hota["AssA"]) * 100.0),
        }
    return scores


def _build_tracker_report(
    raw_by_benchmark: dict[str, dict[str, Any]],
    tracker_name: str,
    benchmarks: tuple[_BenchmarkSpec, ...],
) -> dict[str, Any]:
    sequences: dict[str, dict[str, float]] = {}
    by_benchmark: dict[str, dict[str, dict[str, float]]] = {}
    hota_values: list[float] = []

    for benchmark in benchmarks:
        seq_scores = _sequence_scores(
            raw_by_benchmark[benchmark.name], tracker_name, benchmark.sequences
        )
        by_benchmark[benchmark.name] = seq_scores
        sequences.update(seq_scores)
        hota_values.extend(s["HOTA"] for s in seq_scores.values())

    return {
        "sequences": sequences,
        "benchmarks": by_benchmark,
        "HOTA_mean": float(np.mean(hota_values)) if hota_values else 0.0,
    }


def _save_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    for tracker_name, tracker_data in report["trackers"].items():
        for seq, scores in tracker_data["sequences"].items():
            rows.append({"tracker": tracker_name, "sequence": seq, **scores})
        rows.append(
            {
                "tracker": tracker_name,
                "sequence": "MEAN",
                "HOTA": tracker_data["HOTA_mean"],
                "MOTA": "",
                "IDF1": "",
                "DetA": "",
                "AssA": "",
            }
        )

    fieldnames = ["tracker", "sequence", "HOTA", "MOTA", "IDF1", "DetA", "AssA"]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"csv": str(summary_csv), "json": str(summary_json)}


def run_eval_plan(plan: _EvalPlan, *, use_symlinks: bool = True) -> dict[str, Any]:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"trackers": {}, "metrics": list(plan.metrics)}

    for tracker in plan.trackers:
        raw_by_benchmark: dict[str, dict[str, Any]] = {}
        with _tracker_staging(tracker.name, tracker.results_dir, use_symlinks=use_symlinks) as staging:
            for benchmark in plan.benchmarks:
                raw_by_benchmark[benchmark.name] = _run_benchmark(
                    benchmark,
                    tracker_name=tracker.name,
                    trackers_folder=staging,
                    output_folder=plan.output_dir / benchmark.name / tracker.name,
                    metrics=plan.metrics,
                )
        report["trackers"][tracker.name] = _build_tracker_report(
            raw_by_benchmark, tracker.name, plan.benchmarks
        )

    report["summary_paths"] = _save_report(report, plan.output_dir)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== HOTA summary (TrackEval) ===")
    for tracker_name, tracker_data in report["trackers"].items():
        n_seq = len(tracker_data["sequences"])
        print(f"\nTracker: {tracker_name}")
        print(f"{'Sequence':<20} {'HOTA':>8} {'MOTA':>8} {'IDF1':>8}")
        print("-" * 48)
        for seq, scores in sorted(tracker_data["sequences"].items()):
            print(
                f"{seq:<20} {scores['HOTA']:8.2f} "
                f"{scores['MOTA']:8.2f} {scores['IDF1']:8.2f}"
            )
        print("-" * 48)
        print(f"{'MEAN (' + str(n_seq) + ' videos)':<20} {tracker_data['HOTA_mean']:8.2f}")
    print(f"\nSaved: {report.get('summary_paths', {})}")
