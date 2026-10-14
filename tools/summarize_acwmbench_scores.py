#!/usr/bin/env python3
"""Summarize per-episode ACWMBench results into baseline-level 0-100 scores.

Each dimension is averaged within each task first, then tasks are averaged with
equal weight. This prevents tasks containing more episodes from dominating the
benchmark score.

KeyArea is reconstructed from per-mask PSNR/LPIPS/V-JEPA using the same rule as
``analyze_keyarea_results.py``: normalize each mask, take its three-way
geometric mean, then average masks, frames, sources, episodes, and tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml


DEFAULT_BASELINES: dict[str, str] = {}

DIMENSIONS = (
    "physical_judge",
    "keyarea_visual_quality",
    "interaction_spatial_alignment",
    "multiview_consistency",
)
CORE_DIMENSIONS = (
    "physical_judge",
    "keyarea_visual_quality",
    "interaction_spatial_alignment",
)
RESULT_FILENAMES = {
    dimension: f"{dimension}.json" for dimension in DIMENSIONS
}
SOURCES = ("object", "robot_arm")
KEYAREA_METRICS = ("psnr", "lpips", "vjepa_similarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Baseline root. Repeat for every model to compare. "
            "Example: --baseline MyModel=/path/to/model-results"
        ),
    )
    parser.add_argument(
        "--normalization-config",
        default="configs/analysis/keyarea_results.yaml",
        help="YAML containing normalization.psnr/lpips/vjepa_similarity.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/acwmbench_score_summary",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task allow-list, for example: pickfruit hangtowel.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print file-processing progress every N JSON files (default: 100).",
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def parse_baselines(values: list[str] | None) -> dict[str, Path]:
    if not values:
        raise SystemExit("provide at least one --baseline NAME=PATH")
    baselines: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--baseline expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --baseline value: {value!r}")
        baselines[name] = Path(path)
    return baselines


def load_normalization(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    normalization = config.get("normalization") or {}
    result = {}
    for metric in KEYAREA_METRICS:
        values = normalization.get(metric)
        if not isinstance(values, dict):
            raise ValueError(f"{path}: normalization.{metric} is required")
        low, high = finite(values.get("min")), finite(values.get("max"))
        if low is None or high is None or high <= low:
            raise ValueError(
                f"{path}: normalization.{metric} needs finite min < max"
            )
        result[metric] = {"min": low, "max": high}
    return result


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_mask(
    record: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
) -> float | None:
    psnr = finite(record.get("psnr"))
    lpips = finite(record.get("lpips"))
    vjepa = finite(record.get("vjepa_similarity"))
    if psnr is None:
        # Positive infinity is a perfect PSNR match.
        try:
            if math.isinf(float(record.get("psnr"))) and float(
                record.get("psnr")
            ) > 0:
                psnr = float(normalization["psnr"]["max"])
        except (TypeError, ValueError):
            pass
    if psnr is None or lpips is None or vjepa is None:
        return None

    psnr_cfg = normalization["psnr"]
    lpips_cfg = normalization["lpips"]
    vjepa_cfg = normalization["vjepa_similarity"]
    psnr_unit = min(
        1.0,
        max(
            0.0,
            (psnr - psnr_cfg["min"])
            / (psnr_cfg["max"] - psnr_cfg["min"]),
        ),
    )
    lpips_unit = min(
        1.0,
        max(
            0.0,
            (lpips_cfg["max"] - lpips)
            / (lpips_cfg["max"] - lpips_cfg["min"]),
        ),
    )
    vjepa_unit = min(
        1.0,
        max(
            0.0,
            (vjepa - vjepa_cfg["min"])
            / (vjepa_cfg["max"] - vjepa_cfg["min"]),
        ),
    )
    return 100.0 * (psnr_unit * lpips_unit * vjepa_unit) ** (1.0 / 3.0)


def keyarea_source_score(
    source: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
) -> float | None:
    fallback: dict[int, list[Mapping[str, Any]]] = {}
    for record in source.get("per_mask") or []:
        if isinstance(record, dict) and "frame" in record:
            fallback.setdefault(int(record["frame"]), []).append(record)

    frame_scores = []
    for frame_record in source.get("per_frame") or []:
        if not isinstance(frame_record, dict) or "frame" not in frame_record:
            continue
        masks = frame_record.get("masks")
        if not isinstance(masks, list):
            masks = fallback.get(int(frame_record["frame"]), [])
        frame_score = mean(
            normalize_mask(mask, normalization)
            for mask in masks
            if isinstance(mask, dict)
        )
        if frame_score is not None:
            frame_scores.append(frame_score)
    return mean(frame_scores)


def keyarea_episode_score(
    data: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
) -> float | None:
    mask_sources = data.get("mask_sources")
    if not isinstance(mask_sources, dict):
        return None
    return mean(
        keyarea_source_score(source, normalization)
        for name in SOURCES
        if isinstance((source := mask_sources.get(name)), dict)
    )


def physical_episode_score(data: Mapping[str, Any]) -> float | None:
    trial_scores = []
    for trial in data.values():
        if not isinstance(trial, dict):
            continue
        if trial.get("status") not in (None, "complete"):
            continue
        if int(trial.get("metric_version", 0)) < 1:
            continue
        score = finite(trial.get("score"))
        if score is not None:
            trial_scores.append(100.0 * score if score <= 1.0 else score)
    return mean(trial_scores)


def episode_score(
    dimension: str,
    data: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
) -> float | None:
    if dimension == "physical_judge":
        return physical_episode_score(data)
    if dimension == "keyarea_visual_quality":
        if int(data.get("metric_version", 0)) < 12:
            return None
        return keyarea_episode_score(data, normalization)
    if dimension == "interaction_spatial_alignment":
        if int(data.get("metric_version", 0)) < 2:
            return None
        return finite(data.get("alignment_score"))
    if dimension == "multiview_consistency":
        if int(data.get("metric_version", 0)) < 5:
            return None
        return finite(data.get("multiview_consistency_score"))
    raise ValueError(f"Unsupported dimension: {dimension}")


def discover_episode_files(
    root: Path,
    filename: str,
    tasks: set[str] | None,
) -> list[tuple[str, str, Path]]:
    records = []
    for path in sorted(root.glob(f"*/*/{filename}")):
        relative = path.relative_to(root)
        task, episode = relative.parts[:2]
        if tasks is None or task in tasks:
            records.append((task, episode, path))
    return records


def summarize_dimension(
    root: Path,
    dimension: str,
    normalization: Mapping[str, Mapping[str, float]],
    tasks: set[str] | None,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    task_episodes: dict[str, list[float]] = {}
    invalid_files = []
    files = discover_episode_files(
        root,
        RESULT_FILENAMES[dimension],
        tasks,
    )
    if progress:
        progress(f"{dimension}: found {len(files)} episode JSON file(s)")
    for index, (task, _episode, path) in enumerate(files, start=1):
        try:
            data = load_json(path)
            score = (
                episode_score(dimension, data, normalization)
                if isinstance(data, dict)
                else None
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})
            continue
        if score is None:
            invalid_files.append({"path": str(path), "error": "no valid score"})
            continue
        task_episodes.setdefault(task, []).append(score)
        if progress and (
            index == len(files) or index % max(1, progress_every) == 0
        ):
            progress(
                f"{dimension}: processed {index}/{len(files)} file(s)"
            )

    per_task = {
        task: {
            "score": mean(scores),
            "num_episodes": len(scores),
        }
        for task, scores in sorted(task_episodes.items())
    }
    result = {
        "score": mean(record["score"] for record in per_task.values()),
        "num_tasks": len(per_task),
        "num_episodes": sum(
            record["num_episodes"] for record in per_task.values()
        ),
        "per_task": per_task,
        "invalid_files": invalid_files,
    }
    if progress:
        progress(
            f"{dimension}: score={display(result['score'])}, "
            f"tasks={result['num_tasks']}, episodes={result['num_episodes']}, "
            f"invalid={len(invalid_files)}"
        )
    return result


def complete_mean(
    summaries: Mapping[str, Mapping[str, Any]],
    dimensions: Iterable[str],
) -> float | None:
    dimensions = tuple(dimensions)
    values = [finite(summaries[name].get("score")) for name in dimensions]
    if any(value is None for value in values):
        return None
    return mean(values)


def available_mean(
    summaries: Mapping[str, Mapping[str, Any]],
) -> float | None:
    return mean(summary.get("score") for summary in summaries.values())


def summarize_baseline(
    name: str,
    root: Path,
    normalization: Mapping[str, Mapping[str, float]],
    tasks: set[str] | None,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "baseline": name,
            "root": str(root),
            "error": "baseline root does not exist",
            "dimensions": {},
        }
    dimensions = {}
    for dimension in DIMENSIONS:
        dimensions[dimension] = summarize_dimension(
            root,
            dimension,
            normalization,
            tasks,
            progress=progress,
            progress_every=progress_every,
        )
    return {
        "baseline": name,
        "root": str(root),
        "dimensions": dimensions,
        # Comparable across all baselines that have the three universal metrics.
        "acwmbench_core_score": complete_mean(dimensions, CORE_DIMENSIONS),
        # Comparable only when all four dimensions, including MultiView, exist.
        "acwmbench_full_score": complete_mean(dimensions, DIMENSIONS),
        # Diagnostic only: can be unfair when baselines have different coverage.
        "available_dimensions_mean": available_mean(dimensions),
        "num_available_dimensions": sum(
            finite(summary.get("score")) is not None
            for summary in dimensions.values()
        ),
    }


def csv_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = summary.get("dimensions") or {}
    row = {
        "baseline": summary["baseline"],
        "root": summary["root"],
        "acwmbench_core_score": summary.get("acwmbench_core_score"),
        "acwmbench_full_score": summary.get("acwmbench_full_score"),
        "available_dimensions_mean": summary.get(
            "available_dimensions_mean"
        ),
        "num_available_dimensions": summary.get(
            "num_available_dimensions",
            0,
        ),
        "error": summary.get("error", ""),
    }
    for dimension in DIMENSIONS:
        result = dimensions.get(dimension) or {}
        row[f"{dimension}_score"] = result.get("score")
        row[f"{dimension}_tasks"] = result.get("num_tasks", 0)
        row[f"{dimension}_episodes"] = result.get("num_episodes", 0)
    return row


def display(value: Any) -> str:
    number = finite(value)
    return f"{number:.2f}" if number is not None else "N/A"


def main() -> None:
    args = parse_args()
    baselines = parse_baselines(args.baseline)
    normalization = load_normalization(Path(args.normalization_config))
    tasks = set(args.tasks) if args.tasks else None
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    summaries = []
    total = len(baselines)
    for index, (name, root) in enumerate(baselines.items(), start=1):
        print(
            f"\n[{index}/{total}] Processing {name}: {root}",
            flush=True,
        )

        def report(message: str, *, name=name) -> None:
            print(f"  [{name}] {message}", flush=True)

        summary = summarize_baseline(
            name,
            root,
            normalization,
            tasks,
            progress=report,
            progress_every=args.progress_every,
        )
        summaries.append(summary)
        if summary.get("error"):
            report(f"ERROR: {summary['error']}")
        else:
            report(
                "complete: "
                f"core={display(summary.get('acwmbench_core_score'))}, "
                f"full={display(summary.get('acwmbench_full_score'))}"
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "score_range": [0.0, 100.0],
        "aggregation": {
            "dimension": "task-balanced mean of episode scores",
            "acwmbench_core_score": list(CORE_DIMENSIONS),
            "acwmbench_full_score": list(DIMENSIONS),
            "missing_full_dimension": "N/A; never silently averaged away",
        },
        "keyarea_normalization": normalization,
        "task_filter": sorted(tasks) if tasks else None,
        "baselines": summaries,
    }
    with (output / "acwmbench_scores.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    rows = [csv_row(summary) for summary in summaries]
    fieldnames = list(rows[0]) if rows else []
    with (output / "acwmbench_scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'Baseline':<18} {'Physical':>9} {'KeyArea':>9} "
        f"{'Interact':>9} {'MultiView':>9} {'Core':>9} {'Full':>9}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        dimensions = summary.get("dimensions") or {}
        scores = {
            dimension: (dimensions.get(dimension) or {}).get("score")
            for dimension in DIMENSIONS
        }
        print(
            f"{summary['baseline']:<18} "
            f"{display(scores['physical_judge']):>9} "
            f"{display(scores['keyarea_visual_quality']):>9} "
            f"{display(scores['interaction_spatial_alignment']):>9} "
            f"{display(scores['multiview_consistency']):>9} "
            f"{display(summary.get('acwmbench_core_score')):>9} "
            f"{display(summary.get('acwmbench_full_score')):>9}"
        )
    print(f"\nWrote: {(output / 'acwmbench_scores.csv').resolve()}")
    print(f"Wrote: {(output / 'acwmbench_scores.json').resolve()}")


if __name__ == "__main__":
    main()
