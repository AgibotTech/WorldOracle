#!/usr/bin/env python3
"""Compute cumulative 0-T ACWMBench scores from completed episode JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .summarize_acwmbench_scores import (
        CORE_DIMENSIONS,
        DEFAULT_BASELINES,
        DIMENSIONS,
        RESULT_FILENAMES,
        SOURCES,
        complete_mean,
        discover_episode_files,
        display,
        finite,
        load_json,
        load_normalization,
        mean,
        normalize_mask,
        parse_baselines,
    )
except ImportError:  # Direct execution: python tools/<script>.py
    from summarize_acwmbench_scores import (
        CORE_DIMENSIONS,
        DEFAULT_BASELINES,
        DIMENSIONS,
        RESULT_FILENAMES,
        SOURCES,
        complete_mean,
        discover_episode_files,
        display,
        finite,
        load_json,
        load_normalization,
        mean,
        normalize_mask,
        parse_baselines,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        metavar="NAME=PATH",
        help="Repeat NAME=PATH to override the six built-in baselines.",
    )
    parser.add_argument(
        "--normalization-config",
        default="configs/analysis/keyarea_results.yaml",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--step-seconds", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        default="output/acwmbench_temporal_score_summary",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def _frame_in_window(record: Mapping[str, Any], frame_limit: float) -> bool:
    frame = finite(record.get("frame"))
    return frame is not None and 0.0 <= frame < frame_limit


def physical_window_score(
    data: Mapping[str, Any],
    frame_limit: float,
) -> float | None:
    trial_scores = []
    for trial in data.values():
        if not isinstance(trial, Mapping):
            continue
        if trial.get("status") not in (None, "complete"):
            continue
        step_scores = []
        for step in trial.get("key_steps") or []:
            if not isinstance(step, Mapping) or not _frame_in_window(
                step,
                frame_limit,
            ):
                continue
            score = finite(step.get("score"))
            if score is None:
                scores = [
                    finite(value) for value in step.get("aspect_scores") or []
                ]
                score = mean(value for value in scores if value is not None)
            if score is not None:
                step_scores.append(100.0 * score if score <= 1.0 else score)
        trial_score = mean(step_scores)
        if trial_score is not None:
            trial_scores.append(trial_score)
    return mean(trial_scores)


def keyarea_source_window_score(
    source: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
    frame_limit: float,
) -> float | None:
    fallback: dict[int, list[Mapping[str, Any]]] = {}
    for record in source.get("per_mask") or []:
        if isinstance(record, Mapping) and "frame" in record:
            fallback.setdefault(int(record["frame"]), []).append(record)
    frame_scores = []
    for frame_record in source.get("per_frame") or []:
        if not isinstance(frame_record, Mapping) or not _frame_in_window(
            frame_record,
            frame_limit,
        ):
            continue
        masks = frame_record.get("masks")
        if not isinstance(masks, list):
            masks = fallback.get(int(frame_record["frame"]), [])
        frame_score = mean(
            normalize_mask(mask, normalization)
            for mask in masks
            if isinstance(mask, Mapping)
        )
        if frame_score is not None:
            frame_scores.append(frame_score)
    return mean(frame_scores)


def keyarea_window_score(
    data: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
    frame_limit: float,
) -> float | None:
    if int(data.get("metric_version", 0)) < 12:
        return None
    sources = data.get("mask_sources")
    if not isinstance(sources, Mapping):
        return None
    return mean(
        keyarea_source_window_score(source, normalization, frame_limit)
        for name in SOURCES
        if isinstance((source := sources.get(name)), Mapping)
    )


def interaction_window_score(
    data: Mapping[str, Any],
    frame_limit: float,
) -> float | None:
    if int(data.get("metric_version", 0)) < 2:
        return None
    sources = data.get("mask_sources")
    if not isinstance(sources, Mapping):
        return None
    source_scores = []
    for name in SOURCES:
        source = sources.get(name)
        if not isinstance(source, Mapping):
            continue
        source_scores.append(
            mean(
                record.get("alignment_score")
                for record in source.get("per_frame") or []
                if isinstance(record, Mapping)
                and _frame_in_window(record, frame_limit)
            )
        )
    return mean(source_scores)


def multiview_window_score(
    data: Mapping[str, Any],
    frame_limit: float,
) -> float | None:
    if int(data.get("metric_version", 0)) < 5:
        return None
    return mean(
        record.get("multiview_consistency_score")
        for record in data.get("per_frame") or []
        if isinstance(record, Mapping)
        and _frame_in_window(record, frame_limit)
    )


def episode_window_score(
    dimension: str,
    data: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, float]],
    frame_limit: float,
) -> float | None:
    if dimension == "physical_judge":
        return physical_window_score(data, frame_limit)
    if dimension == "keyarea_visual_quality":
        return keyarea_window_score(data, normalization, frame_limit)
    if dimension == "interaction_spatial_alignment":
        return interaction_window_score(data, frame_limit)
    if dimension == "multiview_consistency":
        return multiview_window_score(data, frame_limit)
    raise ValueError(f"Unsupported dimension: {dimension}")


def summarize_dimension_windows(
    root: Path,
    dimension: str,
    windows: Iterable[float],
    fps: float,
    normalization: Mapping[str, Mapping[str, float]],
    tasks: set[str] | None,
    progress_every: int,
    baseline: str,
) -> dict[float, dict[str, Any]]:
    windows = tuple(windows)
    task_scores: dict[float, dict[str, list[float]]] = {
        window: {} for window in windows
    }
    invalid = 0
    files = discover_episode_files(
        root,
        RESULT_FILENAMES[dimension],
        tasks,
    )
    print(
        f"  [{baseline}] {dimension}: found {len(files)} JSON file(s)",
        flush=True,
    )
    for index, (task, _episode, path) in enumerate(files, start=1):
        try:
            data = load_json(path)
            if not isinstance(data, Mapping):
                raise ValueError("result is not a JSON object")
            for window in windows:
                score = episode_window_score(
                    dimension,
                    data,
                    normalization,
                    window * fps,
                )
                if score is not None:
                    task_scores[window].setdefault(task, []).append(score)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            invalid += 1
        if index == len(files) or index % progress_every == 0:
            print(
                f"  [{baseline}] {dimension}: "
                f"processed {index}/{len(files)}",
                flush=True,
            )

    output = {}
    for window in windows:
        per_task = {
            task: {
                "score": mean(scores),
                "num_episodes": len(scores),
            }
            for task, scores in sorted(task_scores[window].items())
        }
        output[window] = {
            "score": mean(item["score"] for item in per_task.values()),
            "num_tasks": len(per_task),
            "num_episodes": sum(
                item["num_episodes"] for item in per_task.values()
            ),
            "per_task": per_task,
            "invalid_files": invalid,
        }
    return output


def cumulative_windows(step: float, maximum: float) -> list[float]:
    if step <= 0 or maximum <= 0:
        raise ValueError("--step-seconds and --max-seconds must be positive")
    count = int(math.floor(maximum / step + 1e-9))
    windows = [step * index for index in range(1, count + 1)]
    if not windows or not math.isclose(windows[-1], maximum):
        windows.append(maximum)
    return windows


def summarize_baseline(
    name: str,
    root: Path,
    windows: list[float],
    fps: float,
    normalization: Mapping[str, Mapping[str, float]],
    tasks: set[str] | None,
    progress_every: int,
) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "baseline": name,
            "root": str(root),
            "error": "baseline root does not exist",
            "windows": {},
        }
    by_dimension = {
        dimension: summarize_dimension_windows(
            root,
            dimension,
            windows,
            fps,
            normalization,
            tasks,
            progress_every,
            name,
        )
        for dimension in DIMENSIONS
    }
    by_window = {}
    for window in windows:
        dimensions = {
            dimension: by_dimension[dimension][window]
            for dimension in DIMENSIONS
        }
        by_window[window] = {
            "dimensions": dimensions,
            "acwmbench_core_score": complete_mean(
                dimensions,
                CORE_DIMENSIONS,
            ),
            "acwmbench_full_score": complete_mean(
                dimensions,
                DIMENSIONS,
            ),
        }
    return {
        "baseline": name,
        "root": str(root),
        "windows": by_window,
    }


def csv_rows(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for window, result in (summary.get("windows") or {}).items():
            dimensions = result["dimensions"]
            row = {
                "baseline": summary["baseline"],
                "window": f"0-{float(window):g}s",
                "window_end_sec": float(window),
                "acwmbench_core_score": result["acwmbench_core_score"],
                "acwmbench_full_score": result["acwmbench_full_score"],
            }
            for dimension in DIMENSIONS:
                record = dimensions[dimension]
                row[f"{dimension}_score"] = record["score"]
                row[f"{dimension}_tasks"] = record["num_tasks"]
                row[f"{dimension}_episodes"] = record["num_episodes"]
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.progress_every < 1:
        raise ValueError("--fps and --progress-every must be positive")
    windows = cumulative_windows(args.step_seconds, args.max_seconds)
    baselines = parse_baselines(args.baseline)
    # Keep the same built-in ordering as the non-temporal summary.
    if not args.baseline:
        baselines = {
            name: Path(path) for name, path in DEFAULT_BASELINES.items()
        }
    normalization = load_normalization(Path(args.normalization_config))
    tasks = set(args.tasks) if args.tasks else None

    summaries = []
    for index, (name, root) in enumerate(baselines.items(), start=1):
        print(
            f"\n[{index}/{len(baselines)}] Processing {name}: {root}",
            flush=True,
        )
        summary = summarize_baseline(
            name,
            root,
            windows,
            args.fps,
            normalization,
            tasks,
            args.progress_every,
        )
        summaries.append(summary)
        if summary.get("error"):
            print(f"  [{name}] ERROR: {summary['error']}", flush=True)
        else:
            final = summary["windows"][windows[-1]]
            print(
                f"  [{name}] complete at 0-{windows[-1]:g}s: "
                f"core={display(final['acwmbench_core_score'])}, "
                f"full={display(final['acwmbench_full_score'])}",
                flush=True,
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": args.fps,
        "windows_seconds": windows,
        "window_semantics": "0 <= frame < window_end_sec * fps",
        "keyarea_normalization": normalization,
        "task_filter": sorted(tasks) if tasks else None,
        "baselines": summaries,
    }
    json_path = output / "acwmbench_temporal_scores.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = csv_rows(summaries)
    csv_path = output / "acwmbench_temporal_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [],
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote: {csv_path.resolve()}")
    print(f"Wrote: {json_path.resolve()}")


if __name__ == "__main__":
    main()
