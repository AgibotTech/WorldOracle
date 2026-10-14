#!/usr/bin/env python3
"""Analyze completed KeyArea Visual Quality JSON files across baselines.

Unlike ``analyze_keyarea_downsampling.py``, this tool does not simulate other
strides. It reads the completed evaluation files at one expected stride,
normalizes every mask's PSNR/LPIPS/V-JEPA similarity before any averaging, and reports overall,
source-specific, task-specific, and temporal model performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SOURCES = ("object", "robot_arm")
PSNR_METRICS = ("psnr",)
RAW_METRICS = ("psnr", "lpips", "vjepa_similarity")
NORMALIZED_METRICS = (
    "psnr_score",
    "lpips_score",
    "vjepa_score",
    "quality_score",
)
SCORE_COLUMNS = (
    "overall_quality_score",
    "object_quality_score",
    "robot_arm_quality_score",
)
SCORE_LABELS = {
    "overall_quality_score": "Overall quality",
    "object_quality_score": "Object quality",
    "robot_arm_quality_score": "Robot-arm quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed KeyArea results for multiple baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis/keyarea_results.yaml",
        help="YAML with baseline roots, normalization, timing, and output path.",
    )
    return parser.parse_args()


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def normalize_frame(
    metrics: Mapping[str, float],
    normalization: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    psnr_cfg = normalization["psnr"]
    lpips_cfg = normalization["lpips"]
    vjepa_cfg = normalization["vjepa_similarity"]
    psnr_unit = float(
        np.clip(
            (metrics["psnr"] - psnr_cfg["min"])
            / (psnr_cfg["max"] - psnr_cfg["min"]),
            0.0,
            1.0,
        )
    )
    lpips_unit = float(
        np.clip(
            (lpips_cfg["max"] - metrics["lpips"])
            / (lpips_cfg["max"] - lpips_cfg["min"]),
            0.0,
            1.0,
        )
    )
    vjepa_unit = float(
        np.clip(
            (metrics["vjepa_similarity"] - vjepa_cfg["min"])
            / (vjepa_cfg["max"] - vjepa_cfg["min"]),
            0.0,
            1.0,
        )
    )
    return {
        "psnr_score": 100.0 * psnr_unit,
        "lpips_score": 100.0 * lpips_unit,
        "vjepa_score": 100.0 * vjepa_unit,
        "quality_score": 100.0 * (psnr_unit * lpips_unit * vjepa_unit) ** (1.0 / 3.0),
    }


def discover_results(root: str | Path) -> Dict[Tuple[str, str], Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Baseline root does not exist: {root}")
    results = {}
    for path in sorted(root.glob("*/*/keyarea_visual_quality.json")):
        relative = path.relative_to(root)
        results[(relative.parts[0], relative.parts[1])] = path
    if not results:
        raise FileNotFoundError(f"No KeyArea JSON files found under {root}")
    return results


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or len(baselines) < 2:
        raise ValueError("config.baselines must contain at least two models")
    normalization = config.get("normalization") or {}
    for metric in RAW_METRICS:
        values = normalization.get(metric)
        if not isinstance(values, dict) or not {"min", "max"} <= set(values):
            raise ValueError(f"normalization.{metric} requires min and max")
        values["min"] = float(values["min"])
        values["max"] = float(values["max"])
        if values["max"] <= values["min"]:
            raise ValueError(f"normalization.{metric}.max must exceed min")
    config["normalization"] = normalization
    return config


def load_episode(
    path: Path,
    normalization: Mapping[str, Mapping[str, float]],
    expected_stride: int,
    minimum_metric_version: int,
    fps: float,
    bin_seconds: float,
    cumulative_end_seconds: Sequence[float],
) -> Tuple[dict, list[dict], list[dict]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    stride = int(data.get("downsample_rate", -1))
    version = int(data.get("metric_version", -1))
    if stride != expected_stride:
        raise ValueError(
            f"{path}: downsample_rate={stride}, expected {expected_stride}"
        )
    if version < minimum_metric_version:
        raise ValueError(
            f"{path}: metric_version={version}, expected >= {minimum_metric_version}"
        )

    source_frames: Dict[str, Dict[int, dict]] = {}
    for source in SOURCES:
        source_data = (data.get("mask_sources") or {}).get(source)
        if not isinstance(source_data, dict):
            raise ValueError(f"{path}: missing mask_sources.{source}")
        records = source_data.get("per_frame")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: missing {source} per_frame records")
        fallback_masks: Dict[int, list[dict]] = {}
        per_mask = source_data.get("per_mask")
        if isinstance(per_mask, list):
            for mask_record in per_mask:
                if isinstance(mask_record, dict) and "frame" in mask_record:
                    fallback_masks.setdefault(
                        int(mask_record["frame"]),
                        [],
                    ).append(mask_record)
        parsed = {}
        for record in records:
            frame = int(record["frame"])
            mask_records = record.get("masks")
            if not isinstance(mask_records, list):
                mask_records = fallback_masks.get(frame, [])
            normalized_masks = []
            for mask_record in mask_records:
                if not isinstance(mask_record, dict):
                    continue
                raw_values = {
                    metric: float(mask_record[metric])
                    for metric in RAW_METRICS
                }
                for metric in PSNR_METRICS:
                    if not math.isfinite(raw_values[metric]):
                        raw_values[metric] = float(
                            normalization[metric]["max"]
                        )
                if not all(
                    math.isfinite(raw_values[metric])
                    for metric in ("lpips", "vjepa_similarity")
                ):
                    continue
                normalized_masks.append(
                    {
                        **raw_values,
                        **normalize_frame(raw_values, normalization),
                    }
                )
            if not normalized_masks:
                continue
            parsed[frame] = {
                "frame": frame,
                "num_masks": len(normalized_masks),
                **{
                    metric: finite_mean(
                        mask_record[metric]
                        for mask_record in normalized_masks
                    )
                    for metric in (
                        *RAW_METRICS,
                        *NORMALIZED_METRICS,
                    )
                },
            }
        source_frames[source] = parsed

    episode = {
        "metric_version": version,
        "downsample_rate": stride,
        "mask_dilation_pixels": int(data.get("mask_dilation_pixels", 0)),
    }
    for source in SOURCES:
        records = list(source_frames[source].values())
        episode[f"{source}_num_frames"] = len(records)
        for metric in RAW_METRICS:
            episode[f"{source}_{metric}"] = finite_mean(
                r[metric] for r in records
            )
        for metric in NORMALIZED_METRICS:
            episode[f"{source}_{metric}"] = finite_mean(
                r[metric] for r in records
            )
    for metric in NORMALIZED_METRICS:
        episode[f"overall_{metric}"] = finite_mean(
            episode[f"{source}_{metric}"] for source in SOURCES
        )

    temporal_rows = []
    all_frames = sorted(set().union(*(set(frames) for frames in source_frames.values())))
    bins: Dict[int, list[int]] = {}
    for frame in all_frames:
        index = int((frame / fps) // bin_seconds)
        bins.setdefault(index, []).append(frame)
    for index, frames in sorted(bins.items()):
        row = {
            "time_bin_index": index,
            "time_start_sec": index * bin_seconds,
            "time_end_sec": (index + 1) * bin_seconds,
            "time_bin": f"{index * bin_seconds:g}-{(index + 1) * bin_seconds:g}s",
        }
        for source in SOURCES:
            source_frame_ids = [
                frame for frame in frames if frame in source_frames[source]
            ]
            row[f"{source}_num_frames"] = len(source_frame_ids)
            for metric in NORMALIZED_METRICS:
                row[f"{source}_{metric}"] = finite_mean(
                    source_frames[source][frame][metric]
                    for frame in source_frame_ids
                )
        for metric in NORMALIZED_METRICS:
            row[f"overall_{metric}"] = finite_mean(
                row[f"{source}_{metric}"] for source in SOURCES
            )
        temporal_rows.append(row)

    cumulative_rows = []
    for end_sec in cumulative_end_seconds:
        label = f"0-{end_sec:g}s"
        row = {
            "window_end_sec": float(end_sec),
            "time_window": label,
        }
        for source in SOURCES:
            records = [
                record
                for record in source_frames[source].values()
                if 0.0 <= record["frame"] / fps < end_sec
            ]
            row[f"{source}_num_frames"] = len(records)
            for metric in NORMALIZED_METRICS:
                row[f"{source}_{metric}"] = finite_mean(
                    record[metric] for record in records
                )
        for metric in NORMALIZED_METRICS:
            row[f"overall_{metric}"] = finite_mean(
                row[f"{source}_{metric}"] for source in SOURCES
            )
        cumulative_rows.append(row)
    return episode, temporal_rows, cumulative_rows


def summarize(
    data: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    summary = (
        data.groupby(list(group_columns))[list(SCORE_COLUMNS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    for metric in SCORE_COLUMNS:
        summary[f"{metric}_ci95"] = (
            1.96
            * summary[f"{metric}_std"].fillna(0)
            / np.sqrt(summary[f"{metric}_count"].clip(lower=1))
        )
    return summary


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_overview(summary: pd.DataFrame, output_dir: Path) -> None:
    baselines = list(summary["baseline"])
    x = np.arange(len(baselines))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, metric in zip(axes, SCORE_COLUMNS):
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        ci = summary[f"{metric}_ci95"].to_numpy(dtype=float)
        bars = axis.bar(x, mean, yerr=ci, capsize=4)
        axis.set_xticks(x, labels=baselines, rotation=20, ha="right")
        axis.set_title(SCORE_LABELS[metric])
        axis.set_ylabel("Normalized quality score (0-100)")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, mean):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("KeyArea baseline performance at the evaluated stride")
    save_figure(fig, output_dir, "baseline_score_overview")


def plot_episode_distributions(episodes: pd.DataFrame, output_dir: Path) -> None:
    baselines = sorted(episodes["baseline"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, metric in zip(axes, SCORE_COLUMNS):
        values = [
            episodes.loc[episodes["baseline"] == baseline, metric].dropna()
            for baseline in baselines
        ]
        axis.boxplot(values, tick_labels=baselines, showfliers=False)
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(SCORE_LABELS[metric])
        axis.set_ylabel("Episode quality score (0-100)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Episode-level score distributions")
    save_figure(fig, output_dir, "episode_score_distributions")


def plot_source_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    for _, row in summary.iterrows():
        x = row["object_quality_score_mean"]
        y = row["robot_arm_quality_score_mean"]
        axis.scatter(x, y, s=55)
        axis.annotate(row["baseline"], (x, y), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Object quality score (0-100)")
    axis.set_ylabel("Robot-arm quality score (0-100)")
    axis.set_title("Object fidelity versus robot-arm fidelity")
    axis.grid(alpha=0.25)
    save_figure(fig, output_dir, "object_robot_arm_tradeoff")


def plot_temporal(temporal: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for axis, metric in zip(axes, SCORE_COLUMNS):
        for baseline, data in temporal.groupby("baseline"):
            data = data.sort_values("time_bin_index")
            x = data["time_start_sec"].to_numpy(dtype=float)
            mean = data[f"{metric}_mean"].to_numpy(dtype=float)
            ci = data[f"{metric}_ci95"].to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", label=baseline)
            axis.fill_between(x, mean - ci, mean + ci, alpha=0.12)
        axis.set_title(SCORE_LABELS[metric])
        axis.set_xlabel("Episode time (seconds)")
        axis.set_ylabel("Mean quality score ± 95% CI")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Model performance by 10-second episode stage")
    save_figure(fig, output_dir, "temporal_quality_scores")


def annotated_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
) -> None:
    values = matrix.to_numpy(dtype=float)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if math.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < np.nanmedian(values) else "black",
                )
    axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)


def plot_temporal_heatmaps(temporal: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    for axis, metric in zip(axes, SCORE_COLUMNS):
        matrix = temporal.pivot(
            index="baseline",
            columns="time_bin",
            values=f"{metric}_mean",
        )
        ordered = sorted(
            matrix.columns,
            key=lambda value: float(str(value).split("-")[0]),
        )
        annotated_heatmap(axis, matrix[ordered], SCORE_LABELS[metric])
    fig.suptitle("Model × temporal-stage quality heatmaps")
    fig.tight_layout()
    save_figure(fig, output_dir, "temporal_quality_heatmaps")


def save_cumulative_overall_table(
    cumulative_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = cumulative_summary.sort_values("window_end_sec")
    matrix = ordered.pivot(
        index="baseline",
        columns="time_window",
        values="overall_quality_score_mean",
    )
    window_order = (
        ordered[["window_end_sec", "time_window"]]
        .drop_duplicates()
        .sort_values("window_end_sec")["time_window"]
        .tolist()
    )
    matrix = matrix.reindex(columns=window_order)
    matrix.to_csv(output_dir / "cumulative_overall_score_comparison.csv")

    cell_text = [
        [f"{value:.2f}" if math.isfinite(value) else "—" for value in row]
        for row in matrix.to_numpy(dtype=float)
    ]
    figure_width = max(9.0, 1.45 * (len(matrix.columns) + 1))
    fig, axis = plt.subplots(figsize=(figure_width, 1.0 + 0.55 * len(matrix.index)))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        rowLabels=matrix.index,
        colLabels=matrix.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    values = matrix.to_numpy(dtype=float)
    for column in range(values.shape[1]):
        finite_column = values[:, column]
        if np.isfinite(finite_column).any():
            best_row = int(np.nanargmax(finite_column))
            table[(best_row + 1, column)].set_facecolor("#c7e9c0")
    axis.set_title(
        "Baseline overall quality by cumulative episode window",
        pad=14,
        fontsize=12,
    )
    save_figure(fig, output_dir, "cumulative_overall_score_comparison")


def plot_task_heatmaps(task_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for axis, metric in zip(axes, SCORE_COLUMNS):
        matrix = task_summary.pivot(
            index="baseline",
            columns="task_id",
            values=f"{metric}_mean",
        )
        annotated_heatmap(axis, matrix, SCORE_LABELS[metric])
    fig.suptitle("Model × task quality heatmaps")
    fig.tight_layout()
    save_figure(fig, output_dir, "task_quality_heatmaps")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(
        config.get("output_dir", "output/keyarea_results_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_stride = int(config.get("expected_downsample_rate", 10))
    minimum_version = int(config.get("minimum_metric_version", 5))
    fps = float(config.get("fps", 30.0))
    bin_seconds = float(config.get("temporal_bin_seconds", 10.0))
    cumulative_end_seconds = sorted(
        {
            float(value)
            for value in config.get(
                "cumulative_end_seconds",
                (10, 20, 30, 40, 50, 60),
            )
        }
    )
    if fps <= 0 or bin_seconds <= 0:
        raise ValueError("fps and temporal_bin_seconds must be > 0")
    if not cumulative_end_seconds or cumulative_end_seconds[0] <= 0:
        raise ValueError("cumulative_end_seconds must contain positive values")

    discovered = {
        baseline: discover_results(root)
        for baseline, root in config["baselines"].items()
    }
    common = set.intersection(*(set(paths) for paths in discovered.values()))
    if not common:
        raise ValueError("The baselines have no common task/episode results")

    episode_rows, temporal_rows, cumulative_rows, errors = [], [], [], []
    for task_id, episode_id in sorted(common):
        for baseline in config["baselines"]:
            path = discovered[baseline][(task_id, episode_id)]
            try:
                episode, temporal, cumulative = load_episode(
                    path,
                    config["normalization"],
                    expected_stride,
                    minimum_version,
                    fps,
                    bin_seconds,
                    cumulative_end_seconds,
                )
            except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                errors.append(
                    {
                        "baseline": baseline,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "path": str(path),
                        "error": str(error),
                    }
                )
                continue
            identity = {
                "baseline": baseline,
                "task_id": task_id,
                "episode_id": episode_id,
            }
            episode_rows.append({**identity, **episode})
            temporal_rows.extend({**identity, **row} for row in temporal)
            cumulative_rows.extend({**identity, **row} for row in cumulative)

    episodes = pd.DataFrame(episode_rows)
    if episodes.empty:
        raise ValueError("No valid completed evaluation files were loaded")
    # Keep only episodes valid for all baselines after schema checks.
    valid_counts = episodes.groupby(["task_id", "episode_id"])["baseline"].nunique()
    valid_keys = set(valid_counts[valid_counts == len(config["baselines"])].index)
    episodes = episodes[
        episodes.apply(
            lambda row: (row["task_id"], row["episode_id"]) in valid_keys,
            axis=1,
        )
    ].copy()
    temporal = pd.DataFrame(temporal_rows)
    temporal = temporal[
        temporal.apply(
            lambda row: (row["task_id"], row["episode_id"]) in valid_keys,
            axis=1,
        )
    ].copy()
    cumulative = pd.DataFrame(cumulative_rows)
    cumulative = cumulative[
        cumulative.apply(
            lambda row: (row["task_id"], row["episode_id"]) in valid_keys,
            axis=1,
        )
    ].copy()

    baseline_summary = summarize(episodes, ["baseline"])
    for metric in SCORE_COLUMNS:
        baseline_summary[f"{metric}_rank"] = baseline_summary[
            f"{metric}_mean"
        ].rank(method="min", ascending=False)
    task_summary = summarize(episodes, ["baseline", "task_id"])
    temporal_summary = summarize(
        temporal,
        [
            "baseline",
            "time_bin_index",
            "time_start_sec",
            "time_end_sec",
            "time_bin",
        ],
    )
    cumulative_summary = summarize(
        cumulative,
        [
            "baseline",
            "window_end_sec",
            "time_window",
        ],
    )

    episodes.to_csv(output_dir / "episode_scores.csv", index=False)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    task_summary.to_csv(output_dir / "task_summary.csv", index=False)
    temporal.to_csv(output_dir / "temporal_episode_scores.csv", index=False)
    temporal_summary.to_csv(
        output_dir / "temporal_baseline_summary.csv", index=False
    )
    cumulative.to_csv(output_dir / "cumulative_episode_scores.csv", index=False)
    cumulative_summary.to_csv(
        output_dir / "cumulative_baseline_summary.csv",
        index=False,
    )
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)

    coverage = {
        baseline: {
            "discovered": len(paths),
            "common_before_validation": len(common),
            "paired_valid": len(valid_keys),
        }
        for baseline, paths in discovered.items()
    }
    summary = {
        "expected_downsample_rate": expected_stride,
        "minimum_metric_version": minimum_version,
        "fps": fps,
        "temporal_bin_seconds": bin_seconds,
        "cumulative_end_seconds": cumulative_end_seconds,
        "paired_valid_episodes": len(valid_keys),
        "normalization": config["normalization"],
        "coverage": coverage,
        "num_load_errors": len(errors),
        "score_semantics": (
            "Normalize PSNR, LPIPS, and V-JEPA similarity independently for every mask and "
            "compute that mask's three-way geometric-mean quality; average mask scores per frame; "
            "temporally average each source; equally average object and "
            "robot_arm source scores for overall."
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    plot_baseline_overview(baseline_summary, output_dir)
    plot_episode_distributions(episodes, output_dir)
    plot_source_tradeoff(baseline_summary, output_dir)
    plot_temporal(temporal_summary, output_dir)
    plot_temporal_heatmaps(temporal_summary, output_dir)
    save_cumulative_overall_table(cumulative_summary, output_dir)
    plot_task_heatmaps(task_summary, output_dir)
    print(
        f"Saved analysis of {len(valid_keys)} paired episodes to {output_dir}"
    )


if __name__ == "__main__":
    main()
