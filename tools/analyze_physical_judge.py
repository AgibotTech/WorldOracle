#!/usr/bin/env python3
"""Analyze paired per-episode Physical Judge results across baselines."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SUMMARY_METRICS = (
    "keyframe_mean_success",
    "segment_success_rate",
    "format_valid_rate",
)
DISPLAY_METRICS = (
    "keyframe_mean_success",
    "segment_success_rate",
)
METRIC_LABELS = {
    "keyframe_mean_success": "Mean keyframe success",
    "segment_success_rate": "All-criteria keyframe success",
    "format_valid_rate": "Format-valid response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze four-baseline Physical Judge episode results."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis/physical_judge.yaml",
        help="Analysis YAML containing baseline roots and output settings.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or len(baselines) < 2:
        raise ValueError("config.baselines must map at least two names to roots")
    config["minimum_metric_version"] = int(
        config.get("minimum_metric_version", 1)
    )
    config["bootstrap_samples"] = int(config.get("bootstrap_samples", 1000))
    config["seed"] = int(config.get("seed", 0))
    config["fps"] = float(config.get("fps", 30.0))
    config["temporal_bin_seconds"] = float(
        config.get("temporal_bin_seconds", 10.0)
    )
    if config["fps"] <= 0 or config["temporal_bin_seconds"] <= 0:
        raise ValueError("config.fps and temporal_bin_seconds must be positive")
    return config


def discover_results(root: str | Path) -> Dict[Tuple[str, str], Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Baseline root does not exist: {root}")
    results = {}
    for path in sorted(root.glob("*/*/physical_judge.json")):
        relative = path.relative_to(root)
        results[(relative.parts[0], relative.parts[1])] = path
    if not results:
        raise FileNotFoundError(f"No physical_judge.json files under {root}")
    return results


def finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def binary_number(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "pass", "success", "satisfied"}:
            return 1.0
        if normalized in {"0", "false", "no", "fail", "failure", "not satisfied"}:
            return 0.0
    return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def criterion_records(step: Mapping[str, Any]) -> list[dict]:
    constraints = step.get("constraints")
    if isinstance(constraints, list) and constraints:
        return [record for record in constraints if isinstance(record, dict)]
    criteria = step.get("success_criteria")
    if isinstance(criteria, str):
        criteria = [criteria]
    if not isinstance(criteria, list):
        criteria = []
    scores = step.get("aspect_scores")
    if not isinstance(scores, list):
        scores = []
    reasoning = step.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = []
    return [
        {
            "name": f"criterion_{index + 1}",
            "spec": str(criterion),
            "score": scores[index] if index < len(scores) else None,
            "satisfied": (
                bool(scores[index]) if index < len(scores) else None
            ),
            "evidence": str(reasoning[index]) if index < len(reasoning) else "",
        }
        for index, criterion in enumerate(criteria)
    ]


def valid_trial(
    trial: object,
    minimum_version: int,
    require_complete: bool,
) -> bool:
    if not isinstance(trial, dict) or not isinstance(trial.get("key_steps"), list):
        return False
    try:
        version = int(trial.get("metric_version", 0))
    except (TypeError, ValueError):
        return False
    if version < minimum_version:
        return False
    if require_complete and trial.get("status") != "complete":
        return False
    return True


def parse_trial(
    baseline: str,
    task_id: str,
    episode_id: str,
    trial_id: str,
    trial: Mapping[str, Any],
    fps: float,
    temporal_bin_seconds: float,
) -> tuple[dict, list[dict], list[dict]]:
    steps = [step for step in trial.get("key_steps", []) if isinstance(step, dict)]
    criterion_rows, segment_rows = [], []
    criterion_values = []
    for step_order, step in enumerate(steps):
        step_score = finite_number(step.get("score"))
        step_success = binary_number(step.get("satisfied"))
        format_valid = binary_number(step.get("format_valid"))
        frame = finite_number(step.get("frame"))
        time_sec = frame / fps if math.isfinite(frame) else float("nan")
        bin_index = (
            int(math.floor(time_sec / temporal_bin_seconds))
            if math.isfinite(time_sec)
            else None
        )
        bin_start = (
            bin_index * temporal_bin_seconds if bin_index is not None else float("nan")
        )
        bin_end = (
            bin_start + temporal_bin_seconds
            if math.isfinite(bin_start)
            else float("nan")
        )
        segment_rows.append(
            {
                "baseline": baseline,
                "task_id": task_id,
                "episode_id": episode_id,
                "trial_id": trial_id,
                "step_order": step_order,
                "segment_id": step.get("segment_id", step_order),
                "frame": step.get("frame"),
                "source_frame": step.get("source_frame", step.get("frame")),
                "time_sec": time_sec,
                "time_bin_index": bin_index,
                "time_bin_start_sec": bin_start,
                "time_bin_end_sec": bin_end,
                "time_bin": (
                    f"{bin_start:g}-{bin_end:g}s"
                    if math.isfinite(bin_start)
                    else None
                ),
                "is_final": bool(step.get("is_final")),
                "score": step_score,
                "satisfied": step_success,
                "format_valid": format_valid,
            }
        )
        for criterion_index, criterion in enumerate(criterion_records(step)):
            criterion_score = binary_number(
                criterion.get("score", criterion.get("satisfied"))
            )
            criterion_values.append(criterion_score)
            criterion_rows.append(
                {
                    "baseline": baseline,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "trial_id": trial_id,
                    "step_order": step_order,
                    "segment_id": step.get("segment_id", step_order),
                    "frame": step.get("frame"),
                    "source_frame": step.get("source_frame", step.get("frame")),
                    "criterion_index": criterion_index + 1,
                    "criterion": criterion.get("spec", criterion.get("name", "")),
                    "score": criterion_score,
                    "evidence": criterion.get("evidence", ""),
                    "format_valid": format_valid,
                }
            )

    keyframe_mean_success = finite_mean(row["score"] for row in segment_rows)
    summary = {
        "baseline": baseline,
        "task_id": task_id,
        "episode_id": episode_id,
        "trial_id": trial_id,
        "status": trial.get("status"),
        "metric_version": trial.get("metric_version"),
        # Keep the stored value for auditing, but use the value recomputed from
        # all independent keyframes as the canonical analysis metric.
        "trial_score": finite_number(trial.get("score")),
        "keyframe_mean_success": keyframe_mean_success,
        "segment_success_rate": finite_mean(
            row["satisfied"] for row in segment_rows
        ),
        "format_valid_rate": finite_mean(
            row["format_valid"] for row in segment_rows
        ),
        "num_keyframes": len(segment_rows),
        "num_criteria": sum(math.isfinite(value) for value in criterion_values),
        "num_failed_criteria": sum(value == 0.0 for value in criterion_values),
    }
    return summary, segment_rows, criterion_rows


def load_paired_results(
    baselines: Mapping[str, str | Path],
    minimum_version: int,
    require_complete: bool,
    fps: float,
    temporal_bin_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    discovered = {
        baseline: discover_results(root) for baseline, root in baselines.items()
    }
    common_episodes = set.intersection(
        *(set(paths) for paths in discovered.values())
    )
    if not common_episodes:
        raise ValueError("Baselines have no common Physical Judge episodes")
    coverage = {
        baseline: {
            "discovered_episodes": len(paths),
            "paired_episodes": len(common_episodes),
            "excluded_episodes": len(paths) - len(common_episodes),
        }
        for baseline, paths in discovered.items()
    }

    trial_rows, segment_rows, criterion_rows, errors = [], [], [], []
    for task_id, episode_id in sorted(common_episodes):
        episode_data = {}
        for baseline in baselines:
            path = discovered[baseline][(task_id, episode_id)]
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                if not isinstance(data, dict):
                    raise ValueError("episode result is not a JSON object")
                episode_data[baseline] = data
            except (OSError, json.JSONDecodeError, ValueError) as error:
                errors.append(
                    {
                        "baseline": baseline,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "path": str(path),
                        "error": str(error),
                    }
                )
        if len(episode_data) != len(baselines):
            continue
        valid_ids = []
        for baseline, data in episode_data.items():
            ids = {
                str(trial_id)
                for trial_id, trial in data.items()
                if valid_trial(trial, minimum_version, require_complete)
            }
            valid_ids.append(ids)
        common_trials = set.intersection(*valid_ids)
        for trial_id in sorted(common_trials):
            for baseline in baselines:
                summary, segments, criteria = parse_trial(
                    baseline,
                    task_id,
                    episode_id,
                    trial_id,
                    episode_data[baseline][trial_id],
                    fps,
                    temporal_bin_seconds,
                )
                trial_rows.append(summary)
                segment_rows.extend(segments)
                criterion_rows.extend(criteria)

    if not trial_rows:
        raise ValueError("No common valid Physical Judge trials were found")
    return (
        pd.DataFrame(trial_rows),
        pd.DataFrame(segment_rows),
        pd.DataFrame(criterion_rows),
        coverage,
        errors,
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    if len(array) == 1 or samples <= 0:
        value = float(array.mean())
        return value, value
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.choice(array, size=len(array), replace=True)
        estimates[index] = sampled.mean()
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def summarize_baselines(
    trials: pd.DataFrame,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for baseline, group in trials.groupby("baseline"):
        row: Dict[str, Any] = {
            "baseline": baseline,
            "num_trials": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(finite.mean()) if len(finite) else float("nan")
            )
            row[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
            low, high = bootstrap_mean_ci(finite, bootstrap_samples, rng)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "keyframe_mean_success_mean", ascending=False
    )


def summarize_tasks(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["baseline", "task_id"])[list(SUMMARY_METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def summarize_temporal(
    segments: pd.DataFrame,
    group_by_task: bool,
) -> pd.DataFrame:
    valid = segments.dropna(subset=["time_bin_index"]).copy()
    group_columns = ["baseline"]
    if group_by_task:
        group_columns.append("task_id")
    group_columns.extend(
        [
            "time_bin_index",
            "time_bin_start_sec",
            "time_bin_end_sec",
            "time_bin",
        ]
    )
    summary = (
        valid.groupby(group_columns)
        .agg(
            keyframe_score_mean=("score", "mean"),
            keyframe_score_std=("score", "std"),
            keyframe_score_count=("score", "count"),
            keyframe_success_rate=("satisfied", "mean"),
            keyframe_success_count=("satisfied", "count"),
        )
        .reset_index()
        .sort_values(group_columns)
    )
    entity_columns = ["baseline", *(["task_id"] if group_by_task else [])]
    complete_rows = []
    for entity_values, entity_data in valid.groupby(entity_columns):
        if not isinstance(entity_values, tuple):
            entity_values = (entity_values,)
        maximum_bin = int(entity_data["time_bin_index"].max())
        entity = dict(zip(entity_columns, entity_values))
        for bin_index in range(maximum_bin + 1):
            width = float(
                entity_data["time_bin_end_sec"].iloc[0]
                - entity_data["time_bin_start_sec"].iloc[0]
            )
            start = float(bin_index * width)
            complete_rows.append(
                {
                    **entity,
                    "time_bin_index": bin_index,
                    "time_bin_start_sec": start,
                    "time_bin_end_sec": start + width,
                    "time_bin": f"{start:g}-{start + width:g}s",
                }
            )
    complete = pd.DataFrame(complete_rows)
    summary = complete.merge(summary, on=group_columns, how="left")
    for column in ("keyframe_score_count", "keyframe_success_count"):
        summary[column] = summary[column].fillna(0).astype(int)
    return summary.sort_values([*entity_columns, "time_bin_index"])


def correlation_rows(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_columns = ["task_id", "episode_id", "trial_id"]
    for metric in SUMMARY_METRICS:
        pivot = trials.pivot(index=index_columns, columns="baseline", values=metric)
        for left, right in combinations(sorted(pivot.columns), 2):
            paired = pivot[[left, right]].dropna()
            rows.append(
                {
                    "metric": metric,
                    "baseline_a": left,
                    "baseline_b": right,
                    "num_paired_trials": len(paired),
                    "pearson": float(
                        paired[left].corr(paired[right], method="pearson")
                    ),
                    "spearman": float(
                        paired[left].corr(paired[right], method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def pairwise_rows(
    trials: pd.DataFrame,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pivot = trials.pivot(
        index=["task_id", "episode_id", "trial_id"],
        columns="baseline",
        values="keyframe_mean_success",
    )
    rows = []
    for left, right in combinations(sorted(pivot.columns), 2):
        paired = pivot[[left, right]].dropna()
        differences = (paired[left] - paired[right]).to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(differences, bootstrap_samples, rng)
        rows.append(
            {
                "baseline_a": left,
                "baseline_b": right,
                "num_paired_trials": len(differences),
                "mean_score_difference_a_minus_b": finite_mean(differences),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "a_win_rate": float(np.mean(differences > 0)),
                "tie_rate": float(np.mean(differences == 0)),
                "b_win_rate": float(np.mean(differences < 0)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_baseline_overview(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DISPLAY_METRICS), figsize=(15, 4.4))
    names = summary["baseline"].tolist()
    x = np.arange(len(names))
    for axis, metric in zip(axes, DISPLAY_METRICS):
        means = summary[f"{metric}_mean"].to_numpy(dtype=np.float64)
        low = summary[f"{metric}_ci_low"].to_numpy(dtype=np.float64)
        high = summary[f"{metric}_ci_high"].to_numpy(dtype=np.float64)
        errors = np.vstack(
            [
                np.maximum(0.0, means - low),
                np.maximum(0.0, high - means),
            ]
        )
        axis.bar(x, means, yerr=errors, capsize=4)
        axis.set_xticks(x, names, rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("Mean ± bootstrap 95% CI")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Physical Judge baseline comparison on paired trials")
    fig.tight_layout()
    save_figure(fig, output_dir, "baseline_score_overview")


def annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "—" if not math.isfinite(value) else f"{value:.3f}"
            axis.text(column, row, text, ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_task_heatmaps(trials: pd.DataFrame, output_dir: Path) -> None:
    baselines = sorted(trials["baseline"].unique())
    tasks = sorted(trials["task_id"].unique())
    fig, axes = plt.subplots(1, len(DISPLAY_METRICS), figsize=(16, 4.8))
    for axis, metric in zip(axes, DISPLAY_METRICS):
        table = (
            trials.groupby(["task_id", "baseline"])[metric]
            .mean()
            .unstack("baseline")
            .reindex(index=tasks, columns=baselines)
        )
        annotated_heatmap(
            axis,
            table.to_numpy(dtype=np.float64),
            tasks,
            baselines,
            METRIC_LABELS[metric],
        )
    fig.suptitle("Physical Judge performance by task")
    fig.tight_layout()
    save_figure(fig, output_dir, "task_score_heatmaps")


def plot_distributions(trials: pd.DataFrame, output_dir: Path) -> None:
    baselines = sorted(trials["baseline"].unique())
    values = [
        trials.loc[trials["baseline"] == baseline, "keyframe_mean_success"]
        .dropna()
        .to_numpy(dtype=np.float64)
        for baseline in baselines
    ]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.boxplot(values, labels=baselines, showmeans=True)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean keyframe success")
    axis.set_title("Paired episode keyframe-success distributions")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "episode_score_distributions")


def plot_pairwise(pairwise: pd.DataFrame, output_dir: Path) -> None:
    names = sorted(set(pairwise["baseline_a"]) | set(pairwise["baseline_b"]))
    difference = np.zeros((len(names), len(names)), dtype=np.float64)
    win_rate = np.full((len(names), len(names)), 0.5, dtype=np.float64)
    for _, row in pairwise.iterrows():
        left = names.index(row["baseline_a"])
        right = names.index(row["baseline_b"])
        delta = float(row["mean_score_difference_a_minus_b"])
        difference[left, right] = delta
        difference[right, left] = -delta
        win_rate[left, right] = float(row["a_win_rate"])
        win_rate[right, left] = float(row["b_win_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    limit = max(0.01, float(np.nanmax(np.abs(difference))))
    annotated_heatmap(
        axes[0],
        difference,
        names,
        names,
        "Mean score difference (row − column)",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].images[0].set_cmap("coolwarm")
    annotated_heatmap(
        axes[1],
        win_rate,
        names,
        names,
        "Pairwise win rate (row over column)",
    )
    fig.suptitle("Paired Physical Judge model comparison")
    fig.tight_layout()
    save_figure(fig, output_dir, "pairwise_model_comparison")


def plot_correlations(correlations: pd.DataFrame, output_dir: Path) -> None:
    trial_score = correlations[
        correlations["metric"] == "keyframe_mean_success"
    ]
    names = sorted(
        set(trial_score["baseline_a"]) | set(trial_score["baseline_b"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for axis, method in zip(axes, ("pearson", "spearman")):
        matrix = np.eye(len(names), dtype=np.float64)
        for _, row in trial_score.iterrows():
            left = names.index(row["baseline_a"])
            right = names.index(row["baseline_b"])
            matrix[left, right] = matrix[right, left] = float(row[method])
        annotated_heatmap(
            axis,
            matrix,
            names,
            names,
            f"Trial-score {method.capitalize()} correlation",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.images[0].set_cmap("coolwarm")
    fig.suptitle("Same-episode Physical Judge score correlations")
    fig.tight_layout()
    save_figure(fig, output_dir, "baseline_score_correlations")


def temporal_axis(
    axis: plt.Axes,
    temporal: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    starts = sorted(temporal["time_bin_start_sec"].dropna().unique())
    for baseline, data in temporal.groupby("baseline"):
        data = data.sort_values("time_bin_start_sec")
        centers = (
            data["time_bin_start_sec"].to_numpy(dtype=np.float64)
            + data["time_bin_end_sec"].to_numpy(dtype=np.float64)
        ) / 2.0
        axis.plot(
            centers,
            data[value_column],
            marker="o",
            linewidth=1.8,
            label=baseline,
        )
    if starts:
        width = float(
            temporal["time_bin_end_sec"].iloc[0]
            - temporal["time_bin_start_sec"].iloc[0]
        )
        centers = [start + width / 2.0 for start in starts]
        labels = [f"{start:g}-{start + width:g}s" for start in starts]
        axis.set_xticks(centers, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Keyframe time interval")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)


def plot_temporal_trends(
    temporal: pd.DataFrame,
    task_temporal: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    temporal_axis(
        axes[0],
        temporal,
        "keyframe_score_mean",
        "Mean keyframe score",
        "Physical-state score over time",
    )
    temporal_axis(
        axes[1],
        temporal,
        "keyframe_success_rate",
        "Successful keyframe fraction",
        "Full-criterion success over time",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle("Physical Judge trends by 10-second execution interval")
    fig.tight_layout()
    save_figure(fig, output_dir, "temporal_score_trends")

    tasks = sorted(task_temporal["task_id"].unique())
    fig, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(14, 4.0 * len(tasks)),
        squeeze=False,
    )
    for row, task_id in enumerate(tasks):
        task_data = task_temporal[task_temporal["task_id"] == task_id]
        temporal_axis(
            axes[row, 0],
            task_data,
            "keyframe_score_mean",
            "Mean keyframe score",
            f"{task_id}: mean keyframe success",
        )
        temporal_axis(
            axes[row, 1],
            task_data,
            "keyframe_success_rate",
            "All-criteria success fraction",
            f"{task_id}: all-criteria keyframe success",
        )
    if tasks:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle("Task-specific Physical Judge trends by execution interval")
    fig.tight_layout()
    save_figure(fig, output_dir, "task_temporal_score_trends")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(
        config.get("output_dir", "output/physical_judge_results_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    trials, segments, criteria, coverage, errors = load_paired_results(
        config["baselines"],
        config["minimum_metric_version"],
        bool(config.get("require_complete_status", True)),
        config["fps"],
        config["temporal_bin_seconds"],
    )
    baseline_summary = summarize_baselines(
        trials,
        config["bootstrap_samples"],
        rng,
    )
    task_summary = summarize_tasks(trials)
    correlations = correlation_rows(trials)
    pairwise = pairwise_rows(
        trials,
        config["bootstrap_samples"],
        rng,
    )
    temporal = summarize_temporal(segments, group_by_task=False)
    task_temporal = summarize_temporal(segments, group_by_task=True)
    failures = (
        criteria[criteria["score"] == 0.0].copy()
        if "score" in criteria
        else criteria.copy()
    )

    trials.to_csv(output_dir / "trial_scores.csv", index=False)
    segments.to_csv(output_dir / "segment_scores.csv", index=False)
    criteria.to_csv(output_dir / "criterion_results.csv", index=False)
    failures.to_csv(output_dir / "criterion_failures.csv", index=False)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    task_summary.to_csv(output_dir / "task_summary.csv", index=False)
    correlations.to_csv(output_dir / "baseline_correlations.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    temporal.to_csv(output_dir / "temporal_summary.csv", index=False)
    task_temporal.to_csv(output_dir / "task_temporal_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)

    ranking = [
        {
            "rank": index + 1,
            "baseline": str(row["baseline"]),
            "keyframe_mean_success": float(
                row["keyframe_mean_success_mean"]
            ),
            "segment_success_rate": float(row["segment_success_rate_mean"]),
        }
        for index, (_, row) in enumerate(baseline_summary.iterrows())
    ]
    summary = {
        "config": str(Path(args.config).resolve()),
        "baselines": config["baselines"],
        "minimum_metric_version": config["minimum_metric_version"],
        "require_complete_status": bool(
            config.get("require_complete_status", True)
        ),
        "fps": config["fps"],
        "temporal_bin_seconds": config["temporal_bin_seconds"],
        "num_paired_trial_rows": int(len(trials)),
        "num_unique_paired_trials": int(
            len(trials[["task_id", "episode_id", "trial_id"]].drop_duplicates())
        ),
        "num_criteria": int(len(criteria)),
        "num_failed_criteria": int(len(failures)),
        "coverage": coverage,
        "ranking": ranking,
        "score_semantics": (
            "The primary score is the unweighted mean of all independent "
            "keyframe/segment scores in an episode. All-criteria keyframe "
            "success requires every criterion in that keyframe to pass. "
            "Criterion-level satisfaction and final-state success are "
            "intentionally not used as aggregate metrics."
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    plot_baseline_overview(baseline_summary, output_dir)
    plot_task_heatmaps(trials, output_dir)
    plot_distributions(trials, output_dir)
    plot_pairwise(pairwise, output_dir)
    plot_correlations(correlations, output_dir)
    for suffix in (".png", ".pdf"):
        (output_dir / f"segment_score_progression{suffix}").unlink(missing_ok=True)
    if temporal.empty:
        raise ValueError(
            "No valid keyframe frame indices were available for time-bin analysis"
        )
    plot_temporal_trends(temporal, task_temporal, output_dir)
    print(f"Saved Physical Judge analysis to {output_dir}")


if __name__ == "__main__":
    main()
