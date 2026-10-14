#!/usr/bin/env python3
"""Analyze KeyArea downsampling from existing stride=1 episode results.

This script never recomputes LPIPS and never modifies episode JSON files. It
loads the full-frame ``keyarea_visual_quality.json`` files, selects frames whose
index is divisible by each candidate stride, re-applies the existing hierarchy
(masks -> source/frame -> object+robot_arm -> episode), and compares every
candidate against stride=1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PSNR_METRICS = ("psnr",)
RAW_METRICS = ("psnr", "lpips", "vjepa_similarity")
NORMALIZED_METRICS = (
    "quality_score",
    "psnr_score",
    "lpips_score",
    "vjepa_score",
)
SOURCE_QUALITY_METRICS = tuple(
    f"{source}_{metric}"
    for source in ("object", "robot_arm")
    for metric in NORMALIZED_METRICS
)
ALL_METRICS = RAW_METRICS + NORMALIZED_METRICS
SOURCES = ("object", "robot_arm")
QUALITY_COLUMNS = (
    "quality_score",
    "object_quality_score",
    "robot_arm_quality_score",
)
QUALITY_LABELS = {
    "quality_score": "Total quality",
    "object_quality_score": "Object quality",
    "robot_arm_quality_score": "Robot-arm quality",
}


@dataclass
class Reservoir:
    """Fixed-memory uniform sample for raw per-mask distribution plots."""

    limit: int
    rng: random.Random

    def __post_init__(self) -> None:
        self.values: List[float] = []
        self.seen = 0

    def add(self, value: object) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return
        self.seen += 1
        val = float(value)
        if len(self.values) < self.limit:
            self.values.append(val)
            return
        index = self.rng.randrange(self.seen)
        if index < self.limit:
            self.values[index] = val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze KeyArea stride correlations from stride=1 results."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis/keyarea_downsampling.yaml",
        help="Analysis YAML containing baselines, strides, and output_dir.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or len(baselines) < 2:
        raise ValueError("config.baselines must map at least two names to dataset roots")
    strides = sorted({int(value) for value in config.get("strides", [])})
    if not strides or strides[0] != 1 or any(value < 1 for value in strides):
        raise ValueError("config.strides must contain 1 and only positive integers")
    normalization = config.get("normalization") or {}
    for metric, defaults in (
        ("psnr", {"min": 15.0, "max": 35.0}),
        ("lpips", {"min": 0.1, "max": 0.3}),
        ("vjepa_similarity", {"min": 0.70, "max": 0.95}),
    ):
        values = normalization.setdefault(metric, defaults)
        if not isinstance(values, dict) or "min" not in values or "max" not in values:
            raise ValueError(f"normalization.{metric} must define min and max")
        values["min"] = float(values["min"])
        values["max"] = float(values["max"])
        if values["max"] <= values["min"]:
            raise ValueError(f"normalization.{metric}.max must be greater than min")
    config["normalization"] = normalization
    config["strides"] = strides
    return config


def discover_results(root: str | Path) -> Dict[Tuple[str, str], Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Baseline root does not exist: {root}")
    results: Dict[Tuple[str, str], Path] = {}
    for path in sorted(root.glob("*/*/keyarea_visual_quality.json")):
        relative = path.relative_to(root)
        results[(relative.parts[0], relative.parts[1])] = path
    if not results:
        raise FileNotFoundError(f"No keyarea_visual_quality.json files under {root}")
    return results


def paired_result_paths(
    baselines: Mapping[str, str | Path],
) -> Tuple[Dict[str, Dict[Tuple[str, str], Path]], List[Tuple[str, str]], dict]:
    discovered = {
        name: discover_results(root) for name, root in baselines.items()
    }
    common = set.intersection(*(set(paths) for paths in discovered.values()))
    if not common:
        raise ValueError("The baselines have no common task/episode results")
    coverage = {
        name: {
            "discovered": len(paths),
            "paired": len(common),
            "excluded": len(paths) - len(common),
        }
        for name, paths in discovered.items()
    }
    return discovered, sorted(common), coverage


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def normalized_raw_value(
    value: object,
    metric: str,
    psnr_inf_cap: float,
) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isfinite(number):
        return number
    if metric in PSNR_METRICS and number == float("inf"):
        return float(psnr_inf_cap)
    return None


def normalize_quality_scores(
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


def load_full_frame_episode(
    path: Path,
    reservoirs: Mapping[Tuple[str, str, str], Reservoir],
    baseline: str,
    psnr_inf_cap: float,
    normalization: Mapping[str, Mapping[str, float]],
) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if int(data.get("downsample_rate", -1)) != 1:
        raise ValueError(
            f"{path} has downsample_rate={data.get('downsample_rate')}, expected 1. "
            "This tool needs full-frame results to simulate larger strides. "
            "To compare baseline scores from existing downsampled results, run "
            "tools/analyze_keyarea_results.py instead."
        )
    if int(data.get("metric_version", -1)) < 12:
        raise ValueError(
            f"{path} uses metric_version={data.get('metric_version')}; "
            "metric_version >= 12 is required for PSNR/LPIPS/V-JEPA records"
        )

    source_frames: Dict[str, Dict[int, Dict[str, float]]] = {}
    source_counts = {}
    for source in SOURCES:
        source_data = (data.get("mask_sources") or {}).get(source)
        if not isinstance(source_data, dict):
            raise ValueError(f"{path} is missing mask_sources.{source}")
        per_mask = source_data.get("per_mask")
        if not isinstance(per_mask, list):
            raise ValueError(f"{path} is missing mask_sources.{source}.per_mask")

        grouped: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {metric: [] for metric in ALL_METRICS}
        )
        capped_psnr = 0
        invalid_values = 0
        for record in per_mask:
            frame = int(record["frame"])
            raw_values = {}
            for metric in RAW_METRICS:
                raw_value = record.get(metric)
                value = normalized_raw_value(raw_value, metric, psnr_inf_cap)
                if value is None:
                    invalid_values += 1
                    raw_values = {}
                    break
                if metric in PSNR_METRICS and raw_value == float("inf"):
                    capped_psnr += 1
                raw_values[metric] = value
            if not raw_values:
                continue
            scores = normalize_quality_scores(raw_values, normalization)
            for metric, value in {**raw_values, **scores}.items():
                grouped[frame][metric].append(value)
            for metric, value in raw_values.items():
                reservoirs[(baseline, source, metric)].add(value)
        source_frames[source] = {
            frame: {
                metric: finite_mean(metric_values[metric])
                for metric in ALL_METRICS
            }
            for frame, metric_values in grouped.items()
        }
        source_counts[source] = {
            "raw_per_mask_records": len(per_mask),
            "valid_frames": len(source_frames[source]),
            "capped_infinite_psnr": capped_psnr,
            "invalid_metric_values": invalid_values,
        }

    return source_frames, source_counts


def episode_scores_for_strides(
    source_frames: Mapping[str, Mapping[int, Mapping[str, float]]],
    strides: Sequence[int],
) -> List[dict]:
    rows = []
    all_frame_ids = sorted(
        set().union(*(set(source_frames[source]) for source in SOURCES))
    )
    for stride in strides:
        selected = [frame for frame in all_frame_ids if frame % stride == 0]
        row = {
            "stride": stride,
            "num_frames": len(selected),
            "full_num_frames": len(all_frame_ids),
            "frame_fraction": (
                len(selected) / len(all_frame_ids) if all_frame_ids else float("nan")
            ),
        }

        # Per-frame source scores were already averaged from independently
        # normalized per-mask scores. Source episode scores are temporal means.
        for source in SOURCES:
            selected_source_frames = [
                frame for frame in selected if frame in source_frames[source]
            ]
            row[f"{source}_num_frames"] = len(selected_source_frames)
            for metric in RAW_METRICS:
                row[f"{source}_{metric}"] = finite_mean(
                    source_frames[source][frame][metric]
                    for frame in selected_source_frames
                )
            for metric in NORMALIZED_METRICS:
                row[f"{source}_{metric}"] = finite_mean(
                    source_frames[source][frame][metric]
                    for frame in selected_source_frames
                )

        # The total score equally averages the two source-level temporal means.
        # Thus object and robot_arm retain equal weight even if their masks are
        # valid on different numbers of frames.
        for metric in NORMALIZED_METRICS:
            row[metric] = finite_mean(
                row[f"{source}_{metric}"] for source in SOURCES
            )

        # Retain combined raw metrics for diagnostics only. Normalized totals
        # above are intentionally not derived from these combined raw values.
        for metric in RAW_METRICS:
            row[metric] = finite_mean(
                finite_mean(
                    source_frames[source][frame][metric]
                    for source in SOURCES
                    if frame in source_frames[source]
                )
                for frame in selected
            )
        rows.append(row)
    return rows


def temporal_scores_for_strides(
    source_frames: Mapping[str, Mapping[int, Mapping[str, float]]],
    strides: Sequence[int],
    fps: float,
    bin_seconds: float,
) -> List[dict]:
    """Aggregate independently normalized source scores into time windows."""
    if fps <= 0 or bin_seconds <= 0:
        raise ValueError("fps and temporal_bin_seconds must be positive")
    all_frame_ids = sorted(
        set().union(*(set(source_frames[source]) for source in SOURCES))
    )
    rows = []
    for stride in strides:
        selected = [frame for frame in all_frame_ids if frame % stride == 0]
        by_bin: Dict[int, List[int]] = defaultdict(list)
        for frame in selected:
            by_bin[int((frame / fps) // bin_seconds)].append(frame)
        for bin_index, frames in sorted(by_bin.items()):
            row = {
                "stride": int(stride),
                "time_bin_index": int(bin_index),
                "time_start_sec": float(bin_index * bin_seconds),
                "time_end_sec": float((bin_index + 1) * bin_seconds),
                "time_bin": (
                    f"{bin_index * bin_seconds:g}-"
                    f"{(bin_index + 1) * bin_seconds:g}s"
                ),
                "num_frames": len(frames),
            }
            for source in SOURCES:
                source_values = [
                    source_frames[source][frame]["quality_score"]
                    for frame in frames
                    if frame in source_frames[source]
                ]
                row[f"{source}_num_frames"] = len(source_values)
                row[f"{source}_quality_score"] = finite_mean(source_values)
            row["quality_score"] = finite_mean(
                row[f"{source}_quality_score"] for source in SOURCES
            )
            rows.append(row)
    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson(x_rank, y_rank)


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's CCC: correlation penalized by mean and scale disagreement."""
    if len(x) < 2:
        return float("nan")
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    var_x, var_y = float(np.var(x)), float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else float("nan")
    return float(2.0 * covariance / denominator)


def kendall_pairwise(reference: Mapping[str, float], candidate: Mapping[str, float]) -> Tuple[float, float]:
    names = sorted(set(reference) & set(candidate))
    concordant = 0
    discordant = 0
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            ref_sign = np.sign(reference[left] - reference[right])
            cand_sign = np.sign(candidate[left] - candidate[right])
            if ref_sign == 0 or cand_sign == 0:
                continue
            if ref_sign == cand_sign:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan"), float("nan")
    return (concordant - discordant) / total, concordant / total


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    if len(x) < 2 or samples <= 0:
        return {
            "pearson_ci_low": float("nan"),
            "pearson_ci_high": float("nan"),
            "spearman_ci_low": float("nan"),
            "spearman_ci_high": float("nan"),
        }
    pearsons, spearmans = [], []
    for _ in range(samples):
        indices = rng.integers(0, len(x), size=len(x))
        p = pearson(x[indices], y[indices])
        s = spearman(x[indices], y[indices])
        if math.isfinite(p):
            pearsons.append(p)
        if math.isfinite(s):
            spearmans.append(s)
    return {
        "pearson_ci_low": float(np.quantile(pearsons, 0.025)) if pearsons else float("nan"),
        "pearson_ci_high": float(np.quantile(pearsons, 0.975)) if pearsons else float("nan"),
        "spearman_ci_low": float(np.quantile(spearmans, 0.025)) if spearmans else float("nan"),
        "spearman_ci_high": float(np.quantile(spearmans, 0.975)) if spearmans else float("nan"),
    }


def correlation_row(
    candidate: pd.DataFrame,
    metric: str,
    stride: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    reference = candidate[candidate["stride"] == 1][
        ["baseline", "task_id", "episode_id", metric]
    ].rename(columns={metric: "reference"})
    current = candidate[candidate["stride"] == stride][
        ["baseline", "task_id", "episode_id", metric, "frame_fraction"]
    ]
    paired = reference.merge(
        current, on=["baseline", "task_id", "episode_id"], how="inner"
    ).dropna(subset=["reference", metric])
    x = paired["reference"].to_numpy(dtype=np.float64)
    y = paired[metric].to_numpy(dtype=np.float64)
    errors = np.abs(y - x)

    ref_means = reference.groupby("baseline")["reference"].mean().to_dict()
    cand_means = current.groupby("baseline")[metric].mean().to_dict()
    kendall, pairwise = kendall_pairwise(ref_means, cand_means)
    result = {
        "metric": metric,
        "stride": stride,
        "num_paired_episodes": len(paired),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "ccc": concordance_correlation(x, y),
        "mean_bias": float(np.mean(y - x)) if len(errors) else float("nan"),
        "mae": float(errors.mean()) if len(errors) else float("nan"),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))) if len(errors) else float("nan"),
        "p95_abs_error": float(np.quantile(errors, 0.95)) if len(errors) else float("nan"),
        "max_abs_error": float(errors.max()) if len(errors) else float("nan"),
        "mean_frame_fraction": float(paired["frame_fraction"].mean()),
        "theoretical_speedup": (
            float(1.0 / paired["frame_fraction"].mean())
            if len(paired) and paired["frame_fraction"].mean() > 0
            else float("nan")
        ),
        "baseline_kendall": kendall,
        "baseline_pairwise_agreement": pairwise,
    }
    result.update(bootstrap_correlation_ci(x, y, bootstrap_samples, rng))
    return result


def build_task_correlations(scores: pd.DataFrame, strides: Sequence[int]) -> pd.DataFrame:
    rows = []
    for task_id, task_scores in scores.groupby("task_id"):
        for metric in ALL_METRICS:
            reference = task_scores[task_scores["stride"] == 1][
                ["baseline", "episode_id", metric]
            ].rename(columns={metric: "reference"})
            for stride in strides:
                current = task_scores[task_scores["stride"] == stride][
                    ["baseline", "episode_id", metric]
                ]
                paired = reference.merge(
                    current, on=["baseline", "episode_id"], how="inner"
                ).dropna()
                x = paired["reference"].to_numpy(dtype=np.float64)
                y = paired[metric].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "task_id": task_id,
                        "metric": metric,
                        "stride": stride,
                        "num_paired_episodes": len(paired),
                        "pearson": pearson(x, y),
                        "spearman": spearman(x, y),
                        "mae": float(np.mean(np.abs(y - x))) if len(x) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def build_quantiles(
    reservoirs: Mapping[Tuple[str, str, str], Reservoir],
) -> pd.DataFrame:
    rows = []
    quantiles = (0.0, 0.01, 0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99, 1.0)
    groups: List[Tuple[str, str, str, np.ndarray, int]] = []
    for (baseline, source, metric), reservoir in reservoirs.items():
        groups.append(
            (
                baseline,
                source,
                metric,
                np.asarray(reservoir.values, dtype=np.float64),
                reservoir.seen,
            )
        )
    for source in SOURCES:
        for metric in RAW_METRICS:
            matching = [
                reservoir
                for (baseline, src, met), reservoir in reservoirs.items()
                if src == source and met == metric
            ]
            groups.append(
                (
                    "ALL",
                    source,
                    metric,
                    np.concatenate(
                        [np.asarray(reservoir.values, dtype=np.float64) for reservoir in matching]
                    ),
                    sum(reservoir.seen for reservoir in matching),
                )
            )

    for baseline, source, metric, array, population_seen in groups:
        if not len(array):
            continue
        for quantile, value in zip(quantiles, np.quantile(array, quantiles)):
            rows.append(
                {
                    "baseline": baseline,
                    "source": source,
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(value),
                    "sample_size": len(array),
                    "population_seen": population_seen,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_correlations(
    correlations: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    metrics: Sequence[str],
    stem: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        data = correlations[correlations["metric"] == metric].sort_values("stride")
        x = data["stride"].to_numpy(dtype=np.float64)
        axis.plot(x, data["pearson"].to_numpy(dtype=np.float64), marker="o", label="Pearson")
        axis.plot(x, data["spearman"].to_numpy(dtype=np.float64), marker="s", label="Spearman")
        axis.fill_between(
            x,
            data["spearman_ci_low"].to_numpy(dtype=np.float64),
            data["spearman_ci_high"].to_numpy(dtype=np.float64),
            alpha=0.18,
            label="Spearman 95% CI",
        )
        axis.axhline(threshold, linestyle="--", linewidth=1, label=f"target={threshold:g}")
        axis.set_title(f"{metric.upper()} episode correlation")
        axis.set_xlabel("Downsample rate (frame stride)")
        axis.set_ylabel("Correlation with stride=1")
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output_dir, stem)


def plot_errors(
    correlations: pd.DataFrame,
    output_dir: Path,
    metrics: Sequence[str],
    stem: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        data = correlations[correlations["metric"] == metric].sort_values("stride")
        axis.plot(data["stride"], data["mae"], marker="o", label="MAE")
        axis.plot(data["stride"], data["p95_abs_error"], marker="s", label="P95 error")
        axis.set_title(f"{metric.upper()} approximation error")
        axis.set_xlabel("Downsample rate (frame stride)")
        axis.set_ylabel(f"Absolute error ({metric.upper()} units)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output_dir, stem)


def plot_baseline_stability(
    scores: pd.DataFrame,
    output_dir: Path,
    metrics: Sequence[str],
    stem: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        means = (
            scores.groupby(["baseline", "stride"])[metric]
            .mean()
            .reset_index()
        )
        for baseline, data in means.groupby("baseline"):
            data = data.sort_values("stride")
            axis.plot(data["stride"], data[metric], marker="o", label=baseline)
        axis.set_title(f"Baseline mean {metric.upper()}")
        axis.set_xlabel("Downsample rate (frame stride)")
        axis.set_ylabel(f"Episode mean ({metric.upper()} units)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, output_dir, stem)


def plot_quality_score_overview(
    baseline_scores: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot total/object/robot-arm scores and rankings across strides."""
    columns = (
        ("total_quality_score", "Total quality"),
        ("object_quality_score", "Object quality"),
        ("robot_arm_quality_score", "Robot-arm quality"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, (column, label) in zip(axes, columns):
        for baseline, data in baseline_scores.groupby("baseline"):
            data = data.sort_values("stride")
            axis.plot(data["stride"], data[column], marker="o", label=baseline)
        axis.set_title(label)
        axis.set_xlabel("Downsample rate (frame stride)")
        axis.set_ylabel("Quality score (0-100)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Baseline quality scores across downsample rates")
    save_figure(fig, output_dir, "quality_scores_vs_stride")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for axis, (column, label) in zip(axes, columns):
        ranked = baseline_scores.copy()
        ranked["rank"] = ranked.groupby("stride")[column].rank(
            method="min", ascending=False
        )
        for baseline, data in ranked.groupby("baseline"):
            data = data.sort_values("stride")
            axis.plot(data["stride"], data["rank"], marker="o", label=baseline)
        axis.set_title(label)
        axis.set_xlabel("Downsample rate (frame stride)")
        axis.set_ylabel("Rank (1 = best)")
        axis.set_yticks(
            range(1, int(ranked["rank"].max()) + 1)
        )
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Baseline rankings across downsample rates")
    save_figure(fig, output_dir, "quality_rankings_vs_stride")


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    value_format: str = ".1f",
) -> None:
    values = matrix.to_numpy(dtype=np.float64)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set_xlabel(matrix.columns.name or "")
    axis.set_ylabel(matrix.index.name or "")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not math.isfinite(value):
                continue
            text_color = "white" if value < np.nanmedian(values) else "black"
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def plot_quality_score_heatmaps(
    baseline_scores: pd.DataFrame,
    output_dir: Path,
) -> None:
    columns = (
        ("total_quality_score", "Total quality"),
        ("object_quality_score", "Object quality"),
        ("robot_arm_quality_score", "Robot-arm quality"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for axis, (column, label) in zip(axes, columns):
        matrix = baseline_scores.pivot(
            index="baseline", columns="stride", values=column
        )
        _annotated_heatmap(axis, matrix, label, "Quality score")
    fig.suptitle("Baseline score matrix: model × downsample rate")
    fig.tight_layout()
    save_figure(fig, output_dir, "quality_score_heatmaps")


def plot_temporal_quality(
    temporal_summary: pd.DataFrame,
    output_dir: Path,
    strides: Sequence[int],
) -> None:
    temporal_dir = output_dir / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    for stride in strides:
        stride_data = temporal_summary[temporal_summary["stride"] == stride]
        if stride_data.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
        for axis, metric in zip(axes, QUALITY_COLUMNS):
            for baseline, data in stride_data.groupby("baseline"):
                data = data.sort_values("time_bin_index")
                x = data["time_start_sec"].to_numpy(dtype=np.float64)
                mean = data[f"{metric}_mean"].to_numpy(dtype=np.float64)
                std = data[f"{metric}_std"].fillna(0).to_numpy(dtype=np.float64)
                count = data[f"{metric}_count"].to_numpy(dtype=np.float64)
                ci = 1.96 * std / np.sqrt(np.maximum(count, 1))
                axis.plot(x, mean, marker="o", label=baseline)
                axis.fill_between(x, mean - ci, mean + ci, alpha=0.12)
            axis.set_title(QUALITY_LABELS[metric])
            axis.set_xlabel("Episode time (seconds)")
            axis.set_ylabel("Quality score (mean ± 95% CI)")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Temporal model performance — stride {stride}")
        save_figure(
            fig,
            temporal_dir,
            f"temporal_quality_stride_{stride:03d}",
        )

    # A detailed stage-by-model heatmap at full-frame stride=1.
    full = temporal_summary[temporal_summary["stride"] == 1]
    if not full.empty:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        for axis, metric in zip(axes, QUALITY_COLUMNS):
            matrix = full.pivot(
                index="baseline",
                columns="time_bin",
                values=f"{metric}_mean",
            )
            matrix = matrix.reindex(
                sorted(matrix.columns, key=lambda value: float(value.split("-")[0])),
                axis=1,
            )
            _annotated_heatmap(
                axis,
                matrix,
                QUALITY_LABELS[metric],
                "Quality score",
            )
            axis.tick_params(axis="x", labelrotation=45)
        fig.suptitle("Temporal quality by 10-second stage — stride 1")
        fig.tight_layout()
        save_figure(fig, temporal_dir, "temporal_quality_heatmaps_stride_001")


def plot_raw_ecdf(
    reservoirs: Mapping[Tuple[str, str, str], Reservoir],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(RAW_METRICS),
        len(SOURCES),
        figsize=(12, 3.4 * len(RAW_METRICS)),
        squeeze=False,
    )
    for row, metric in enumerate(RAW_METRICS):
        for col, source in enumerate(SOURCES):
            axis = axes[row, col]
            for (baseline, src, met), reservoir in reservoirs.items():
                if src != source or met != metric or not reservoir.values:
                    continue
                values = np.sort(np.asarray(reservoir.values, dtype=np.float64))
                y = np.arange(1, len(values) + 1) / len(values)
                axis.plot(values, y, label=baseline)
            axis.set_title(f"{source}: {metric.upper()} per-mask ECDF")
            axis.set_xlabel(f"Raw {metric.upper()} value")
            axis.set_ylabel("Empirical cumulative probability")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Full-frame KeyArea distributions for empirical min/max selection")
    save_figure(fig, output_dir, "full_frame_raw_ecdf")


def recommend_stride(
    correlations: pd.DataFrame,
    recommendation_config: Mapping[str, object],
) -> dict:
    """Choose a conservative stride using correlation and absolute agreement."""
    thresholds = {
        "pearson_min": float(recommendation_config.get("pearson_min", 0.99)),
        "spearman_min": float(recommendation_config.get("spearman_min", 0.99)),
        "ccc_min": float(recommendation_config.get("ccc_min", 0.99)),
        "mae_max": float(recommendation_config.get("mae_max", 0.5)),
        "rmse_max": float(recommendation_config.get("rmse_max", 0.6)),
        "p95_abs_error_max": float(
            recommendation_config.get("p95_abs_error_max", 1.0)
        ),
        "max_abs_error_max": float(
            recommendation_config.get("max_abs_error_max", 2.0)
        ),
        "abs_mean_bias_max": float(
            recommendation_config.get("abs_mean_bias_max", 0.5)
        ),
        "baseline_pairwise_agreement_min": float(
            recommendation_config.get(
                "baseline_pairwise_agreement_min", 1.0
            )
        ),
    }
    require_prefix = bool(
        recommendation_config.get("require_all_smaller_strides_pass", True)
    )
    eligible = []
    diagnostics = []
    quality = correlations[correlations["metric"] == "quality_score"]
    prefix_still_passes = True
    for stride, rows in quality.groupby("stride", sort=True):
        if len(rows) != 1:
            diagnostics.append(
                {
                    "stride": int(stride),
                    "passes": False,
                    "independent_passes": False,
                    "prefix_passes": False,
                    "failed_checks": ["expected_exactly_one_quality_row"],
                }
            )
            prefix_still_passes = False
            continue
        row = rows.iloc[0]
        checks = {
            "pearson": bool(row["pearson"] >= thresholds["pearson_min"]),
            "spearman": bool(row["spearman"] >= thresholds["spearman_min"]),
            "ccc": bool(row["ccc"] >= thresholds["ccc_min"]),
            "mae": bool(row["mae"] <= thresholds["mae_max"]),
            "rmse": bool(row["rmse"] <= thresholds["rmse_max"]),
            "p95_abs_error": bool(
                row["p95_abs_error"] <= thresholds["p95_abs_error_max"]
            ),
            "max_abs_error": bool(
                row["max_abs_error"] <= thresholds["max_abs_error_max"]
            ),
            "abs_mean_bias": bool(
                abs(row["mean_bias"]) <= thresholds["abs_mean_bias_max"]
            ),
            "baseline_pairwise_agreement": bool(
                row["baseline_pairwise_agreement"]
                >= thresholds["baseline_pairwise_agreement_min"]
            ),
        }
        independent_passes = all(checks.values())
        prefix_still_passes = prefix_still_passes and independent_passes
        passes = independent_passes and (
            prefix_still_passes if require_prefix else True
        )
        failed_checks = [
            name for name, check_passes in checks.items()
            if not check_passes
        ]
        if (
            require_prefix
            and independent_passes
            and not prefix_still_passes
        ):
            failed_checks.append("prior_smaller_stride_failed")
        diagnostics.append(
            {
                "stride": int(stride),
                "passes": passes,
                "independent_passes": independent_passes,
                "prefix_passes": prefix_still_passes,
                "failed_checks": failed_checks,
                "pearson": float(row["pearson"]),
                "spearman": float(row["spearman"]),
                "ccc": float(row["ccc"]),
                "mean_bias": float(row["mean_bias"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "p95_abs_error": float(row["p95_abs_error"]),
                "max_abs_error": float(row["max_abs_error"]),
                "baseline_kendall": float(row["baseline_kendall"]),
                "baseline_pairwise_agreement": float(
                    row["baseline_pairwise_agreement"]
                ),
                "mean_frame_fraction": float(row["mean_frame_fraction"]),
                "theoretical_speedup": float(row["theoretical_speedup"]),
                "checks": checks,
            }
        )
        if passes:
            eligible.append(int(stride))
    return {
        "recommended_stride": max(eligible) if eligible else 1,
        "thresholds": thresholds,
        "require_all_smaller_strides_pass": require_prefix,
        "rule": (
            "Largest stride satisfying every correlation, absolute-agreement, "
            "and baseline-ranking threshold"
            + (
                ", with every evaluated smaller stride also required to pass."
                if require_prefix
                else "."
            )
        ),
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baselines = {name: str(path) for name, path in config["baselines"].items()}
    strides = config["strides"]
    output_dir = Path(config.get("output_dir", "output/keyarea_downsampling_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    sample_limit = int(config.get("distribution_sample_limit", 200_000))
    bootstrap_samples = int(config.get("bootstrap_samples", 1000))
    psnr_inf_cap = float(config.get("psnr_inf_cap", 100.0))
    normalization = config["normalization"]
    fps = float(config.get("fps", 30.0))
    temporal_bin_seconds = float(config.get("temporal_bin_seconds", 10.0))

    discovered, paired_keys, coverage = paired_result_paths(baselines)
    reservoirs = {
        (baseline, source, metric): Reservoir(
            limit=sample_limit,
            rng=random.Random(f"{seed}:{baseline}:{source}:{metric}"),
        )
        for baseline in baselines
        for source in SOURCES
        for metric in RAW_METRICS
    }

    score_rows, source_count_rows, temporal_rows = [], [], []
    for task_id, episode_id in paired_keys:
        for baseline in baselines:
            path = discovered[baseline][(task_id, episode_id)]
            source_frames, source_counts = load_full_frame_episode(
                path,
                reservoirs,
                baseline,
                psnr_inf_cap,
                normalization,
            )
            for row in episode_scores_for_strides(source_frames, strides):
                score_rows.append(
                    {
                        "baseline": baseline,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        **row,
                    }
                )
            for row in temporal_scores_for_strides(
                source_frames,
                strides,
                fps,
                temporal_bin_seconds,
            ):
                temporal_rows.append(
                    {
                        "baseline": baseline,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        **row,
                    }
                )
            for source, counts in source_counts.items():
                source_count_rows.append(
                    {
                        "baseline": baseline,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "source": source,
                        **counts,
                    }
                )

    scores = pd.DataFrame(score_rows)
    temporal_scores = pd.DataFrame(temporal_rows)
    rng = np.random.default_rng(seed)
    correlation_rows = [
        correlation_row(scores, metric, stride, bootstrap_samples, rng)
        for metric in ALL_METRICS
        for stride in strides
    ]
    correlations = pd.DataFrame(correlation_rows)
    task_correlations = build_task_correlations(scores, strides)
    baseline_scores = (
        scores.melt(
            id_vars=["baseline", "task_id", "episode_id", "stride"],
            value_vars=[*ALL_METRICS, *SOURCE_QUALITY_METRICS],
            var_name="metric",
            value_name="value",
        )
        .groupby(["baseline", "stride", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    quantiles = build_quantiles(reservoirs)

    scores.to_csv(output_dir / "episode_scores_by_stride.csv", index=False)
    correlations.to_csv(output_dir / "correlations.csv", index=False)
    task_correlations.to_csv(output_dir / "task_correlations.csv", index=False)
    baseline_scores.to_csv(output_dir / "baseline_scores.csv", index=False)
    baseline_quality_scores = (
        scores.groupby(["baseline", "stride"])[list(QUALITY_COLUMNS)]
        .mean()
        .reset_index()
        .rename(columns={"quality_score": "total_quality_score"})
    )
    baseline_quality_scores.to_csv(
        output_dir / "baseline_quality_scores_by_stride.csv", index=False
    )
    final_scores = baseline_quality_scores.pivot(
        index="stride",
        columns="baseline",
        values=[
            "total_quality_score",
            "object_quality_score",
            "robot_arm_quality_score",
        ],
    )
    final_scores.columns = [
        f"{baseline}_{metric}"
        for metric, baseline in final_scores.columns
    ]
    final_scores.reset_index().to_csv(
        output_dir / "final_baseline_scores_by_stride.csv", index=False
    )
    ranking_columns = {
        "total_quality_score": "total_rank",
        "object_quality_score": "object_rank",
        "robot_arm_quality_score": "robot_arm_rank",
    }
    baseline_rankings = baseline_quality_scores.copy()
    for score_column, rank_column in ranking_columns.items():
        baseline_rankings[rank_column] = baseline_rankings.groupby("stride")[
            score_column
        ].rank(method="min", ascending=False)
    baseline_rankings.to_csv(
        output_dir / "baseline_rankings_by_stride.csv", index=False
    )
    temporal_scores.to_csv(
        output_dir / "temporal_episode_scores_by_stride.csv", index=False
    )
    temporal_summary = (
        temporal_scores.groupby(
            [
                "baseline",
                "stride",
                "time_bin_index",
                "time_start_sec",
                "time_end_sec",
                "time_bin",
            ]
        )[list(QUALITY_COLUMNS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    temporal_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in temporal_summary.columns
    ]
    temporal_summary.to_csv(
        output_dir / "temporal_baseline_scores_by_stride.csv", index=False
    )
    quantiles.to_csv(output_dir / "full_frame_raw_quantiles.csv", index=False)
    source_diagnostics = pd.DataFrame(source_count_rows)
    source_diagnostics.to_csv(output_dir / "source_coverage.csv", index=False)
    normalization_diagnostics = scores[scores["stride"] == 1][
        [
            "baseline",
            "task_id",
            "episode_id",
            *RAW_METRICS,
            *NORMALIZED_METRICS,
        ]
    ].copy()
    for metric in NORMALIZED_METRICS:
        normalization_diagnostics[f"{metric}_clipped_zero"] = (
            normalization_diagnostics[metric] <= 0.0
        )
        normalization_diagnostics[f"{metric}_clipped_hundred"] = (
            normalization_diagnostics[metric] >= 100.0
        )
    normalization_diagnostics.to_csv(
        output_dir / "normalization_diagnostics.csv", index=False
    )
    flag_columns = [
        f"{metric}_clipped_{boundary}"
        for metric in NORMALIZED_METRICS
        for boundary in ("zero", "hundred")
    ]
    saturation_summary = (
        normalization_diagnostics.groupby("baseline")[flag_columns]
        .agg(["sum", "mean"])
        .reset_index()
    )
    saturation_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in saturation_summary.columns
    ]
    saturation_summary.to_csv(
        output_dir / "normalization_saturation_summary.csv", index=False
    )

    threshold = float(config.get("correlation_threshold", 0.98))
    recommendation = recommend_stride(
        correlations,
        config.get("recommendation") or {},
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "baselines": baselines,
        "strides": strides,
        "fps": fps,
        "temporal_bin_seconds": temporal_bin_seconds,
        "psnr_inf_cap": psnr_inf_cap,
        "normalization": normalization,
        "max_saturation_rates": {
            column: float(saturation_summary[column].max())
            for column in saturation_summary
            if column.endswith("_mean")
        },
        "paired_episodes": len(paired_keys),
        "coverage": coverage,
        "recommendation": recommendation,
        "note": (
            "All candidate strides are simulated from stride=1 per-mask records; "
            "no LPIPS recomputation or episode JSON modification was performed."
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(output_dir / "recommended_stride.json", "w", encoding="utf-8") as file:
        json.dump(recommendation, file, indent=2, ensure_ascii=False)

    plot_correlations(
        correlations,
        output_dir,
        threshold,
        RAW_METRICS,
        "raw_correlation_vs_stride",
        "Raw metric fidelity versus downsampling",
    )
    plot_correlations(
        correlations,
        output_dir,
        threshold,
        NORMALIZED_METRICS,
        "normalized_correlation_vs_stride",
        "Normalized score fidelity versus downsampling",
    )
    plot_errors(
        correlations,
        output_dir,
        RAW_METRICS,
        "raw_error_vs_stride",
        "Raw episode-score error relative to stride=1",
    )
    plot_errors(
        correlations,
        output_dir,
        NORMALIZED_METRICS,
        "normalized_error_vs_stride",
        "Normalized episode-score error relative to stride=1",
    )
    plot_baseline_stability(
        scores,
        output_dir,
        RAW_METRICS,
        "raw_baseline_stability",
        "Raw baseline means across downsample rates",
    )
    plot_baseline_stability(
        scores,
        output_dir,
        ("quality_score",),
        "normalized_baseline_stability",
        "Normalized geometric quality across downsample rates",
    )
    plot_quality_score_overview(baseline_quality_scores, output_dir)
    plot_quality_score_heatmaps(baseline_quality_scores, output_dir)
    plot_temporal_quality(temporal_summary, output_dir, strides)
    plot_raw_ecdf(reservoirs, output_dir)
    print(f"Saved KeyArea downsampling analysis to {output_dir}")


if __name__ == "__main__":
    main()
