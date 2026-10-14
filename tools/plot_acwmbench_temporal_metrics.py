#!/usr/bin/env python3
"""Export publication-ready cumulative ACWMBench metric curves as PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "acwmbench_core": (
        "acwmbench_core_score",
        "ACWMBench Core",
    ),
    "physical_judge": (
        "physical_judge_score",
        "Physical Judge",
    ),
    "keyarea_visual_quality": (
        "keyarea_visual_quality_score",
        "KeyArea Visual Quality",
    ),
    "interaction_spatial_alignment": (
        "interaction_spatial_alignment_score",
        "Interaction Spatial Alignment",
    ),
    "multiview_consistency": (
        "multiview_consistency_score",
        "MultiView Consistency",
    ),
}
BASELINE_ORDER = (
    "Cosmos3-G02",
    "CtrlWorld",
    "DreamDojo",
    "GESim-G02",
    "GESim-G02-DMD",
    "IRASim",
    "GESim-G01",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINE_STYLES = ("-", "--", "-.", ":", "-", "--", "-.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "output/acwmbench_temporal_score_summary/"
            "acwmbench_temporal_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/acwmbench_temporal_score_summary/plots",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"baseline", "window_end_sec"}
    required.update(column for column, _label in METRICS.values())
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    data["window_end_sec"] = pd.to_numeric(
        data["window_end_sec"],
        errors="raise",
    )
    return data


def ordered_baselines(data: pd.DataFrame) -> list[str]:
    available = set(data["baseline"].astype(str))
    ordered = [name for name in BASELINE_ORDER if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
        }
    )


def draw_metric(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric_key: str,
    *,
    compact: bool = False,
) -> None:
    column, label = METRICS[metric_key]
    colors = plt.get_cmap("tab10").colors
    plotted = 0
    for index, baseline in enumerate(ordered_baselines(data)):
        group = data[data["baseline"] == baseline].sort_values(
            "window_end_sec"
        )
        values = pd.to_numeric(group[column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        axis.plot(
            group.loc[valid, "window_end_sec"],
            values.loc[valid],
            label=baseline,
            color=colors[index % len(colors)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markersize=3.4 if compact else 4.0,
            linewidth=1.45 if compact else 1.7,
            markevery=1,
        )
        plotted += 1
    axis.set_title(f"{label} over cumulative evaluation horizon")
    axis.set_xlabel("Cumulative window end (seconds)")
    axis.set_ylabel(f"{label} score (0–100)")
    axis.set_xlim(8, 122)
    axis.set_ylim(0, 100)
    axis.set_xticks(range(10, 121, 10))
    axis.grid(True, color="0.85", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if plotted:
        axis.legend(
            loc="best",
            ncol=2 if not compact else 1,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.0,
        )


def export_individual(
    data: pd.DataFrame,
    output: Path,
) -> list[Path]:
    paths = []
    for metric_key in METRICS:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        draw_metric(axis, data, metric_key)
        note = (
            "Note: GESim-G01 Physical Judge covers only 1 task / 6 episodes; "
            "other GESim-G01 dimensions cover 4 tasks / 100 episodes."
            if metric_key == "physical_judge"
            else (
                "Only baselines with valid MultiView results are shown."
                if metric_key == "multiview_consistency"
                else "Scores are task-balanced cumulative means."
            )
        )
        figure.text(0.01, 0.012, note, fontsize=7.4, color="0.35")
        figure.tight_layout(rect=(0, 0.045, 1, 1))
        path = output / f"{metric_key}_curve.pdf"
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def export_combined(data: pd.DataFrame, output: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    combined_metrics = (
        "acwmbench_core",
        "physical_judge",
        "keyarea_visual_quality",
        "interaction_spatial_alignment",
    )
    for axis, metric_key in zip(axes.flat, combined_metrics):
        draw_metric(axis, data, metric_key, compact=True)
    figure.text(
        0.01,
        0.008,
        (
            "Cumulative windows at FPS=30. GESim-G01 Physical Judge coverage "
            "is limited to 1 task / 6 episodes."
        ),
        fontsize=7.5,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    path = output / "core_physical_keyarea_interaction_curves.pdf"
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_results(input_path)
    paths = export_individual(data, output)
    paths.append(export_combined(data, output))
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
