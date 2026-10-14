"""Per-episode metric result storage under the generated dataset (val_base).

Instead of one monolithic ``<save_path>/<name>_results.json`` written only at
the very end of a run, each metric persists one JSON per episode right next to
that episode's generated frames, e.g.::

    <val_base>/<task_id>/<episode_id>/keyarea_visual_quality.json
    <val_base>/<task_id>/<episode_id>/interaction_spatial_alignment.json
    <val_base>/<task_id>/<episode_id>/multiview_consistency.json
    <val_base>/<task_id>/<episode_id>/physical_judge.json

Visualizations (when ``--visualize``) go under::

    <val_base>/<task_id>/<episode_id>/visualize/

This makes each episode durable as soon as it finishes (crash-safe) and lets
resume skip episodes whose per-metric JSON already exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .io_utils import load_json, save_json


def episode_dir(val_base: str | os.PathLike, task_id: str, episode_id: str) -> Path:
    return Path(val_base) / str(task_id) / str(episode_id)


def metric_path(
    val_base: str | os.PathLike, task_id: str, episode_id: str, metric_name: str
) -> Path:
    return episode_dir(val_base, task_id, episode_id) / f"{metric_name}.json"


def visualize_dir(val_base: str | os.PathLike, task_id: str, episode_id: str) -> Path:
    return episode_dir(val_base, task_id, episode_id) / "visualize"


def has_metric(
    val_base: str | os.PathLike, task_id: str, episode_id: str, metric_name: str
) -> bool:
    return metric_path(val_base, task_id, episode_id, metric_name).is_file()


def remove_metric(
    val_base: str | os.PathLike, task_id: str, episode_id: str, metric_name: str
) -> None:
    """Remove one cached metric file (used by explicit ``--overwrite``)."""
    metric_path(val_base, task_id, episode_id, metric_name).unlink(missing_ok=True)


def clear_visualizations(
    val_base: str | os.PathLike, task_id: str, episode_id: str, metric_name: str
) -> None:
    """Remove visualization PNGs belonging to one metric, preserving others."""
    viz_dir = visualize_dir(val_base, task_id, episode_id)
    if not viz_dir.is_dir():
        return
    for suffix in ("png", "json"):
        for path in viz_dir.glob(f"{metric_name}_*.{suffix}"):
            path.unlink()


def save_metric(
    val_base: str | os.PathLike,
    task_id: str,
    episode_id: str,
    metric_name: str,
    data: Any,
) -> Path:
    path = metric_path(val_base, task_id, episode_id, metric_name)
    save_json(data, path)
    return path


def load_metric(
    val_base: str | os.PathLike, task_id: str, episode_id: str, metric_name: str
) -> Optional[Any]:
    path = metric_path(val_base, task_id, episode_id, metric_name)
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        # A corrupt/incomplete cache is treated as absent and recomputed.
        return None
