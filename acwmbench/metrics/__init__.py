"""Metric entrypoints, loaded lazily to keep optional ML dependencies isolated."""

from typing import Any

# Entrypoints are provided dynamically by __getattr__ so lightweight metrics do
# not import every optional model stack.
# pylint: disable=undefined-all-variable
__all__ = [
    "compute_keyarea_visual_quality",
    "compute_interaction_spatial_alignment",
    "compute_multiview_consistency",
    "compute_physical_judge",
    "compute_policy_evaluation",
]


def __getattr__(name: str) -> Any:
    if name == "compute_keyarea_visual_quality":
        from .keyarea_visual_quality import compute_keyarea_visual_quality

        return compute_keyarea_visual_quality
    if name == "compute_interaction_spatial_alignment":
        from .interaction_spatial_alignment import (
            compute_interaction_spatial_alignment,
        )

        return compute_interaction_spatial_alignment
    if name == "compute_multiview_consistency":
        from .multiview_consistency import compute_multiview_consistency

        return compute_multiview_consistency
    if name == "compute_physical_judge":
        from .physical_judge import compute_physical_judge

        return compute_physical_judge
    if name == "compute_policy_evaluation":
        from .policy_eval import compute_policy_evaluation

        return compute_policy_evaluation
    raise AttributeError(name)
