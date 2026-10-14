"""Incremental DINOv3 Multi-View Consistency metric.

``interaction_spatial_alignment`` already computes the top-head robot-arm
alignment. Multi-View reuses that persisted result, computes only hand-left and
hand-right with the exact same DINO/matcher/calibration settings, and averages
the three view scores for every frame. There is no threshold or binary gate.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from ..dataset import EpisodeTrial
from .interaction_spatial_alignment import (
    DinoExtractor,
    SUMMARY_METRICS,
    _discover_frames,
    _extract_paired_features,
    _load_masks,
    _mean,
    _prepare_frame_batch,
    _prefetched_frame_batches,
    _source_frames,
    match_feature_maps,
)

logger = logging.getLogger(__name__)

TOP_VIEW = "top_head"
HAND_VIEWS = ("hand_left", "hand_right")
DEFAULT_VIEWS = (TOP_VIEW, *HAND_VIEWS)
METRIC_VERSION = 5
INTERACTION_METRIC_VERSION = 2
MATCH_SETTING_KEYS = (
    "min_mask_patches",
    "min_mutual_matches",
    "inlier_radius_px",
    "confidence_bias",
    "confidence_cosine_weight",
    "confidence_margin_weight",
    "confidence_mutual_weight",
    "confidence_inlier_weight",
    "match_support_target_ratio",
    "match_support_count_scale",
    "match_support_power",
    "confidence_bad",
    "confidence_good",
    "distance_good_px",
    "distance_bad_px",
    "alignment_confidence_power",
)


def _strict_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _top_robot_frames(
    interaction: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    source = (
        (interaction.get("mask_sources") or {}).get("robot_arm") or {}
    )
    records = source.get("per_frame")
    if not isinstance(records, list) or not records:
        raise ValueError(
            "interaction_spatial_alignment.json has no robot_arm.per_frame"
        )
    return records


def interaction_robot_arm_digest(
    interaction: Mapping[str, Any],
) -> str:
    """Hash exactly the persisted top-head values Multi-View consumes."""
    payload = {
        "uid": interaction.get("uid"),
        "metric_version": interaction.get("metric_version"),
        "evaluation_settings": interaction.get("evaluation_settings"),
        "robot_arm_alignment": [
            [
                int(record["frame"]),
                _strict_float(record.get("alignment_score")),
            ]
            for record in _top_robot_frames(interaction)
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _interaction_settings(
    interaction: Mapping[str, Any],
) -> Dict[str, Any]:
    if int(interaction.get("metric_version", -1)) != INTERACTION_METRIC_VERSION:
        raise ValueError(
            "interaction_spatial_alignment metric_version=2 is required"
        )
    settings = interaction.get("evaluation_settings")
    if not isinstance(settings, dict):
        raise ValueError(
            "interaction_spatial_alignment.json has no evaluation_settings"
        )
    if str(settings.get("eval_view")) != TOP_VIEW:
        raise ValueError("Interaction source must use eval_view=top_head")
    required = (
        "downsample_rate",
        "fps",
        "checkpoint",
        "model_name",
        *MATCH_SETTING_KEYS,
    )
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(
            f"Interaction source is missing settings: {missing}"
        )
    return dict(settings)


def _normalize_execution(
    execution: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Use source batch/fusion identity; other knobs only affect throughput."""
    result = dict(execution or {})
    result["frame_batch_size"] = int(
        settings.get("feature_batch_size", 1)
    )
    result["fuse_gt_generated"] = bool(
        settings.get("fuse_gt_generated", False)
    )
    result.setdefault("mode", "serial")
    result.setdefault("loader_workers", 1)
    result.setdefault("prefetch_batches", 1)
    result.setdefault("matcher_workers", 1)
    result.setdefault("pin_memory", False)
    for key in (
        "frame_batch_size",
        "loader_workers",
        "prefetch_batches",
        "matcher_workers",
    ):
        result[key] = int(result[key])
        if result[key] < 1:
            raise ValueError(f"execution.{key} must be >= 1")
    result["mode"] = str(result["mode"]).lower()
    if result["mode"] not in {"serial", "pipeline"}:
        raise ValueError("execution.mode must be 'serial' or 'pipeline'")
    if result["mode"] == "serial":
        # Preserve source batch/fusion identity even in serial mode.
        result.update(
            loader_workers=1,
            prefetch_batches=1,
            matcher_workers=1,
            pin_memory=False,
        )
    return result


def _match_hand_frame(
    *,
    frame: int,
    gt_feature: np.ndarray,
    generated_feature: np.ndarray,
    masks: np.ndarray,
    fps: float,
    match_kwargs: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Match one hand-camera frame using Interaction's mask semantics."""
    records = []
    for mask_index in range(masks.shape[0]):
        match = match_feature_maps(
            gt_feature,
            generated_feature,
            masks[mask_index, frame],
            **match_kwargs,
        )
        records.append(
            {
                "frame": int(frame),
                "time_sec": float(frame / fps),
                "mask_index": int(mask_index),
                "mask_source": "robot_arm",
                **match,
            }
        )
    return records


def _compute_hand_views(
    *,
    extractor: DinoExtractor,
    selected_frames: Sequence[int],
    resources: Mapping[str, Mapping[str, Any]],
    fps: float,
    match_kwargs: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Batch both hand views through one bounded load/GPU/match pipeline."""
    frame_batch_size = int(execution["frame_batch_size"])
    matcher_workers = int(execution["matcher_workers"])
    work_by_size: Dict[tuple[int, int], List[tuple[str, int]]] = {}
    for frame in selected_frames:
        for view in HAND_VIEWS:
            resource = resources[view]
            if (
                frame not in resource["gt_frames"]
                or frame not in resource["generated_frames"]
                or frame >= resource["masks"].shape[1]
            ):
                continue
            height, width = resource["masks"].shape[-2:]
            work_by_size.setdefault((width, height), []).append((view, frame))

    def prepared_work():
        for size, work_items in work_by_size.items():
            item_by_id = dict(enumerate(work_items))
            gt_paths = {
                item_id: resources[view]["gt_frames"][frame]
                for item_id, (view, frame) in item_by_id.items()
            }
            generated_paths = {
                item_id: resources[view]["generated_frames"][frame]
                for item_id, (view, frame) in item_by_id.items()
            }

            def prepare(
                item_ids: Sequence[int],
                *,
                gt_paths=gt_paths,
                generated_paths=generated_paths,
                size=size,
            ):
                return _prepare_frame_batch(
                    extractor,
                    item_ids,
                    gt_paths,
                    generated_paths,
                    size,
                )

            yield from (
                (item_by_id, prepared)
                for prepared in _prefetched_frame_batches(
                    list(item_by_id),
                    frame_batch_size=frame_batch_size,
                    loader_workers=int(execution["loader_workers"]),
                    prefetch_batches=int(execution["prefetch_batches"]),
                    prepare=prepare,
                )
            )

    def match_args(view, frame, gt_feature, generated_feature):
        return {
            "frame": frame,
            "gt_feature": gt_feature,
            "generated_feature": generated_feature,
            "masks": resources[view]["masks"],
            "fps": fps,
            "match_kwargs": match_kwargs,
        }

    per_view: Dict[str, List[Dict[str, Any]]] = {
        view: [] for view in HAND_VIEWS
    }
    if matcher_workers <= 1:
        for item_by_id, prepared in prepared_work():
            gt_features, generated_features = _extract_paired_features(
                extractor,
                prepared,
                fuse_gt_generated=bool(
                    execution["fuse_gt_generated"]
                ),
                pin_memory=bool(execution["pin_memory"]),
            )
            for item_id, gt_feature, generated_feature in zip(
                prepared.frames,
                gt_features,
                generated_features,
            ):
                view, frame = item_by_id[item_id]
                per_view[view].extend(
                    _match_hand_frame(
                        **match_args(
                            view,
                            frame,
                            gt_feature,
                            generated_feature,
                        )
                    )
                )
        return per_view

    backlog_limit = max(
        frame_batch_size,
        frame_batch_size * int(execution["prefetch_batches"]),
    )
    pending: deque[
        tuple[str, Future[List[Dict[str, Any]]]]
    ] = deque()
    with ThreadPoolExecutor(
        max_workers=matcher_workers,
        thread_name_prefix="multiview-matcher",
    ) as pool:
        for item_by_id, prepared in prepared_work():
            gt_features, generated_features = _extract_paired_features(
                extractor,
                prepared,
                fuse_gt_generated=bool(
                    execution["fuse_gt_generated"]
                ),
                pin_memory=bool(execution["pin_memory"]),
            )
            for item_id, gt_feature, generated_feature in zip(
                prepared.frames,
                gt_features,
                generated_features,
            ):
                view, frame = item_by_id[item_id]
                pending.append(
                    (
                        view,
                        pool.submit(
                            _match_hand_frame,
                            **match_args(
                                view,
                                frame,
                                gt_feature,
                                generated_feature,
                            ),
                        ),
                    )
                )
                if len(pending) >= backlog_limit:
                    completed_view, future = pending.popleft()
                    per_view[completed_view].extend(future.result())
        while pending:
            completed_view, future = pending.popleft()
            per_view[completed_view].extend(future.result())
    return per_view


def _combine_view_scores(
    *,
    frame: int,
    view_scores: Mapping[str, float | None],
    fps: float,
) -> Dict[str, Any]:
    """Always average three views; unavailable view scores contribute zero."""
    normalized = {
        view: _strict_float(view_scores.get(view)) for view in DEFAULT_VIEWS
    }
    values = [
        normalized[view] if normalized[view] is not None else 0.0
        for view in DEFAULT_VIEWS
    ]
    return {
        "frame": int(frame),
        "time_sec": float(frame / fps),
        "view_alignment_scores": normalized,
        "view_available": {
            view: normalized[view] is not None for view in DEFAULT_VIEWS
        },
        "multiview_consistency_score": float(np.mean(values)),
    }


def compute_multiview_consistency(
    episodes: List[EpisodeTrial],
    gt_root: str | os.PathLike,
    *,
    device: str,
    save_root: str | os.PathLike,
    progress: bool = True,
    execution: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Reuse top-head Interaction output and compute only two hand views."""
    del gt_root  # GT location comes from each EpisodeTrial.
    from .. import episode_store

    results = []
    extractor_cache: Dict[tuple, DinoExtractor] = {}
    iterator = (
        tqdm(episodes, desc="Multi-View Consistency")
        if progress
        else episodes
    )
    for episode in iterator:
        interaction = episode_store.load_metric(
            save_root,
            episode.task_id,
            episode.episode_id,
            "interaction_spatial_alignment",
        )
        if not isinstance(interaction, dict):
            logger.warning(
                "Skipping %s: interaction_spatial_alignment.json is required",
                episode.uid,
            )
            continue
        try:
            settings = _interaction_settings(interaction)
            top_frames = _top_robot_frames(interaction)
            top_lookup = {
                int(record["frame"]): record for record in top_frames
            }
            selected_frames = sorted(top_lookup)
            if not selected_frames:
                raise ValueError("Interaction source has no top-head frames")
            interaction_digest = interaction_robot_arm_digest(interaction)
            fps = float(settings["fps"])
            match_kwargs = {
                key: settings[key] for key in MATCH_SETTING_KEYS
            }
            runtime_execution = _normalize_execution(execution, settings)
            gt_video_dir = Path(
                episode.gt_video_dir or episode.video_dir
            )
            gt_episode = gt_video_dir.parent
            resources: Dict[str, Dict[str, Any]] = {}
            for view in HAND_VIEWS:
                gt_frames = _discover_frames(gt_video_dir, view)
                generated_frames = _discover_frames(
                    episode.video_dir,
                    view,
                )
                masks = _load_masks(
                    gt_episode
                    / "multiview_masks"
                    / view
                    / "masks.npz"
                )
                resources[view] = {
                    "gt_frames": gt_frames,
                    "generated_frames": generated_frames,
                    "masks": masks,
                }
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("Skipping %s: %s", episode.uid, exc)
            continue

        model_key = (
            str(settings["model_name"]),
            str(settings["checkpoint"]),
            str(settings.get("dinov3_repo") or ""),
            str(device),
        )
        extractor = extractor_cache.get(model_key)
        if extractor is None:
            extractor = DinoExtractor(
                model_name=str(settings["model_name"]),
                checkpoint=str(settings["checkpoint"]),
                dinov3_repo=settings.get("dinov3_repo"),
                device=device,
            )
            extractor_cache[model_key] = extractor

        try:
            per_view = {
                TOP_VIEW: {
                    "alignment_score": _mean(
                        top_frames,
                        "alignment_score",
                    ),
                    "num_frames_evaluated": len(top_frames),
                    "source": "interaction_spatial_alignment.json",
                    "per_frame": top_frames,
                    "per_mask": (
                        (
                            interaction.get("mask_sources") or {}
                        ).get("robot_arm")
                        or {}
                    ).get("per_mask", []),
                }
            }
            lookups = {TOP_VIEW: top_lookup}
            hand_per_mask = _compute_hand_views(
                extractor=extractor,
                selected_frames=selected_frames,
                resources=resources,
                fps=fps,
                match_kwargs=match_kwargs,
                execution=runtime_execution,
            )
            for view in HAND_VIEWS:
                per_mask = hand_per_mask[view]
                per_frame = _source_frames(per_mask)
                lookups[view] = {
                    int(record["frame"]): record
                    for record in per_frame
                }
                per_view[view] = {
                    **{
                        metric: _mean(per_frame, metric)
                        for metric in SUMMARY_METRICS
                    },
                    "num_frames_evaluated": len(per_frame),
                    "source": "computed",
                    "per_frame": per_frame,
                    "per_mask": per_mask,
                }
            per_frame = [
                _combine_view_scores(
                    frame=frame,
                    view_scores={
                        view: (
                            lookups[view].get(frame) or {}
                        ).get("alignment_score")
                        for view in DEFAULT_VIEWS
                    },
                    fps=fps,
                )
                for frame in selected_frames
            ]
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            cuda_oom = getattr(
                getattr(extractor.torch, "cuda", None),
                "OutOfMemoryError",
                (),
            )
            if cuda_oom and isinstance(exc, cuda_oom):
                raise
            logger.exception(
                "Skipping %s after Multi-View pipeline failure: %s",
                episode.uid,
                exc,
            )
            continue

        score = _mean(per_frame, "multiview_consistency_score")
        result_settings = {
            "views": list(DEFAULT_VIEWS),
            "top_head_source": "interaction_spatial_alignment.json",
            "interaction_metric_version": INTERACTION_METRIC_VERSION,
            "interaction_evaluation_settings": settings,
            "feature_batch_size": int(
                runtime_execution["frame_batch_size"]
            ),
            "fuse_gt_generated": bool(
                runtime_execution["fuse_gt_generated"]
            ),
            "hand_batch_strategy": "cross_view_interleaved",
        }
        result = {
            "uid": episode.uid,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "metrics": ["multiview_consistency_score"],
            "metric_version": METRIC_VERSION,
            "score_range": [0.0, 100.0],
            "score_formula": (
                "(top_head_alignment + hand_left_alignment + "
                "hand_right_alignment) / 3; unavailable view contributes 0"
            ),
            "multiview_consistency_score": score,
            "alignment_score": score,
            "num_selected_frames": len(selected_frames),
            "top_head_source_digest": interaction_digest,
            "evaluation_settings": result_settings,
            "per_view": per_view,
            "per_frame": per_frame,
        }
        path = episode_store.save_metric(
            save_root,
            episode.task_id,
            episode.episode_id,
            "multiview_consistency",
            result,
        )
        logger.info("Saved %s Multi-View result to %s", episode.uid, path)
        results.append(result)

    return {
        "per_episode": results,
        "aggregated": {
            "multiview_consistency_score": _mean(
                results,
                "multiview_consistency_score",
            ),
            "alignment_score": _mean(results, "alignment_score"),
            "num_episodes": len(results),
            "metric_version": METRIC_VERSION,
        },
    }
