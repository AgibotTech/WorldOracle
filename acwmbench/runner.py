"""Top-level ACWMBench runner: discover data, dispatch metrics, save results."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import episode_store
from .aggregation import merge_results_to_csv, write_policy_eval_reports
from .dataset import build_trial_list
from .distributed import print0
from .metrics import (
    compute_interaction_spatial_alignment,
    compute_keyarea_visual_quality,
    compute_multiview_consistency,
    compute_physical_judge,
    compute_policy_evaluation,
)
from .models import VLMBackend, build_vlm_backend
from .policy_eval import HTTPWorldModelClient, discover_policy_rollouts
from .policy_rollout import build_policy_client, generate_wm_policy_rollouts
from .runtime import (
    resolve_interaction_execution,
    resolve_keyarea_execution,
)

logger = logging.getLogger(__name__)

SUPPORTED_DIMENSIONS = (
    "physical_judge",
    "keyarea_visual_quality",
    "interaction_spatial_alignment",
    "multiview_consistency",
    "policy_eval",
)

_VISUAL_METRIC_KEYS = (
    "ssim",
    "psnr",
    "lpips",
    "vjepa_similarity",
    "alignment_score",
    "confidence_score",
    "confidence_quality_score",
    "distance_score",
    "normalized_displacement",
)
_DINO_DEFAULT_CHECKPOINT = (
    "./ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def _interaction_execution_config(dim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve legacy top-level batch size into the execution namespace."""

    execution = dict(dim_cfg.get("execution", {}) or {})
    if (
        "frame_batch_size" not in execution
        and dim_cfg.get("batch_size") is not None
    ):
        execution["frame_batch_size"] = int(dim_cfg["batch_size"])
    return execution


def _dino_result_settings(
    dim_cfg: Dict[str, Any],
    *,
    interaction_execution: Any | None = None,
) -> Dict[str, Any]:
    """Result-identity settings shared by DINO interaction dimensions."""
    settings: Dict[str, Any] = {
        "downsample_rate": int(dim_cfg.get("downsample_rate", 10)),
        "fps": float(dim_cfg.get("fps", 30.0)),
        "checkpoint": str(
            dim_cfg.get("checkpoint", _DINO_DEFAULT_CHECKPOINT)
        ),
        "dinov3_repo": (
            str(dim_cfg["dinov3_repo"])
            if dim_cfg.get("dinov3_repo")
            else None
        ),
        "model_name": str(
            dim_cfg.get("model_name", "dinov3_vitb16")
        ),
        "min_mask_patches": int(dim_cfg.get("min_mask_patches", 2)),
        "min_mutual_matches": int(
            dim_cfg.get("min_mutual_matches", 3)
        ),
        "inlier_radius_px": float(
            dim_cfg.get("inlier_radius_px", 32.0)
        ),
        "confidence_bias": float(dim_cfg.get("confidence_bias", -2.0)),
        "confidence_cosine_weight": float(
            dim_cfg.get("confidence_cosine_weight", 3.0)
        ),
        "confidence_margin_weight": float(
            dim_cfg.get("confidence_margin_weight", 8.0)
        ),
        "confidence_mutual_weight": float(
            dim_cfg.get("confidence_mutual_weight", 1.5)
        ),
        "confidence_inlier_weight": float(
            dim_cfg.get("confidence_inlier_weight", 1.5)
        ),
        "match_support_target_ratio": float(
            dim_cfg.get("match_support_target_ratio", 0.3)
        ),
        "match_support_count_scale": float(
            dim_cfg.get("match_support_count_scale", 8.0)
        ),
        "match_support_power": float(
            dim_cfg.get("match_support_power", 1.0)
        ),
        "confidence_bad": float(dim_cfg.get("confidence_bad", 0.5)),
        "confidence_good": float(dim_cfg.get("confidence_good", 0.8)),
        "distance_good_px": float(
            dim_cfg.get("distance_good_px", 30.0)
        ),
        "distance_bad_px": float(
            dim_cfg.get("distance_bad_px", 150.0)
        ),
        "alignment_confidence_power": float(
            dim_cfg.get("alignment_confidence_power", 3.0)
        ),
    }
    settings["eval_view"] = str(dim_cfg.get("eval_view", "top_head"))
    if interaction_execution is not None:
        settings.update(
            feature_batch_size=int(
                interaction_execution.frame_batch_size
            ),
            fuse_gt_generated=bool(
                interaction_execution.fuse_gt_generated
            ),
        )
    return settings


def _current_interaction_robot_arm_digest(
    val_base: str | os.PathLike,
    task_id: str,
    episode_id: str,
) -> str | None:
    interaction = episode_store.load_metric(
        val_base,
        task_id,
        episode_id,
        "interaction_spatial_alignment",
    )
    if not isinstance(interaction, dict):
        return None
    try:
        from .metrics.multiview_consistency import (
            interaction_robot_arm_digest,
        )

        return interaction_robot_arm_digest(interaction)
    except (KeyError, TypeError, ValueError):
        return None


def _done_uids_from_episode_files(
    val_base: str | os.PathLike,
    trials: List[Any],
    metric_name: str,
    require_visualization: bool = False,
    expected_metric_settings: Dict[str, Any] | None = None,
) -> set:
    """Return trial uids already persisted in per-episode metric files."""
    done = set()
    for trial in trials:
        data = episode_store.load_metric(
            val_base, trial.task_id, trial.episode_id, metric_name
        )
        if not isinstance(data, dict):
            continue
        if metric_name == "physical_judge":
            trial_result = data.get(trial.trial_id)
            settings = (
                trial_result.get("evaluation_settings")
                if isinstance(trial_result, dict)
                else None
            )
            if (
                isinstance(trial_result, dict)
                and trial_result.get("metric_version") == 1
                and trial_result.get("status") == "complete"
                and isinstance(settings, dict)
                and all(
                    settings.get(key) == value
                    for key, value in (expected_metric_settings or {}).items()
                )
            ):
                done.add(trial.uid)
        elif data.get("uid") == trial.uid:
            if (
                metric_name == "keyarea_visual_quality"
                and (
                    data.get("metrics")
                    != ["psnr", "lpips", "vjepa_similarity"]
                    or data.get("metric_version") != 12
                    or any(
                        data.get(key) != value
                        for key, value in (expected_metric_settings or {}).items()
                    )
                )
            ):
                # Invalidate legacy files and results produced with other mask settings.
                episode_store.remove_metric(
                    val_base, trial.task_id, trial.episode_id, metric_name
                )
                episode_store.clear_visualizations(
                    val_base, trial.task_id, trial.episode_id, metric_name
                )
                continue
            if (
                metric_name == "multiview_consistency"
                and (
                    data.get("metric_version") != 5
                    or data.get("top_head_source_digest")
                    != _current_interaction_robot_arm_digest(
                        val_base,
                        trial.task_id,
                        trial.episode_id,
                    )
                )
            ):
                episode_store.remove_metric(
                    val_base, trial.task_id, trial.episode_id, metric_name
                )
                continue
            if (
                metric_name == "interaction_spatial_alignment"
                and (
                    data.get("metric_version") != 2
                    or any(
                        (data.get("evaluation_settings") or {}).get(key)
                        != value
                        for key, value in (
                            expected_metric_settings or {}
                        ).items()
                    )
                )
            ):
                episode_store.remove_metric(
                    val_base,
                    trial.task_id,
                    trial.episode_id,
                    metric_name,
                )
                continue
            if require_visualization:
                viz_dir = episode_store.visualize_dir(
                    val_base, trial.task_id, trial.episode_id
                )
                has_source_panels = (
                    viz_dir.is_dir()
                    and all(
                        any(
                            viz_dir.glob(
                                f"keyarea_visual_quality_{source}_frame*_mask*.png"
                            )
                        )
                        and (
                            viz_dir
                            / f"keyarea_visual_quality_contact_sheet_{source}.png"
                        ).is_file()
                        for source in ("object", "robot_arm")
                    )
                    and (
                        viz_dir / "keyarea_visual_quality_manifest.json"
                    ).is_file()
                )
                if not has_source_panels:
                    continue
            done.add(trial.uid)
    return done


def _merge_usage(existing: Any, new: Any) -> Any:
    """Sum token-usage counters across resume passes."""
    if not isinstance(new, dict):
        return existing
    if not isinstance(existing, dict):
        return new
    out = dict(existing)
    for k, v in new.items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _assemble_per_episode_from_files(
    val_base: str | os.PathLike, cand_eps: set, metric_name: str
) -> Dict[str, Any]:
    """Load every candidate episode's ``<metric>.json`` into a per-episode block."""
    per_episode = []
    for task_id, episode_id in sorted(cand_eps):
        data = episode_store.load_metric(val_base, task_id, episode_id, metric_name)
        if isinstance(data, dict) and data:
            per_episode.append(data)
    return {"per_episode": per_episode, "aggregated": {}}


def _assemble_physical_from_files(
    val_base: str | os.PathLike, cand_eps: set
) -> Dict[str, Any]:
    """Load every candidate episode's ``physical_judge.json`` into a nested dict."""
    nested: Dict[str, Any] = {}
    for task_id, episode_id in sorted(cand_eps):
        data = episode_store.load_metric(val_base, task_id, episode_id, "physical_judge")
        if isinstance(data, dict) and data:
            nested.setdefault(str(task_id), {})[str(episode_id)] = data
    return nested


def _merge_per_episode_dim(existing: Any, new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two per-episode-style dim results, re-aggregating the means.

    New records for a uid replace old ones (so ``--overwrite`` semantics also
    work when a caller re-runs a subset).
    """
    ex_list = existing.get("per_episode", []) if isinstance(existing, dict) else []
    new_list = new.get("per_episode", []) if isinstance(new, dict) else []

    new_uids = {r.get("uid") for r in new_list}
    merged = [r for r in ex_list if r.get("uid") not in new_uids] + list(new_list)

    # Prefer the fresh aggregated block for its config metadata (thresholds, ...).
    base_agg = dict(
        (new.get("aggregated") or {})
        if isinstance(new, dict) and new.get("aggregated")
        else (existing.get("aggregated") or {} if isinstance(existing, dict) else {})
    )
    for k in _VISUAL_METRIC_KEYS:
        if not any(k in r for r in merged):
            base_agg.pop(k, None)
            continue
        vals = [
            r[k]
            for r in merged
            if isinstance(r.get(k), (int, float))
            and not (isinstance(r[k], float) and math.isnan(r[k]))
        ]
        base_agg[k] = float(np.nanmean(vals)) if vals else float("nan")
    base_agg["num_episodes"] = len(merged)

    return {"per_episode": merged, "aggregated": base_agg}


class ACWMBenchmark:
    """Main entrypoint object, analogous to EWMBench's `EmbodiedWorldModelBenchmark`."""

    def __init__(self, device: torch.device | str, output_path: str | os.PathLike):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.output_path = str(output_path)
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def full_dimension_list() -> List[str]:
        return list(SUPPORTED_DIMENSIONS)

    def _load_vlm(self, dim_cfg: Dict[str, Any]) -> VLMBackend:
        vlm_cfg = dict(dim_cfg.get("vlm", {}))
        if "model_path" not in vlm_cfg:
            raise ValueError(
                "configs must define `physical_judge.vlm.model_path` "
                "(e.g. ./ckpt/Qwen3.5-27B-GPTQ-Int4)"
            )
        return build_vlm_backend(vlm_cfg)

    def evaluate(
        self,
        data_base: str | os.PathLike,
        data_name: str,
        dimension_list: List[str],
        gt_path: str | os.PathLike,
        dim_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        hardware_config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        limit_episodes: Optional[int] = None,
        limit_episodes_per_task: Optional[int] = None,
        task_filter: Optional[List[str]] = None,
        episode_filter: Optional[List[str]] = None,
        include_episodes: Optional[set] = None,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        dim_configs = dim_configs or {}
        hardware_config = hardware_config or {}

        for dim in dimension_list:
            if dim not in SUPPORTED_DIMENSIONS:
                raise ValueError(
                    f"Unsupported dimension {dim!r}. "
                    f"Supported: {SUPPORTED_DIMENSIONS}"
                )
        if {
            "interaction_spatial_alignment",
            "multiview_consistency",
        }.issubset(dimension_list):
            dimension_list = list(dimension_list)
            dimension_list.remove("interaction_spatial_alignment")
            multiview_index = dimension_list.index(
                "multiview_consistency"
            )
            dimension_list.insert(
                multiview_index,
                "interaction_spatial_alignment",
            )

        # Raw metric results are persisted per episode under data_base. This
        # in-memory dict is assembled from those files for aggregate reports.
        results_dict: Dict[str, Any] = {}

        trials = []
        standard_dimensions = [d for d in dimension_list if d != "policy_eval"]
        if standard_dimensions:
            print0(f"[acwmbench] Building trial list from {data_base} (gt={gt_path}) ...")
            trials = build_trial_list(data_base, gt_path)
            print0(f"[acwmbench] Found {len(trials)} trials.")

            # Show discovered subtasks and success-criteria coverage.
            by_task: Dict[str, Dict[str, int]] = {}
            for t in trials:
                rec = by_task.setdefault(
                    str(t.task_id), {"trials": 0, "with_criteria": 0}
                )
                rec["trials"] += 1
                if t.prompt_segments:
                    rec["with_criteria"] += 1
            for task_id in sorted(by_task):
                rec = by_task[task_id]
                print0(
                    f"[acwmbench]   task={task_id}: {rec['trials']} trial(s), "
                    f"{rec['with_criteria']} with success_criteria"
                )

            if task_filter:
                wanted = {str(x).strip().lower() for x in task_filter}
                before = len(trials)
                trials = [
                    t for t in trials if str(t.task_id).strip().lower() in wanted
                ]
                print0(
                    f"[acwmbench] --task {sorted(wanted)}: kept "
                    f"{len(trials)}/{before} trial(s)."
                )
                if not trials:
                    raise ValueError(
                        f"--task {sorted(wanted)} matched no subtasks. "
                        f"Available: {sorted(by_task)}"
                    )

            if episode_filter:
                def _norm_ep(x: str) -> str:
                    s = str(x).strip().lower()
                    return s[2:] if s.startswith("ep") else s

                wanted_eps = {_norm_ep(x) for x in episode_filter}
                before = len(trials)
                trials = [t for t in trials if _norm_ep(t.episode_id) in wanted_eps]
                print0(
                    f"[acwmbench] --episode {sorted(wanted_eps)}: kept "
                    f"{len(trials)}/{before} trial(s)."
                )
                if not trials:
                    available_eps = sorted(
                        {t.episode_id for t in build_trial_list(data_base, gt_path)}
                    )
                    raise ValueError(
                        f"--episode {sorted(wanted_eps)} matched no episodes. "
                        f"Available episode ids: {available_eps}"
                    )

            if include_episodes is not None:
                # Allow-list of (task_id, episode_id) that passed the
                # data-completeness check; drop everything else.
                allowed = {(str(t), str(e)) for (t, e) in include_episodes}
                before = len(trials)
                trials = [
                    t
                    for t in trials
                    if (str(t.task_id), str(t.episode_id)) in allowed
                ]
                print0(
                    f"[acwmbench] data-check allow-list: kept {len(trials)}/{before} "
                    f"trial(s) from {len(allowed)} complete episode(s)."
                )

            if limit_episodes_per_task is not None and limit_episodes_per_task > 0:
                # Keep every trial belonging to the first N episode IDs in each task.
                episode_keys_by_task: Dict[str, List[str]] = {}
                for trial in trials:
                    task_id = str(trial.task_id)
                    episode_id = str(trial.episode_id)
                    task_episodes = episode_keys_by_task.setdefault(task_id, [])
                    if episode_id not in task_episodes:
                        task_episodes.append(episode_id)
                selected = {
                    (task_id, episode_id)
                    for task_id, episode_ids in episode_keys_by_task.items()
                    for episode_id in sorted(episode_ids)[:limit_episodes_per_task]
                }
                before = len(trials)
                trials = [
                    trial
                    for trial in trials
                    if (str(trial.task_id), str(trial.episode_id)) in selected
                ]
                selected_counts = {
                    task_id: sum(1 for task, _ in selected if task == task_id)
                    for task_id in sorted(episode_keys_by_task)
                }
                print0(
                    f"[acwmbench] --top-n-per-task {limit_episodes_per_task}: "
                    f"kept {len(trials)}/{before} trial(s) across "
                    f"{len(selected)} episode(s): {selected_counts}."
                )

            if limit_episodes is not None and limit_episodes > 0:
                # Keep all trials for the first N distinct task/episode pairs.
                seen: List[tuple] = []
                kept = []
                for t in trials:
                    key = (t.task_id, t.episode_id)
                    if key not in seen:
                        if len(seen) >= limit_episodes:
                            break
                        seen.append(key)
                    kept.append(t)
                trials = kept
                print0(
                    f"[acwmbench] --top-n {limit_episodes}: limited to {len(seen)} "
                    f"episode(s) / {len(trials)} trial(s)."
                )

        candidate_episodes = {
            (str(t.task_id), str(t.episode_id)) for t in trials
        }

        for dim in dimension_list:
            print0(f"[acwmbench] Evaluating dimension: {dim}")
            dim_cfg = dim_configs.get(dim, {})
            interaction_execution = (
                resolve_interaction_execution(
                    hardware_config,
                    _interaction_execution_config(dim_cfg),
                    device=self.device,
                )
                if dim == "interaction_spatial_alignment"
                else None
            )

            if dim == "policy_eval":
                rollout_root = dim_cfg.get("rollout_root")
                if not rollout_root:
                    raise ValueError(
                        "dimensions.policy_eval.rollout_root must point to paired "
                        "real/WM policy rollouts"
                    )
                policy_dataset_override = os.environ.get("POLICY_EVAL_ROOT")
                policy_gt_root = (
                    policy_dataset_override
                    or dim_cfg.get("gt_root")
                    or gt_path
                )
                policy_eval_mode = str(
                    dim_cfg.get("mode", "paired_fidelity")
                ).strip().lower()
                if policy_eval_mode not in {
                    "paired_fidelity",
                    "wm_generate",
                    "wm_score",
                }:
                    raise ValueError(
                        "dimensions.policy_eval.mode must be "
                        "'paired_fidelity', 'wm_generate', or 'wm_score'"
                    )
                if policy_eval_mode == "wm_generate":
                    dataset_root = (
                        policy_dataset_override
                        or dim_cfg.get("dataset_root")
                    )
                    if not dataset_root:
                        raise ValueError(
                            "dimensions.policy_eval.dataset_root is required for "
                            "wm_generate mode"
                        )
                    initial_states_path = dim_cfg.get("initial_states_path")
                    if not initial_states_path:
                        raise ValueError(
                            "dimensions.policy_eval.initial_states_path is "
                            "required for wm_generate mode"
                        )
                    policy_cfg = dict(dim_cfg.get("policy", {}) or {})
                    world_model_cfg = dict(
                        dim_cfg.get("world_model", {}) or {}
                    )
                    policy = build_policy_client(policy_cfg)
                    world_model = HTTPWorldModelClient(
                        str(
                            world_model_cfg.get(
                                "server_url", "http://127.0.0.1:9000"
                            )
                        ),
                        timeout=float(
                            world_model_cfg.get("timeout_sec", 900)
                        ),
                        wire_action_layout=str(
                            world_model_cfg.get(
                                "wire_action_layout", "policy"
                            )
                        ),
                    )
                    generated_paths = generate_wm_policy_rollouts(
                        dataset_root=dataset_root,
                        rollout_root=rollout_root,
                        initial_states_path=initial_states_path,
                        policy=policy,
                        world_model=world_model,
                        policy_id=str(
                            policy_cfg.get("policy_id", "pi05_agibot")
                        ),
                        policy_checkpoint=str(
                            policy_cfg.get("checkpoint", "")
                        ),
                        world_model_id=str(
                            os.environ.get("WORLD_MODEL_ID")
                            or world_model_cfg.get(
                                "world_model_id", "ge_sim_ar"
                            )
                        ),
                        world_model_checkpoint=str(
                            os.environ.get("WORLD_MODEL_CHECKPOINT")
                            or world_model_cfg.get("checkpoint", "")
                        ),
                        action_horizon=int(
                            policy_cfg.get("action_horizon", 50)
                        ),
                        wm_chunk_size=int(
                            world_model_cfg.get("chunk_size", 25)
                        ),
                        compress_actions=bool(
                            world_model_cfg.get("compress_actions", True)
                        ),
                        state_source=str(
                            dim_cfg.get("state_source", "commanded")
                        ),
                        rollout_number=int(
                            dim_cfg.get("rollout_number", 3)
                        ),
                        task_filter=task_filter,
                        episode_filter=episode_filter,
                        limit_episodes=limit_episodes,
                        limit_episodes_per_task=limit_episodes_per_task,
                        overwrite=overwrite,
                    )
                    results_dict[dim] = {
                        "mode": "wm_generate",
                        "rollout_root": str(Path(rollout_root).resolve()),
                        "n_rollouts": len(generated_paths),
                        "summary": {
                            "status": "complete",
                            "seed": 42,
                            "rollout_number": int(
                                dim_cfg.get("rollout_number", 3)
                            ),
                            "message": (
                                "Generation only; run policy_eval in wm_score "
                                "mode for offline VLM evaluation."
                            ),
                        },
                    }
                    continue
                judge_cfg = dict(dim_configs.get("physical_judge", {}))
                judge_cfg.update(dim_cfg.get("judge", {}) or {})
                vlm_cfg = dim_cfg.get("vlm") or judge_cfg.get("vlm")
                discovered = discover_policy_rollouts(
                    rollout_root,
                    task_filter=task_filter,
                    init_filter=episode_filter,
                )
                needs_vlm = overwrite or any(
                    r.judge_path is None for r in discovered
                )
                vlms = []
                if needs_vlm and vlm_cfg:
                    execution_cfg = dict(
                        judge_cfg.get("execution", {}) or {}
                    )
                    api_workers = int(
                        execution_cfg.get("api_workers", 1)
                    )
                    if api_workers < 1:
                        raise ValueError(
                            "policy_eval judge api_workers must be >= 1"
                        )
                    backend_name = str(
                        vlm_cfg.get("backend", "transformers")
                    ).lower()
                    if (
                        api_workers > 1
                        and backend_name
                        not in {
                            "openai",
                            "vllm",
                            "anthropic",
                            "gemini",
                        }
                    ):
                        print0(
                            "[acwmbench] policy_eval: forcing "
                            f"api_workers=1 for local backend {backend_name!r}."
                        )
                        api_workers = 1
                    missing_count = (
                        len(discovered)
                        if overwrite
                        else sum(
                            r.judge_path is None for r in discovered
                        )
                    )
                    api_workers = min(api_workers, max(1, missing_count))
                    vlms = [
                        self._load_vlm({"vlm": vlm_cfg})
                        for _ in range(api_workers)
                    ]
                    print0(
                        "[acwmbench] policy_eval: "
                        f"{missing_count} rollout(s) need GPT judging; "
                        f"api_workers={len(vlms)}."
                    )
                try:
                    results_dict[dim] = compute_policy_evaluation(
                        rollout_root=rollout_root,
                        gt_root=policy_gt_root,
                        vlm=vlms[0] if vlms else None,
                        worker_vlms=vlms or None,
                        task_filter=task_filter,
                        init_filter=episode_filter,
                        judge_config=judge_cfg,
                        threshold=float(dim_cfg.get("threshold", 0.5)),
                        bootstrap_samples=int(dim_cfg.get("bootstrap_samples", 1000)),
                        bootstrap_seed=int(dim_cfg.get("bootstrap_seed", 0)),
                        require_paired=policy_eval_mode == "paired_fidelity",
                        overwrite=overwrite,
                    )
                finally:
                    for worker_vlm in vlms:
                        worker_vlm.close()
                continue

            if overwrite:
                for task_id, episode_id in candidate_episodes:
                    episode_store.remove_metric(
                        data_base, task_id, episode_id, dim
                    )
                    episode_store.clear_visualizations(
                        data_base, task_id, episode_id, dim
                    )
                print0(
                    f"[acwmbench] {dim}: --overwrite cleared cached result(s) "
                    f"for {len(candidate_episodes)} selected episode(s)."
                )

            # Resume from each episode's <metric>.json. A corrupt or absent file
            # is treated as pending; --overwrite recomputes every selected unit.
            done = (
                set()
                if overwrite
                else _done_uids_from_episode_files(
                    data_base,
                    trials,
                    dim,
                    require_visualization=(
                        visualize and dim == "keyarea_visual_quality"
                    ),
                    expected_metric_settings=(
                        (
                            {
                                "n_fallback_keysteps": int(
                                    dim_cfg.get("n_fallback_keysteps", 1)
                                ),
                                "max_image_long_side": int(
                                    dim_cfg.get("max_image_long_side", 768)
                                ),
                                "eval_view": str(
                                    dim_cfg.get("eval_view", "top_head")
                                ),
                                "format_retries": int(
                                    dim_cfg.get("format_retries", 2)
                                ),
                                "use_reference_frame": bool(
                                    dim_cfg.get("use_first_frame_reference", True)
                                ),
                            }
                            if dim == "physical_judge"
                            else {
                                "downsample_rate": int(
                                    dim_cfg.get("downsample_rate", 1)
                                ),
                                "fps": (
                                    float(dim_cfg["fps"])
                                    if dim_cfg.get("fps") is not None
                                    else None
                                ),
                                "secondary_mask_view": str(
                                    dim_cfg.get("secondary_mask_view", "top_head")
                                ),
                                "eval_view": str(
                                    dim_cfg.get("eval_view", "top_head")
                                ),
                                "crop_to_bbox": bool(
                                    dim_cfg.get("crop_to_bbox", True)
                                ),
                                "bbox_pad": int(dim_cfg.get("bbox_pad", 10)),
                                "lpips_min_crop_size": int(
                                    dim_cfg.get("lpips_min_crop_size", 64)
                                ),
                                "mask_dilation_pixels": int(
                                    dim_cfg.get("mask_dilation_pixels", 0)
                                ),
                                "mask_aware_psnr": bool(
                                    dim_cfg.get("mask_aware_psnr", True)
                                ),
                                "vjepa_model_name": str(
                                    (dim_cfg.get("vjepa", {}) or {}).get(
                                        "model_name",
                                        "vjepa2_1_vit_base_384",
                                    )
                                ),
                                "vjepa_model_path": (
                                    str(
                                        (dim_cfg.get("vjepa", {}) or {}).get(
                                            "model_path"
                                        )
                                    )
                                    if (dim_cfg.get("vjepa", {}) or {}).get(
                                        "model_path"
                                    )
                                    is not None
                                    else None
                                ),
                                "vjepa_repo": (
                                    str(
                                        (dim_cfg.get("vjepa", {}) or {}).get(
                                            "repo"
                                        )
                                    )
                                    if (dim_cfg.get("vjepa", {}) or {}).get(
                                        "repo"
                                    )
                                    is not None
                                    else None
                                ),
                            }
                        )
                        if dim in ("physical_judge", "keyarea_visual_quality")
                        else (
                            {}
                            if dim == "multiview_consistency"
                            else (
                                _dino_result_settings(
                                    dim_cfg,
                                    interaction_execution=interaction_execution,
                                )
                                if dim
                                == "interaction_spatial_alignment"
                                else None
                            )
                        )
                    ),
                )
            )
            dim_trials = [t for t in trials if t.uid not in done]
            if done:
                print0(
                    f"[acwmbench] {dim}: resume from per-episode files — "
                    f"{len(done)} unit(s) already done, "
                    f"{len(dim_trials)} to evaluate."
                )
            if not dim_trials:
                print0(
                    f"[acwmbench] {dim}: nothing new to evaluate; "
                    "assembling existing per-episode files."
                )
                if dim == "physical_judge":
                    results_dict[dim] = _assemble_physical_from_files(
                        data_base, candidate_episodes
                    )
                else:
                    assembled = _assemble_per_episode_from_files(
                        data_base, candidate_episodes, dim
                    )
                    results_dict[dim] = _merge_per_episode_dim(None, assembled)
                continue

            fps = float(dim_cfg["fps"]) if dim_cfg.get("fps") else None

            if dim == "physical_judge":
                physical_execution = dict(
                    dim_cfg.get("execution", {}) or {}
                )
                api_workers = int(
                    physical_execution.get("api_workers", 1)
                )
                if api_workers < 1:
                    raise ValueError(
                        "physical_judge.execution.api_workers must be >= 1"
                    )
                backend_name = str(
                    (dim_cfg.get("vlm", {}) or {}).get(
                        "backend", "transformers"
                    )
                ).lower()
                if (
                    api_workers > 1
                    and backend_name
                    not in {"openai", "vllm", "anthropic", "gemini"}
                ):
                    print0(
                        "[acwmbench] physical_judge: forcing api_workers=1 "
                        f"for local backend {backend_name!r}."
                    )
                    api_workers = 1
                api_workers = min(api_workers, len(dim_trials))
                vlms = [
                    self._load_vlm(dim_cfg) for _ in range(api_workers)
                ]
                new_usage = None
                try:
                    new_dim = compute_physical_judge(
                        vlm=vlms[0],
                        worker_vlms=vlms,
                        trials=dim_trials,
                        n_fallback_keysteps=int(dim_cfg.get("n_fallback_keysteps", 1)),
                        max_image_long_side=int(dim_cfg.get("max_image_long_side", 768)),
                        eval_view=str(dim_cfg.get("eval_view", "top_head")),
                        format_retries=int(dim_cfg.get("format_retries", 2)),
                        api_retries=int(dim_cfg.get("api_retries", 5)),
                        api_retry_backoff=float(dim_cfg.get("api_retry_backoff", 2.0)),
                        use_reference_frame=bool(dim_cfg.get("use_first_frame_reference", True)),
                        save_root=data_base,
                    )
                    for worker_vlm in vlms:
                        usage_getter = getattr(
                            worker_vlm, "get_usage_summary", None
                        )
                        if callable(usage_getter):
                            new_usage = _merge_usage(
                                new_usage, usage_getter()
                            )
                finally:
                    for worker_vlm in vlms:
                        worker_vlm.close()
                results_dict[dim] = _assemble_physical_from_files(
                    data_base, candidate_episodes
                )
                if new_usage:
                    results_dict["physical_judge_usage"] = _merge_usage(
                        results_dict.get("physical_judge_usage"), new_usage
                    )
            elif dim == "keyarea_visual_quality":
                vjepa_cfg = dim_cfg.get("vjepa", {}) or {}
                execution = resolve_keyarea_execution(
                    hardware_config,
                    dim_cfg.get("execution", {}) or {},
                    device=self.device,
                )
                new_dim = compute_keyarea_visual_quality(
                    episodes=dim_trials,
                    gt_root=gt_path,
                    device=str(self.device),
                    eval_view=str(dim_cfg.get("eval_view", "top_head")),
                    downsample_rate=int(dim_cfg.get("downsample_rate", 1)),
                    fps=fps,
                    crop_to_bbox=bool(dim_cfg.get("crop_to_bbox", True)),
                    bbox_pad=int(dim_cfg.get("bbox_pad", 10)),
                    lpips_min_crop_size=int(dim_cfg.get("lpips_min_crop_size", 64)),
                    mask_dilation_pixels=int(
                        dim_cfg.get("mask_dilation_pixels", 0)
                    ),
                    mask_aware_psnr=bool(dim_cfg.get("mask_aware_psnr", True)),
                    vjepa_model_path=vjepa_cfg.get("model_path"),
                    vjepa_model_name=str(
                        vjepa_cfg.get("model_name", "vjepa2_1_vit_base_384")
                    ),
                    vjepa_repo=vjepa_cfg.get("repo"),
                    secondary_mask_view=str(
                        dim_cfg.get("secondary_mask_view", "top_head")
                    ),
                    visualize=visualize,
                    save_root=data_base,
                    visualize_max_frames=int(dim_cfg.get("visualize_max_frames", 8)),
                    execution=execution.to_dict(),
                )
                assembled = _assemble_per_episode_from_files(
                    data_base, candidate_episodes, dim
                )
                results_dict[dim] = _merge_per_episode_dim(new_dim, assembled)
            elif dim == "interaction_spatial_alignment":
                if interaction_execution is None:
                    raise RuntimeError(
                        "Interaction execution settings were not resolved"
                    )
                new_dim = compute_interaction_spatial_alignment(
                    episodes=dim_trials,
                    gt_root=gt_path,
                    device=str(self.device),
                    eval_view=str(dim_cfg.get("eval_view", "top_head")),
                    downsample_rate=int(
                        dim_cfg.get("downsample_rate", 10)
                    ),
                    fps=float(dim_cfg.get("fps", 30.0)),
                    checkpoint=str(
                        dim_cfg.get(
                            "checkpoint",
                            _DINO_DEFAULT_CHECKPOINT,
                        )
                    ),
                    dinov3_repo=dim_cfg.get("dinov3_repo"),
                    model_name=str(
                        dim_cfg.get("model_name", "dinov3_vitb16")
                    ),
                    batch_size=interaction_execution.frame_batch_size,
                    min_mask_patches=int(
                        dim_cfg.get("min_mask_patches", 2)
                    ),
                    min_mutual_matches=int(
                        dim_cfg.get("min_mutual_matches", 3)
                    ),
                    inlier_radius_px=float(
                        dim_cfg.get("inlier_radius_px", 32.0)
                    ),
                    confidence_bias=float(
                        dim_cfg.get("confidence_bias", -2.0)
                    ),
                    confidence_cosine_weight=float(
                        dim_cfg.get("confidence_cosine_weight", 3.0)
                    ),
                    confidence_margin_weight=float(
                        dim_cfg.get("confidence_margin_weight", 8.0)
                    ),
                    confidence_mutual_weight=float(
                        dim_cfg.get("confidence_mutual_weight", 1.5)
                    ),
                    confidence_inlier_weight=float(
                        dim_cfg.get("confidence_inlier_weight", 1.5)
                    ),
                    match_support_target_ratio=float(
                        dim_cfg.get(
                            "match_support_target_ratio",
                            0.3,
                        )
                    ),
                    match_support_count_scale=float(
                        dim_cfg.get(
                            "match_support_count_scale",
                            8.0,
                        )
                    ),
                    match_support_power=float(
                        dim_cfg.get("match_support_power", 1.0)
                    ),
                    confidence_bad=float(
                        dim_cfg.get("confidence_bad", 0.5)
                    ),
                    confidence_good=float(
                        dim_cfg.get("confidence_good", 0.8)
                    ),
                    distance_good_px=float(
                        dim_cfg.get("distance_good_px", 30.0)
                    ),
                    distance_bad_px=float(
                        dim_cfg.get("distance_bad_px", 150.0)
                    ),
                    alignment_confidence_power=float(
                        dim_cfg.get(
                            "alignment_confidence_power",
                            3.0,
                        )
                    ),
                    save_root=data_base,
                    execution=interaction_execution.to_dict(),
                )
                assembled = _assemble_per_episode_from_files(
                    data_base,
                    candidate_episodes,
                    dim,
                )
                results_dict[dim] = _merge_per_episode_dim(
                    new_dim,
                    assembled,
                )
            elif dim == "multiview_consistency":
                multiview_execution_cfg = dict(
                    dim_cfg.get("execution", {}) or {}
                )
                if (
                    "frame_batch_size" not in multiview_execution_cfg
                    and dim_cfg.get("batch_size") is not None
                ):
                    multiview_execution_cfg["frame_batch_size"] = int(
                        dim_cfg["batch_size"]
                    )
                execution = resolve_interaction_execution(
                    hardware_config,
                    multiview_execution_cfg,
                    device=self.device,
                )
                new_dim = compute_multiview_consistency(
                    episodes=dim_trials,
                    gt_root=gt_path,
                    device=str(self.device),
                    save_root=data_base,
                    execution=execution.to_dict(),
                )
                assembled = _assemble_per_episode_from_files(
                    data_base, candidate_episodes, dim
                )
                results_dict[dim] = _merge_per_episode_dim(new_dim, assembled)
            else:  # pragma: no cover - guarded above
                raise ValueError(f"Unsupported dimension {dim!r}")

        csv_path = Path(self.output_path) / "acwm_final_table.csv"
        merge_results_to_csv(results_dict, str(csv_path))
        print0(f"[acwmbench] Wrote CSV to {csv_path}")
        policy_paths = write_policy_eval_reports(results_dict, self.output_path, data_name)
        if policy_paths:
            print0(f"[acwmbench] Wrote policy-eval reports: {policy_paths}")
        return results_dict
