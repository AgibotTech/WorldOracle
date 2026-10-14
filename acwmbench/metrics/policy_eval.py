"""Policy Evaluation Ability: real-vs-world-model outcome fidelity.

The primary score compares distributions of task criteria, not pixels. Each
world-model rollout remains independent of the real trajectory; oracle branch
selection using real observations is intentionally excluded to avoid leakage.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..policy_eval import PolicyRollout, discover_policy_rollouts, validate_paired_rollouts


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _criterion_scores(result: Mapping[str, Any]) -> dict[str, int]:
    """Flatten Physical Judge output into stable criterion keys."""

    scores: dict[str, int] = {}
    for step_index, step in enumerate(result.get("key_steps") or []):
        if not isinstance(step, Mapping):
            continue
        criteria = step.get("success_criteria") or []
        if isinstance(criteria, str):
            criteria = [criteria]
        values = step.get("aspect_scores") or []
        segment_id = step.get("segment_id", step_index)
        for criterion_index, criterion in enumerate(criteria):
            if criterion_index >= len(values):
                continue
            value = int(values[criterion_index])
            if value not in (0, 1):
                continue
            key = f"{segment_id}:{criterion_index}:{str(criterion).strip()}"
            scores[key] = value
    if scores:
        return scores

    final = result.get("final") or {}
    fallback = _finite_float(final.get("score", result.get("score")))
    if fallback is not None:
        scores["final:0:task_success"] = int(fallback >= 0.5)
    return scores


def _persist_episode_judge(
    rollout: PolicyRollout,
    result: Mapping[str, Any],
    payloads: dict[Path, dict[str, Any]] | None = None,
) -> Path:
    """Atomically store one rollout result in its episode policy_eval.json."""
    judge_path = rollout.root.parent / "policy_eval.json"
    payloads = payloads if payloads is not None else {}
    if judge_path not in payloads:
        existing = {}
        if judge_path.is_file():
            try:
                loaded = json.loads(judge_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, ValueError, json.JSONDecodeError):
                existing = {}
        payloads[judge_path] = existing
    payloads[judge_path][rollout.metadata.rollout_id] = dict(result)
    temporary = judge_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            payloads[judge_path],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(judge_path)
    return judge_path


def _load_precomputed_judges(
    rollouts: Sequence[PolicyRollout],
    *,
    overwrite: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[PolicyRollout]]:
    judged: dict[str, dict[str, Any]] = {}
    missing: list[PolicyRollout] = []
    for rollout in rollouts:
        if overwrite or rollout.judge_path is None:
            missing.append(rollout)
            continue
        with rollout.judge_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if rollout.judge_path.name == "policy_eval.json":
            result = (
                payload.get(rollout.metadata.rollout_id)
                or payload.get(rollout.uid)
                if isinstance(payload, Mapping)
                else None
            )
        else:
            result = payload
        if not isinstance(result, dict):
            missing.append(rollout)
            continue
        judged[rollout.uid] = result
        if rollout.judge_path.name == "physical_judge.json":
            _persist_episode_judge(rollout, result)
    return judged, missing


def _make_judge_trial(rollout: PolicyRollout, gt_root: Path) -> Any:
    from ..dataset import (
        EpisodeTrial,
        _init_frame,
        _read_grounding_objects,
        _read_instruction,
        _read_keysteps,
        _read_prompt_segments,
        _read_task_spec,
    )

    if rollout.video_dir is None:
        raise ValueError(f"{rollout.root} has no video for VLM judging")
    m = rollout.metadata
    gt_task_dir = gt_root / m.task_id
    gt_episode_dir = gt_task_dir / m.init_id
    if not gt_episode_dir.is_dir():
        raise FileNotFoundError(
            f"missing GT metadata for policy rollout {rollout.uid}: {gt_episode_dir}"
        )
    gt_video_dir = gt_episode_dir / "video"
    return EpisodeTrial(
        task_id=m.task_id,
        episode_id=m.init_id,
        trial_id=m.rollout_id,
        video_dir=str(rollout.video_dir),
        instruction=m.prompt or _read_instruction(gt_episode_dir),
        init_frame=_init_frame(gt_episode_dir),
        task_spec=_read_task_spec(gt_task_dir, gt_episode_dir),
        keysteps=_read_keysteps(gt_episode_dir),
        prompt_segments=_read_prompt_segments(gt_episode_dir),
        grounding_objects=_read_grounding_objects(gt_episode_dir),
        gt_video_dir=str(gt_video_dir) if gt_video_dir.is_dir() else None,
        meta=rollout.metadata.to_dict(),
    )


def _judge_missing_rollouts(
    rollouts: Sequence[PolicyRollout],
    *,
    vlm: Any,
    worker_vlms: Sequence[Any] | None,
    gt_root: Path,
    judge_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not rollouts:
        return {}
    if vlm is None:
        raise ValueError(
            "rollouts without physical_judge.json require a VLM configuration"
        )
    from .physical_judge import compute_physical_judge

    trials = [_make_judge_trial(rollout, gt_root) for rollout in rollouts]
    nested = compute_physical_judge(
        vlm=vlm,
        worker_vlms=worker_vlms,
        trials=trials,
        n_fallback_keysteps=int(judge_config.get("n_fallback_keysteps", 1)),
        max_image_long_side=int(judge_config.get("max_image_long_side", 768)),
        eval_view=str(judge_config.get("eval_view", "top_head")),
        format_retries=int(judge_config.get("format_retries", 2)),
        api_retries=int(judge_config.get("api_retries", 5)),
        api_retry_backoff=float(judge_config.get("api_retry_backoff", 2.0)),
        use_reference_frame=bool(judge_config.get("use_first_frame_reference", True)),
    )
    judged: dict[str, dict[str, Any]] = {}
    episode_payloads: dict[Path, dict[str, Any]] = {}
    for rollout in rollouts:
        m = rollout.metadata
        result = (
            nested.get(m.task_id, {})
            .get(m.init_id, {})
            .get(m.rollout_id)
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Physical Judge returned no result for {rollout.uid}")
        judged[rollout.uid] = result
        _persist_episode_judge(rollout, result, episode_payloads)
    return judged


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""

    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    start = 0
    while start < len(arr):
        stop = start + 1
        while stop < len(arr) and arr[order[stop]] == arr[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx = np.sign(xs[i] - xs[j])
            dy = np.sign(ys[i] - ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return (concordant - discordant) / denominator if denominator else None


def _cohen_kappa(real: Sequence[int], wm: Sequence[int]) -> float | None:
    if not real or len(real) != len(wm):
        return None
    r = np.asarray(real, dtype=np.int64)
    w = np.asarray(wm, dtype=np.int64)
    observed = float(np.mean(r == w))
    p_r = float(np.mean(r))
    p_w = float(np.mean(w))
    expected = p_r * p_w + (1.0 - p_r) * (1.0 - p_w)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else None


def _bootstrap_correlations(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, list[float] | None]:
    if len(rows) < 2 or samples <= 0:
        return {"pearson_95ci": None, "spearman_95ci": None}
    rng = np.random.default_rng(seed)
    pearson: list[float] = []
    spearman: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        real = [float(rows[i]["real_score"]) for i in indices]
        wm = [float(rows[i]["wm_score"]) for i in indices]
        p = _pearson(real, wm)
        s = _spearman(real, wm)
        if p is not None:
            pearson.append(p)
        if s is not None:
            spearman.append(s)

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [float(x) for x in np.percentile(values, [2.5, 97.5])]

    return {"pearson_95ci": interval(pearson), "spearman_95ci": interval(spearman)}


def _calibration_summary(
    cells: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    if not cells:
        return {
            "n_criterion_cells": 0,
            "brier_score": None,
            "mae": None,
            "threshold_agreement": None,
            "cohen_kappa": None,
        }
    real_prob = np.asarray([float(row["real_probability"]) for row in cells])
    wm_prob = np.asarray([float(row["wm_probability"]) for row in cells])
    real_binary = (real_prob >= threshold).astype(np.int64)
    wm_binary = (wm_prob >= threshold).astype(np.int64)
    # Evaluate each empirical real observation against the WM probability.
    brier_terms: list[float] = []
    for row in cells:
        brier_terms.extend(
            (float(row["wm_probability"]) - float(y)) ** 2
            for y in row["real_observations"]
        )
    return {
        "n_criterion_cells": len(cells),
        "n_real_observations": len(brier_terms),
        "brier_score": float(np.mean(brier_terms)) if brier_terms else None,
        "mae": float(np.mean(np.abs(real_prob - wm_prob))),
        "threshold": float(threshold),
        "threshold_agreement": float(np.mean(real_binary == wm_binary)),
        "cohen_kappa": _cohen_kappa(real_binary.tolist(), wm_binary.tolist()),
    }


def _build_criterion_cells(
    rollouts: Sequence[PolicyRollout],
    scores_by_uid: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, list[int]]] = {}
    for rollout in rollouts:
        m = rollout.metadata
        for criterion_key, score in scores_by_uid[rollout.uid].items():
            key = (m.policy_id, m.task_id, m.init_id, criterion_key)
            groups.setdefault(key, {"real": [], "wm": []})[m.domain].append(score)

    cells: list[dict[str, Any]] = []
    for (policy_id, task_id, init_id, criterion_key), values in sorted(groups.items()):
        if not values["real"] or not values["wm"]:
            continue
        cells.append(
            {
                "policy_id": policy_id,
                "task_id": task_id,
                "init_id": init_id,
                "criterion_key": criterion_key,
                "n_real": len(values["real"]),
                "n_wm": len(values["wm"]),
                "real_observations": values["real"],
                "wm_observations": values["wm"],
                "real_probability": float(np.mean(values["real"])),
                "wm_probability": float(np.mean(values["wm"])),
            }
        )
    return cells


def evaluate_judged_rollouts(
    rollouts: Sequence[PolicyRollout],
    judged: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate already-judged rollouts; useful without a VLM and in tests."""

    validate_paired_rollouts(rollouts)
    rollout_scores: dict[str, float] = {}
    scores_by_uid: dict[str, dict[str, int]] = {}

    for rollout in rollouts:
        result = judged.get(rollout.uid)
        if not isinstance(result, Mapping):
            raise ValueError(f"missing judge result for rollout {rollout.uid}")
        scores = _criterion_scores(result)
        if not scores:
            raise ValueError(f"judge result has no usable criteria for {rollout.uid}")
        scores_by_uid[rollout.uid] = scores
        rollout_scores[rollout.uid] = float(np.mean(list(scores.values())))
    cells = _build_criterion_cells(rollouts, scores_by_uid)

    per_policy_task: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"real": [], "wm": []}
    )
    for rollout in rollouts:
        m = rollout.metadata
        per_policy_task[(m.policy_id, m.task_id)][m.domain].append(
            rollout_scores[rollout.uid]
        )
    policy_task_rows: list[dict[str, Any]] = []
    for (policy_id, task_id), values in sorted(per_policy_task.items()):
        if not values["real"] or not values["wm"]:
            continue
        policy_task_rows.append(
            {
                "policy_id": policy_id,
                "task_id": task_id,
                "n_real": len(values["real"]),
                "n_wm": len(values["wm"]),
                "real_score": float(np.mean(values["real"])),
                "wm_score": float(np.mean(values["wm"])),
            }
        )

    real_scores = [row["real_score"] for row in policy_task_rows]
    wm_scores = [row["wm_score"] for row in policy_task_rows]
    correlation = {
        "n_policy_task_cells": len(policy_task_rows),
        "n_policies": len({row["policy_id"] for row in policy_task_rows}),
        "pearson": _pearson(real_scores, wm_scores),
        "spearman": _spearman(real_scores, wm_scores),
        "kendall_tau_b": _kendall_tau_b(real_scores, wm_scores),
    }
    correlation.update(
        _bootstrap_correlations(policy_task_rows, bootstrap_samples, bootstrap_seed)
    )

    # Protocol ablations are descriptive slices of the same paired cells.
    cell_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        cell_lookup[(cell["policy_id"], cell["task_id"], cell["init_id"])].append(cell)
    ablation_groups: dict[tuple[int, str, bool], set[tuple[str, str, str]]] = defaultdict(set)
    for rollout in rollouts:
        m = rollout.metadata
        if m.domain != "wm":
            continue
        protocol = m.protocol
        ablation_groups[
            (protocol.replan_horizon, protocol.state_source, protocol.action_compression)
        ].add((m.policy_id, m.task_id, m.init_id))
    ablations: list[dict[str, Any]] = []
    for (horizon, state_source, compression), keys in sorted(ablation_groups.items()):
        subset = [cell for key in keys for cell in cell_lookup.get(key, [])]
        ablations.append(
            {
                "replan_horizon": horizon,
                "state_source": state_source,
                "action_compression": compression,
                **_calibration_summary(subset, threshold),
            }
        )

    seed_ablations: list[dict[str, Any]] = []
    wm_seeds = sorted(
        {
            rollout.metadata.seed
            for rollout in rollouts
            if rollout.metadata.domain == "wm" and rollout.metadata.seed is not None
        }
    )
    for seed in wm_seeds:
        seed_wm = [
            rollout
            for rollout in rollouts
            if rollout.metadata.domain == "wm" and rollout.metadata.seed == seed
        ]
        paired_keys = {
            (r.metadata.policy_id, r.metadata.task_id, r.metadata.init_id) for r in seed_wm
        }
        real = [
            rollout
            for rollout in rollouts
            if rollout.metadata.domain == "real"
            and (
                rollout.metadata.policy_id,
                rollout.metadata.task_id,
                rollout.metadata.init_id,
            )
            in paired_keys
        ]
        seed_cells = _build_criterion_cells(real + seed_wm, scores_by_uid)
        seed_ablations.append(
            {"wm_seed": seed, **_calibration_summary(seed_cells, threshold)}
        )

    return {
        "summary": {
            "calibration": _calibration_summary(cells, threshold),
            "correlation": correlation,
            "warning": (
                "Only one policy_id is present; correlation measures task variation, "
                "not general policy-ranking ability."
                if correlation["n_policies"] < 2
                else None
            ),
        },
        "criterion_cells": cells,
        "policy_task": policy_task_rows,
        "protocol_ablations": ablations,
        "wm_seed_ablations": seed_ablations,
        "rollout_scores": rollout_scores,
        "judge_results": dict(judged),
    }


def evaluate_wm_judged_rollouts(
    rollouts: Sequence[PolicyRollout],
    judged: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate world-model-only policy rollouts.

    This mode measures imagined task success and chooses the highest-scoring
    rollout for each policy/task/episode before dataset-level aggregation. It
    deliberately does not manufacture a real-domain baseline or correlation.
    """

    wm_rollouts = [r for r in rollouts if r.metadata.domain == "wm"]
    if not wm_rollouts:
        raise ValueError("wm_rollout evaluation found no world-model rollouts")

    rows: list[dict[str, Any]] = []
    rollout_scores: dict[str, float] = {}
    scores_by_uid: dict[str, dict[str, int]] = {}
    for rollout in wm_rollouts:
        result = judged.get(rollout.uid)
        if not isinstance(result, Mapping):
            raise ValueError(f"missing judge result for rollout {rollout.uid}")
        scores = _criterion_scores(result)
        if not scores:
            raise ValueError(f"judge result has no usable criteria for {rollout.uid}")
        score = float(np.mean(list(scores.values())))
        scores_by_uid[rollout.uid] = scores
        rollout_scores[rollout.uid] = score
        m = rollout.metadata
        rows.append(
            {
                "uid": rollout.uid,
                "world_model_id": m.world_model_id or "unknown",
                "policy_id": m.policy_id,
                "task_id": m.task_id,
                "init_id": m.init_id,
                "rollout_id": m.rollout_id,
                "seed": m.seed,
                "state_source": m.protocol.state_source,
                "score": score,
                "n_criteria": len(scores),
            }
        )
    episode_groups: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        episode_groups[
            (
                str(row["world_model_id"]),
                str(row["policy_id"]),
                str(row["task_id"]),
                str(row["init_id"]),
            )
        ].append(row)
    selected_rows: list[dict[str, Any]] = []
    for candidates in episode_groups.values():
        best = max(candidates, key=lambda row: (float(row["score"]), str(row["rollout_id"])))
        best["selected_best"] = True
        selected_rows.append(best)
        for row in candidates:
            row.setdefault("selected_best", False)

    criterion_groups: dict[
        tuple[str, str, str, str], list[int]
    ] = defaultdict(list)
    policy_task_groups: dict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
    for row in selected_rows:
        key = (
            str(row["world_model_id"]),
            str(row["policy_id"]),
            str(row["task_id"]),
        )
        policy_task_groups[key].append(float(row["score"]))
        for criterion_key, value in scores_by_uid[str(row["uid"])].items():
            criterion_groups[(*key, criterion_key)].append(value)

    policy_task = [
        {
            "world_model_id": world_model_id,
            "policy_id": policy_id,
            "task_id": task_id,
            "n_episodes": len(values),
            "wm_score": float(np.mean(values)),
        }
        for (
            world_model_id,
            policy_id,
            task_id,
        ), values in sorted(policy_task_groups.items())
    ]
    criterion_cells = [
        {
            "world_model_id": world_model_id,
            "policy_id": policy_id,
            "task_id": task_id,
            "criterion_key": criterion_key,
            "n_wm": len(values),
            "wm_observations": values,
            "wm_probability": float(np.mean(values)),
        }
        for (world_model_id, policy_id, task_id, criterion_key), values in sorted(
            criterion_groups.items()
        )
    ]
    task_scores = {
        task_id: float(
            np.mean(
                [
                    row["score"]
                    for row in selected_rows
                    if row["task_id"] == task_id
                ]
            )
        )
        for task_id in sorted({str(row["task_id"]) for row in selected_rows})
    }
    return {
        "mode": "wm_score",
        "summary": {
            "n_rollouts": len(rows),
            "n_episodes": len(selected_rows),
            "n_policies": len({str(row["policy_id"]) for row in rows}),
            "n_world_models": len(
                {str(row["world_model_id"]) for row in rows}
            ),
            "n_tasks": len(task_scores),
            "mean_wm_success": float(
                np.mean([row["score"] for row in selected_rows])
            ),
            "per_task_wm_success": task_scores,
            "selection": "best_rollout_per_episode",
            "warning": (
                "WM-only success does not measure world-model fidelity; add paired "
                "real rollouts before interpreting policy rankings as calibrated."
            ),
        },
        "wm_rollouts": rows,
        "criterion_cells": criterion_cells,
        "policy_task": policy_task,
        "protocol_ablations": [],
        "wm_seed_ablations": [],
        "rollout_scores": rollout_scores,
        "criterion_scores": scores_by_uid,
        "judge_results": dict(judged),
    }


def compute_policy_evaluation(
    *,
    rollout_root: str | Path,
    gt_root: str | Path,
    vlm: Any = None,
    worker_vlms: Sequence[Any] | None = None,
    task_filter: Iterable[str] | None = None,
    init_filter: Iterable[str] | None = None,
    judge_config: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    require_paired: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Discover, judge, and aggregate policy rollouts.

    ``require_paired=False`` is the closed-loop ACWM-only experiment. It judges
    only ``wm`` rollouts and reports imagined task success without claiming
    real-vs-world-model calibration.
    """

    rollouts = discover_policy_rollouts(
        rollout_root, task_filter=task_filter, init_filter=init_filter
    )
    if require_paired:
        validate_paired_rollouts(rollouts)
    else:
        rollouts = [rollout for rollout in rollouts if rollout.metadata.domain == "wm"]
        if not rollouts:
            raise ValueError("no world-model rollouts found")
    judged, missing = _load_precomputed_judges(
        rollouts,
        overwrite=overwrite,
    )
    judged.update(
        _judge_missing_rollouts(
            missing,
            vlm=vlm,
            worker_vlms=worker_vlms,
            gt_root=Path(gt_root),
            judge_config=judge_config or {},
        )
    )
    result = (
        evaluate_judged_rollouts(
            rollouts,
            judged,
            threshold=threshold,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if require_paired
        else evaluate_wm_judged_rollouts(rollouts, judged)
    )
    result["rollout_root"] = str(Path(rollout_root).resolve())
    result["n_rollouts"] = len(rollouts)
    return result
