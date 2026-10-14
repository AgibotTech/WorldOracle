"""Aggregate per-dimension results into a flat CSV table."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _collect_triplets(results: Dict[str, Any]) -> List[tuple]:
    triplets = set()
    for dim_name, dim_data in results.items():
        if dim_name == "policy_eval" or dim_name.endswith("_usage"):
            continue
        if not isinstance(dim_data, dict):
            continue
        for task_id, episodes in dim_data.items():
            if not isinstance(episodes, dict):
                continue
            for episode_id, trials in episodes.items():
                if not isinstance(trials, dict):
                    continue
                for trial_id in trials.keys():
                    triplets.add((str(task_id), str(episode_id), str(trial_id)))
    return sorted(triplets)


def _safe_get(nested: Dict[str, Any], *keys, default=None):
    cur = nested
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def merge_results_to_csv(
    results: Dict[str, Any],
    save_path: str,
) -> str:
    """Flatten the nested results dict into one CSV row per (task, episode, trial).

    Returns the absolute path of the saved CSV.
    """
    triplets = _collect_triplets(results)

    rows: List[Dict[str, Any]] = []
    metric_columns: set[str] = set()

    for task_id, episode_id, trial_id in triplets:
        row: Dict[str, Any] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "trial_id": trial_id,
        }

        pj = _safe_get(results, "physical_judge", task_id, episode_id, trial_id)
        if isinstance(pj, dict):
            row["pj_score"] = pj.get("score")
            final = pj.get("final") or {}
            row["pj_final_satisfied"] = final.get("satisfied")
            row["pj_final_score"] = final.get("score")
            row["pj_n_keysteps"] = len(pj.get("key_steps") or [])
            row["pj_explanation"] = (final.get("explanation") or "")[:200]
            metric_columns.update(
                {
                    "pj_score",
                    "pj_final_satisfied",
                    "pj_final_score",
                    "pj_n_keysteps",
                    "pj_explanation",
                }
            )

        rows.append(row)

    columns = ["task_id", "episode_id", "trial_id"] + sorted(metric_columns)

    # Build the per-trial table.
    df = pd.DataFrame(rows, columns=columns)

    # Append a "MEAN" row over numeric columns.
    mean_row: Dict[str, Any] = {c: "" for c in columns}
    mean_row["task_id"] = "MEAN"
    for c in columns:
        if c in {"task_id", "episode_id", "trial_id"}:
            continue
        if df[c].dtype.kind in "fiub":
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            mean_row[c] = round(float(vals.mean()), 6) if len(vals) else ""
        else:
            # Try numeric coercion for object columns too (handles bools-as-strings).
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                mean_row[c] = round(float(vals.mean()), 6)

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    df.to_csv(save_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# physical_judge accuracy summary
#
# When the prediction side IS the ground truth (val_base == gt_path), every
# success_criterion was authored from that exact key frame, so a perfect judge
# should mark all of them satisfied (criterion accuracy == 1.0). This summary
# quantifies how close we are and surfaces the failing criteria / episodes so
# they can drive prompt-engineering and data-filtering.
# ---------------------------------------------------------------------------


def summarize_physical_judge(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute criterion-level accuracy + per-episode + failure breakdown.

    ``results`` is the full results dict (i.e. it has a ``physical_judge`` key).
    """
    pj = results.get("physical_judge")
    if not isinstance(pj, dict):
        return {}

    n_criteria = 0
    n_satisfied = 0
    n_segments = 0
    n_segments_full = 0
    n_format_invalid = 0
    trial_scores: List[float] = []
    per_episode: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    for task_id, episodes in pj.items():
        if not isinstance(episodes, dict):
            continue
        for episode_id, trials in episodes.items():
            if not isinstance(trials, dict):
                continue
            ep_key = f"{task_id}/{episode_id}"
            ep = per_episode.setdefault(
                ep_key, {"n_criteria": 0, "n_satisfied": 0, "n_segments": 0, "n_segments_full": 0}
            )
            for trial_id, per_trial in trials.items():
                if not isinstance(per_trial, dict):
                    continue
                try:
                    trial_scores.append(float(per_trial.get("score", 0.0)))
                except (TypeError, ValueError):
                    pass
                for step in per_trial.get("key_steps") or []:
                    if not isinstance(step, dict):
                        continue
                    criteria = step.get("success_criteria")
                    if not criteria:
                        # No success criteria on this step (fallback final frame): skip.
                        continue
                    if isinstance(criteria, str):
                        criteria = [criteria]
                    scores = step.get("aspect_scores") or []
                    n_segments += 1
                    ep["n_segments"] += 1
                    if not step.get("format_valid"):
                        n_format_invalid += 1
                    seg_full = bool(scores) and all(int(s) == 1 for s in scores)
                    if seg_full:
                        n_segments_full += 1
                        ep["n_segments_full"] += 1
                    reasoning = step.get("reasoning") or []
                    for i, crit in enumerate(criteria):
                        score = int(scores[i]) if i < len(scores) else 0
                        n_criteria += 1
                        ep["n_criteria"] += 1
                        if score == 1:
                            n_satisfied += 1
                            ep["n_satisfied"] += 1
                        else:
                            failures.append(
                                {
                                    "task_id": str(task_id),
                                    "episode_id": str(episode_id),
                                    "trial_id": str(trial_id),
                                    "segment_id": step.get("segment_id"),
                                    "frame": step.get("frame"),
                                    "source_frame": step.get("source_frame"),
                                    "criterion_index": i + 1,
                                    "criterion": str(crit),
                                    "score": score,
                                    "format_valid": bool(step.get("format_valid")),
                                    "evidence": str(reasoning[i]) if i < len(reasoning) else "",
                                }
                            )

    for ep in per_episode.values():
        ep["criterion_accuracy"] = (
            ep["n_satisfied"] / ep["n_criteria"] if ep["n_criteria"] else None
        )

    overall = {
        "n_episodes": len(per_episode),
        "n_segments": n_segments,
        "n_segments_fully_satisfied": n_segments_full,
        "segment_accuracy": (n_segments_full / n_segments) if n_segments else None,
        "n_criteria": n_criteria,
        "n_satisfied": n_satisfied,
        "criterion_accuracy": (n_satisfied / n_criteria) if n_criteria else None,
        "n_format_invalid_segments": n_format_invalid,
        "mean_trial_score": (sum(trial_scores) / len(trial_scores)) if trial_scores else None,
    }
    return {"overall": overall, "per_episode": per_episode, "failures": failures}


def write_physical_judge_accuracy_report(
    results: Dict[str, Any],
    out_dir: str,
    data_name: str,
) -> Dict[str, Any]:
    """Write the accuracy summary JSON + a failures CSV; return the summary dict."""
    summary = summarize_physical_judge(results)
    if not summary:
        return {}

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{data_name}_physical_judge_accuracy.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    failures = summary.get("failures") or []
    csv_path = os.path.join(out_dir, f"{data_name}_physical_judge_failures.csv")
    fieldnames = [
        "task_id", "episode_id", "trial_id", "segment_id", "frame", "source_frame",
        "criterion_index", "criterion", "score", "format_valid", "evidence",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in failures:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary["_paths"] = {"summary_json": os.path.abspath(json_path), "failures_csv": os.path.abspath(csv_path)}
    return summary


def write_policy_eval_reports(
    results: Dict[str, Any],
    out_dir: str,
    data_name: str,
) -> Dict[str, str]:
    """Write paired-fidelity or WM-only policy-eval reports."""

    policy_eval = results.get("policy_eval")
    if not isinstance(policy_eval, dict):
        return {}
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{data_name}_policy_eval")
    paths = {
        "summary_json": os.path.abspath(f"{prefix}_summary.json"),
        "criteria_csv": os.path.abspath(f"{prefix}_criteria.csv"),
        "policy_task_csv": os.path.abspath(f"{prefix}_policy_task.csv"),
        "protocol_ablations_csv": os.path.abspath(f"{prefix}_protocol_ablations.csv"),
        "wm_seed_ablations_csv": os.path.abspath(f"{prefix}_wm_seed_ablations.csv"),
        "wm_rollouts_csv": os.path.abspath(f"{prefix}_wm_rollouts.csv"),
    }
    summary_payload = {
        "mode": policy_eval.get("mode", "paired_fidelity"),
        "summary": policy_eval.get("summary", {}),
        "n_rollouts": policy_eval.get("n_rollouts"),
        "rollout_root": policy_eval.get("rollout_root"),
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    criteria_rows = []
    for row in policy_eval.get("criterion_cells") or []:
        flat = dict(row)
        flat["real_observations"] = json.dumps(flat.get("real_observations", []))
        flat["wm_observations"] = json.dumps(flat.get("wm_observations", []))
        criteria_rows.append(flat)
    pd.DataFrame(criteria_rows).to_csv(paths["criteria_csv"], index=False)
    pd.DataFrame(policy_eval.get("policy_task") or []).to_csv(
        paths["policy_task_csv"], index=False
    )
    pd.DataFrame(policy_eval.get("protocol_ablations") or []).to_csv(
        paths["protocol_ablations_csv"], index=False
    )
    pd.DataFrame(policy_eval.get("wm_seed_ablations") or []).to_csv(
        paths["wm_seed_ablations_csv"], index=False
    )
    pd.DataFrame(policy_eval.get("wm_rollouts") or []).to_csv(
        paths["wm_rollouts_csv"], index=False
    )
    return paths
