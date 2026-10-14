#!/usr/bin/env python3
"""Benchmark episode-level concurrency for the GPT Physical Status Judge.

This script never writes production ``physical_judge.json`` files. Each worker
owns an independent VLM backend/client and processes complete trials serially.
Paid API calls require the explicit ``--confirm-paid-api`` flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import statistics
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acwmbench.dataset import EpisodeTrial, build_trial_list  # noqa: E402
from acwmbench.metrics.physical_judge import judge_one_trial  # noqa: E402
from acwmbench.models.vlm import build_vlm_backend  # noqa: E402
from evaluate import load_merged_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_physical_judge_concurrency")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _merge_usage(
    summaries: list[dict[str, Any] | None],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for summary in summaries:
        if not summary:
            continue
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
            elif key not in merged:
                merged[key] = value
    return merged


def _trial_workload(trial: EpisodeTrial) -> tuple[int, int]:
    segments = trial.prompt_segments or []
    if not segments:
        return 1, 0
    criteria = 0
    for segment in segments:
        values = segment.get("success_criteria") or []
        if isinstance(values, str):
            values = [values] if values.strip() else []
        criteria += len(values)
    return len(segments), criteria


def _select_trials(
    trials: list[EpisodeTrial],
    *,
    tasks: list[str] | None,
    episodes: int,
) -> list[EpisodeTrial]:
    wanted = set(tasks or [])
    selected = [
        trial for trial in trials if not wanted or trial.task_id in wanted
    ]
    selected.sort(key=lambda trial: (
        trial.task_id,
        trial.episode_id,
        trial.trial_id,
    ))
    return selected[: max(1, episodes)]


def _limit_trial_steps(
    trials: list[EpisodeTrial],
    max_steps: int | None,
) -> list[EpisodeTrial]:
    """Select evenly spaced annotated segments, including the final segment."""

    if max_steps is None:
        return trials
    if max_steps < 1:
        raise SystemExit("--max-steps-per-trial must be >= 1")
    limited = []
    for trial in trials:
        segments = list(trial.prompt_segments or [])
        if len(segments) <= max_steps:
            limited.append(trial)
            continue
        if max_steps == 1:
            indices = [len(segments) - 1]
        else:
            indices = [
                round(index * (len(segments) - 1) / (max_steps - 1))
                for index in range(max_steps)
            ]
        limited.append(
            replace(
                trial,
                prompt_segments=[segments[index] for index in indices],
            )
        )
    return limited


def _judge_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_fallback_keysteps": int(
            config.get("n_fallback_keysteps", 1)
        ),
        "max_image_long_side": int(
            config.get("max_image_long_side", 768)
        ),
        "eval_view": str(config.get("eval_view", "top_head")),
        "format_retries": int(config.get("format_retries", 2)),
        "api_retries": int(config.get("api_retries", 5)),
        "api_retry_backoff": float(
            config.get("api_retry_backoff", 2.0)
        ),
        "use_reference_frame": bool(
            config.get("use_first_frame_reference", True)
        ),
    }


def _normalized_vlm_config(
    physical_config: dict[str, Any],
    *,
    model: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    vlm_config = dict(physical_config.get("vlm") or {})
    if not vlm_config:
        raise SystemExit("physical_judge.vlm config is missing")
    if model:
        vlm_config["served_model_name"] = model
    if base_url:
        vlm_config["base_url"] = base_url
    key_file = vlm_config.get("api_key_file")
    if key_file and not Path(key_file).is_absolute():
        vlm_config["api_key_file"] = str(
            (REPO_ROOT / key_file).resolve()
        )
    return vlm_config


def _run_concurrent(
    trials: list[EpisodeTrial],
    *,
    workers: int,
    vlm_config: dict[str, Any],
    judge_kwargs: dict[str, Any],
) -> dict[str, Any]:
    requested_workers = workers
    workers = min(workers, len(trials))
    work_queue: queue.Queue[tuple[int, EpisodeTrial]] = queue.Queue()
    for index, trial in enumerate(trials):
        work_queue.put((index, trial))
    records: list[dict[str, Any] | None] = [None] * len(trials)
    usage_by_worker: list[dict[str, Any] | None] = [None] * workers

    def worker(worker_index: int) -> None:
        backend = build_vlm_backend(dict(vlm_config))
        try:
            while True:
                try:
                    index, trial = work_queue.get_nowait()
                except queue.Empty:
                    break
                started = time.perf_counter()
                try:
                    result = judge_one_trial(
                        backend, trial, **judge_kwargs
                    )
                    record = {
                        "uid": trial.uid,
                        "task_id": trial.task_id,
                        "episode_id": trial.episode_id,
                        "trial_id": trial.trial_id,
                        "status": "complete",
                        "duration_sec": time.perf_counter() - started,
                        "result": result,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "worker=%d failed trial=%s", worker_index, trial.uid
                    )
                    record = {
                        "uid": trial.uid,
                        "task_id": trial.task_id,
                        "episode_id": trial.episode_id,
                        "trial_id": trial.trial_id,
                        "status": "error",
                        "duration_sec": time.perf_counter() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records[index] = record
                work_queue.task_done()
        finally:
            getter = getattr(backend, "get_usage_summary", None)
            usage_by_worker[worker_index] = (
                getter() if callable(getter) else None
            )
            backend.close()

    started = time.perf_counter()
    threads = [
        threading.Thread(
            target=worker,
            args=(index,),
            name=f"physical-judge-{index}",
        )
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    completed_records = [record for record in records if record is not None]
    durations = [
        float(record["duration_sec"]) for record in completed_records
    ]
    usage = _merge_usage(usage_by_worker)
    complete = sum(
        record.get("status") == "complete" for record in completed_records
    )
    return {
        "requested_workers": requested_workers,
        "active_workers": workers,
        "elapsed_sec": elapsed,
        "trials_per_hour": complete * 3600.0 / elapsed if elapsed else 0.0,
        "successful_trials": complete,
        "failed_trials": len(completed_records) - complete,
        "trial_latency_sec": {
            "mean": statistics.fmean(durations) if durations else 0.0,
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations, default=0.0),
        },
        "usage": usage,
        "records": completed_records,
    }


def _step_key(step: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        step.get("segment_id"),
        step.get("source_frame"),
        step.get("frame"),
    )


def _compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = {record["uid"]: record for record in baseline["records"]}
    right = {record["uid"]: record for record in candidate["records"]}
    common = sorted(set(left) & set(right))
    score_deltas: list[float] = []
    matching_aspects = 0
    compared_aspects = 0
    matching_steps = 0
    compared_steps = 0
    for uid in common:
        if (
            left[uid].get("status") != "complete"
            or right[uid].get("status") != "complete"
        ):
            continue
        left_result = left[uid]["result"]
        right_result = right[uid]["result"]
        score_deltas.append(
            abs(
                float(left_result.get("score", 0.0))
                - float(right_result.get("score", 0.0))
            )
        )
        left_steps = {
            _step_key(step): step
            for step in left_result.get("key_steps", [])
        }
        right_steps = {
            _step_key(step): step
            for step in right_result.get("key_steps", [])
        }
        for key in sorted(set(left_steps) & set(right_steps), key=str):
            expected = left_steps[key].get("aspect_scores") or []
            actual = right_steps[key].get("aspect_scores") or []
            if len(expected) != len(actual):
                continue
            compared_steps += 1
            if expected == actual:
                matching_steps += 1
            for left_value, right_value in zip(expected, actual):
                compared_aspects += 1
                if left_value == right_value:
                    matching_aspects += 1
    return {
        "common_trials": len(common),
        "score_mae": (
            statistics.fmean(score_deltas) if score_deltas else None
        ),
        "max_score_delta": max(score_deltas, default=None),
        "step_exact_agreement": (
            matching_steps / compared_steps if compared_steps else None
        ),
        "criterion_agreement": (
            matching_aspects / compared_aspects
            if compared_aspects
            else None
        ),
        "compared_steps": compared_steps,
        "compared_criteria": compared_aspects,
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--task", nargs="*", default=["pickfruit"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--max-steps-per-trial",
        type=int,
        default=None,
        help="Evenly sample at most N annotated segments per trial.",
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--output",
        default="output/physical_judge_concurrency_benchmark.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-paid-api",
        action="store_true",
        help="Required before any VLM request is sent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(workers < 1 for workers in args.workers):
        raise SystemExit("--workers values must be >= 1")
    config = load_merged_config(args.base_config, args.config_path)
    data_config = config.get("data", {})
    root = Path(args.root or data_config["val_base"]).resolve()
    gt_root = Path(args.gt_root or data_config["gt_path"]).resolve()
    physical_config = (config.get("dimensions") or {}).get(
        "physical_judge", {}
    )
    trials = _limit_trial_steps(
        _select_trials(
            build_trial_list(root, gt_root),
            tasks=args.task,
            episodes=args.episodes,
        ),
        args.max_steps_per_trial,
    )
    if not trials:
        raise SystemExit("No trials selected")
    segments, criteria = 0, 0
    per_trial_workload = {}
    for trial in trials:
        trial_segments, trial_criteria = _trial_workload(trial)
        segments += trial_segments
        criteria += trial_criteria
        per_trial_workload[trial.uid] = {
            "api_calls_without_retries": trial_segments,
            "criteria": trial_criteria,
        }
    vlm_config = _normalized_vlm_config(
        physical_config,
        model=args.model,
        base_url=args.base_url,
    )
    payload: dict[str, Any] = {
        "root": str(root),
        "gt_root": str(gt_root),
        "backend": vlm_config.get("backend"),
        "model": (
            vlm_config.get("served_model_name")
            or vlm_config.get("model_path")
        ),
        "trials": [trial.uid for trial in trials],
        "workload": {
            "trials": len(trials),
            "api_calls_per_run_without_retries": segments,
            "criteria_per_run": criteria,
            "requested_worker_runs": args.workers,
            "effective_workers_per_run": {
                str(workers): min(workers, len(trials))
                for workers in args.workers
            },
            "total_minimum_api_calls": segments * len(args.workers),
            "per_trial": per_trial_workload,
        },
        "runs": [],
    }
    output = Path(args.output)
    if args.dry_run:
        _write_output(output, payload)
        print(json.dumps(payload, indent=2))
        print("Dry run only; no API requests were sent.")
        return
    if not args.confirm_paid_api:
        _write_output(output, payload)
        raise SystemExit(
            "Refusing paid API calls without --confirm-paid-api. "
            "Run with --dry-run first to inspect request count."
        )

    baseline = None
    for workers in args.workers:
        logger.info(
            "Starting paid run: workers=%d trials=%d minimum_requests=%d",
            workers,
            len(trials),
            segments,
        )
        run = _run_concurrent(
            trials,
            workers=workers,
            vlm_config=vlm_config,
            judge_kwargs=_judge_kwargs(physical_config),
        )
        if baseline is None:
            baseline = run
            run["comparison_to_first_run"] = None
        else:
            run["comparison_to_first_run"] = _compare_runs(baseline, run)
            baseline_elapsed = float(baseline["elapsed_sec"])
            run["speedup_vs_first_run"] = (
                baseline_elapsed / float(run["elapsed_sec"])
                if run["elapsed_sec"]
                else None
            )
        payload["runs"].append(run)
        _write_output(output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
