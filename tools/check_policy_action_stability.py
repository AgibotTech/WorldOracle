#!/usr/bin/env python3
"""Measure repeated first-chunk PI0.5 outputs on policy_eval_dataset episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acwmbench.policy_eval import OpenPIWebsocketPolicyClient
from acwmbench.policy_rollout import (
    discover_policy_eval_episodes,
    load_initial_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data/policy_eval_dataset",
    )
    parser.add_argument(
        "--initial-states",
        type=Path,
        default=REPO_ROOT / "configs/states/initial_states.yaml",
    )
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--episode", action="append", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output/policy_action_stability.json",
    )
    parser.add_argument(
        "--actions-root",
        type=Path,
        default=REPO_ROOT / "output/policy_action_stability_actions",
    )
    return parser.parse_args()


def _episode_metrics(
    actions: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    reference = actions[0]
    deltas = actions[1:] - reference[None]
    if len(deltas):
        abs_delta = np.abs(deltas)
        max_abs = float(abs_delta.max())
        mean_abs = float(abs_delta.mean())
        rmse = float(np.sqrt(np.mean(np.square(deltas))))
        repeat_max = [
            float(np.max(np.abs(actions[index] - reference)))
            for index in range(1, len(actions))
        ]
    else:
        max_abs = mean_abs = rmse = 0.0
        repeat_max = []
    pairwise_max = 0.0
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            pairwise_max = max(
                pairwise_max,
                float(np.max(np.abs(actions[left] - actions[right]))),
            )
    return {
        "shape": list(actions.shape),
        "exactly_identical": bool(
            all(np.array_equal(reference, value) for value in actions[1:])
        ),
        "allclose": bool(
            all(
                np.allclose(reference, value, atol=atol, rtol=rtol)
                for value in actions[1:]
            )
        ),
        "max_abs_diff_from_first": max_abs,
        "mean_abs_diff_from_first": mean_abs,
        "rmse_from_first": rmse,
        "max_pairwise_abs_diff": pairwise_max,
        "per_repeat_max_abs_diff": repeat_max,
        "reference_abs_max": float(np.max(np.abs(reference))),
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    episodes = discover_policy_eval_episodes(
        args.dataset_root,
        task_filter=args.task,
        episode_filter=args.episode,
        limit_episodes=args.limit,
        limit_episodes_per_task=args.limit_per_task,
    )
    client = OpenPIWebsocketPolicyClient(args.server_url)
    client.reset(seed=args.seed)
    rows: list[dict[str, Any]] = []
    global_max = 0.0
    stable_count = 0

    for index, episode in enumerate(episodes, start=1):
        observation = load_initial_observation(
            episode, args.initial_states
        )
        chunks = []
        timings = []
        print(
            f"[{index}/{len(episodes)}] {episode.task_id}/"
            f"{episode.episode_id}: {args.repeats} first-chunk calls",
            flush=True,
        )
        for repeat in range(args.repeats):
            start = time.monotonic()
            chunks.append(client.infer(observation))
            timings.append(time.monotonic() - start)
            print(
                f"  repeat {repeat}: {timings[-1]:.2f}s",
                flush=True,
            )
        stacked = np.stack(chunks).astype(np.float32, copy=False)
        metrics = _episode_metrics(
            stacked, atol=float(args.atol), rtol=float(args.rtol)
        )
        stable_count += int(metrics["allclose"])
        global_max = max(global_max, metrics["max_pairwise_abs_diff"])
        action_path = (
            args.actions_root
            / episode.task_id
            / episode.episode_id
            / "first_action_chunks.npy"
        )
        action_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(action_path, stacked, allow_pickle=False)
        row = {
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "timing_sec": timings,
            "actions_path": str(action_path.resolve()),
            **metrics,
        }
        rows.append(row)
        print(
            "  "
            f"allclose={metrics['allclose']} "
            f"exact={metrics['exactly_identical']} "
            f"max_pairwise={metrics['max_pairwise_abs_diff']:.8g}",
            flush=True,
        )

    report = {
        "server_url": args.server_url,
        "dataset_root": str(args.dataset_root.resolve()),
        "repeats": int(args.repeats),
        "policy_seed": int(args.seed),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "summary": {
            "n_episodes": len(rows),
            "n_allclose": stable_count,
            "allclose_rate": stable_count / len(rows),
            "global_max_pairwise_abs_diff": global_max,
        },
        "seed_note": (
            "The server-side policy RNG is reset to the requested fixed seed "
            "before every inference, so all calls use the same initial "
            "diffusion noise."
        ),
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
