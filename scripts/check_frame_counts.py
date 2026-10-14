#!/usr/bin/env python3
"""Check whether slice lengths match the prompt frame counts.

Example:
    python scripts/check_frame_counts.py

By default this compares:
  /path/to/main-ground-truth
  /path/to/model-predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("pickfruit", "pourwater", "hangtowel", "pulldrawer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify end_frame - start_frame == n_frames for each episode."
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=Path(
            "/path/to/main-ground-truth"
        ),
        help="Root containing <task>/<episode>/prompt/prompts.json.",
    )
    parser.add_argument(
        "--slice-root",
        type=Path,
        default=Path("/path/to/model-predictions"),
        help="Root containing <task>/<episode>/slice.json.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=100,
        help="Expected number of episodes across all tasks (default: 100).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("JSON root is not an object")
    return data


def episode_names(task_prompt_root: Path, task_slice_root: Path) -> list[str]:
    """Return the union so a missing file is reported as an incorrect episode."""
    names = {
        path.parent.parent.name
        for path in task_prompt_root.glob("ep*/prompt/prompts.json")
    }
    names.update(path.parent.name for path in task_slice_root.glob("ep*/slice.json"))
    return sorted(names)


def main() -> int:
    args = parse_args()
    total = 0
    correct = 0
    errors: list[str] = []

    for task in TASKS:
        task_total = 0
        task_correct = 0
        prompt_task_root = args.prompt_root / task
        slice_task_root = args.slice_root / task

        for episode in episode_names(prompt_task_root, slice_task_root):
            total += 1
            task_total += 1
            prompt_path = prompt_task_root / episode / "prompt" / "prompts.json"
            slice_path = slice_task_root / episode / "slice.json"

            if not prompt_path.is_file() or not slice_path.is_file():
                missing = []
                if not prompt_path.is_file():
                    missing.append(str(prompt_path))
                if not slice_path.is_file():
                    missing.append(str(slice_path))
                errors.append(f"{task}/{episode}: missing {', '.join(missing)}")
                continue

            try:
                prompts = load_json(prompt_path)
                frame_slice = load_json(slice_path)
                n_frames = int(prompts["n_frames"])
                start_frame = int(frame_slice["start_frame"])
                end_frame = int(frame_slice["end_frame"])
                slice_length = end_frame - start_frame
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{task}/{episode}: {type(exc).__name__}: {exc}")
                continue

            if slice_length == n_frames:
                correct += 1
                task_correct += 1
            else:
                errors.append(
                    f"{task}/{episode}: end_frame({end_frame}) - "
                    f"start_frame({start_frame}) = {slice_length}, "
                    f"n_frames = {n_frames}"
                )

        task_accuracy = 100.0 * task_correct / task_total if task_total else 0.0
        print(
            f"{task}: {task_correct}/{task_total} correct "
            f"({task_accuracy:.2f}%)"
        )

    print()
    if total != args.expected_total:
        print(f"WARNING: found {total} episodes; expected {args.expected_total}.")

    accuracy = 100.0 * correct / total if total else 0.0
    print(f"Overall accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Incorrect episodes: {len(errors)}")
    for error in errors:
        print(f"  - {error}")

    return 0 if not errors and total == args.expected_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
