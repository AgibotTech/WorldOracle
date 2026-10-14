#!/usr/bin/env python3
"""Profile ISA CPU loading, DINO GPU inference, and CPU matching stages."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import acwmbench.metrics.interaction_spatial_alignment as isa  # noqa: E402
from acwmbench.dataset import build_trial_list  # noqa: E402
from benchmark_interaction_spatial_alignment import _run  # noqa: E402
from evaluate import load_merged_config  # noqa: E402


class StageTimings:
    """Thread-safe accumulated wall-work timings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._seconds[stage] += float(seconds)
            self._calls[stage] += 1

    def wrap(self, stage: str, function: Callable):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.record(stage, time.perf_counter() - started)

        return timed

    def report(
        self,
        *,
        elapsed: float,
        loader_workers: int,
        matcher_workers: int,
    ) -> dict[str, Any]:
        stages = {}
        for name in sorted(self._seconds):
            seconds = self._seconds[name]
            calls = self._calls[name]
            stages[name] = {
                "work_sec": seconds,
                "calls": calls,
                "mean_ms": seconds * 1000.0 / max(calls, 1),
                "work_pct_of_wall": (
                    seconds * 100.0 / elapsed if elapsed else 0.0
                ),
            }

        load_work = self._seconds["image_read_resize"] + self._seconds[
            "cpu_normalize"
        ]
        estimates = {
            "model_load": self._seconds["model_load"],
            "cpu_load_normalize": load_work / max(loader_workers, 1),
            # GPU extraction has one owner, so accumulated wall time is already
            # a critical-path estimate rather than parallel worker work.
            "gpu_h2d_dino_d2h": self._seconds["gpu_h2d_dino_d2h"],
            "cpu_matching": self._seconds["cpu_matching"]
            / max(matcher_workers, 1),
        }
        bottleneck = max(estimates, key=estimates.get)
        return {
            "stages": stages,
            "critical_path_estimates_sec": estimates,
            "estimated_bottleneck": bottleneck,
            "gpu_stage_pct_of_wall": (
                100.0
                * self._seconds["gpu_h2d_dino_d2h"]
                / elapsed
                if elapsed
                else 0.0
            ),
        }


def _profile_run(
    trials,
    *,
    dim_cfg: dict[str, Any],
    gt_root: Path,
    device: str,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    timings = StageTimings()
    original_init = isa.DinoExtractor.__init__
    original_read = isa._read_rgb
    original_prepare = isa.DinoExtractor.prepare_batch
    original_extract = isa._extract_paired_features
    original_match = isa._match_frame

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                isa.DinoExtractor,
                "__init__",
                timings.wrap("model_load", original_init),
            )
        )
        stack.enter_context(
            patch.object(
                isa,
                "_read_rgb",
                timings.wrap("image_read_resize", original_read),
            )
        )
        stack.enter_context(
            patch.object(
                isa.DinoExtractor,
                "prepare_batch",
                timings.wrap("cpu_normalize", original_prepare),
            )
        )
        stack.enter_context(
            patch.object(
                isa,
                "_extract_paired_features",
                timings.wrap("gpu_h2d_dino_d2h", original_extract),
            )
        )
        stack.enter_context(
            patch.object(
                isa,
                "_match_frame",
                timings.wrap("cpu_matching", original_match),
            )
        )
        result, resources = _run(
            trials,
            dim_cfg=dim_cfg,
            gt_root=gt_root,
            device=device,
            execution=execution,
        )

    resources["stage_profile"] = timings.report(
        elapsed=float(resources["elapsed_sec"]),
        loader_workers=int(execution.get("loader_workers", 1)),
        matcher_workers=int(execution.get("matcher_workers", 1)),
    )
    return result, resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--task", nargs="*", default=["pickfruit"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("serial", "pipeline"),
        default=["serial", "pipeline"],
    )
    parser.add_argument(
        "--output",
        default="output/interaction_spatial_alignment_profile.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.base_config, args.config_path)
    data_cfg = config.get("data", {})
    root = Path(args.root or data_cfg["val_base"]).resolve()
    gt_root = Path(args.gt_root or data_cfg["gt_path"]).resolve()
    trials = build_trial_list(root, gt_root)
    if args.task:
        wanted = set(args.task)
        trials = [trial for trial in trials if trial.task_id in wanted]
    trials = trials[: max(1, args.episodes)]
    if not trials:
        raise SystemExit("No episodes selected")
    dim_cfg = (config.get("dimensions") or {})[
        "interaction_spatial_alignment"
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    executions = {
        "serial": {
            "mode": "serial",
            "frame_batch_size": 4,
        },
        "pipeline": {
            "mode": "pipeline",
            "frame_batch_size": 4,
            "loader_workers": 4,
            "prefetch_batches": 2,
            "matcher_workers": 2,
            "pin_memory": device == "cuda",
            "fuse_gt_generated": False,
        },
    }
    payload = {
        "root": str(root),
        "gt_root": str(gt_root),
        "device": device,
        "episodes": [trial.uid for trial in trials],
        "runs": [],
    }
    output = Path(args.output)
    for mode in args.modes:
        execution = executions[mode]
        _, resources = _profile_run(
            trials,
            dim_cfg=dim_cfg,
            gt_root=gt_root,
            device=device,
            execution=execution,
        )
        payload["runs"].append(
            {
                "name": mode,
                "execution": execution,
                **resources,
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
