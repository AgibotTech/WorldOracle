#!/usr/bin/env python3
"""Tune DINO batching/fusion for Interaction Spatial Alignment on real data."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acwmbench.dataset import build_trial_list  # noqa: E402
from acwmbench.metrics.interaction_spatial_alignment import (  # noqa: E402
    SUMMARY_METRICS,
    compute_interaction_spatial_alignment,
)
from evaluate import load_merged_config  # noqa: E402


@dataclass
class ResourceSample:
    rss_gb: float
    gpu_utilization_pct: float | None
    gpu_memory_gb: float | None


class ResourceMonitor:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _rss_gb() -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 1024**2 if sys.platform == "darwin" else 1024**3
            return value / divisor

    @staticmethod
    def _gpu_sample() -> tuple[float | None, float | None]:
        if not torch.cuda.is_available():
            return None, None
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(torch.cuda.current_device()),
                ],
                text=True,
                timeout=2,
            ).strip()
            utilization, memory_mb = output.split(",", 1)
            return float(utilization), float(memory_mb) / 1024.0
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, None

    def __enter__(self):
        def sample() -> None:
            while not self._stop.wait(self.interval):
                utilization, gpu_memory = self._gpu_sample()
                self.samples.append(
                    ResourceSample(
                        self._rss_gb(), utilization, gpu_memory
                    )
                )

        self._thread = threading.Thread(
            target=sample, name="isa-resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        utilization = [
            sample.gpu_utilization_pct
            for sample in self.samples
            if sample.gpu_utilization_pct is not None
        ]
        gpu_memory = [
            sample.gpu_memory_gb
            for sample in self.samples
            if sample.gpu_memory_gb is not None
        ]
        return {
            "peak_rss_gb": max(
                (sample.rss_gb for sample in self.samples), default=0.0
            ),
            "mean_gpu_utilization_pct": (
                sum(utilization) / len(utilization)
                if utilization
                else None
            ),
            "peak_gpu_utilization_pct": (
                max(utilization) if utilization else None
            ),
            "peak_nvidia_smi_memory_gb": (
                max(gpu_memory) if gpu_memory else None
            ),
        }


def _result_index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        episode["uid"]: episode
        for episode in result.get("per_episode", [])
    }


def _numeric_delta(left: Any, right: Any) -> float:
    a, b = float(left), float(right)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if not math.isfinite(a) or not math.isfinite(b):
        return 0.0 if a == b else float("inf")
    return abs(a - b)


def _max_metric_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float:
    left = _result_index(baseline)
    right = _result_index(candidate)
    if left.keys() != right.keys():
        return float("inf")
    delta = 0.0
    metrics = (*SUMMARY_METRICS, "relative_object_robot_displacement_px")
    for uid in left:
        if left[uid].get("selected_frames") != right[uid].get(
            "selected_frames"
        ):
            return float("inf")
        for metric in metrics:
            delta = max(
                delta,
                _numeric_delta(left[uid][metric], right[uid][metric]),
            )
        left_frames = left[uid].get("per_frame") or []
        right_frames = right[uid].get("per_frame") or []
        if len(left_frames) != len(right_frames):
            return float("inf")
        for expected, actual in zip(left_frames, right_frames):
            if expected["frame"] != actual["frame"]:
                return float("inf")
            for metric in metrics:
                delta = max(
                    delta,
                    _numeric_delta(expected[metric], actual[metric]),
                )
    return delta


def _run(
    trials,
    *,
    dim_cfg: dict[str, Any],
    gt_root: Path,
    device: str,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with ResourceMonitor() as monitor:
        result = compute_interaction_spatial_alignment(
            episodes=trials,
            gt_root=gt_root,
            device=device,
            eval_view=str(dim_cfg.get("eval_view", "top_head")),
            downsample_rate=int(dim_cfg.get("downsample_rate", 10)),
            fps=float(dim_cfg.get("fps", 30.0)),
            checkpoint=str(dim_cfg["checkpoint"]),
            dinov3_repo=dim_cfg.get("dinov3_repo"),
            model_name=str(dim_cfg.get("model_name", "dinov3_vitb16")),
            min_mask_patches=int(dim_cfg.get("min_mask_patches", 2)),
            min_mutual_matches=int(dim_cfg.get("min_mutual_matches", 3)),
            inlier_radius_px=float(dim_cfg.get("inlier_radius_px", 32.0)),
            confidence_bias=float(dim_cfg.get("confidence_bias", -2.0)),
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
                dim_cfg.get("match_support_target_ratio", 0.3)
            ),
            match_support_count_scale=float(
                dim_cfg.get("match_support_count_scale", 8.0)
            ),
            match_support_power=float(
                dim_cfg.get("match_support_power", 1.0)
            ),
            confidence_bad=float(dim_cfg.get("confidence_bad", 0.5)),
            confidence_good=float(dim_cfg.get("confidence_good", 0.8)),
            distance_good_px=float(
                dim_cfg.get("distance_good_px", 30.0)
            ),
            distance_bad_px=float(dim_cfg.get("distance_bad_px", 120.0)),
            alignment_confidence_power=float(
                dim_cfg.get("alignment_confidence_power", 3.0)
            ),
            save_root=None,
            progress=True,
            execution=execution,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    resources = monitor.summary()
    resources.update(
        {
            "elapsed_sec": elapsed,
            "episodes_per_hour": (
                len(result.get("per_episode", [])) * 3600.0 / elapsed
                if elapsed
                else 0.0
            ),
            "torch_peak_allocated_gb": (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "torch_peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    return result, resources


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--gt-root", default=None)
    parser.add_argument("--task", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--frame-batches", nargs="+", type=int, default=[4, 6, 8]
    )
    parser.add_argument(
        "--fusions", nargs="+", choices=("off", "on"), default=["off", "on"]
    )
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--matcher-workers", type=int, default=2)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument(
        "--output",
        default="output/interaction_spatial_alignment_benchmark.json",
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
    output = Path(args.output)
    payload: dict[str, Any] = {
        "root": str(root),
        "gt_root": str(gt_root),
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        ),
        "episodes": [trial.uid for trial in trials],
        "runs": [],
    }

    baseline_execution = {
        "mode": "serial",
        "frame_batch_size": 4,
    }
    baseline, resources = _run(
        trials,
        dim_cfg=dim_cfg,
        gt_root=gt_root,
        device=device,
        execution=baseline_execution,
    )
    payload["runs"].append(
        {
            "name": "serial_batch4",
            "execution": baseline_execution,
            **resources,
            "max_metric_delta": 0.0,
        }
    )
    _write_output(output, payload)

    for batch_size in args.frame_batches:
        for fusion in args.fusions:
            execution = {
                "mode": "pipeline",
                "frame_batch_size": batch_size,
                "loader_workers": args.loader_workers,
                "prefetch_batches": args.prefetch_batches,
                "matcher_workers": args.matcher_workers,
                "pin_memory": device == "cuda",
                "fuse_gt_generated": fusion == "on",
            }
            name = f"pipeline_batch{batch_size}_fuse{fusion}"
            try:
                candidate, resources = _run(
                    trials,
                    dim_cfg=dim_cfg,
                    gt_root=gt_root,
                    device=device,
                    execution=execution,
                )
                row = {
                    "name": name,
                    "execution": execution,
                    **resources,
                    "max_metric_delta": _max_metric_delta(
                        baseline, candidate
                    ),
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "name": name,
                    "execution": execution,
                    "status": "cuda_oom",
                    "error": str(exc),
                }
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "name": name,
                    "execution": execution,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            payload["runs"].append(row)
            _write_output(output, payload)
            gc.collect()
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
