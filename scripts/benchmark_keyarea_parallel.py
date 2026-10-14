#!/usr/bin/env python3
"""Benchmark serial vs pipelined KeyArea on a few real episodes.

The benchmark never writes per-episode metric files. It reports throughput,
peak host/GPU memory, sampled GPU utilization, and numerical deltas against the
serial result.
"""

from __future__ import annotations

import argparse
import json
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
from acwmbench.metrics.keyarea_visual_quality import (  # noqa: E402
    RESULT_METRICS,
    compute_keyarea_visual_quality,
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
            # Linux reports KiB; macOS reports bytes.
            return value / (1024**2 if sys.platform == "darwin" else 1024**3)

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
        def run() -> None:
            while not self._stop.wait(self.interval):
                util, gpu_mem = self._gpu_sample()
                self.samples.append(ResourceSample(self._rss_gb(), util, gpu_mem))

        self._thread = threading.Thread(
            target=run, name="keyarea-resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        util = [
            sample.gpu_utilization_pct
            for sample in self.samples
            if sample.gpu_utilization_pct is not None
        ]
        gpu_mem = [
            sample.gpu_memory_gb
            for sample in self.samples
            if sample.gpu_memory_gb is not None
        ]
        return {
            "peak_rss_gb": max((s.rss_gb for s in self.samples), default=0.0),
            "mean_gpu_utilization_pct": sum(util) / len(util) if util else None,
            "peak_gpu_utilization_pct": max(util) if util else None,
            "peak_nvidia_smi_memory_gb": max(gpu_mem) if gpu_mem else None,
        }


def _result_index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["uid"]: item for item in result.get("per_episode", [])}


def _max_metric_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float:
    left = _result_index(baseline)
    right = _result_index(candidate)
    if left.keys() != right.keys():
        return float("inf")
    delta = 0.0
    for uid in left:
        for metric in RESULT_METRICS:
            delta = max(delta, abs(float(left[uid][metric]) - float(right[uid][metric])))
        left_frames = left[uid].get("per_frame") or []
        right_frames = right[uid].get("per_frame") or []
        if len(left_frames) != len(right_frames):
            return float("inf")
        for a, b in zip(left_frames, right_frames):
            if a["frame"] != b["frame"]:
                return float("inf")
            for metric in RESULT_METRICS:
                delta = max(delta, abs(float(a[metric]) - float(b[metric])))
    return delta


def _run(
    trials,
    *,
    dim_cfg: dict[str, Any],
    gt_root: Path,
    device: str,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    vjepa = dim_cfg.get("vjepa", {}) or {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with ResourceMonitor() as monitor:
        result = compute_keyarea_visual_quality(
            episodes=trials,
            gt_root=gt_root,
            device=device,
            eval_view=str(dim_cfg.get("eval_view", "top_head")),
            downsample_rate=int(dim_cfg.get("downsample_rate", 1)),
            fps=float(dim_cfg["fps"]) if dim_cfg.get("fps") else None,
            crop_to_bbox=bool(dim_cfg.get("crop_to_bbox", True)),
            bbox_pad=int(dim_cfg.get("bbox_pad", 10)),
            lpips_min_crop_size=int(dim_cfg.get("lpips_min_crop_size", 64)),
            mask_dilation_pixels=int(dim_cfg.get("mask_dilation_pixels", 0)),
            mask_aware_psnr=bool(dim_cfg.get("mask_aware_psnr", True)),
            vjepa_model_path=vjepa.get("model_path"),
            vjepa_model_name=str(
                vjepa.get("model_name", "vjepa2_1_vit_base_384")
            ),
            vjepa_repo=vjepa.get("repo"),
            secondary_mask_view=str(
                dim_cfg.get("secondary_mask_view", "top_head")
            ),
            visualize=False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--root", default=None, help="Override data.val_base.")
    parser.add_argument("--gt-root", default=None, help="Override data.gt_path.")
    parser.add_argument("--task", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--vjepa-batches", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--in-flight", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--episode-workers", type=int, default=2)
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    parser.add_argument("--output", default="output/keyarea_parallel_benchmark.json")
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
    dim_cfg = (config.get("dimensions") or {})["keyarea_visual_quality"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline, baseline_resources = _run(
        trials,
        dim_cfg=dim_cfg,
        gt_root=gt_root,
        device=device,
        execution={"mode": "serial"},
    )
    rows: list[dict[str, Any]] = [
        {
            "name": "serial",
            "execution": {"mode": "serial"},
            **baseline_resources,
            "max_metric_delta": 0.0,
        }
    ]
    for batch_size in args.vjepa_batches:
        for in_flight in args.in_flight:
            execution = {
                "mode": "pipeline",
                "episode_workers": args.episode_workers,
                "max_in_flight_episodes": in_flight,
                "prefetch_depth": 1,
                "vjepa_batch_size": batch_size,
                "lpips_batch_size": args.lpips_batch_size,
                "pin_memory": device == "cuda",
            }
            candidate, resources = _run(
                trials,
                dim_cfg=dim_cfg,
                gt_root=gt_root,
                device=device,
                execution=execution,
            )
            rows.append(
                {
                    "name": f"pipeline_vjepa{batch_size}_inflight{in_flight}",
                    "execution": execution,
                    **resources,
                    "max_metric_delta": _max_metric_delta(
                        baseline, candidate
                    ),
                }
            )
    payload = {
        "root": str(root),
        "gt_root": str(gt_root),
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        ),
        "episodes": [trial.uid for trial in trials],
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
