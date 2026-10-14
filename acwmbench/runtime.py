"""Hardware discovery and execution-profile resolution.

Hardware policy is kept separate from metric semantics. Settings that are
observably batch-shape-sensitive are still persisted in metric cache identity.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    """Runtime facts for the CUDA device used by this process."""

    available: bool
    device_count: int
    index: int | None
    name: str
    total_memory_gb: float


@dataclass(frozen=True)
class KeyAreaExecutionConfig:
    """Resolved, concrete KeyArea execution settings."""

    mode: str
    episode_workers: int
    max_in_flight_episodes: int
    prefetch_depth: int
    vjepa_batch_size: int
    lpips_batch_size: int
    pin_memory: bool
    gpu_profile: str
    memory_utilization_target: float
    gpu: GPUInfo

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InteractionExecutionConfig:
    """Resolved pipeline settings for Interaction Spatial Alignment."""

    mode: str
    frame_batch_size: int
    loader_workers: int
    prefetch_batches: int
    matcher_workers: int
    pin_memory: bool
    fuse_gt_generated: bool
    gpu_profile: str
    memory_utilization_target: float
    gpu: GPUInfo

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES: dict[str, dict[str, int]] = {
    # Conservative defaults: full episodes are currently resident in host RAM.
    "rtx_4090": {
        "episode_workers": 2,
        "max_in_flight_episodes": 2,
        "prefetch_depth": 1,
        "vjepa_batch_size": 32,
        "lpips_batch_size": 8,
    },
    "generic_16gb": {
        "episode_workers": 2,
        "max_in_flight_episodes": 2,
        "prefetch_depth": 1,
        "vjepa_batch_size": 16,
        "lpips_batch_size": 8,
    },
    "generic_8gb": {
        "episode_workers": 1,
        "max_in_flight_episodes": 1,
        "prefetch_depth": 1,
        "vjepa_batch_size": 8,
        "lpips_batch_size": 4,
    },
    "cpu": {
        "episode_workers": 1,
        "max_in_flight_episodes": 1,
        "prefetch_depth": 1,
        "vjepa_batch_size": 1,
        "lpips_batch_size": 1,
    },
}

_INTERACTION_PROFILES: dict[str, dict[str, int | bool]] = {
    "rtx_4090": {
        # Real-data regression: batch 4 is bit-identical to the historical
        # baseline; larger batches can flip near-tied DINO patch matches.
        "frame_batch_size": 4,
        "loader_workers": 4,
        "prefetch_batches": 2,
        "matcher_workers": 2,
        "fuse_gt_generated": False,
    },
    "generic_16gb": {
        "frame_batch_size": 4,
        "loader_workers": 2,
        "prefetch_batches": 2,
        "matcher_workers": 2,
        "fuse_gt_generated": False,
    },
    "generic_8gb": {
        "frame_batch_size": 2,
        "loader_workers": 2,
        "prefetch_batches": 1,
        "matcher_workers": 1,
        "fuse_gt_generated": False,
    },
    "cpu": {
        "frame_batch_size": 1,
        "loader_workers": 1,
        "prefetch_batches": 1,
        "matcher_workers": 1,
        "fuse_gt_generated": False,
    },
}


def detect_gpu(device: str | torch.device = "cuda") -> GPUInfo:
    """Inspect the selected CUDA device without allocating model tensors."""

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        return GPUInfo(False, 0, None, "cpu", 0.0)

    count = torch.cuda.device_count()
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= count:
        raise ValueError(f"CUDA device index {index} is outside available range 0..{count - 1}")
    props = torch.cuda.get_device_properties(index)
    return GPUInfo(
        available=True,
        device_count=count,
        index=index,
        name=str(props.name),
        total_memory_gb=float(props.total_memory) / (1024**3),
    )


def _auto_profile(configured: str, gpu: GPUInfo, auto_detect: bool) -> str:
    if not gpu.available:
        return "cpu"
    if not auto_detect:
        return configured if configured in _PROFILES else "rtx_4090"
    name = gpu.name.lower()
    if "4090" in name and gpu.total_memory_gb >= 20:
        return "rtx_4090"
    if gpu.total_memory_gb >= 14:
        return "generic_16gb"
    return "generic_8gb"


def _positive_int(value: Any, default: int, name: str) -> int:
    if value in (None, "auto"):
        return int(default)
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return result


def resolve_keyarea_execution(
    hardware_cfg: Mapping[str, Any] | None,
    execution_cfg: Mapping[str, Any] | None,
    *,
    device: str | torch.device,
) -> KeyAreaExecutionConfig:
    """Resolve ``auto`` values using actual hardware plus the configured profile."""

    hardware = dict(hardware_cfg or {})
    execution = dict(execution_cfg or {})
    gpu = detect_gpu(device)
    configured_profile = str(hardware.get("gpu_profile", "rtx_4090")).lower()
    auto_detect = bool(hardware.get("auto_detect", True))
    profile_name = _auto_profile(configured_profile, gpu, auto_detect)
    defaults = _PROFILES[profile_name]

    target = float(hardware.get("memory_utilization_target", 0.85))
    if not 0.0 < target <= 1.0:
        raise ValueError("hardware.memory_utilization_target must be in (0, 1]")

    mode = str(execution.get("mode", "pipeline")).lower()
    if mode not in {"serial", "pipeline"}:
        raise ValueError("keyarea execution.mode must be 'serial' or 'pipeline'")

    resolved = KeyAreaExecutionConfig(
        mode=mode,
        episode_workers=_positive_int(
            execution.get("episode_workers"), defaults["episode_workers"], "episode_workers"
        ),
        max_in_flight_episodes=_positive_int(
            execution.get("max_in_flight_episodes"),
            defaults["max_in_flight_episodes"],
            "max_in_flight_episodes",
        ),
        prefetch_depth=_positive_int(
            execution.get("prefetch_depth"), defaults["prefetch_depth"], "prefetch_depth"
        ),
        vjepa_batch_size=_positive_int(
            execution.get("vjepa_batch_size"),
            defaults["vjepa_batch_size"],
            "vjepa_batch_size",
        ),
        lpips_batch_size=_positive_int(
            execution.get("lpips_batch_size"),
            defaults["lpips_batch_size"],
            "lpips_batch_size",
        ),
        pin_memory=bool(execution.get("pin_memory", gpu.available)),
        gpu_profile=profile_name,
        memory_utilization_target=target,
        gpu=gpu,
    )
    if mode == "serial":
        resolved = KeyAreaExecutionConfig(
            **{
                **resolved.to_dict(),
                "episode_workers": 1,
                "max_in_flight_episodes": 1,
                "prefetch_depth": 1,
                "vjepa_batch_size": 1,
                "lpips_batch_size": 1,
                "gpu": gpu,
            }
        )
    logger.info(
        "KeyArea runtime: profile=%s gpu=%s %.1fGB count=%d mode=%s "
        "episode_workers=%d in_flight=%d prefetch=%d vjepa_batch=%d lpips_batch=%d",
        resolved.gpu_profile,
        gpu.name,
        gpu.total_memory_gb,
        gpu.device_count,
        resolved.mode,
        resolved.episode_workers,
        resolved.max_in_flight_episodes,
        resolved.prefetch_depth,
        resolved.vjepa_batch_size,
        resolved.lpips_batch_size,
    )
    return resolved


def _bool_or_auto(value: Any, default: bool) -> bool:
    if value in (None, "auto"):
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        raise ValueError(f"Expected boolean or 'auto', got {value!r}")
    return bool(value)


def resolve_interaction_execution(
    hardware_cfg: Mapping[str, Any] | None,
    execution_cfg: Mapping[str, Any] | None,
    *,
    device: str | torch.device,
) -> InteractionExecutionConfig:
    """Resolve the bounded DINO loader/GPU/matcher pipeline."""

    hardware = dict(hardware_cfg or {})
    execution = dict(execution_cfg or {})
    gpu = detect_gpu(device)
    configured_profile = str(
        hardware.get("gpu_profile", "rtx_4090")
    ).lower()
    profile_name = _auto_profile(
        configured_profile, gpu, bool(hardware.get("auto_detect", True))
    )
    defaults = _INTERACTION_PROFILES[profile_name]
    target = float(hardware.get("memory_utilization_target", 0.85))
    if not 0.0 < target <= 1.0:
        raise ValueError("hardware.memory_utilization_target must be in (0, 1]")
    mode = str(execution.get("mode", "pipeline")).lower()
    if mode not in {"serial", "pipeline"}:
        raise ValueError(
            "interaction execution.mode must be 'serial' or 'pipeline'"
        )

    resolved = InteractionExecutionConfig(
        mode=mode,
        frame_batch_size=_positive_int(
            execution.get("frame_batch_size"),
            int(defaults["frame_batch_size"]),
            "frame_batch_size",
        ),
        loader_workers=_positive_int(
            execution.get("loader_workers"),
            int(defaults["loader_workers"]),
            "loader_workers",
        ),
        prefetch_batches=_positive_int(
            execution.get("prefetch_batches"),
            int(defaults["prefetch_batches"]),
            "prefetch_batches",
        ),
        matcher_workers=_positive_int(
            execution.get("matcher_workers"),
            int(defaults["matcher_workers"]),
            "matcher_workers",
        ),
        pin_memory=bool(execution.get("pin_memory", gpu.available)),
        fuse_gt_generated=_bool_or_auto(
            execution.get("fuse_gt_generated"),
            bool(defaults["fuse_gt_generated"]),
        ),
        gpu_profile=profile_name,
        memory_utilization_target=target,
        gpu=gpu,
    )
    if mode == "serial":
        resolved = InteractionExecutionConfig(
            **{
                **resolved.to_dict(),
                "loader_workers": 1,
                "prefetch_batches": 1,
                "matcher_workers": 1,
                "pin_memory": False,
                "fuse_gt_generated": False,
                "gpu": gpu,
            }
        )
    logger.info(
        "Interaction runtime: profile=%s gpu=%s %.1fGB mode=%s batch=%d "
        "loaders=%d prefetch=%d matchers=%d pin=%s fuse=%s",
        profile_name,
        gpu.name,
        gpu.total_memory_gb,
        mode,
        resolved.frame_batch_size,
        resolved.loader_workers,
        resolved.prefetch_batches,
        resolved.matcher_workers,
        resolved.pin_memory,
        resolved.fuse_gt_generated,
    )
    return resolved

