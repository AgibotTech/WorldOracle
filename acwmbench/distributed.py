"""Minimal distributed helpers, single-process by default.

For the MVP we only need rank-aware printing; multi-rank support is preserved
so we can later parallelize VLM inference across GPUs.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def print0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def dist_init() -> None:
    """Initialize torch.distributed if `WORLD_SIZE > 1`, otherwise no-op."""
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def distribute_list_to_rank(data_list):
    return data_list[get_rank() :: get_world_size()]
