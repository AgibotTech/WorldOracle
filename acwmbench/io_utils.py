"""Lightweight I/O helpers shared across metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Common view subdirectories used in test datasets
VIEW_SUBDIRS = ("top_head", "hand_left", "hand_right", "wrist", "ego", "exocentric")
DEFAULT_VIEW = "top_head"


def _resolve_frame_dir(video_dir: Path, preferred_view: str | None = None) -> Path:
    """Return the directory that actually contains frame images.

    Handles two layouts:
      1. <video_dir>/frame_*.png          (direct)
      2. <video_dir>/<view>/frame_*.png   (e.g. top_head/)

    If `preferred_view` is given, it is tried first.
    """
    if not video_dir.is_dir():
        return video_dir

    # Case 1: images directly inside video_dir
    direct = [f for f in os.listdir(video_dir) if f.lower().endswith(IMAGE_EXTS)]
    if direct:
        return video_dir

    # Build candidate view list (preferred first if provided)
    candidates = []
    if preferred_view:
        candidates.append(preferred_view)
    candidates.extend(v for v in VIEW_SUBDIRS if v != preferred_view)

    for sub in candidates:
        subdir = video_dir / sub
        if subdir.is_dir():
            frames = [f for f in os.listdir(subdir) if f.lower().endswith(IMAGE_EXTS)]
            if frames:
                return subdir

    # Fallback: return original (will produce empty list later)
    return video_dir


def list_frame_files(
    video_dir: str | os.PathLike, preferred_view: str | None = None
) -> List[str]:
    """Return sorted absolute paths of frame images inside `video_dir`.

    Automatically handles the common layout where frames live in a view
    subdirectory (e.g. video/top_head/).

    If `preferred_view` is provided (e.g. "top_head"), it is preferred.
    """
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        return []

    frame_dir = _resolve_frame_dir(video_dir, preferred_view=preferred_view)

    frames = [
        str(frame_dir / f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    frames.sort()
    return frames


def uniform_indices(n_total: int, n_sample: int) -> List[int]:
    """Uniformly sample `n_sample` indices from [0, n_total)."""
    if n_total <= 0:
        return []
    if n_sample >= n_total:
        return list(range(n_total))
    if n_sample <= 1:
        return [n_total - 1]
    return [int(round(i * (n_total - 1) / (n_sample - 1))) for i in range(n_sample)]


def load_frames_as_pil(
    video_dir: str | os.PathLike,
    indices: Sequence[int] | None = None,
    max_long_side: int | None = None,
    preferred_view: str | None = None,
) -> List[Image.Image]:
    """Load selected frames from a video directory as PIL Images (RGB)."""
    files = list_frame_files(video_dir, preferred_view=preferred_view)
    if not files:
        raise FileNotFoundError(f"No frames found in {video_dir}")

    if indices is None:
        indices = range(len(files))
    indices = [i for i in indices if 0 <= i < len(files)]

    images: List[Image.Image] = []
    for i in indices:
        img = Image.open(files[i]).convert("RGB")
        if max_long_side and max(img.size) > max_long_side:
            scale = max_long_side / float(max(img.size))
            new_w = int(round(img.size[0] * scale))
            new_h = int(round(img.size[1] * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)
        images.append(img)
    return images


def load_frames_as_array(
    video_dir: str | os.PathLike,
    indices: Sequence[int] | None = None,
    preferred_view: str | None = None,
) -> np.ndarray:
    """Load frames as a (T, H, W, 3) uint8 array."""
    images = load_frames_as_pil(video_dir, indices, preferred_view=preferred_view)
    arr = np.stack([np.asarray(img) for img in images], axis=0)
    return arr


def save_json(data, path: str | os.PathLike, indent: int = 2) -> None:
    """Atomically write JSON so an interrupted run cannot leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: str | os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def iter_episode_dirs(data_base: str | os.PathLike) -> Iterable[dict]:
    """Yield {task_id, episode_id, episode_dir} for every episode under `data_base`.

    `data_base` is expected to be a `gt_dataset`-style root containing
    <task_id>/<episode_id>/... directories.
    """
    data_base = Path(data_base)
    for task_id in sorted(os.listdir(data_base)):
        task_dir = data_base / task_id
        if not task_dir.is_dir():
            continue
        for episode_id in sorted(os.listdir(task_dir)):
            episode_dir = task_dir / episode_id
            if not episode_dir.is_dir():
                continue
            yield {
                "task_id": task_id,
                "episode_id": episode_id,
                "episode_dir": str(episode_dir),
            }
