#!/usr/bin/env python3
"""Play Ego GT top-head videos and pause at prompts.json keyframes."""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import visualize_policy_eval_prompts as viewer


DEFAULT_ROOT = Path("/path/to/ego-ground-truth")
DEFAULT_CACHE = Path(
    os.environ.get(
        "GT_EGO_KEYFRAME_CACHE",
        "/path/to/viewer-cache/ego-keyframes",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task filter, for example hangtowel pulldrawer.",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preview-height", type=int, default=600)
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    if args.clear_cache and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    viewer.ROOT = root
    viewer.FPS = args.fps
    viewer.FRAME_VIEW = "top_head"
    viewer.TASK_FILTER = list(args.tasks) if args.tasks else None
    viewer.CLIP_CACHE_DIR = cache
    viewer.CLIP_CRF = args.crf
    viewer.CLIP_PRESET = args.preset
    viewer.CLIP_SCALE_H = max(0, args.preview_height)
    viewer._EPISODES_CACHE = None  # pylint: disable=protected-access
    viewer._CLIP_EXECUTOR = ThreadPoolExecutor(  # pylint: disable=protected-access
        max_workers=max(1, args.prewarm_workers)
    )

    episodes = viewer._get_episodes()  # pylint: disable=protected-access
    keyframes = sum(len(episode.segments) for episode in episodes)
    annotated = sum(
        bool(segment.success_criteria)
        for episode in episodes
        for segment in episode.segments
    )
    print(f"[gt-ego-keyframes] root       = {root}", flush=True)
    print("[gt-ego-keyframes] view       = top_head", flush=True)
    print(f"[gt-ego-keyframes] episodes   = {len(episodes)}", flush=True)
    print(
        f"[gt-ego-keyframes] keyframes  = {keyframes} "
        f"({annotated} with success criteria)",
        flush=True,
    )
    print(f"[gt-ego-keyframes] cache      = {cache}", flush=True)
    if not episodes:
        raise SystemExit("No episodes with prompts.json were found")
    if args.prewarm:
        viewer._prewarm(args.prewarm_workers)  # pylint: disable=protected-access

    print(
        f"[gt-ego-keyframes] serving    = http://{args.host}:{args.port}",
        flush=True,
    )
    viewer.app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
