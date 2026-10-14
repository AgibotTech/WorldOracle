#!/usr/bin/env python3
"""View GT episodes and pause playback at every annotated keyframe.

This is a GT-dataset entry point for ``visualize_policy_eval_prompts.py``.
It displays ``success_criteria`` while paused and continues to the next
keyframe when the user clicks Continue.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import visualize_policy_eval_prompts as viewer


DEFAULT_ROOT = Path("/path/to/yam-ground-truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task whitelist, for example hangtowel pickfruit.",
    )
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "GT_KEYFRAME_VIEWER_CACHE",
                "~/.cache/acwmbench_gt_keyframes",
            )
        ),
    )
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")

    viewer.ROOT = root
    viewer.FPS = args.fps
    viewer.FRAME_VIEW = args.view
    viewer.TASK_FILTER = list(args.tasks) if args.tasks else None
    viewer.CLIP_CACHE_DIR = cache_dir
    viewer.CLIP_CRF = args.crf
    viewer.CLIP_PRESET = args.preset
    viewer.CLIP_SCALE_H = max(0, args.preview_height)
    viewer._EPISODES_CACHE = None  # pylint: disable=protected-access

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    viewer._CLIP_EXECUTOR = ThreadPoolExecutor(  # pylint: disable=protected-access
        max_workers=max(1, args.prewarm_workers)
    )

    episodes = viewer._get_episodes()  # pylint: disable=protected-access
    annotated = sum(
        bool(segment.success_criteria)
        for episode in episodes
        for segment in episode.segments
    )
    keyframes = sum(len(episode.segments) for episode in episodes)
    print(f"[gt-keyframes] root       = {root}", flush=True)
    print(f"[gt-keyframes] view       = {args.view}", flush=True)
    print(f"[gt-keyframes] episodes   = {len(episodes)}", flush=True)
    print(
        f"[gt-keyframes] annotated  = {annotated}/{keyframes} keyframes",
        flush=True,
    )
    print(f"[gt-keyframes] cache      = {cache_dir}", flush=True)
    if not episodes:
        raise SystemExit("no episodes with prompts.json were found")
    if args.prewarm:
        viewer._prewarm(args.prewarm_workers)  # pylint: disable=protected-access

    print(
        f"[gt-keyframes] serving on http://{args.host}:{args.port}",
        flush=True,
    )
    viewer.app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
