#!/usr/bin/env python3
"""Browse Ego GT episodes as cached top-head videos.

The source dataset stores JPEG frames, so the first request for an episode
creates an H.264 MP4 in the cache. Later playback reuses that clip.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import visualize_policy_eval_dataset as viewer


DEFAULT_ROOT = Path("/path/to/ego-ground-truth")
DEFAULT_CACHE = Path(
    os.environ.get(
        "GT_EGO_VIEWER_CACHE",
        "/path/to/viewer-cache/ego",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
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

    viewer.DATASET_ROOT = root
    viewer.CACHE_ROOT = cache
    viewer.VIEW = "top_head"
    viewer.FPS = args.fps
    viewer.DATASET_NAME = "gt_dataset_ego"
    viewer.CRF = args.crf
    viewer.PRESET = args.preset
    viewer.REQUESTED_TASKS = list(args.tasks) if args.tasks else None
    viewer.refresh_dataset("top_head")

    task_count = len({task for task, _ in viewer._EPISODES})  # pylint: disable=protected-access
    episode_count = len(viewer._EPISODES)  # pylint: disable=protected-access
    print(f"[gt-ego-viewer] root     = {root}", flush=True)
    print(f"[gt-ego-viewer] view     = top_head @ {args.fps:g} FPS", flush=True)
    print(
        f"[gt-ego-viewer] dataset  = {task_count} tasks, "
        f"{episode_count} episodes",
        flush=True,
    )
    print(f"[gt-ego-viewer] cache    = {cache}", flush=True)
    if not episode_count:
        raise SystemExit("No top_head episodes were found")
    if args.prewarm:
        viewer.prewarm_clips(max(1, args.prewarm_workers))
        print(
            f"[gt-ego-viewer] prewarming with "
            f"{max(1, args.prewarm_workers)} workers",
            flush=True,
        )

    print(
        f"[gt-ego-viewer] serving  = http://{args.host}:{args.port}",
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
