#!/usr/bin/env python3
"""Remove redundant artifacts from ACWMBench ground-truth episodes.

The script is dry-run by default. Pass --apply to perform deletions.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/path/to/main-ground-truth")
VIEWS = ("hand_left", "hand_right", "top_head")


def episode_dirs(
    root: Path,
    selected_tasks: set[str],
    selected_episodes: set[str],
) -> Iterable[Path]:
    """Yield real task/episode directories without following symlinks."""
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or task_dir.is_symlink():
            continue
        if selected_tasks and task_dir.name not in selected_tasks:
            continue
        for episode_dir in sorted(task_dir.iterdir()):
            if (
                not episode_dir.is_dir()
                or episode_dir.is_symlink()
                or not episode_dir.name.startswith("ep")
            ):
                continue
            if selected_episodes and episode_dir.name not in selected_episodes:
                continue
            yield episode_dir


def redundant_targets(episode_dir: Path) -> Iterable[Path]:
    """Yield only the artifacts approved for deletion."""
    yield episode_dir / "raw_video"
    yield episode_dir / "prompt" / "masks_meta.json"
    yield episode_dir / "prompt" / "overlay.mp4"
    for view in VIEWS:
        view_dir = episode_dir / "multiview_masks" / view
        yield view_dir / "masks_meta.json"
        yield view_dir / "overlay.mp4"


def remove_target(path: Path, root: Path, apply: bool) -> bool:
    """Report or remove one target, returning whether it exists."""
    if not path.exists() and not path.is_symlink():
        return False

    # Symlinks are unlinked directly and are never followed.
    if not path.is_symlink() and not path.resolve().is_relative_to(root):
        raise ValueError(f"refusing to delete path outside dataset root: {path}")

    action = "remove" if apply else "would remove"
    print(f"[{action}] {path}")
    if not apply:
        return True

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise ValueError(f"unsupported target type: {path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune redundant raw videos and mask preview artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"GT dataset root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Process only this task; repeat to select multiple tasks.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Process only this episode ID; repeat to select multiple episodes.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    episodes = 0
    targets = 0
    for episode_dir in episode_dirs(root, set(args.task), set(args.episode)):
        episodes += 1
        for target in redundant_targets(episode_dir):
            targets += int(remove_target(target, root, args.apply))

    mode = "deleted" if args.apply else "found"
    print(f"{mode} {targets} target(s) across {episodes} episode(s)")
    if not args.apply:
        print("Dry run only. Re-run with --apply to perform these deletions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
