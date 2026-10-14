#!/usr/bin/env python3
"""Create writable, cache-free evaluation trees backed by source symlinks.

Only immutable inputs (videos and rollout metadata/arrays) are linked. Metric
JSON files are deliberately not copied, so evaluation starts from zero and
writes results under the project-local data/eval_workspace directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STANDARD_DATASETS = {
    "cosmos3": "cosmos3_g02_dataset",
    "ctrlworld": "ctrlworld_dataset",
    "dreamdojo": "dreamdojo_dataset",
    "gesim_g01_23000": "gesim_g01_23000",
    "gesim_g02_14000": "gesim_g02_14000",
    "gesim_g02_dmd_14000": "gesim_g02_dmd_14000",
    "irasim": "irasim_dataset",
}


def ensure_link(source: Path, target: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"wrong existing symlink: {target} -> {target.resolve()}")
        return
    if target.exists():
        raise FileExistsError(f"refusing to replace existing path: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare_standard(source_root: Path, workspace_root: Path) -> int:
    count = 0
    for workspace_name, source_name in STANDARD_DATASETS.items():
        source = source_root / source_name
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = workspace_root / workspace_name
        for task_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            for episode_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
                video = episode_dir / "video"
                if not video.is_dir():
                    continue
                target_episode = target / task_dir.name / episode_dir.name
                target_episode.mkdir(parents=True, exist_ok=True)
                ensure_link(video, target_episode / "video")
                count += 1
    return count


def prepare_policy(source_root: Path, workspace_root: Path) -> int:
    count = 0
    for meta_path in sorted(source_root.rglob("meta.json")):
        rollout_dir = meta_path.parent
        relative = rollout_dir.relative_to(source_root)
        target = workspace_root / "policy_eval_rollouts" / relative
        target.mkdir(parents=True, exist_ok=True)
        for name in ("meta.json", "actions.npy", "states.npy", "video"):
            source = rollout_dir / name
            if source.exists():
                ensure_link(source, target / name)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/path/to/source-data"),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "eval_workspace",
    )
    args = parser.parse_args()
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    standard_count = prepare_standard(args.source_root, args.workspace_root)
    policy_count = prepare_policy(
        args.source_root / "policy_eval_rollouts",
        args.workspace_root,
    )
    print(f"standard episodes linked: {standard_count}")
    print(f"policy rollouts linked: {policy_count}")
    print(f"workspace: {args.workspace_root}")


if __name__ == "__main__":
    main()
