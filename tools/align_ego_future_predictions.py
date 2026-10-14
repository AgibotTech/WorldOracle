#!/usr/bin/env python3
"""Prepend Ego's initial observation to future-only prediction sequences.

Ego rollouts contain N-1 generated future frames for an N-frame episode:
generated frame i corresponds to GT frame i+1.  ACWMBench expects a complete
0..N-1 trajectory.  This tool creates that trajectory in place without
regenerating or copying the generated frames:

* aligned frame 0 is GT frame 0, resized to the model output resolution;
* aligned frame i+1 is a hard link to generated frame i.

Each new ``top_head`` directory is staged and validated before an atomic
directory swap.  ``rollout.json`` records the alignment, making the operation
restartable and idempotent.

Example:
    sudo /path/to/python tools/align_ego_future_predictions.py \
      --prediction cosmos=/path/to/cosmos_pinhole \
      --prediction jointcontrol=/path/to/jointcontrolvideo \
      --prediction oscar=/path/to/oscar_pinhole \
      --apply
"""

# pylint: disable=broad-exception-caught,import-error

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2


DEFAULT_GT_ROOT = Path("/path/to/ego-ground-truth")
ALIGNMENT_TYPE = "initial_observation_plus_future_predictions"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def atomic_json(
    path: Path,
    value: dict[str, Any],
    *,
    owner: tuple[int, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    if owner is not None:
        os.chown(temporary, owner[0], owner[1])
    os.replace(temporary, path)


def frame_paths(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"{directory}: no frame_*.jpg")
    return frames


def frame_indices(frames: list[Path]) -> list[int]:
    return [int(path.stem.rsplit("_", 1)[1]) for path in frames]


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image.shape[1], image.shape[0]


def expected_frames(gt_episode: Path) -> int:
    prompt = read_json(gt_episode / "prompt" / "prompts.json")
    return int(prompt["n_frames"])


def is_aligned(prediction_episode: Path) -> bool:
    rollout_path = prediction_episode / "rollout.json"
    if not rollout_path.is_file():
        return False
    rollout = read_json(rollout_path)
    alignment = rollout.get("frame_alignment")
    return (
        isinstance(alignment, dict)
        and alignment.get("type") == ALIGNMENT_TYPE
    )


def validate_episode(
    prediction_episode: Path,
    gt_episode: Path,
    *,
    expect_aligned: bool,
) -> dict[str, Any]:
    video_dir = prediction_episode / "video" / "top_head"
    frames = frame_paths(video_dir)
    indices = frame_indices(frames)
    gt_count = expected_frames(gt_episode)
    expected_count = gt_count if expect_aligned else gt_count - 1
    if len(frames) != expected_count:
        raise ValueError(
            f"{prediction_episode}: {len(frames)} frames, "
            f"expected {expected_count}"
        )
    if indices != list(range(expected_count)):
        raise ValueError(f"{video_dir}: frame indices are not contiguous")
    first_size = image_size(frames[0])
    last_size = image_size(frames[-1])
    if first_size != last_size:
        raise ValueError(
            f"{video_dir}: first/last dimensions differ "
            f"({first_size} vs {last_size})"
        )
    rollout_path = prediction_episode / "rollout.json"
    rollout = read_json(rollout_path) if rollout_path.is_file() else {}
    if expect_aligned:
        if not is_aligned(prediction_episode):
            raise ValueError(f"{rollout_path}: alignment metadata missing")
        if int(rollout.get("frames", -1)) != gt_count:
            raise ValueError(f"{rollout_path}: frame count was not updated")
    else:
        declared_episode_frames = rollout.get("episode_num_frames")
        if declared_episode_frames is not None:
            if int(declared_episode_frames) != gt_count:
                raise ValueError(
                    f"{rollout_path}: episode_num_frames="
                    f"{declared_episode_frames}, GT={gt_count}"
                )
        declared_frames = rollout.get("frames")
        if declared_frames is not None and int(declared_frames) != gt_count - 1:
            raise ValueError(
                f"{rollout_path}: frames={declared_frames}, GT={gt_count}"
            )
    return {
        "frames": len(frames),
        "width": first_size[0],
        "height": first_size[1],
    }


def stage_aligned_directory(
    prediction_episode: Path,
    gt_episode: Path,
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    source_dir = prediction_episode / "video" / "top_head"
    source_stat = source_dir.stat()
    owner = (source_stat.st_uid, source_stat.st_gid)
    source_frames = frame_paths(source_dir)
    target_width, target_height = image_size(source_frames[0])
    stage = source_dir.parent / (
        f".top_head.aligned_{os.getpid()}_{time.time_ns()}"
    )
    stage.mkdir()
    os.chown(stage, owner[0], owner[1])
    try:
        initial_path = gt_episode / "video" / "top_head" / "frame_00000.jpg"
        initial = cv2.imread(str(initial_path), cv2.IMREAD_COLOR)
        if initial is None:
            raise RuntimeError(f"cannot decode {initial_path}")
        if (initial.shape[1], initial.shape[0]) != (
            target_width,
            target_height,
        ):
            initial = cv2.resize(
                initial,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        staged_initial = stage / "frame_00000.jpg"
        if not cv2.imwrite(
            str(staged_initial),
            initial,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"failed to write {staged_initial}")
        os.chown(staged_initial, owner[0], owner[1])

        for source in source_frames:
            source_index = int(source.stem.rsplit("_", 1)[1])
            destination = stage / f"frame_{source_index + 1:05d}.jpg"
            os.link(source, destination)

        staged_frames = frame_paths(stage)
        gt_count = expected_frames(gt_episode)
        if len(staged_frames) != gt_count:
            raise RuntimeError(
                f"{stage}: staged {len(staged_frames)}, expected {gt_count}"
            )
        if frame_indices(staged_frames) != list(range(gt_count)):
            raise RuntimeError(f"{stage}: staged indices are not contiguous")
        if image_size(staged_frames[0]) != (target_width, target_height):
            raise RuntimeError(f"{stage}: initial frame has wrong dimensions")
        if image_size(staged_frames[-1]) != (target_width, target_height):
            raise RuntimeError(f"{stage}: final frame has wrong dimensions")
        info = {
            "frames": gt_count,
            "width": target_width,
            "height": target_height,
            "source_initial_frame": str(initial_path),
        }
        return stage, owner, info
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def install_aligned_directory(
    prediction_episode: Path,
    stage: Path,
    owner: tuple[int, int],
    info: dict[str, Any],
) -> None:
    source_dir = prediction_episode / "video" / "top_head"
    backup = source_dir.parent / (
        f".top_head.future_only_{os.getpid()}_{time.time_ns()}"
    )
    rollout_path = prediction_episode / "rollout.json"
    original_rollout = read_json(rollout_path) if rollout_path.is_file() else {}
    installed = False
    try:
        os.replace(source_dir, backup)
        os.replace(stage, source_dir)
        installed = True
        rollout = dict(original_rollout)
        rollout["frames"] = int(info["frames"])
        rollout["episode_num_frames"] = int(info["frames"])
        rollout["frame_alignment"] = {
            "type": ALIGNMENT_TYPE,
            "initial_frame_index": 0,
            "initial_frame_source": info["source_initial_frame"],
            "generated_source_index_offset": 1,
            "generated_frame_mapping": (
                "aligned_frame_i_plus_1 = original_generated_frame_i"
            ),
            "initial_frame_resize": {
                "width": int(info["width"]),
                "height": int(info["height"]),
                "interpolation": "lanczos4",
            },
            "note": (
                "The generated rollout was not regenerated. Frame 0 is the "
                "conditioning observation; original future frames were shifted "
                "forward by one index using hard links."
            ),
        }
        atomic_json(rollout_path, rollout, owner=owner)
    except BaseException:
        if installed and source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
        if backup.exists() and not source_dir.exists():
            os.replace(backup, source_dir)
        if original_rollout:
            atomic_json(rollout_path, original_rollout, owner=owner)
        raise
    shutil.rmtree(backup)


def stage_aligned_output_episode(
    source_episode: Path,
    destination_episode: Path,
    gt_episode: Path,
) -> Path:
    source_dir = source_episode / "video" / "top_head"
    source_frames = frame_paths(source_dir)
    target_width, target_height = image_size(source_frames[0])
    destination_episode.parent.mkdir(parents=True, exist_ok=True)
    stage = destination_episode.parent / (
        f".{destination_episode.name}.aligned_{os.getpid()}_{time.time_ns()}"
    )
    try:
        output_dir = stage / "video" / "top_head"
        output_dir.mkdir(parents=True)
        initial_path = gt_episode / "video" / "top_head" / "frame_00000.jpg"
        initial = cv2.imread(str(initial_path), cv2.IMREAD_COLOR)
        if initial is None:
            raise RuntimeError(f"cannot decode {initial_path}")
        if (initial.shape[1], initial.shape[0]) != (
            target_width,
            target_height,
        ):
            initial = cv2.resize(
                initial,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
        if not cv2.imwrite(
            str(output_dir / "frame_00000.jpg"),
            initial,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"failed to write aligned initial frame")
        for source in source_frames:
            source_index = int(source.stem.rsplit("_", 1)[1])
            destination = output_dir / f"frame_{source_index + 1:05d}.jpg"
            destination.symlink_to(source.resolve())

        source_rollout = source_episode / "rollout.json"
        rollout = read_json(source_rollout) if source_rollout.is_file() else {}
        count = expected_frames(gt_episode)
        rollout["frames"] = count
        rollout["episode_num_frames"] = count
        rollout["frame_alignment"] = {
            "type": ALIGNMENT_TYPE,
            "initial_frame_index": 0,
            "initial_frame_source": str(initial_path),
            "generated_source_index_offset": 1,
            "generated_frame_mapping": (
                "aligned_frame_i_plus_1 = original_generated_frame_i"
            ),
            "source_prediction_episode": str(source_episode),
            "initial_frame_resize": {
                "width": target_width,
                "height": target_height,
                "interpolation": "lanczos4",
            },
            "note": (
                "Compatibility view: frame 0 is the conditioning observation; "
                "future frames are symlinks to the immutable source rollout."
            ),
        }
        atomic_json(stage / "rollout.json", rollout)
        validate_episode(stage, gt_episode, expect_aligned=True)
        return stage
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def align_output_episode(
    source_episode: Path,
    destination_episode: Path,
    gt_episode: Path,
    *,
    apply: bool,
) -> str:
    if destination_episode.exists():
        validate_episode(
            destination_episode, gt_episode, expect_aligned=True
        )
        return "already_aligned"
    validate_episode(source_episode, gt_episode, expect_aligned=False)
    if not apply:
        return "pending"
    stage = stage_aligned_output_episode(
        source_episode, destination_episode, gt_episode
    )
    os.replace(stage, destination_episode)
    validate_episode(destination_episode, gt_episode, expect_aligned=True)
    return "aligned"


def align_episode(
    prediction_episode: Path,
    gt_episode: Path,
    *,
    apply: bool,
) -> str:
    if is_aligned(prediction_episode):
        validate_episode(
            prediction_episode, gt_episode, expect_aligned=True
        )
        return "already_aligned"
    validate_episode(
        prediction_episode, gt_episode, expect_aligned=False
    )
    if not apply:
        return "pending"
    stage, owner, info = stage_aligned_directory(
        prediction_episode, gt_episode
    )
    install_aligned_directory(
        prediction_episode, stage, owner, info
    )
    validate_episode(
        prediction_episode, gt_episode, expect_aligned=True
    )
    return "aligned"


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path).expanduser()


def discover_jobs(
    gt_root: Path,
    predictions: list[tuple[str, Path]],
    tasks: set[str] | None,
    episodes: set[int] | None,
) -> list[tuple[str, Path, Path]]:
    jobs = []
    for name, prediction_root in predictions:
        for prediction_episode in sorted(prediction_root.glob("*/ep*")):
            task = prediction_episode.parent.name
            if tasks and task not in tasks:
                continue
            try:
                episode_index = int(
                    prediction_episode.name.removeprefix("ep")
                )
            except ValueError:
                continue
            if episodes and episode_index not in episodes:
                continue
            gt_episode = gt_root / task / prediction_episode.name
            if not gt_episode.is_dir():
                raise FileNotFoundError(
                    f"{name}: missing GT {task}/{prediction_episode.name}"
                )
            jobs.append((name, prediction_episode, gt_episode))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument(
        "--prediction",
        type=parse_prediction,
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help=(
            "Create NAME/task/episode compatibility trees here instead of "
            "modifying the source prediction directories."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = args.gt_root.expanduser().resolve()
    predictions = [
        (name, path.resolve()) for name, path in args.prediction
    ]
    jobs = discover_jobs(
        gt_root,
        predictions,
        set(args.tasks) if args.tasks else None,
        set(args.episodes) if args.episodes else None,
    )
    if not jobs:
        raise SystemExit("no prediction episodes selected")
    print(
        f"[ego-align] jobs={len(jobs)} apply={args.apply} "
        f"workers={args.workers} output_base={args.output_base}",
        flush=True,
    )

    counts = {"aligned": 0, "already_aligned": 0, "pending": 0, "failed": 0}

    def run(job: tuple[str, Path, Path]) -> tuple[str, str]:
        name, prediction_episode, gt_episode = job
        label = (
            f"{name}/{prediction_episode.parent.name}/"
            f"{prediction_episode.name}"
        )
        destination_episode = (
            args.output_base.expanduser().resolve()
            / name
            / prediction_episode.parent.name
            / prediction_episode.name
            if args.output_base is not None
            else prediction_episode
        )
        if args.validate_only:
            validate_episode(
                destination_episode, gt_episode, expect_aligned=True
            )
            return label, "already_aligned"
        if args.output_base is not None:
            return label, align_output_episode(
                prediction_episode,
                destination_episode,
                gt_episode,
                apply=args.apply,
            )
        return label, align_episode(
            prediction_episode,
            gt_episode,
            apply=args.apply,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            name, prediction_episode, _ = futures[future]
            fallback_label = (
                f"{name}/{prediction_episode.parent.name}/"
                f"{prediction_episode.name}"
            )
            try:
                label, status = future.result()
                counts[status] += 1
                if (
                    status == "aligned"
                    and (counts["aligned"] <= 5 or index % 25 == 0)
                ):
                    print(
                        f"[{index}/{len(jobs)}] {label}: {status}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                print(
                    f"[{index}/{len(jobs)}] {fallback_label}: FAILED "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
    print(f"[ego-align] done: {counts}", flush=True)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
