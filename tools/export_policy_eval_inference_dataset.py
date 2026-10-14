#!/usr/bin/env python3
"""Export selected policy-eval episodes in ACWMBench inference format.

This adapts the one-file-per-episode LeRobot v3 layout used by
policy_eval_dataset/robot_local to the portable inference dataset schema.
Episode indices are read from <selection-dir>/<task>.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import export_inference_dataset as exporter


DEFAULT_SOURCE = Path(
    "/path/to/policy-source"
)
DEFAULT_OUTPUT = Path(
    "/path/to/policy-inference-output"
)
DEFAULT_SELECTION_DIR = Path("/path/to/selection-output")
DEFAULT_TASKS = ("hangtowel", "pickfruit", "pourwater")
FPS = 30.0
CAMERA_CALIBRATION_NAMES = {
    "top_head": "head",
    "hand_left": "hand_left",
    "hand_right": "hand_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def read_selection(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes: list[int] = []
    seen: set[int] = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            episode = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid episode index {value!r}"
            ) from exc
        if episode < 0 or episode in seen:
            raise ValueError(
                f"{path}:{line_number}: negative or duplicate episode {episode}"
            )
        seen.add(episode)
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"{path}: no episodes selected")
    return episodes


def episode_parquet(task: exporter.TaskSpec, episode: int) -> Path:
    return (
        task.source_root
        / "data"
        / "chunk-000"
        / f"file-{episode:03d}.parquet"
    )


def episode_video(
    task: exporter.TaskSpec, episode: int, camera: str
) -> Path:
    return (
        task.source_root
        / "videos"
        / f"observation.images.{camera}"
        / "chunk-000"
        / f"file-{episode:03d}.mp4"
    )


def video_frame_count(path: Path) -> int:
    import cv2  # type: ignore[import-not-found]  # pylint: disable=import-error

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        return int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def validate_episode_source(
    task: exporter.TaskSpec, episode: int
) -> int:
    parquet_path = episode_parquet(task, episode)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(
        parquet_path,
        columns=["episode_index", "frame_index", "timestamp"],
    )
    length = table.num_rows
    if length < 2:
        raise ValueError(f"{parquet_path}: expected at least two frames")
    if table["episode_index"].to_pylist() != [episode] * length:
        raise ValueError(f"{parquet_path}: episode_index mismatch")
    if table["frame_index"].to_pylist() != list(range(length)):
        raise ValueError(f"{parquet_path}: frame_index is not contiguous")
    for camera in exporter.CAMERAS:
        path = episode_video(task, episode, camera)
        if not path.is_file():
            raise FileNotFoundError(path)
        count = video_frame_count(path)
        if count != length:
            raise ValueError(
                f"{path}: {count} video frames != {length} trajectory rows"
            )
    return length


def policy_episode_metadata(
    _task: exporter.TaskSpec, episode: int
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "data/chunk_index": 0,
        "data/file_index": episode,
    }
    for camera in exporter.CAMERAS:
        feature = exporter.CAMERA_FEATURES[camera]
        metadata[f"videos/{feature}/chunk_index"] = 0
        metadata[f"videos/{feature}/file_index"] = episode
        metadata[f"videos/{feature}/from_timestamp"] = 0.0
    return metadata


def policy_calibration(
    task: exporter.TaskSpec, _episode: int
) -> dict[str, Any]:
    source = read_json(task.source_root / "meta/agibot/g2_calib.json")
    resolutions = source.get("resolution", {})
    intrinsics = source.get("intrinsics", {})
    extrinsics = source.get("extrinsics", {})
    serial_numbers = source.get("sn", {})
    cameras: dict[str, Any] = {}
    for camera, source_name in CAMERA_CALIBRATION_NAMES.items():
        resolution = resolutions[source_name]
        intrinsic = dict(intrinsics[source_name])
        if source_name in serial_numbers:
            intrinsic["SN"] = serial_numbers[source_name]
        cameras[camera] = {
            "intrinsics": intrinsic,
            "resolution": {
                "height": int(resolution[0]),
                "width": int(resolution[1]),
            },
            "transforms": [
                {
                    "name": "transform_00",
                    "matrix": extrinsics[source_name],
                }
            ],
        }
    result = {
        "schema_version": exporter.SCHEMA_VERSION,
        "coordinate_convention": (
            "Matrices retain the robot calibration convention."
        ),
        "cameras": cameras,
    }
    exporter._reject_nonportable(result, "calibration")  # pylint: disable=protected-access
    return result


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    selection_dir = args.selection_dir.expanduser().resolve()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    if args.validate_only:
        task_roots = sorted(
            path
            for path in output.iterdir()
            if path.is_dir() and path.name != "shared"
        )
        if not task_roots:
            raise ValueError(f"{output}: no task directories")
        task_data = read_json(task_roots[0] / "task.json")
        exporter.ACTION_WIDTH = int(
            task_data["features"]["action"]["shape"][0]
        )
        exporter.STATE_WIDTH = int(
            task_data["features"]["observation.state"]["shape"][0]
        )
        episodes = exporter._validate_root(output)  # pylint: disable=protected-access
        print(f"[valid] {output}: {len(episodes)} episodes")
        return
    if output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it")

    tasks: dict[str, exporter.TaskSpec] = {}
    infos: dict[str, dict[str, Any]] = {}
    selections: list[exporter.Selection] = []
    action_width: int | None = None
    state_width: int | None = None

    for task_name in args.tasks:
        task = exporter.TaskSpec(task_name, source / task_name)
        info_path = task.source_root / "meta/info.json"
        info = read_json(info_path)
        info["fps"] = float(args.fps)
        current_action_width = int(info["features"]["action"]["shape"][0])
        current_state_width = int(
            info["features"]["observation.state"]["shape"][0]
        )
        action_width = action_width or current_action_width
        state_width = state_width or current_state_width
        if (
            current_action_width != action_width
            or current_state_width != state_width
        ):
            raise ValueError("all tasks must use the same action/state widths")
        tasks[task_name] = task
        infos[task_name] = info

        selection_path = selection_dir / f"{task_name}.txt"
        episodes = read_selection(selection_path)
        for episode in episodes:
            length = validate_episode_source(task, episode)
            selections.append(
                exporter.Selection(
                    task=task,
                    episode_name=f"ep{episode:06d}",
                    episode_index=episode,
                    start=0,
                    end=length,
                    slice_path=selection_path,
                )
            )
        print(f"[selection] {task_name}: {len(episodes)} episodes")

    if action_width is None or state_width is None:
        raise SystemExit("no tasks selected")
    exporter.ACTION_WIDTH = action_width
    exporter.STATE_WIDTH = state_width
    exporter._read_episode_metadata = (  # pylint: disable=protected-access
        policy_episode_metadata
    )
    exporter._portable_calibration = (  # pylint: disable=protected-access
        policy_calibration
    )

    for task_name, info in infos.items():
        exporter._validate_source_info(  # pylint: disable=protected-access
            info, tasks[task_name].source_root / "meta/info.json"
        )

    print(f"[source] {source}")
    print(f"[output] {output}")
    print(f"[episodes] {len(selections)}")
    print(f"[features] action={action_width}, observation.state={state_width}")
    if args.dry_run:
        print("[dry-run] source validation succeeded; no files written")
        return

    first_task = tasks[args.tasks[0]]
    urdf = (
        first_task.source_root
        / "meta/agibot/G2_t2v2_crsB_ctek90d.urdf"
    )
    exporter._build_dataset(  # pylint: disable=protected-access
        output, selections, infos, urdf
    )
    print(f"[done] {output}: {len(selections)} episodes")


if __name__ == "__main__":
    main()
