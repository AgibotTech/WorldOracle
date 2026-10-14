#!/usr/bin/env python3
"""Build ACWMBench GT and inference datasets from Ego LeRobot-v3 data."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image  # type: ignore[import-not-found]


DEFAULT_GT_ROOT = Path("/path/to/ego-ground-truth")
DEFAULT_INFERENCE_ROOT = Path(
    "/path/to/ego-inference-data"
)
DEFAULT_SOURCES = {
    "hangtowel": Path("/path/to/ego-source/hangtowel"),
    "pourwater": Path("/path/to/ego-source/pourwater"),
    "pickfruit": Path("/path/to/ego-source/pickfruit"),
    "pulldrawer": Path("/path/to/ego-source/pulldrawer"),
}
VIEW_TO_SOURCE = {
    "top_head": "head_fisheye",
    "hand_left": "head_fisheye_left",
    "hand_right": "head_fisheye_right",
    "head_stereo_left": "head_stereo_left",
    "head_stereo_right": "head_stereo_right",
}
WRIST_FRAME_ALIGNMENT = np.asarray(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
IDENTITY_9D = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


@dataclass(frozen=True)
class Episode:
    task: str
    source_root: Path
    row: dict[str, Any]

    @property
    def index(self) -> int:
        return int(self.row["episode_index"])

    @property
    def name(self) -> str:
        return f"ep{self.index:06d}"

    @property
    def length(self) -> int:
        return int(self.row["length"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument(
        "--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--tasks", nargs="*", choices=sorted(DEFAULT_SOURCES))
    parser.add_argument("--episodes", nargs="*", type=int)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--jpg-quality", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--without-source-state",
        action="store_true",
        help="Do not archive the original 635D state in trajectory.parquet.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_parquet_parts(paths: list[Path]) -> pa.Table:
    if not paths:
        raise FileNotFoundError("No parquet parts found")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def discover_episodes(tasks: list[str] | None, selected: list[int] | None) -> list[Episode]:
    result: list[Episode] = []
    selected_set = set(selected or [])
    for task in tasks or list(DEFAULT_SOURCES):
        root = DEFAULT_SOURCES[task]
        if not root.is_dir():
            raise FileNotFoundError(root)
        table = read_parquet_parts(
            sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
        )
        for row in table.to_pylist():
            episode = Episode(task, root, row)
            if selected_set and episode.index not in selected_set:
                continue
            result.append(episode)
    return result


def state_fields(source_root: Path) -> dict[str, dict[str, Any]]:
    info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
    return info["features"]["observation.state"]["field_descriptions"]


def indices(fields: dict[str, dict[str, Any]], name: str) -> list[int]:
    values = fields[name].get("indices")
    if not values:
        raise KeyError(f"Missing state indices for {name}")
    return [int(value) for value in values]


def episode_data_path(episode: Episode) -> Path:
    chunk = int(episode.row["data/chunk_index"])
    file_index = int(episode.row["data/file_index"])
    return episode.source_root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"


def episode_video_path(episode: Episode, source_view: str) -> Path:
    prefix = f"videos/observation.images.{source_view}"
    chunk = int(episode.row[f"{prefix}/chunk_index"])
    file_index = int(episode.row[f"{prefix}/file_index"])
    return (
        episode.source_root
        / f"videos/observation.images.{source_view}"
        / f"chunk-{chunk:03d}/file-{file_index:03d}.mp4"
    )


def source_stride(episode: Episode, target_fps: float) -> tuple[float, int]:
    info = json.loads(
        (episode.source_root / "meta/info.json").read_text(encoding="utf-8")
    )
    source_fps = float(info["fps"])
    ratio = source_fps / target_fps
    if not ratio.is_integer():
        raise ValueError(
            f"{episode.task}: source FPS {source_fps} is not an integer "
            f"multiple of target FPS {target_fps}"
        )
    return source_fps, int(ratio)


def selected_source_frames(length: int, stride: int) -> np.ndarray:
    return np.arange(0, length, stride, dtype=np.int64)


def extract_video(
    source: Path,
    destination: Path,
    count: int,
    stride: int,
    jpg_quality: int,
    overwrite: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    existing = list(destination.glob("frame_*.jpg"))
    if len(existing) == count and not overwrite:
        return
    for path in existing:
        path.unlink()
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=not(mod(n\\,{stride}))",
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(count),
        "-q:v",
        str(jpg_quality),
        "-start_number",
        "0",
        str(destination / "frame_%05d.jpg"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {source}:\n{result.stderr.strip()}")
    actual = sum(1 for _ in destination.glob("frame_*.jpg"))
    if actual != count:
        raise RuntimeError(f"{source}: extracted {actual} frames, expected {count}")


def quaternion_to_matrix(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(quaternions, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1)
    valid = norms > 1e-8
    repaired = values.copy()
    previous = np.asarray([0, 0, 0, 1], dtype=np.float32)
    for index in range(len(repaired)):
        if valid[index]:
            previous = repaired[index] / norms[index]
        repaired[index] = previous
    x, y, z, w = np.moveaxis(repaired, -1, 0)
    matrices = np.empty((len(repaired), 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices, valid


def poses(position: np.ndarray, quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation, quaternion_valid = quaternion_to_matrix(quaternion)
    result = np.repeat(np.eye(4, dtype=np.float32)[None], len(position), axis=0)
    result[:, :3, :3] = rotation
    result[:, :3, 3] = position
    return result, quaternion_valid


def relative_pose_9d(absolute: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(absolute[:-1])
    delta = inverse @ absolute[1:]
    translation = delta[:, :3, 3]
    rot6d = delta[:, :3, :2].transpose(0, 2, 1).reshape(-1, 6)
    return np.concatenate([translation, rot6d], axis=-1).astype(np.float32)


def fingertips_in_wrist(
    hand_positions: np.ndarray, wrist_camera: np.ndarray
) -> np.ndarray:
    fingertips = hand_positions[:, FINGERTIP_INDICES, :]
    homogeneous = np.concatenate(
        [fingertips, np.ones((*fingertips.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    transformed = np.einsum(
        "tij,tnj->tni", np.linalg.inv(wrist_camera), homogeneous
    )[..., :3]
    return transformed.reshape(len(hand_positions), 15).astype(np.float32)


def build_action(
    canonical_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    left_ori = canonical_state[:, 0:84].reshape(-1, 21, 4)
    left_pos = canonical_state[:, 84:147].reshape(-1, 21, 3)
    left_valid = canonical_state[:, 147] > 0.5
    right_ori = canonical_state[:, 148:232].reshape(-1, 21, 4)
    right_pos = canonical_state[:, 232:295].reshape(-1, 21, 3)
    right_valid = canonical_state[:, 295] > 0.5
    camera_ori = canonical_state[:, 296:300]
    camera_pos = canonical_state[:, 300:303]
    camera_valid = canonical_state[:, 303] > 0.5

    camera_pose, camera_quaternion_valid = poses(camera_pos, camera_ori)
    right_wrist_camera, right_quaternion_valid = poses(
        right_pos[:, 0], right_ori[:, 0]
    )
    left_wrist_camera, left_quaternion_valid = poses(left_pos[:, 0], left_ori[:, 0])
    right_wrist_camera = right_wrist_camera @ WRIST_FRAME_ALIGNMENT
    left_wrist_camera = left_wrist_camera @ WRIST_FRAME_ALIGNMENT
    right_wrist_world = camera_pose @ right_wrist_camera
    left_wrist_world = camera_pose @ left_wrist_camera

    right_fingertips = fingertips_in_wrist(right_pos, right_wrist_camera)
    left_fingertips = fingertips_in_wrist(left_pos, left_wrist_camera)
    transition_action = np.concatenate(
        [
            relative_pose_9d(camera_pose),
            relative_pose_9d(right_wrist_world),
            right_fingertips[1:],
            relative_pose_9d(left_wrist_world),
            left_fingertips[1:],
        ],
        axis=-1,
    ).astype(np.float32)
    initial_action = np.concatenate(
        [
            IDENTITY_9D,
            IDENTITY_9D,
            right_fingertips[0],
            IDENTITY_9D,
            left_fingertips[0],
        ]
    )
    action = np.concatenate([initial_action[None], transition_action], axis=0)
    pose_valid = (
        left_valid
        & right_valid
        & camera_valid
        & left_quaternion_valid
        & right_quaternion_valid
        & camera_quaternion_valid
    )
    action_valid = np.zeros(len(action), dtype=bool)
    action_valid[1:] = pose_valid[:-1] & pose_valid[1:]
    controls = {
        "left_positions_camera": left_pos,
        "right_positions_camera": right_pos,
        "left_quaternions_camera_xyzw": left_ori,
        "right_quaternions_camera_xyzw": right_ori,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "camera_position_world": camera_pos,
        "camera_quaternion_world_xyzw": camera_ori,
        "camera_valid": camera_valid,
    }
    if action.shape != (len(canonical_state), 57):
        raise ValueError(f"Unexpected action shape: {action.shape}")
    return action, action_valid, controls


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    width = int(values.shape[1])
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), width
    )


def write_trajectory(
    path: Path,
    source_frames: np.ndarray,
    target_fps: float,
    source_state: np.ndarray,
    canonical_state_indices: list[int],
    aux_indices: list[int],
    include_source_state: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    canonical = source_state[:, canonical_state_indices].astype(np.float32)
    auxiliary = source_state[:, aux_indices].astype(np.float32)
    action, action_valid, controls = build_action(canonical)
    columns: dict[str, pa.Array] = {
        "frame_index": pa.array(np.arange(len(source_frames)), type=pa.int64()),
        "source_frame_index": pa.array(source_frames, type=pa.int64()),
        "timestamp": pa.array(
            np.arange(len(source_frames), dtype=np.float32) / target_fps,
            type=pa.float32(),
        ),
        "action": fixed_list(action),
        "action_valid": pa.array(action_valid, type=pa.bool_()),
        "observation.state": fixed_list(canonical),
        "observation.aux": fixed_list(auxiliary),
    }
    if include_source_state:
        columns["observation.source_state"] = fixed_list(
            source_state.astype(np.float32)
        )
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        row_group_size=512,
        store_schema=True,
    )
    os.replace(temporary, path)
    return action, controls


def keyframes_for_episode(episode: Episode, stride: int, count: int) -> list[int]:
    table = pq.read_table(
        episode.source_root / "meta/agibot/keyframes.parquet",
        filters=[("episode_index", "=", episode.index)],
        columns=["start_frame_index"],
    )
    source_keyframes = [int(value) for value in table.column(0).to_pylist()]
    mapped = {
        min(count - 1, max(0, int(round(frame / stride))))
        for frame in source_keyframes
    }
    return sorted(mapped)


def episode_calibration(episode: Episode) -> dict[str, Any]:
    all_calibration = json.loads(
        (episode.source_root / "meta/agibot/camera_params.json").read_text(
            encoding="utf-8"
        )
    )
    raw = all_calibration[str(episode.index)]
    cameras: dict[str, Any] = {}
    for output_view, source_view in VIEW_TO_SOURCE.items():
        intrinsic_key = f"intrinsic_{source_view}"
        matching_extrinsics = {
            key: value
            for key, value in raw.items()
            if key.startswith(f"extrinsic_{source_view}_")
        }
        intrinsic = raw.get(intrinsic_key)
        if intrinsic is None:
            raise KeyError(f"{episode.task}/{episode.name}: {intrinsic_key}")
        cameras[output_view] = {
            "source_camera": source_view,
            "camera_mount": "head",
            "projection_model": (
                "opencv_fisheye" if "fisheye" in source_view else "pinhole"
            ),
            "intrinsics": intrinsic,
            "extrinsics": matching_extrinsics,
            "resolution": {
                "width": int(intrinsic["width"]),
                "height": int(intrinsic["height"]),
            },
        }
    return {
        "schema_version": "acwmbench.ego.calibration.v1",
        "episode_index": episode.index,
        "view_aliases": {
            "top_head": "head_fisheye",
            "hand_left": "head_fisheye_left",
            "hand_right": "head_fisheye_right",
        },
        "alias_warning": (
            "hand_left and hand_right are compatibility aliases for head-mounted "
            "fisheye cameras; they are not wrist cameras."
        ),
        "cameras": cameras,
        "source_calibration": raw,
    }


def save_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def copy_initial_observations(gt_episode: Path, inference_episode: Path) -> None:
    observations = inference_episode / "observations"
    observations.mkdir(parents=True, exist_ok=True)
    for view in VIEW_TO_SOURCE:
        source = gt_episode / "video" / view / "frame_00000.jpg"
        destination = observations / f"{view}.png"
        with Image.open(source) as image:
            image.save(destination, format="PNG", optimize=True)


def write_checksums(episode_dir: Path) -> None:
    lines = []
    for path in sorted(episode_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(episode_dir)}")
    temporary = episode_dir / "checksums.sha256.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, episode_dir / "checksums.sha256")


def build_episode(
    episode: Episode,
    gt_root: Path,
    inference_root: Path,
    target_fps: float,
    jpg_quality: int,
    overwrite: bool,
    include_source_state: bool,
) -> str:
    source_fps, stride = source_stride(episode, target_fps)
    source_frames = selected_source_frames(episode.length, stride)
    count = len(source_frames)
    gt_episode = gt_root / episode.task / episode.name
    inference_episode = inference_root / episode.task / episode.name
    for view, source_view in VIEW_TO_SOURCE.items():
        extract_video(
            episode_video_path(episode, source_view),
            gt_episode / "video" / view,
            count,
            stride,
            jpg_quality,
            overwrite,
        )
    for view in VIEW_TO_SOURCE:
        (gt_episode / "multiview_masks" / view).mkdir(parents=True, exist_ok=True)

    keyframes = keyframes_for_episode(episode, stride, count)
    atomic_json(
        gt_episode / "prompt/prompts.json",
        {
            "n_frames": count,
            "grounding_objects": [],
            "segments": [
                {"key_frame": frame, "success_criteria": []}
                for frame in keyframes
            ],
        },
    )
    source_map = {
        "source_dataset": str(episode.source_root),
        "source_episode_index": episode.index,
        "source_fps": source_fps,
        "canonical_fps": target_fps,
        "sampling_stride": stride,
        "source_frame_indices": source_frames.tolist(),
        "source_keyframes": [
            int(value)
            for value in pq.read_table(
                episode.source_root / "meta/agibot/keyframes.parquet",
                filters=[("episode_index", "=", episode.index)],
                columns=["start_frame_index"],
            )
            .column(0)
            .to_pylist()
        ],
        "canonical_keyframes": keyframes,
    }
    atomic_json(gt_episode / "source_map.json", source_map)

    data = pq.read_table(
        episode_data_path(episode),
        columns=["observation.state"],
    )
    if len(data) != episode.length:
        raise ValueError(
            f"{episode.task}/{episode.name}: data rows {len(data)} != {episode.length}"
        )
    source_state = np.asarray(
        data.column("observation.state").to_pylist(), dtype=np.float32
    )[source_frames]
    fields = state_fields(episode.source_root)
    canonical_names = [
        "hand_state/left/ori",
        "hand_state/left/pos",
        "hand_state/left/valid",
        "hand_state/right/ori",
        "hand_state/right/pos",
        "hand_state/right/valid",
        "head_camera/orientation",
        "head_camera/position",
        "head_camera/valid",
    ]
    canonical_indices = [
        item for name in canonical_names for item in indices(fields, name)
    ]
    aux_names = [
        "head_imu/acceleration",
        "head_imu/angle",
        "head_imu/gyro",
        "head_imu/relative_time",
        "head_imu/sample_index",
        "head_state/ori",
        "head_state/pos",
        "head_state/valid",
    ]
    aux_state_indices = [item for name in aux_names for item in indices(fields, name)]
    _, controls = write_trajectory(
        inference_episode / "trajectory.parquet",
        source_frames,
        target_fps,
        source_state,
        canonical_indices,
        aux_state_indices,
        include_source_state,
    )
    controls["source_frame_index"] = source_frames
    save_npz(inference_episode / "controls/hand_camera_poses.npz", **controls)
    atomic_json(
        inference_episode / "controls/action_57d_spec.json",
        {
            "schema": "cosmos3.human_hand_pose.backward_framewise.v1",
            "width": 57,
            "layout": [
                {"name": "camera_delta", "indices": [0, 9], "encoding": "xyz+rot6d"},
                {
                    "name": "right_wrist_world_delta",
                    "indices": [9, 18],
                    "encoding": "xyz+rot6d",
                },
                {
                    "name": "right_fingertips_in_wrist",
                    "indices": [18, 33],
                    "joint_indices": list(FINGERTIP_INDICES),
                },
                {
                    "name": "left_wrist_world_delta",
                    "indices": [33, 42],
                    "encoding": "xyz+rot6d",
                },
                {
                    "name": "left_fingertips_in_wrist",
                    "indices": [42, 57],
                    "joint_indices": list(FINGERTIP_INDICES),
                },
            ],
            "alignment": "Cosmos3 HumanHandPoseLeRobotDataset",
            "row_alignment": (
                "row 0 is an initial identity transition; rows i>0 encode "
                "source frame i-1 to i"
            ),
        },
    )
    copy_initial_observations(gt_episode, inference_episode)
    calibration = episode_calibration(episode)
    atomic_json(inference_episode / "calibration.json", calibration)
    shutil.copy2(gt_episode / "source_map.json", inference_episode / "source_map.json")
    atomic_json(
        inference_episode / "metadata.json",
        {
            "schema_version": "acwmbench.inference.ego.v1",
            "task_name": episode.task,
            "episode_index": episode.index,
            "episode_name": episode.name,
            "fps": target_fps,
            "num_frames": count,
            "num_transitions": max(0, count - 1),
            "seed_local_index": 0,
            "embodiment": "ego_human_bimanual",
            "files": {
                "trajectory": "trajectory.parquet",
                "calibration": "calibration.json",
                "source_map": "source_map.json",
                "observations": {
                    view: f"observations/{view}.png" for view in VIEW_TO_SOURCE
                },
                "controls": {
                    "hand_camera_poses": "controls/hand_camera_poses.npz",
                    "action_57d_spec": "controls/action_57d_spec.json",
                },
            },
        },
    )
    write_checksums(inference_episode)
    return f"{episode.task}/{episode.name}: {count} frames, {len(keyframes)} keyframes"


def range_description(start: int, stop: int, description: str) -> dict[str, Any]:
    return {
        "description": description,
        "dimensions": stop - start,
        "indices": list(range(start, stop)),
    }


def dataset_schema() -> dict[str, Any]:
    action_fields = {
        "camera_delta": range_description(0, 9, "camera xyz + rot6d delta"),
        "right_wrist_world_delta": range_description(
            9, 18, "right wrist world xyz + rot6d delta"
        ),
        "right_fingertips_in_wrist": range_description(
            18, 33, "five right fingertip XYZ positions in aligned wrist frame"
        ),
        "left_wrist_world_delta": range_description(
            33, 42, "left wrist world xyz + rot6d delta"
        ),
        "left_fingertips_in_wrist": range_description(
            42, 57, "five left fingertip XYZ positions in aligned wrist frame"
        ),
    }
    state_fields_description = {
        "hand_state/left/ori": range_description(0, 84, "21 XYZW quaternions"),
        "hand_state/left/pos": range_description(
            84, 147, "21 XYZ positions in camera frame"
        ),
        "hand_state/left/valid": range_description(147, 148, "left hand valid"),
        "hand_state/right/ori": range_description(148, 232, "21 XYZW quaternions"),
        "hand_state/right/pos": range_description(
            232, 295, "21 XYZ positions in camera frame"
        ),
        "hand_state/right/valid": range_description(295, 296, "right hand valid"),
        "head_camera/orientation": range_description(
            296, 300, "camera-to-world XYZW quaternion"
        ),
        "head_camera/position": range_description(
            300, 303, "camera-to-world XYZ position"
        ),
        "head_camera/valid": range_description(303, 304, "camera pose valid"),
    }
    return {
        "schema_version": "acwmbench.inference.ego.v1",
        "embodiment": "ego_human_bimanual",
        "fps": 15.0,
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [57],
                "field_descriptions": action_fields,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [304],
                "field_descriptions": state_fields_description,
            },
            "observation.aux": {"dtype": "float32", "shape": [19]},
            "observation.source_state": {"dtype": "float32", "shape": [635]},
        },
        "views": {
            output: {
                "source": source,
                "mount": "head",
                "compatibility_alias": output in {"top_head", "hand_left", "hand_right"},
            }
            for output, source in VIEW_TO_SOURCE.items()
        },
    }


def prepare_roots(gt_root: Path, inference_root: Path, tasks: list[str]) -> None:
    gt_root.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)
    (gt_root / "README.md").write_text(
        "# Ego-human ACWMBench ground truth dataset\n\n"
        "Five synchronized head-mounted views sampled at 15 FPS. "
        "`hand_left` and `hand_right` are compatibility aliases for the left "
        "and right head fisheye cameras, not wrist cameras.\n",
        encoding="utf-8",
    )
    schema = dataset_schema()
    atomic_json(inference_root / "schema.json", schema)
    atomic_json(inference_root / "shared/schema.json", schema)
    atomic_json(
        inference_root / "shared/ego_human/joint_order.json",
        {
            "number_of_joints_per_hand": 21,
            "wrist_joint_index": 0,
            "fingertip_joint_indices": list(FINGERTIP_INDICES),
            "orientation": "quaternion_xyzw",
            "note": "Full semantic joint names are not declared by the source schema.",
        },
    )
    atomic_json(
        inference_root / "shared/ego_human/baseline_adapters.json",
        {
            "canonical_view": "top_head",
            "canonical_fps": 15.0,
            "adapters": {
                "cosmos3_nano": {
                    "status": "ready",
                    "inputs": ["observations/top_head.png", "trajectory.parquet:action"],
                },
                "jointcontrolvideo": {
                    "status": "adapter_ready",
                    "inputs": ["controls/hand_camera_poses.npz"],
                },
                "oscar": {
                    "status": "requires_2d_skeleton_renderer",
                    "inputs": ["controls/hand_camera_poses.npz", "calibration.json"],
                },
                "rynnworld_teleop": {
                    "status": "requires_depth_skeleton_renderer",
                    "inputs": ["controls/hand_camera_poses.npz", "calibration.json"],
                },
                "egohoi": {
                    "status": "requires_object_masks_and_pose_map_adapter",
                    "inputs": ["controls/hand_camera_poses.npz", "calibration.json"],
                },
                "generated_reality": {
                    "status": "adapter_ready_when_checkpoint_available",
                    "inputs": ["controls/hand_camera_poses.npz"],
                },
                "lome": {
                    "status": "requires_action_map_and_camera_ray_renderer",
                    "inputs": ["controls/hand_camera_poses.npz", "calibration.json"],
                },
            },
        },
    )
    (inference_root / "README.md").write_text(
        "# Ego-human ACWMBench inference dataset\n\n"
        "Canonical 15 FPS bimanual human-hand trajectories. `top_head`, "
        "`hand_left`, and `hand_right` are compatibility aliases for head-mounted "
        "fisheye cameras. See `shared/ego_human/baseline_adapters.json`.\n",
        encoding="utf-8",
    )
    for task in tasks:
        atomic_json(inference_root / task / "schema.json", schema)


def write_dataset_manifests(
    episodes: list[Episode], gt_root: Path, inference_root: Path, fps: float
) -> None:
    task_counts: dict[str, int] = {}
    task_frames: dict[str, int] = {}
    for episode in episodes:
        _, stride = source_stride(episode, fps)
        task_counts[episode.task] = task_counts.get(episode.task, 0) + 1
        task_frames[episode.task] = task_frames.get(episode.task, 0) + math.ceil(
            episode.length / stride
        )
    common = {
        "canonical_fps": fps,
        "total_episodes": len(episodes),
        "total_frames": sum(task_frames.values()),
        "tasks": {
            task: {
                "episodes": task_counts[task],
                "frames": task_frames[task],
                "source_dataset": str(DEFAULT_SOURCES[task]),
            }
            for task in sorted(task_counts)
        },
        "views": VIEW_TO_SOURCE,
    }
    atomic_json(
        gt_root / "dataset_manifest.json",
        {"schema_version": "acwmbench.gt.ego.v1", **common},
    )
    atomic_json(
        inference_root / "dataset_manifest.json",
        {"schema_version": "acwmbench.inference.ego.v1", **common},
    )


def validate_episode(
    episode: Episode, gt_root: Path, inference_root: Path, fps: float
) -> list[str]:
    errors: list[str] = []
    _, stride = source_stride(episode, fps)
    expected = math.ceil(episode.length / stride)
    gt_episode = gt_root / episode.task / episode.name
    inference_episode = inference_root / episode.task / episode.name
    for view in VIEW_TO_SOURCE:
        actual = sum(1 for _ in (gt_episode / "video" / view).glob("frame_*.jpg"))
        if actual != expected:
            errors.append(f"{episode.task}/{episode.name}/{view}: {actual} != {expected}")
        if not (gt_episode / "multiview_masks" / view).is_dir():
            errors.append(f"{episode.task}/{episode.name}: missing mask dir {view}")
    prompt_path = gt_episode / "prompt/prompts.json"
    if not prompt_path.is_file():
        errors.append(f"{episode.task}/{episode.name}: missing prompts.json")
    else:
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        if int(prompt.get("n_frames", -1)) != expected:
            errors.append(f"{episode.task}/{episode.name}: prompt n_frames mismatch")
        if any(segment.get("success_criteria") != [] for segment in prompt["segments"]):
            errors.append(f"{episode.task}/{episode.name}: criteria are not blank")
    trajectory = inference_episode / "trajectory.parquet"
    if not trajectory.is_file():
        errors.append(f"{episode.task}/{episode.name}: missing trajectory")
    else:
        schema = pq.ParquetFile(trajectory).schema_arrow
        if pq.ParquetFile(trajectory).metadata.num_rows != expected:
            errors.append(f"{episode.task}/{episode.name}: trajectory rows mismatch")
        for name in ["action", "observation.state", "observation.aux"]:
            if name not in schema.names:
                errors.append(f"{episode.task}/{episode.name}: missing {name}")
    for name in ["metadata.json", "calibration.json", "source_map.json", "checksums.sha256"]:
        if not (inference_episode / name).is_file():
            errors.append(f"{episode.task}/{episode.name}: missing {name}")
    for view in VIEW_TO_SOURCE:
        if not (inference_episode / "observations" / f"{view}.png").is_file():
            errors.append(f"{episode.task}/{episode.name}: missing observation {view}")
    return errors


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args.tasks, args.episodes)
    tasks = sorted({episode.task for episode in episodes})
    if not episodes:
        raise SystemExit("No episodes selected")
    if args.validate_only:
        all_errors = [
            error
            for episode in episodes
            for error in validate_episode(
                episode, args.gt_root, args.inference_root, args.fps
            )
        ]
        if all_errors:
            raise SystemExit("\n".join(all_errors))
        print(f"Validated {len(episodes)} episodes")
        return

    prepare_roots(args.gt_root, args.inference_root, tasks)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_to_episode = {
            pool.submit(
                build_episode,
                episode,
                args.gt_root,
                args.inference_root,
                args.fps,
                args.jpg_quality,
                args.overwrite,
                not args.without_source_state,
            ): episode
            for episode in episodes
        }
        for future in concurrent.futures.as_completed(future_to_episode):
            episode = future_to_episode[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                message = f"{episode.task}/{episode.name}: {exc}"
                failures.append(message)
                print(f"ERROR {message}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} episode(s) failed:\n" + "\n".join(failures))

    errors = [
        error
        for episode in episodes
        for error in validate_episode(
            episode, args.gt_root, args.inference_root, args.fps
        )
    ]
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors))
    write_dataset_manifests(episodes, args.gt_root, args.inference_root, args.fps)
    print(
        f"Built and validated {len(episodes)} episodes in {args.gt_root} "
        f"and {args.inference_root}"
    )


if __name__ == "__main__":
    main()
