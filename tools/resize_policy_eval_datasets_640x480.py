#!/usr/bin/env python3
"""Resize paired policy-eval GT/inference datasets to 640x480 in place.

The conversion keeps all coordinate-bearing resources aligned:

* GT RGB frames and the occasional MP4
* prompt and three-view ``masks.npz`` archives (nearest-neighbor)
* mask metadata and referenced SAM point annotations
* inference initial observations
* camera resolutions and intrinsics (extrinsics remain unchanged)
* episode-level and dataset-level SHA-256 manifests

Each episode is staged and validated before its files are atomically replaced.
Without ``--apply`` this command only audits and prints the planned work.
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIEWS = ("top_head", "hand_left", "hand_right")
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
DEFAULT_GT_ROOT = Path(
    "/path/to/policy-ground-truth"
)
DEFAULT_INFERENCE_ROOT = Path(
    "/path/to/policy-inference-data"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image.shape[1], image.shape[0]


def frame_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"{directory}: no frame_*.jpg")
    return paths


def scale_points(
    objects: Any,
    scale_x: float,
    scale_y: float,
) -> None:
    if not isinstance(objects, dict):
        return
    for record in objects.values():
        if not isinstance(record, dict):
            continue
        points = record.get("points")
        if not isinstance(points, list):
            continue
        record["points"] = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]


def stage_resized_frames(
    source_dir: Path,
    destination_dir: Path,
    workers: int,
) -> int:
    sources = frame_paths(source_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    def resize_one(source: Path) -> None:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode image: {source}")
        resized = cv2.resize(
            image,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        destination = destination_dir / source.name
        if not cv2.imwrite(
            str(destination),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            raise RuntimeError(f"failed to write image: {destination}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(resize_one, sources))
    outputs = frame_paths(destination_dir)
    if len(outputs) != len(sources):
        raise RuntimeError(
            f"{destination_dir}: wrote {len(outputs)} of {len(sources)} frames"
        )
    if image_size(outputs[0]) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise RuntimeError(f"{outputs[0]}: invalid resized dimensions")
    if image_size(outputs[-1]) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise RuntimeError(f"{outputs[-1]}: invalid resized dimensions")
    return len(outputs)


def extract_npz_arrays(
    source: Path,
    workspace: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.endswith(".npy")
        }
        if "masks" not in members:
            raise KeyError(f"{source}: missing masks array")
        extracted = workspace / "masks.npy"
        with archive.open(members["masks"]) as input_file:
            with extracted.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=8 << 20)
    masks = np.load(extracted, mmap_mode="r", allow_pickle=False)
    with np.load(source, allow_pickle=False) as archive:
        others = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "masks"
        }
    return masks, others


def npz_array_shape(path: Path, key: str = "masks") -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive:
        member = next(
            (
                name
                for name in archive.namelist()
                if name.endswith(".npy") and Path(name).stem == key
            ),
            None,
        )
        if member is None:
            raise KeyError(f"{path}: missing {key!r} array")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape)


def stage_resized_masks(
    source: Path,
    destination: Path,
    workspace: Path,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    masks, others = extract_npz_arrays(source, workspace)
    if masks.ndim not in (3, 4):
        raise ValueError(f"{source}: expected [T,H,W] or [N,T,H,W], got {masks.shape}")
    old_shape = tuple(int(value) for value in masks.shape)
    leading = masks.shape[:-2]
    resized_path = workspace / "resized_masks.npy"
    resized = np.lib.format.open_memmap(
        resized_path,
        mode="w+",
        dtype=masks.dtype,
        shape=(*leading, TARGET_HEIGHT, TARGET_WIDTH),
    )
    source_flat = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    output_flat = resized.reshape((-1, TARGET_HEIGHT, TARGET_WIDTH))
    for index in range(source_flat.shape[0]):
        output_flat[index] = cv2.resize(
            source_flat[index],
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
    resized.flush()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, masks=resized, **others)
    new_shape = (*leading, TARGET_HEIGHT, TARGET_WIDTH)
    if npz_array_shape(destination) != new_shape:
        raise RuntimeError(f"{destination}: staged mask validation failed")
    del output_flat, resized, source_flat, masks
    return old_shape, tuple(int(value) for value in new_shape)


def updated_mask_metadata(
    source: Path,
    new_shape: tuple[int, ...],
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    metadata = read_json(source)
    metadata["shape"] = list(new_shape)
    scale_points(metadata.get("objects"), scale_x, scale_y)
    metadata["resize"] = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "mask_interpolation": "nearest",
    }
    return metadata


def updated_point_annotation(
    path: Path,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    data = read_json(path)
    scale_points(data.get("objects"), scale_x, scale_y)
    data["frame_width"] = TARGET_WIDTH
    data["frame_height"] = TARGET_HEIGHT
    data["resize"] = {
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return data


def stage_resized_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=area",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)


def atomic_replace_many(
    replacements: list[tuple[Path, Path]],
    backup_root: Path,
) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    committed: list[tuple[Path, Path]] = []
    try:
        for index, (staged, target) in enumerate(replacements):
            if not staged.exists():
                raise FileNotFoundError(staged)
            backup = backup_root / f"{index:03d}"
            if target.exists():
                os.replace(target, backup)
            os.replace(staged, target)
            committed.append((target, backup))
    except BaseException:
        for target, backup in reversed(committed):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            if backup.exists():
                os.replace(backup, target)
        raise


def resize_gt_episode(
    episode: Path,
    workers: int,
    update_point_annotations: bool,
) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix=".resize_640x480_", dir=episode))
    staged_root = temp_root / "staged"
    replacements: list[tuple[Path, Path]] = []
    try:
        view_sizes: dict[str, tuple[int, int]] = {}
        frame_counts: dict[str, int] = {}
        for view in VIEWS:
            source_dir = episode / "video" / view
            first = frame_paths(source_dir)[0]
            view_sizes[view] = image_size(first)
            destination = staged_root / "video" / view
            frame_counts[view] = stage_resized_frames(
                source_dir,
                destination,
                workers,
            )
            replacements.append((destination, source_dir))
        if len(set(frame_counts.values())) != 1:
            raise ValueError(f"{episode}: view frame counts differ: {frame_counts}")

        mask_specs = [
            (episode / "prompt", "top_head"),
            *[
                (episode / "multiview_masks" / view, view)
                for view in VIEWS
            ],
        ]
        staged_points: set[Path] = set()
        for index, (mask_dir, view) in enumerate(mask_specs):
            old_width, old_height = view_sizes[view]
            scale_x = TARGET_WIDTH / old_width
            scale_y = TARGET_HEIGHT / old_height
            staged_mask = (
                staged_root / mask_dir.relative_to(episode) / "masks.npz"
            )
            _, new_shape = stage_resized_masks(
                mask_dir / "masks.npz",
                staged_mask,
                temp_root / "mask_workspace" / str(index),
            )
            if new_shape[-3] != frame_counts[view]:
                raise ValueError(
                    f"{mask_dir}: {new_shape[-3]} mask frames != "
                    f"{frame_counts[view]} RGB frames"
                )
            staged_meta = (
                staged_root / mask_dir.relative_to(episode) / "masks_meta.json"
            )
            metadata = updated_mask_metadata(
                mask_dir / "masks_meta.json",
                new_shape,
                scale_x,
                scale_y,
            )
            write_json(staged_meta, metadata)
            replacements.extend(
                [
                    (staged_mask, mask_dir / "masks.npz"),
                    (staged_meta, mask_dir / "masks_meta.json"),
                ]
            )
            points_value = metadata.get("points_path")
            if points_value and update_point_annotations:
                points_path = Path(str(points_value))
                if points_path.is_file() and points_path not in staged_points:
                    staged_point = staged_root / "_points" / f"{index}.json"
                    write_json(
                        staged_point,
                        updated_point_annotation(points_path, scale_x, scale_y),
                    )
                    replacements.append((staged_point, points_path))
                    staged_points.add(points_path)

        for mp4 in sorted((episode / "video").glob("*.mp4")):
            staged_mp4 = staged_root / "video" / mp4.name
            stage_resized_video(mp4, staged_mp4)
            replacements.append((staged_mp4, mp4))
        atomic_replace_many(replacements, temp_root / "backup")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def resized_calibration(path: Path) -> dict[str, Any]:
    data = read_json(path)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"{path}: missing cameras")
    for view in VIEWS:
        camera = cameras.get(view)
        if not isinstance(camera, dict):
            raise ValueError(f"{path}: missing camera {view}")
        resolution = camera.get("resolution")
        intrinsics = camera.get("intrinsics")
        if not isinstance(resolution, dict) or not isinstance(intrinsics, dict):
            raise ValueError(f"{path}: incomplete camera {view}")
        old_width = float(resolution["width"])
        old_height = float(resolution["height"])
        scale_x = TARGET_WIDTH / old_width
        scale_y = TARGET_HEIGHT / old_height
        for key in ("Fx", "Cx"):
            intrinsics[key] = float(intrinsics[key]) * scale_x
        for key in ("Fy", "Cy"):
            intrinsics[key] = float(intrinsics[key]) * scale_y
        resolution["width"] = TARGET_WIDTH
        resolution["height"] = TARGET_HEIGHT
        camera["resize"] = {
            "source_width": int(old_width),
            "source_height": int(old_height),
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
    data["resize_note"] = (
        "Intrinsics were scaled for direct 640x480 image resizing; "
        "extrinsic matrices are unchanged."
    )
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_text(root: Path, excluded: set[Path]) -> str:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    )
    return "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    )


def resize_inference_episode(episode: Path) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix=".resize_640x480_", dir=episode))
    staged_root = temp_root / "staged"
    replacements: list[tuple[Path, Path]] = []
    try:
        for view in VIEWS:
            source = episode / "observations" / f"{view}.png"
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode image: {source}")
            resized = cv2.resize(
                image,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            staged = staged_root / "observations" / source.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(staged), resized):
                raise RuntimeError(f"failed to write image: {staged}")
            replacements.append((staged, source))

        staged_calibration = staged_root / "calibration.json"
        write_json(
            staged_calibration,
            resized_calibration(episode / "calibration.json"),
        )
        replacements.append((staged_calibration, episode / "calibration.json"))

        checksum_stage = temp_root / "checksum_view"
        shutil.copytree(episode, checksum_stage, ignore=shutil.ignore_patterns(
            temp_root.name,
        ))
        for staged, target in replacements:
            relative = target.relative_to(episode)
            destination = checksum_stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, destination)
        checksum_path = checksum_stage / "checksums.sha256"
        checksum_path.write_text(
            checksum_text(checksum_stage, {checksum_path}),
            encoding="utf-8",
        )
        staged_checksums = staged_root / "checksums.sha256"
        shutil.copy2(checksum_path, staged_checksums)
        replacements.append((staged_checksums, episode / "checksums.sha256"))
        atomic_replace_many(replacements, temp_root / "backup")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def episode_pairs(
    gt_root: Path,
    inference_root: Path,
    tasks: set[str] | None,
    episodes: set[str] | None,
) -> list[tuple[Path, Path]]:
    pairs = []
    for task_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        if tasks and task_dir.name not in tasks:
            continue
        for gt_episode in sorted(
            path
            for path in task_dir.iterdir()
            if path.is_dir() and path.name.startswith("ep")
        ):
            if episodes and gt_episode.name not in episodes:
                continue
            inference_episode = (
                inference_root / task_dir.name / gt_episode.name
            )
            if not inference_episode.is_dir():
                raise FileNotFoundError(inference_episode)
            pairs.append((gt_episode, inference_episode))
    return pairs


def validate_resized_pair(gt_episode: Path, inference_episode: Path) -> None:
    expected_count: int | None = None
    for view in VIEWS:
        frames = frame_paths(gt_episode / "video" / view)
        if image_size(frames[0]) != (TARGET_WIDTH, TARGET_HEIGHT):
            raise ValueError(f"{frames[0]}: not 640x480")
        expected_count = expected_count or len(frames)
        if len(frames) != expected_count:
            raise ValueError(f"{gt_episode}: inconsistent frame count")
        observation = inference_episode / "observations" / f"{view}.png"
        if image_size(observation) != (TARGET_WIDTH, TARGET_HEIGHT):
            raise ValueError(f"{observation}: not 640x480")
    for mask_path in [
        gt_episode / "prompt/masks.npz",
        *[
            gt_episode / "multiview_masks" / view / "masks.npz"
            for view in VIEWS
        ],
    ]:
        shape = npz_array_shape(mask_path)
        if shape[-2:] != (TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(f"{mask_path}: invalid shape {shape}")
        if shape[-3] != expected_count:
            raise ValueError(f"{mask_path}: frame count mismatch {shape}")
    calibration = read_json(inference_episode / "calibration.json")
    for view in VIEWS:
        resolution = calibration["cameras"][view]["resolution"]
        if resolution != {"height": TARGET_HEIGHT, "width": TARGET_WIDTH}:
            raise ValueError(f"{inference_episode}: invalid {view} resolution")


def rewrite_root_checksums(root: Path) -> None:
    target = root / "checksums.sha256"
    temporary = root / ".checksums.sha256.tmp"
    temporary.write_text(
        checksum_text(root, {target, temporary}),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def rewrite_inference_manifest(root: Path) -> None:
    path = root / "manifest.json"
    if not path.is_file():
        return
    manifest = read_json(path)
    episodes = []
    for task_dir in sorted(
        item
        for item in root.iterdir()
        if item.is_dir() and item.name != "shared"
    ):
        for episode_dir in sorted(task_dir.glob("ep[0-9][0-9][0-9][0-9][0-9][0-9]")):
            metadata = read_json(episode_dir / "metadata.json")
            episodes.append(
                {
                    "task": str(metadata["task_name"]),
                    "episode": str(metadata["episode_name"]),
                    "path": (
                        f"{metadata['task_name']}/{metadata['episode_name']}"
                    ),
                    "num_frames": int(metadata["num_frames"]),
                    "fps": float(metadata["fps"]),
                }
            )
    manifest["episodes"] = episodes
    manifest["tasks"] = sorted({item["task"] for item in episodes})
    temporary = path.with_name(".manifest.json.tmp")
    write_json(temporary, manifest)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=DEFAULT_INFERENCE_ROOT,
    )
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--update-point-annotations",
        action="store_true",
        help="Also scale JSON files referenced by masks_meta.json points_path.",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    gt_root = args.gt_root.expanduser().resolve()
    inference_root = args.inference_root.expanduser().resolve()
    pairs = episode_pairs(
        gt_root,
        inference_root,
        set(args.tasks) if args.tasks else None,
        set(args.episodes) if args.episodes else None,
    )
    if not pairs:
        raise SystemExit("No paired episodes selected")
    print(
        f"[resize] {len(pairs)} paired episode(s), target "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        flush=True,
    )
    if not args.apply:
        for gt_episode, _ in pairs:
            print(f"  would resize {gt_episode.relative_to(gt_root)}")
        print("[resize] preview only; pass --apply to modify data")
        return
    for index, (gt_episode, inference_episode) in enumerate(pairs, 1):
        label = gt_episode.relative_to(gt_root)
        print(f"[{index}/{len(pairs)}] {label}: GT", flush=True)
        resize_gt_episode(
            gt_episode,
            args.workers,
            args.update_point_annotations,
        )
        print(f"[{index}/{len(pairs)}] {label}: inference", flush=True)
        resize_inference_episode(inference_episode)
        validate_resized_pair(gt_episode, inference_episode)
        print(f"[{index}/{len(pairs)}] {label}: OK", flush=True)
    rewrite_inference_manifest(inference_root)
    print("[resize] rebuilt inference manifest", flush=True)
    rewrite_root_checksums(inference_root)
    print("[resize] updated inference root checksums", flush=True)
    print("[resize] complete", flush=True)


if __name__ == "__main__":
    main()
