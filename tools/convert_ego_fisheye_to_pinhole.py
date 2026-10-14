#!/usr/bin/env python3
"""Migrate paired Ego GT/inference datasets from fisheye to pinhole views.

Only ``top_head``, ``hand_left`` and ``hand_right`` are transformed.  Source
assets are retained inside each episode:

* GT frames: ``video/fisheye/<view>/``
* GT masks: ``masks_fisheye.npz`` and ``masks_meta_fisheye.json``
* inference observations: ``observations/fisheye/<view>.png``
* inference calibration: ``calibration_fisheye.json``

The replacement view is a 640x480, 90-degree-HFOV pinhole pitched downward
38 degrees.  An episode pair is built in sibling staging directories and then
swapped into place; the original live episode remains untouched until staging
and validation both succeed.  Re-running skips validated migrated episodes.

Examples:
    # Audit only.
    python tools/convert_ego_fisheye_to_pinhole.py

    # Pilot one episode.
    python tools/convert_ego_fisheye_to_pinhole.py \
      --tasks hangtowel --episodes 0 --apply \
      --preview-dir /path/to/preview-output

    # Migrate every paired episode.
    python tools/convert_ego_fisheye_to_pinhole.py --apply --update-root-metadata
"""

# pylint: disable=import-error,too-many-lines

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VIEWS = ("top_head", "hand_left", "hand_right")
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
HFOV_DEG = 90.0
PITCH_DEG = 38.0
GT_ROOT_DEFAULT = Path("/path/to/ego-ground-truth")
INFERENCE_ROOT_DEFAULT = Path(
    "/path/to/ego-inference-data"
)
MIGRATION_SCHEMA = "acwmbench.ego.pinhole_migration.v1"


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class ViewGeometry:
    camera: Camera
    new_K: np.ndarray
    rotation: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(episode: Path) -> None:
    target = episode / "checksums.sha256"
    lines = []
    for path in sorted(episode.rglob("*")):
        if not path.is_file() or path == target:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(episode).as_posix()}")
    temporary = target.with_suffix(".sha256.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def pinhole_for_fov(
    width: int, height: int, hfov_deg: float = HFOV_DEG
) -> np.ndarray:
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_x(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def project_fisheye(
    points: np.ndarray, K: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    a = points[:, 0] / points[:, 2]
    b = points[:, 1] / points[:, 2]
    radius = np.sqrt(a * a + b * b)
    theta = np.arctan(radius)
    theta2 = theta * theta
    theta_distorted = theta * (
        1
        + distortion[0] * theta2
        + distortion[1] * theta2**2
        + distortion[2] * theta2**3
        + distortion[3] * theta2**4
    )
    scale = np.where(
        radius > 1e-9,
        theta_distorted / np.maximum(radius, 1e-12),
        1.0,
    )
    return np.stack(
        [
            K[0, 0] * a * scale + K[0, 2],
            K[1, 1] * b * scale + K[1, 2],
        ],
        axis=-1,
    )


def unproject_fisheye(
    pixels: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
    iterations: int = 12,
) -> np.ndarray:
    a = (pixels[:, 0] - K[0, 2]) / K[0, 0]
    b = (pixels[:, 1] - K[1, 2]) / K[1, 1]
    theta_distorted = np.sqrt(a * a + b * b)
    theta = theta_distorted.copy()
    for _ in range(iterations):
        theta2 = theta * theta
        polynomial = (
            1
            + distortion[0] * theta2
            + distortion[1] * theta2**2
            + distortion[2] * theta2**3
            + distortion[3] * theta2**4
        )
        derivative = (
            1
            + 3 * distortion[0] * theta2
            + 5 * distortion[1] * theta2**2
            + 7 * distortion[2] * theta2**3
            + 9 * distortion[3] * theta2**4
        )
        theta -= (theta * polynomial - theta_distorted) / np.maximum(
            derivative, 1e-9
        )
    theta = np.clip(theta, 0.0, np.pi / 2 - 1e-3)
    radius = np.tan(theta)
    scale = np.where(
        theta_distorted > 1e-9,
        radius / np.maximum(theta_distorted, 1e-12),
        1.0,
    )
    return np.stack([a * scale, b * scale, np.ones_like(a)], axis=-1)


def rectify_maps(
    camera: Camera,
    new_K: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(TARGET_WIDTH, dtype=np.float64),
        np.arange(TARGET_HEIGHT, dtype=np.float64),
    )
    x = (u - new_K[0, 2]) / new_K[0, 0]
    y = (v - new_K[1, 2]) / new_K[1, 1]
    virtual_rays = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    source_rays = virtual_rays @ rotation
    pixels = project_fisheye(source_rays, camera.K, camera.D).reshape(
        TARGET_HEIGHT, TARGET_WIDTH, 2
    )
    return (
        pixels[..., 0].astype(np.float32),
        pixels[..., 1].astype(np.float32),
    )


def camera_from_calibration(calibration: dict[str, Any], view: str) -> Camera:
    camera_data = calibration.get("cameras", {}).get(view)
    if not isinstance(camera_data, dict):
        raise KeyError(f"calibration missing camera {view}")
    intrinsics = camera_data.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise KeyError(f"calibration camera {view} missing intrinsics")
    K = np.asarray(
        [
            [intrinsics["Fx"], 0.0, intrinsics["Cx"]],
            [0.0, intrinsics["Fy"], intrinsics["Cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [intrinsics.get(f"k{index}", 0.0) for index in range(1, 5)],
        dtype=np.float64,
    )
    return Camera(
        K=K,
        D=distortion,
        width=int(intrinsics["width"]),
        height=int(intrinsics["height"]),
    )


def all_geometries(calibration: dict[str, Any]) -> dict[str, ViewGeometry]:
    new_K = pinhole_for_fov(TARGET_WIDTH, TARGET_HEIGHT)
    rotation = rotation_x(PITCH_DEG)
    result = {}
    for view in VIEWS:
        camera = camera_from_calibration(calibration, view)
        map_x, map_y = rectify_maps(camera, new_K, rotation)
        result[view] = ViewGeometry(
            camera=camera,
            new_K=new_K.copy(),
            rotation=rotation.copy(),
            map_x=map_x,
            map_y=map_y,
        )
    return result


def frame_paths(directory: Path) -> list[Path]:
    frames = sorted(directory.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"{directory}: no frame_*.jpg")
    return frames


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image.shape[1], image.shape[0]


def remap_image(
    source: Path,
    destination: Path,
    geometry: ViewGeometry,
    *,
    png: bool = False,
) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {source}")
    if (image.shape[1], image.shape[0]) != (
        geometry.camera.width,
        geometry.camera.height,
    ):
        raise ValueError(
            f"{source}: {image.shape[1]}x{image.shape[0]} does not match "
            f"calibration {geometry.camera.width}x{geometry.camera.height}"
        )
    output = cv2.remap(
        image,
        geometry.map_x,
        geometry.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = (
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if png
        else [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not cv2.imwrite(str(destination), output, options):
        raise RuntimeError(f"failed to write image: {destination}")


def convert_frame_directory(
    source: Path,
    destination: Path,
    geometry: ViewGeometry,
    workers: int,
) -> int:
    sources = frame_paths(source)
    destination.mkdir(parents=True, exist_ok=True)

    def convert_one(path: Path) -> None:
        remap_image(path, destination / path.name, geometry)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(convert_one, sources))
    outputs = frame_paths(destination)
    if len(outputs) != len(sources):
        raise RuntimeError(
            f"{destination}: wrote {len(outputs)} of {len(sources)} frames"
        )
    for path in (outputs[0], outputs[-1]):
        if image_size(path) != (TARGET_WIDTH, TARGET_HEIGHT):
            raise RuntimeError(f"{path}: invalid output dimensions")
    return len(outputs)


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
            raise KeyError(f"{path}: missing {key!r}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape)


def extract_npz(
    source: Path, workspace: Path
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.endswith(".npy")
        }
        if "masks" not in members:
            raise KeyError(f"{source}: missing masks")
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


def write_npz_fast(
    destination: Path,
    masks: np.ndarray,
    others: dict[str, np.ndarray],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for key, array in [("masks", masks), *sorted(others.items())]:
            with archive.open(f"{key}.npy", "w", force_zip64=True) as handle:
                np.lib.format.write_array(
                    handle, np.asanyarray(array), allow_pickle=False
                )


def convert_masks(
    source: Path,
    destination: Path,
    workspace: Path,
    geometry: ViewGeometry,
    workers: int,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, float]]:
    masks, others = extract_npz(source, workspace)
    if masks.ndim not in (3, 4):
        raise ValueError(f"{source}: expected [T,H,W] or [N,T,H,W], got {masks.shape}")
    if tuple(masks.shape[-2:]) != (
        geometry.camera.height,
        geometry.camera.width,
    ):
        raise ValueError(
            f"{source}: mask size {masks.shape[-2:]} does not match calibration"
        )
    old_shape = tuple(int(value) for value in masks.shape)
    leading = old_shape[:-2]
    output_path = workspace / "pinhole_masks.npy"
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=masks.dtype,
        shape=(*leading, TARGET_HEIGHT, TARGET_WIDTH),
    )
    source_flat = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    output_flat = output.reshape((-1, TARGET_HEIGHT, TARGET_WIDTH))

    def convert_one(index: int) -> tuple[int, int]:
        source_mask = source_flat[index]
        converted = cv2.remap(
            source_mask,
            geometry.map_x,
            geometry.map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        output_flat[index] = converted
        return int(np.count_nonzero(source_mask)), int(np.count_nonzero(converted))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        areas = list(executor.map(convert_one, range(source_flat.shape[0])))
    output.flush()
    source_area = sum(value[0] for value in areas)
    output_area = sum(value[1] for value in areas)
    write_npz_fast(destination, output, others)
    new_shape = (*leading, TARGET_HEIGHT, TARGET_WIDTH)
    if npz_array_shape(destination) != new_shape:
        raise RuntimeError(f"{destination}: output mask shape validation failed")
    stats = {
        "source_nonzero": float(source_area),
        "pinhole_nonzero": float(output_area),
        "area_ratio": (
            float(output_area / source_area) if source_area else 1.0
        ),
    }
    del output_flat, output, source_flat, masks
    return old_shape, tuple(int(value) for value in new_shape), stats


def transform_points(
    points: Any,
    labels: Any,
    geometry: ViewGeometry,
) -> tuple[list[list[float]], list[Any], int]:
    if not isinstance(points, list):
        return [], [], 0
    valid_points = []
    valid_labels = []
    dropped = 0
    label_values = labels if isinstance(labels, list) else [1] * len(points)
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            dropped += 1
            continue
        source_pixel = np.asarray([[float(point[0]), float(point[1])]])
        source_ray = unproject_fisheye(
            source_pixel, geometry.camera.K, geometry.camera.D
        )[0]
        virtual_ray = geometry.rotation @ source_ray
        if virtual_ray[2] <= 1e-8:
            dropped += 1
            continue
        u = (
            geometry.new_K[0, 0] * virtual_ray[0] / virtual_ray[2]
            + geometry.new_K[0, 2]
        )
        v = (
            geometry.new_K[1, 1] * virtual_ray[1] / virtual_ray[2]
            + geometry.new_K[1, 2]
        )
        if not (0 <= u < TARGET_WIDTH and 0 <= v < TARGET_HEIGHT):
            dropped += 1
            continue
        valid_points.append([round(float(u), 4), round(float(v), 4)])
        valid_labels.append(label_values[index] if index < len(label_values) else 1)
    return valid_points, valid_labels, dropped


def updated_mask_metadata(
    source_meta: Path,
    final_gt_episode: Path,
    final_mask_dir: Path,
    new_shape: tuple[int, ...],
    geometry: ViewGeometry,
    area_stats: dict[str, float],
) -> dict[str, Any]:
    metadata = read_json(source_meta)
    metadata["shape"] = list(new_shape)
    metadata["resource_path"] = str(
        final_gt_episode / "video" / metadata.get("view", "top_head")
    )
    metadata["mask_path"] = str(final_mask_dir / "masks.npz")
    objects = metadata.get("objects")
    dropped_total = 0
    if isinstance(objects, dict):
        for record in objects.values():
            if not isinstance(record, dict):
                continue
            old_points = copy.deepcopy(record.get("points", []))
            old_labels = copy.deepcopy(record.get("labels", []))
            new_points, new_labels, dropped = transform_points(
                old_points, old_labels, geometry
            )
            record["source_fisheye_points"] = old_points
            record["points"] = new_points
            record["labels"] = new_labels
            dropped_total += dropped
    metadata["projection"] = {
        "model": "pinhole",
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "hfov_deg": HFOV_DEG,
        "view_pitch_deg": PITCH_DEG,
        "K": geometry.new_K.tolist(),
        "source_model": "opencv_fisheye",
        "source_to_virtual_rotation": geometry.rotation.tolist(),
        "rgb_interpolation": "linear",
        "mask_interpolation": "nearest",
        "dropped_annotation_points": dropped_total,
        **area_stats,
    }
    metadata["points_path"] = str(final_mask_dir / "points_pinhole.json")
    return metadata


def write_local_points(path: Path, metadata: dict[str, Any]) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "frame_width": TARGET_WIDTH,
            "frame_height": TARGET_HEIGHT,
            "projection": metadata["projection"],
            "objects": metadata.get("objects", {}),
        },
    )


def pinhole_calibration(
    source: dict[str, Any],
    geometries: dict[str, ViewGeometry],
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result["schema_version"] = "acwmbench.ego.calibration.pinhole.v1"
    result["source_calibration_file"] = "calibration_fisheye.json"
    for view in VIEWS:
        camera = result["cameras"][view]
        old_intrinsics = copy.deepcopy(camera["intrinsics"])
        geometry = geometries[view]
        camera["projection_model"] = "pinhole"
        camera["intrinsics"] = {
            "Fx": float(geometry.new_K[0, 0]),
            "Fy": float(geometry.new_K[1, 1]),
            "Cx": float(geometry.new_K[0, 2]),
            "Cy": float(geometry.new_K[1, 2]),
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
        }
        camera["resolution"] = {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
        }
        camera["rectification"] = {
            "source_projection_model": "opencv_fisheye",
            "source_intrinsics": old_intrinsics,
            "source_to_virtual_rotation": geometry.rotation.tolist(),
            "hfov_deg": HFOV_DEG,
            "view_pitch_deg": PITCH_DEG,
            "note": (
                "Physical extrinsics remain expressed in the source fisheye "
                "camera frame. Apply source_to_virtual_rotation when projecting "
                "physical controls into this virtual pinhole image."
            ),
        }
        camera["extrinsics_reference_frame"] = "source_fisheye_camera"
    result["active_views"] = {
        view: {
            "projection_model": "pinhole",
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "hfov_deg": HFOV_DEG,
            "view_pitch_deg": PITCH_DEG,
        }
        for view in VIEWS
    }
    return result


def migration_record(
    task: str,
    episode: str,
    mask_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "schema_version": MIGRATION_SCHEMA,
        "task": task,
        "episode": episode,
        "views": list(VIEWS),
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "hfov_deg": HFOV_DEG,
        "view_pitch_deg": PITCH_DEG,
        "controls_transformed": False,
        "controls_note": (
            "Physical controls and 57D actions are unchanged. Calibration "
            "contains the source-to-virtual rotation required for projection."
        ),
        "mask_stats": mask_stats,
        "created_unix": time.time(),
    }


def hardlink_copytree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def stage_pair(
    gt_episode: Path,
    inference_episode: Path,
    workers: int,
) -> tuple[Path, Path]:
    token = f"{os.getpid()}_{int(time.time() * 1000)}"
    gt_stage = gt_episode.parent / f".{gt_episode.name}.pinhole_stage_{token}"
    inference_stage = (
        inference_episode.parent
        / f".{inference_episode.name}.pinhole_stage_{token}"
    )
    try:
        hardlink_copytree(gt_episode, gt_stage)
        hardlink_copytree(inference_episode, inference_stage)
        source_calibration = read_json(
            inference_stage / "calibration.json"
        )
        geometries = all_geometries(source_calibration)

        fisheye_video = gt_stage / "video" / "fisheye"
        fisheye_video.mkdir(parents=True, exist_ok=True)
        frame_counts: dict[str, int] = {}
        for view in VIEWS:
            source = gt_stage / "video" / view
            backup = fisheye_video / view
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(source, backup)
            frame_counts[view] = convert_frame_directory(
                backup, source, geometries[view], workers
            )
        if len(set(frame_counts.values())) != 1:
            raise ValueError(
                f"{gt_episode}: view frame counts differ: {frame_counts}"
            )

        mask_specs = [
            (gt_stage / "prompt", "top_head", gt_episode / "prompt"),
            *[
                (
                    gt_stage / "multiview_masks" / view,
                    view,
                    gt_episode / "multiview_masks" / view,
                )
                for view in VIEWS
            ],
        ]
        mask_stats: dict[str, dict[str, float]] = {}
        for index, (mask_dir, view, final_mask_dir) in enumerate(mask_specs):
            source_mask = mask_dir / "masks.npz"
            backup_mask = mask_dir / "masks_fisheye.npz"
            source_meta = mask_dir / "masks_meta.json"
            backup_meta = mask_dir / "masks_meta_fisheye.json"
            os.replace(source_mask, backup_mask)
            os.replace(source_meta, backup_meta)
            workspace = gt_stage / ".pinhole_mask_workspace" / str(index)
            _, new_shape, stats = convert_masks(
                backup_mask,
                source_mask,
                workspace,
                geometries[view],
                workers,
            )
            if new_shape[-3] != frame_counts[view]:
                raise ValueError(
                    f"{mask_dir}: {new_shape[-3]} masks != "
                    f"{frame_counts[view]} frames"
                )
            metadata = updated_mask_metadata(
                backup_meta,
                gt_episode,
                final_mask_dir,
                new_shape,
                geometries[view],
                stats,
            )
            atomic_json(source_meta, metadata)
            write_local_points(mask_dir / "points_pinhole.json", metadata)
            mask_stats[str(final_mask_dir.relative_to(gt_episode))] = stats
            shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(
            gt_stage / ".pinhole_mask_workspace", ignore_errors=True
        )

        observation_backup = inference_stage / "observations" / "fisheye"
        observation_backup.mkdir(parents=True, exist_ok=True)
        for view in VIEWS:
            source = inference_stage / "observations" / f"{view}.png"
            backup = observation_backup / f"{view}.png"
            os.replace(source, backup)
            remap_image(backup, source, geometries[view], png=True)

        calibration_path = inference_stage / "calibration.json"
        calibration_backup = inference_stage / "calibration_fisheye.json"
        os.replace(calibration_path, calibration_backup)
        atomic_json(
            calibration_path,
            pinhole_calibration(source_calibration, geometries),
        )
        metadata_path = inference_stage / "metadata.json"
        metadata = read_json(metadata_path)
        files = metadata.setdefault("files", {})
        files["fisheye_calibration"] = "calibration_fisheye.json"
        files["fisheye_observations"] = {
            view: f"observations/fisheye/{view}.png" for view in VIEWS
        }
        metadata["active_projection"] = {
            "views": list(VIEWS),
            "model": "pinhole",
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "hfov_deg": HFOV_DEG,
            "view_pitch_deg": PITCH_DEG,
        }
        atomic_json(metadata_path, metadata)

        record = migration_record(
            gt_episode.parent.name, gt_episode.name, mask_stats
        )
        atomic_json(gt_stage / "pinhole_migration.json", record)
        atomic_json(inference_stage / "pinhole_migration.json", record)
        write_checksums(inference_stage)
        validate_pair(gt_stage, inference_stage, expect_migrated=True)
        return gt_stage, inference_stage
    except BaseException:
        shutil.rmtree(gt_stage, ignore_errors=True)
        shutil.rmtree(inference_stage, ignore_errors=True)
        raise


def swap_pair(
    gt_live: Path,
    inference_live: Path,
    gt_stage: Path,
    inference_stage: Path,
) -> None:
    token = f"{os.getpid()}_{int(time.time() * 1000)}"
    gt_old = gt_live.parent / f".{gt_live.name}.pre_pinhole_{token}"
    inference_old = (
        inference_live.parent / f".{inference_live.name}.pre_pinhole_{token}"
    )
    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        os.replace(gt_live, gt_old)
        moved.append((gt_old, gt_live))
        os.replace(inference_live, inference_old)
        moved.append((inference_old, inference_live))
        os.replace(gt_stage, gt_live)
        installed.append(gt_live)
        os.replace(inference_stage, inference_live)
        installed.append(inference_live)
    except BaseException:
        for path in reversed(installed):
            shutil.rmtree(path, ignore_errors=True)
        for old, live in reversed(moved):
            if old.exists() and not live.exists():
                os.replace(old, live)
        raise
    shutil.rmtree(gt_old)
    shutil.rmtree(inference_old)


def migrated(episode: Path) -> bool:
    marker = episode / "pinhole_migration.json"
    if not marker.is_file():
        return False
    try:
        return read_json(marker).get("schema_version") == MIGRATION_SCHEMA
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def validate_pair(
    gt_episode: Path,
    inference_episode: Path,
    *,
    expect_migrated: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"frames": {}, "masks": {}}
    calibration_name = (
        "calibration_fisheye.json" if expect_migrated else "calibration.json"
    )
    calibration = read_json(inference_episode / calibration_name)
    expected_frames = None
    for view in VIEWS:
        active = frame_paths(gt_episode / "video" / view)
        result["frames"][view] = len(active)
        if expected_frames is None:
            expected_frames = len(active)
        elif expected_frames != len(active):
            raise ValueError(f"{gt_episode}: frame counts differ")
        if expect_migrated:
            backup = frame_paths(gt_episode / "video" / "fisheye" / view)
            if len(backup) != len(active):
                raise ValueError(f"{gt_episode}: {view} backup count differs")
            if image_size(active[0]) != (TARGET_WIDTH, TARGET_HEIGHT):
                raise ValueError(f"{active[0]}: not 640x480")
            camera = camera_from_calibration(calibration, view)
            if image_size(backup[0]) != (camera.width, camera.height):
                raise ValueError(f"{backup[0]}: source dimensions differ")
            active_observation = (
                inference_episode / "observations" / f"{view}.png"
            )
            backup_observation = (
                inference_episode
                / "observations"
                / "fisheye"
                / f"{view}.png"
            )
            if image_size(active_observation) != (
                TARGET_WIDTH,
                TARGET_HEIGHT,
            ):
                raise ValueError(f"{active_observation}: not 640x480")
            if image_size(backup_observation) != (
                camera.width,
                camera.height,
            ):
                raise ValueError(f"{backup_observation}: invalid backup")

    mask_dirs = [
        gt_episode / "prompt",
        *[gt_episode / "multiview_masks" / view for view in VIEWS],
    ]
    for mask_dir in mask_dirs:
        active_shape = npz_array_shape(mask_dir / "masks.npz")
        result["masks"][str(mask_dir.relative_to(gt_episode))] = active_shape
        if expect_migrated:
            backup_shape = npz_array_shape(mask_dir / "masks_fisheye.npz")
            if active_shape[-2:] != (TARGET_HEIGHT, TARGET_WIDTH):
                raise ValueError(f"{mask_dir}: active mask is not 480x640")
            if active_shape[:-2] != backup_shape[:-2]:
                raise ValueError(f"{mask_dir}: mask leading dimensions changed")
            if active_shape[-3] != expected_frames:
                raise ValueError(f"{mask_dir}: mask/frame count mismatch")
            metadata = read_json(mask_dir / "masks_meta.json")
            if tuple(metadata.get("shape", [])) != active_shape:
                raise ValueError(f"{mask_dir}: metadata shape mismatch")
            if not (mask_dir / "masks_meta_fisheye.json").is_file():
                raise FileNotFoundError(mask_dir / "masks_meta_fisheye.json")
    if expect_migrated:
        active_calibration = read_json(
            inference_episode / "calibration.json"
        )
        for view in VIEWS:
            camera = active_calibration["cameras"][view]
            if camera.get("projection_model") != "pinhole":
                raise ValueError(f"{view}: calibration is not pinhole")
            if camera.get("resolution") != {
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
            }:
                raise ValueError(f"{view}: invalid pinhole resolution")
        if not migrated(gt_episode) or not migrated(inference_episode):
            raise ValueError("migration marker missing")
    return result


def make_preview(
    gt_episode: Path,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_masks, _ = extract_npz(
        gt_episode / "prompt" / "masks.npz",
        output_dir / ".preview_mask_workspace",
    )
    mask_frames = prompt_masks.shape[-3]
    frame_numbers = sorted(
        {0, max(0, mask_frames // 2), max(0, mask_frames - 1)}
    )
    outputs = []
    for view in VIEWS:
        panels = []
        for frame_number in frame_numbers:
            source = cv2.imread(
                str(
                    gt_episode
                    / "video"
                    / "fisheye"
                    / view
                    / f"frame_{frame_number:05d}.jpg"
                )
            )
            pinhole = cv2.imread(
                str(
                    gt_episode
                    / "video"
                    / view
                    / f"frame_{frame_number:05d}.jpg"
                )
            )
            if source is None or pinhole is None:
                raise RuntimeError(f"preview frame {view}/{frame_number} missing")
            source = cv2.resize(source, (TARGET_WIDTH, TARGET_HEIGHT))
            overlay = pinhole.copy()
            if view == "top_head":
                frame_masks = (
                    prompt_masks[:, frame_number]
                    if prompt_masks.ndim == 4
                    else prompt_masks[frame_number][None]
                )
                union = np.any(frame_masks != 0, axis=0)
                color = np.zeros_like(overlay)
                color[..., 1] = 255
                overlay[union] = (
                    0.55 * overlay[union] + 0.45 * color[union]
                ).astype(np.uint8)
            label = f"frame {frame_number}"
            for image, suffix in (
                (source, f"{label} fisheye"),
                (pinhole, f"{label} pinhole"),
                (overlay, f"{label} mask"),
            ):
                cv2.putText(
                    image,
                    suffix,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            panels.append(np.hstack([source, pinhole, overlay]))
        contact = np.vstack(panels)
        destination = output_dir / f"{view}.jpg"
        cv2.imwrite(
            str(destination),
            contact,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        outputs.append(destination)
    del prompt_masks
    shutil.rmtree(output_dir / ".preview_mask_workspace", ignore_errors=True)
    return outputs


def discover_pairs(
    gt_root: Path,
    inference_root: Path,
    tasks: list[str] | None,
    episodes: list[int] | None,
) -> list[tuple[Path, Path]]:
    selected_episodes = set(episodes or [])
    pairs = []
    for gt_episode in sorted(gt_root.glob("*/ep*")):
        task = gt_episode.parent.name
        if tasks and task not in tasks:
            continue
        try:
            episode_index = int(gt_episode.name.removeprefix("ep"))
        except ValueError:
            continue
        if selected_episodes and episode_index not in selected_episodes:
            continue
        inference_episode = inference_root / task / gt_episode.name
        if not inference_episode.is_dir():
            raise FileNotFoundError(
                f"missing inference pair for {task}/{gt_episode.name}"
            )
        pairs.append((gt_episode, inference_episode))
    return pairs


def update_root_metadata(gt_root: Path, inference_root: Path) -> None:
    projection = {
        "model": "pinhole",
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "hfov_deg": HFOV_DEG,
        "view_pitch_deg": PITCH_DEG,
        "source_model": "opencv_fisheye",
        "source_backup": {
            "gt": "video/fisheye/<view>",
            "inference": "observations/fisheye/<view>.png",
        },
    }
    for root in (gt_root, inference_root):
        manifest_path = root / "dataset_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            task_counts: dict[str, dict[str, int]] = {}
            total_frames = 0
            total_episodes = 0
            for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                episodes = sorted(task_dir.glob("ep*"))
                if not episodes:
                    continue
                frames = 0
                for episode in episodes:
                    if root == gt_root:
                        frames += len(frame_paths(episode / "video" / "top_head"))
                    else:
                        frames += int(read_json(episode / "metadata.json")["num_frames"])
                task_counts[task_dir.name] = {
                    "episodes": len(episodes),
                    "frames": frames,
                }
                total_episodes += len(episodes)
                total_frames += frames
            manifest["total_episodes"] = total_episodes
            manifest["total_frames"] = total_frames
            for task, counts in task_counts.items():
                manifest.setdefault("tasks", {}).setdefault(task, {}).update(counts)
            manifest["active_projection"] = projection
            manifest["views"].update(
                {
                    view: {
                        "source_camera": manifest["views"].get(view),
                        **projection,
                    }
                    for view in VIEWS
                }
            )
            atomic_json(manifest_path, manifest)

    gt_readme = (
        "# Ego-human ACWMBench ground truth dataset\n\n"
        "Five synchronized head-mounted views sampled at 15 FPS. The active "
        "`top_head`, `hand_left`, and `hand_right` views are rectified 640x480 "
        "pinhole images (90 degree HFOV, 38 degree downward pitch). Their "
        "original fisheye frames are preserved under `video/fisheye/`. "
        "`hand_left` and `hand_right` are head-mounted side cameras, not wrist "
        "cameras.\n"
    )
    (gt_root / "README.md").write_text(gt_readme, encoding="utf-8")
    inference_readme = (
        "# Ego-human ACWMBench inference dataset\n\n"
        "Canonical 15 FPS bimanual human-hand trajectories. The active "
        "`top_head`, `hand_left`, and `hand_right` observations are rectified "
        "640x480 pinhole images (90 degree HFOV, 38 degree downward pitch). "
        "Original images and calibration are retained in `observations/fisheye/` "
        "and `calibration_fisheye.json`. Physical controls remain unchanged; "
        "read `calibration.json` for the source-to-virtual rotation.\n"
    )
    (inference_root / "README.md").write_text(
        inference_readme, encoding="utf-8"
    )
    schema_paths = [
        inference_root / "schema.json",
        inference_root / "shared" / "schema.json",
        *sorted(inference_root.glob("*/schema.json")),
    ]
    for path in schema_paths:
        if not path.is_file():
            continue
        schema = read_json(path)
        views = schema.get("views")
        if isinstance(views, dict):
            for view in VIEWS:
                if isinstance(views.get(view), dict):
                    views[view]["active_projection"] = projection
                    views[view]["source_projection"] = "opencv_fisheye"
        schema["active_projection"] = projection
        atomic_json(path, schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT_DEFAULT)
    parser.add_argument(
        "--inference-root", type=Path, default=INFERENCE_ROOT_DEFAULT
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--update-root-metadata", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = args.gt_root.resolve()
    inference_root = args.inference_root.resolve()
    pairs = discover_pairs(
        gt_root,
        inference_root,
        args.tasks,
        args.episodes,
    )
    if not pairs:
        raise SystemExit("no paired episodes selected")
    print(
        f"[ego-pinhole] selected={len(pairs)} apply={args.apply} "
        f"geometry={TARGET_WIDTH}x{TARGET_HEIGHT} hfov={HFOV_DEG:g} "
        f"pitch={PITCH_DEG:g}",
        flush=True,
    )
    if not args.apply and not args.validate_only:
        states = {
            "migrated": sum(
                migrated(gt) and migrated(inference)
                for gt, inference in pairs
            ),
            "pending": sum(
                not (migrated(gt) and migrated(inference))
                for gt, inference in pairs
            ),
        }
        print(f"[ego-pinhole] audit: {states}", flush=True)
        return

    completed = skipped = failed = 0
    for index, (gt_episode, inference_episode) in enumerate(pairs, 1):
        name = f"{gt_episode.parent.name}/{gt_episode.name}"
        try:
            if args.validate_only:
                validate_pair(
                    gt_episode,
                    inference_episode,
                    expect_migrated=True,
                )
                completed += 1
                print(f"[{index}/{len(pairs)}] {name}: valid", flush=True)
                continue
            if (
                migrated(gt_episode)
                and migrated(inference_episode)
                and not args.overwrite
            ):
                validate_pair(
                    gt_episode,
                    inference_episode,
                    expect_migrated=True,
                )
                skipped += 1
                print(f"[{index}/{len(pairs)}] {name}: already migrated", flush=True)
                continue
            if migrated(gt_episode) or migrated(inference_episode):
                raise RuntimeError(
                    f"{name}: paired migration state differs; manual recovery required"
                )
            validate_pair(
                gt_episode,
                inference_episode,
                expect_migrated=False,
            )
            started = time.monotonic()
            gt_stage, inference_stage = stage_pair(
                gt_episode,
                inference_episode,
                max(1, args.workers),
            )
            swap_pair(
                gt_episode,
                inference_episode,
                gt_stage,
                inference_stage,
            )
            validate_pair(
                gt_episode,
                inference_episode,
                expect_migrated=True,
            )
            if args.preview_dir is not None:
                make_preview(
                    gt_episode,
                    args.preview_dir / gt_episode.parent.name / gt_episode.name,
                )
            completed += 1
            print(
                f"[{index}/{len(pairs)}] {name}: migrated in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(
                f"[{index}/{len(pairs)}] {name}: FAILED "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if len(pairs) == 1:
                raise
    if args.update_root_metadata and failed == 0:
        update_root_metadata(gt_root, inference_root)
    print(
        f"[ego-pinhole] done: completed={completed} skipped={skipped} "
        f"failed={failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
