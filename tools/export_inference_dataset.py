#!/usr/bin/env python3
"""Export a portable, self-contained ACWMBench inference dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "acwmbench.inference.v1"
DEFAULT_SELECTION_ROOT = Path("/path/to/selected-episodes")
DEFAULT_OUTPUT_ROOT = Path("/path/to/inference-output")
CAMERAS = ("top_head", "hand_left", "hand_right")
CAMERA_FEATURES = {name: f"observation.images.{name}" for name in CAMERAS}
CAMERA_CALIBRATION_STEMS = {
    "top_head": ("head_front_rgb", "head_front_rgbd"),
    "hand_left": ("hand_left_rgb", "hand_left_rgbd"),
    "hand_right": ("hand_right_rgb", "hand_right_rgbd"),
}
PUBLIC_TRAJECTORY_COLUMNS = (
    "frame_index",
    "timestamp",
    "action",
    "observation.state",
)
FORBIDDEN_SUBSTRINGS = (
    "/home/",
    "/mnt/",
    "eval_dataset",
    "source_root",
    "source_video",
    "selection",
    "lerobot",
    "internal source",
)
ACTION_WIDTH = 52
STATE_WIDTH = 208


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source_root: Path
    dataset_id: int = 0
    task_id: int = 0


DEFAULT_TASKS = {
    "pickfruit": TaskSpec(
        "pickfruit", Path("/path/to/lerobot/pickfruit")
    ),
    "pourwater": TaskSpec(
        "pourwater", Path("/path/to/lerobot/pourwater")
    ),
    "hangtowel": TaskSpec(
        "hangtowel", Path("/path/to/lerobot/hangtowel")
    ),
    "pulldrawer": TaskSpec(
        "pulldrawer", Path("/path/to/lerobot/pulldrawer")
    ),
}


@dataclass(frozen=True)
class Selection:
    task: TaskSpec
    episode_name: str
    episode_index: int
    start: int
    end: int
    slice_path: Path

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def public_episode_index(self) -> int:
        return int(self.episode_name[2:])

    @property
    def source_episode_index(self) -> int:
        return self.episode_index


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum_file(
    root: Path, relative_paths: Iterable[Path], output: Path
) -> None:
    lines = []
    for relative in sorted(set(relative_paths), key=lambda item: item.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path: {relative}")
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        lines.append(f"{_sha256(target)}  {relative.as_posix()}")
    if not lines:
        raise ValueError("cannot write an empty checksum manifest")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_checksum_file(path: Path, root: Path) -> set[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty checksum manifest")
    covered = set()
    for number, line in enumerate(lines, 1):
        try:
            digest, relative_text = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: malformed checksum") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}:{number}: invalid SHA-256")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{path}:{number}: unsafe path")
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"{path}: missing {relative}")
        if _sha256(target) != digest:
            raise ValueError(f"{path}: checksum mismatch for {relative}")
        if relative in covered:
            raise ValueError(f"{path}: duplicate checksum path {relative}")
        covered.add(relative)
    return covered


def _slice_bounds(data: Mapping[str, Any], path: Path) -> tuple[int, int]:
    start_value = data.get("start_frame", data.get("start_frame_inclusive"))
    end_value = data.get("end_frame", data.get("end_frame_exclusive"))
    if start_value is None or end_value is None:
        raise KeyError(f"{path}: expected start_frame and end_frame")
    start, end = int(start_value), int(end_value)
    if start < 0 or end <= start:
        raise ValueError(f"{path}: invalid half-open range [{start}, {end})")
    return start, end


def discover_selections(
    selection_root: Path, tasks: dict[str, TaskSpec], limit: int | None
) -> list[Selection]:
    result = []
    for task_name in sorted(tasks):
        task_dir = selection_root / task_name
        if not task_dir.is_dir():
            raise FileNotFoundError(task_dir)
        for episode_dir in sorted(task_dir.glob("ep[0-9][0-9][0-9][0-9][0-9][0-9]")):
            data_path = episode_dir / "slice.json"
            data = _json_load(data_path)
            start, end = _slice_bounds(data, data_path)
            private_index = data.get("source_episode_index", data.get("episode_index"))
            if private_index is None:
                private_index = int(episode_dir.name[2:])
            result.append(
                Selection(
                    tasks[task_name],
                    episode_dir.name,
                    int(private_index),
                    start,
                    end,
                    data_path,
                )
            )
            if limit is not None and len(result) >= limit:
                return result
    if not result:
        raise RuntimeError(f"no episodes found under {selection_root}")
    return result


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required") from exc
    return pa, pc, pq


def _validate_source_info(info: Mapping[str, Any], path: Path) -> None:
    try:
        fps = float(info["fps"])
        features = info["features"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid fps/features") from exc
    if fps <= 0:
        raise ValueError(f"{path}: fps must be positive")
    for name, width in (("action", ACTION_WIDTH), ("observation.state", STATE_WIDTH)):
        shape = features.get(name, {}).get("shape")
        if shape != [width]:
            raise ValueError(f"{path}: {name} expected [{width}], got {shape}")
    missing = [
        CAMERA_FEATURES[name]
        for name in CAMERAS
        if CAMERA_FEATURES[name] not in features
    ]
    if missing:
        raise ValueError(f"{path}: missing camera features {missing}")


def _read_episode_metadata(task: TaskSpec, episode_index: int) -> dict[str, Any]:
    _, pc, pq = _import_pyarrow()
    paths = sorted((task.source_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(task.source_root / "meta" / "episodes")
    for path in paths:
        table = pq.read_table(path)
        if "episode_index" not in table.column_names:
            continue
        rows = table.filter(pc.equal(table["episode_index"], episode_index))
        if rows.num_rows == 1:
            return rows.to_pylist()[0]
        if rows.num_rows > 1:
            raise ValueError(f"{path}: duplicate episode metadata")
    raise KeyError(f"{task.name}: episode {episode_index} not found")


def _first(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    raise KeyError(f"missing fields {names}")


def _source_parquet(task: TaskSpec, metadata: Mapping[str, Any]) -> Path:
    chunk = int(_first(metadata, "data/chunk_index", "data_chunk_index"))
    file_index = int(_first(metadata, "data/file_index", "data_file_index"))
    return task.source_root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"


def _matrix(column: Any, width: int, name: str, path: Path):
    import numpy as np

    try:
        values = np.stack(
            [np.asarray(row, dtype=np.float32) for row in column.to_pylist()]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} is not a dense vector") from exc
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{path}: {name} must have width {width}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: {name} contains non-finite values")
    return values


def _public_trajectory(
    selected: Selection, metadata: Mapping[str, Any], fps: float
):
    import numpy as np

    pa, pc, pq = _import_pyarrow()
    input_path = _source_parquet(selected.task, metadata)
    required = {"episode_index", "frame_index", "action", "observation.state"}
    if missing := sorted(required - set(pq.read_schema(input_path).names)):
        raise ValueError(f"{input_path}: missing columns {missing}")
    table = pq.read_table(input_path, columns=sorted(required))
    table = table.filter(pc.equal(table["episode_index"], selected.episode_index))
    table = table.take(pc.sort_indices(table, sort_keys=[("frame_index", "ascending")]))
    if selected.end > table.num_rows:
        raise ValueError(f"{selected.slice_path}: crop exceeds episode")
    crop = table.slice(selected.start, selected.length)
    if crop["frame_index"].to_pylist() != list(range(selected.start, selected.end)):
        raise ValueError(f"{input_path}: frame timeline is not contiguous")
    actions = _matrix(crop["action"], ACTION_WIDTH, "action", input_path)
    states = _matrix(crop["observation.state"], STATE_WIDTH, "observation.state", input_path)
    return pa.table(
        {
            "frame_index": pa.array(range(selected.length), type=pa.int64()),
            "timestamp": pa.array(
                np.arange(selected.length, dtype=np.float32) / np.float32(fps),
                type=pa.float32(),
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(actions.reshape(-1), type=pa.float32()), ACTION_WIDTH
            ),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(states.reshape(-1), type=pa.float32()), STATE_WIDTH
            ),
        }
    ).replace_schema_metadata(None)


def _video_location(
    task: TaskSpec, metadata: Mapping[str, Any], camera: str, fps: float
) -> tuple[Path, int]:
    feature = CAMERA_FEATURES[camera]
    chunk = int(metadata[f"videos/{feature}/chunk_index"])
    file_index = int(metadata[f"videos/{feature}/file_index"])
    offset = float(metadata.get(f"videos/{feature}/from_timestamp", 0.0) or 0.0)
    path = (
        task.source_root
        / "videos"
        / feature
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    return path, int(round(offset * fps))


def _decode_rgb(path: Path, frame_index: int):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise RuntimeError(f"cannot decode frame {frame_index} from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_rgb_png(rgb: Any, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required") from exc
    if getattr(rgb, "ndim", None) != 3 or rgb.shape[2] != 3:
        raise ValueError("decoded image must be HxWx3 RGB")
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", optimize=False)


def _calibration_block(task: TaskSpec, episode_index: int) -> Mapping[str, Any]:
    data = _json_load(task.source_root / "meta" / "agibot" / "camera_params.json")
    key = str(episode_index)
    if key not in data:
        key = "0" if "0" in data else sorted(data)[0]
    result = data[key]
    if not isinstance(result, Mapping):
        raise TypeError("camera calibration entry must be an object")
    return result


def _portable_calibration(task: TaskSpec, episode_index: int) -> dict[str, Any]:
    block = _calibration_block(task, episode_index)
    resolutions = block.get("camera_resolution", {})
    cameras = {}
    for camera, (rgb, rgbd) in CAMERA_CALIBRATION_STEMS.items():
        intrinsic = block.get(f"intrinsic_{rgb}", block.get(f"{rgb}_intrinsic"))
        if intrinsic is None:
            raise KeyError(f"missing {camera} intrinsics")
        transform_values = [
            value
            for key, value in sorted(block.items())
            if "extrinsic" in key.lower() and rgbd in key.lower()
        ]
        transforms = [
            {"name": f"transform_{index:02d}", "matrix": value}
            for index, value in enumerate(transform_values)
        ]
        cameras[camera] = {
            "intrinsics": intrinsic,
            "resolution": resolutions.get(rgb),
            "transforms": transforms,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_convention": "Matrices retain the robot calibration convention.",
        "cameras": cameras,
    }
    _reject_nonportable(result, "calibration")
    _reject_calibration_identity_keys(result)
    return result


def _strip_urdf(input_path: Path, output_path: Path) -> dict[str, Any]:
    root = ET.parse(input_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{input_path}: expected robot root")
    robot = ET.Element("robot", {"name": "g02"})
    links = []
    for link in root.findall("link"):
        if not (name := link.get("name")):
            raise ValueError(f"{input_path}: unnamed link")
        target = ET.SubElement(robot, "link", {"name": name})
        inertial = link.find("inertial")
        if inertial is not None:
            target.append(ET.fromstring(ET.tostring(inertial)))
        links.append(name)
    joints, movable, children = [], [], set()
    allowed = {
        "parent",
        "child",
        "origin",
        "axis",
        "limit",
        "dynamics",
        "calibration",
        "mimic",
        "safety_controller",
    }
    for joint in root.findall("joint"):
        name, kind = joint.get("name"), joint.get("type")
        if not name or not kind:
            raise ValueError(f"{input_path}: invalid joint")
        target = ET.SubElement(robot, "joint", {"name": name, "type": kind})
        for child in joint:
            if child.tag in allowed:
                target.append(ET.fromstring(ET.tostring(child)))
        child_element = target.find("child")
        if target.find("parent") is None or child_element is None:
            raise ValueError(f"{input_path}: joint {name} lacks parent/child")
        children.add(str(child_element.get("link")))
        joints.append(name)
        if kind != "fixed":
            movable.append(name)
            child_link = robot.find(f"./link[@name='{child_element.get('link')}']")
            inertial = child_link.find("inertial") if child_link is not None else None
            mass = inertial.find("mass") if inertial is not None else None
            inertia = inertial.find("inertia") if inertial is not None else None
            try:
                mass_value = float(mass.get("value", "")) if mass is not None else 0.0
                diagonal = (
                    float(inertia.get("ixx", "")),
                    float(inertia.get("iyy", "")),
                    float(inertia.get("izz", "")),
                ) if inertia is not None else (0.0, 0.0, 0.0)
            except ValueError as exc:
                raise ValueError(
                    f"{input_path}: invalid inertial values for movable link "
                    f"{child_element.get('link')}"
                ) from exc
            if mass_value <= 0 or any(value <= 0 for value in diagonal):
                raise ValueError(
                    f"{input_path}: movable link {child_element.get('link')} "
                    "requires positive mass and diagonal inertia"
                )
    ET.indent(robot, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    text = output_path.read_text(encoding="utf-8")
    if any(token in text.lower() for token in ("<mesh", "filename=", "package://")):
        raise ValueError("public URDF has a resource reference")
    _reject_text(text, output_path)
    for element in robot.iter():
        _reject_nonportable(element.attrib, f"{output_path}:{element.tag}")
    return {
        "schema_version": SCHEMA_VERSION,
        "robot": "g02",
        "urdf": "kinematics.urdf",
        "link_count": len(links),
        "joint_count": len(joints),
        "links": links,
        "joints": joints,
        "movable_joints": movable,
        "base_links": [name for name in links if name not in children],
    }


def _reject_text(text: str, location: Any) -> None:
    lowered = text.lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in lowered:
            raise ValueError(f"{location}: forbidden text {forbidden!r}")


def _looks_absolute(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("\\\\")
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or value.lower().startswith("file://")
    )


def _reject_absolute_paths_in_text(text: str, location: Any) -> None:
    windows = re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s`\"']+", text)
    posix = re.search(
        r"(?<![A-Za-z0-9:])/(?:[A-Za-z0-9._~-]+/)+[A-Za-z0-9._~-]+",
        text,
    )
    if windows or posix:
        match = windows or posix
        raise ValueError(f"{location}: absolute path text {match.group(0)!r}")


def _reject_nonportable(value: Any, location: str = "value") -> None:
    if isinstance(value, str):
        _reject_text(value, location)
        if _looks_absolute(value):
            raise ValueError(f"{location}: absolute path string is not portable")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonportable(str(key), f"{location}.<key>")
            _reject_nonportable(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonportable(child, f"{location}[{index}]")


def _reject_calibration_identity_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if "source" in lowered or "session" in lowered:
                raise ValueError(f"calibration: private identity key {key!r}")
            _reject_calibration_identity_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_calibration_identity_keys(child)


def _descriptor(feature: Mapping[str, Any], width: int) -> dict[str, Any]:
    result: dict[str, Any] = {"dtype": "float32", "shape": [width]}
    descriptions = feature.get("field_descriptions")
    if isinstance(descriptions, Mapping):
        # FK/action adapters consume this established mapping directly.  Keep
        # its portable semantic labels and indices, but never source metadata.
        result["field_descriptions"] = {
            str(key): value
            for key, value in sorted(
                descriptions.items(), key=lambda pair: str(pair[0])
            )
        }
        _reject_nonportable(result["field_descriptions"], "feature descriptor")
    return result


def _dataset_card() -> str:
    return """# ACWMBench Inference Dataset

This self-contained package provides initial RGB observations and aligned
action/state trajectories for action-conditioned world-model evaluation.

Only the initial observation is included. No future RGB frames are provided;
predicting future visual outcomes is the responsibility of each model adapter.
Every episode starts at frame zero, with timestamp equal to `frame_index / fps`.
Action and state vectors retain their recorded values.

Use `manifest.json` for episode discovery, the JSON schemas for structure, and
`shared/g02/kinematics.json` for the mesh-free robot description. SHA-256
manifests support integrity checks.
"""


def _episode_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ACWMBench episode metadata",
        "type": "object",
        "required": [
            "schema_version",
            "episode_name",
            "episode_index",
            "task_name",
            "num_frames",
            "num_transitions",
            "seed_local_index",
            "fps",
            "files",
        ],
        "additionalProperties": False,
    }


def _root_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ACWMBench inference dataset",
        "type": "object",
        "required": ["schema_version", "tasks", "episodes", "kinematics"],
        "additionalProperties": False,
    }


def _portable_task_schema(
    info: Mapping[str, Any], fps: float
) -> dict[str, Any]:
    features = info["features"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "timeline": {
            "fps": float(fps),
            "frame_index": {"dtype": "int64", "origin": 0, "step": 1},
            "timestamp": {
                "dtype": "float32",
                "definition": "frame_index / fps",
            },
        },
        "features": {
            "action": _descriptor(features["action"], ACTION_WIDTH),
            "observation.state": _descriptor(
                features["observation.state"], STATE_WIDTH
            ),
        },
        "observations": {
            camera: {
                "dtype": "uint8",
                "encoding": "png",
                "color_space": "RGB",
                "availability": "initial frame only",
            }
            for camera in CAMERAS
        },
    }
    _reject_nonportable(result, "task schema")
    return result


def _validate_trajectory(path: Path, length: int, fps: float) -> None:
    import numpy as np

    pa, _, pq = _import_pyarrow()
    table = pq.read_table(path)
    if table.column_names != list(PUBLIC_TRAJECTORY_COLUMNS):
        raise ValueError(f"{path}: unexpected columns")
    if table.num_rows != length:
        raise ValueError(f"{path}: invalid row count")
    if not pa.types.is_int64(table.schema.field("frame_index").type):
        raise ValueError(f"{path}: frame_index must be int64")
    if not pa.types.is_float32(table.schema.field("timestamp").type):
        raise ValueError(f"{path}: timestamp must be float32")
    for name, width in (("action", ACTION_WIDTH), ("observation.state", STATE_WIDTH)):
        kind = table.schema.field(name).type
        if (
            not pa.types.is_fixed_size_list(kind)
            or kind.list_size != width
            or not pa.types.is_float32(kind.value_type)
        ):
            raise ValueError(f"{path}: {name} must be float32[{width}]")
    if table["frame_index"].to_pylist() != list(range(length)):
        raise ValueError(f"{path}: non-local frame_index")
    actual = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    expected = np.arange(length, dtype=np.float32) / np.float32(fps)
    if not np.array_equal(actual, expected):
        raise ValueError(f"{path}: invalid timestamp")


def _episode_files() -> set[Path]:
    return {
        Path("metadata.json"),
        Path("trajectory.parquet"),
        Path("observations/top_head.png"),
        Path("observations/hand_left.png"),
        Path("observations/hand_right.png"),
        Path("calibration.json"),
        Path("checksums.sha256"),
    }


def validate_episode(path: Path) -> dict[str, Any]:
    actual = {item.relative_to(path) for item in path.rglob("*") if item.is_file()}
    if actual != _episode_files():
        raise ValueError(f"{path}: invalid episode layout")
    covered = _verify_checksum_file(path / "checksums.sha256", path)
    if covered != actual - {Path("checksums.sha256")}:
        raise ValueError(f"{path}: incomplete episode checksums")
    metadata = _json_load(path / "metadata.json")
    _reject_nonportable(metadata, f"{path}/metadata.json")
    if set(metadata) != {
        "schema_version",
        "episode_name",
        "episode_index",
        "task_name",
        "num_frames",
        "num_transitions",
        "seed_local_index",
        "fps",
        "files",
    }:
        raise ValueError(f"{path}: invalid metadata fields")
    expected_images = {name: f"observations/{name}.png" for name in CAMERAS}
    files = metadata["files"]
    if (
        metadata["episode_name"] != path.name
        or files != {
            "trajectory": "trajectory.parquet",
            "observations": expected_images,
            "calibration": "calibration.json",
        }
        or int(metadata["num_transitions"]) != int(metadata["num_frames"]) - 1
        or int(metadata["seed_local_index"]) != 0
    ):
        raise ValueError(f"{path}: invalid relative references")
    _validate_trajectory(
        path / "trajectory.parquet",
        int(metadata["num_frames"]),
        float(metadata["fps"]),
    )
    calibration = _json_load(path / "calibration.json")
    _reject_nonportable(calibration, f"{path}/calibration.json")
    if set(calibration.get("cameras", {})) != set(CAMERAS):
        raise ValueError(f"{path}: invalid calibration")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required") from exc
    for camera in CAMERAS:
        image_path = path / "observations" / f"{camera}.png"
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"{image_path}: expected RGB PNG")
    return {
        "task": str(metadata["task_name"]),
        "episode": str(metadata["episode_name"]),
        "path": f"{metadata['task_name']}/{metadata['episode_name']}",
        "num_frames": int(metadata["num_frames"]),
        "fps": float(metadata["fps"]),
    }


def _all_episodes(root: Path) -> list[Path]:
    return sorted(
        episode
        for task in root.iterdir()
        if task.is_dir() and task.name != "shared"
        for episode in task.glob("ep[0-9][0-9][0-9][0-9][0-9][0-9]")
        if episode.is_dir()
    )


def _validate_no_leaks(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        _reject_text(relative.as_posix(), relative)
        if path.suffix.lower() in {".json", ".md", ".urdf", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            _reject_text(text, relative)
            if path.suffix.lower() == ".json":
                _reject_nonportable(json.loads(text), relative.as_posix())
            elif path.suffix.lower() == ".md":
                _reject_absolute_paths_in_text(text, relative)
        elif path.suffix.lower() == ".parquet":
            _, _, pq = _import_pyarrow()
            for key, value in (pq.read_schema(path).metadata or {}).items():
                _reject_text(key.decode(errors="replace"), relative)
                _reject_text(value.decode(errors="replace"), relative)


def _validate_root(root: Path) -> list[dict[str, Any]]:
    required = {"README.md", "manifest.json", "schema.json", "checksums.sha256", "shared"}
    names = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if not required.issubset(names):
        raise ValueError(f"{root}: incomplete root")
    covered = _verify_checksum_file(root / "checksums.sha256", root)
    root_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path != root / "checksums.sha256"
    }
    if covered != root_files:
        raise ValueError(f"{root}: incomplete root checksums")
    manifest = _json_load(root / "manifest.json")
    _reject_nonportable(manifest, "manifest.json")
    episodes = [validate_episode(path) for path in _all_episodes(root)]
    if manifest.get("episodes") != episodes:
        raise ValueError(f"{root}: manifest mismatch")
    task_names = sorted({item["task"] for item in episodes})
    if manifest.get("tasks") != task_names:
        raise ValueError(f"{root}: task mismatch")
    if names != required | set(task_names):
        raise ValueError(f"{root}: unexpected root layout")
    for task_name in task_names:
        task_root = root / task_name
        expected = {"task.json", "schema.json"} | {
            item["episode"] for item in episodes if item["task"] == task_name
        }
        if {path.name for path in task_root.iterdir()} != expected:
            raise ValueError(f"{task_root}: unexpected layout")
        task_data = _json_load(task_root / "task.json")
        _reject_nonportable(task_data, f"{task_name}/task.json")
        if task_data.get("episodes") != [
            item["episode"] for item in episodes if item["task"] == task_name
        ]:
            raise ValueError(f"{task_root}: episode list mismatch")
    urdf = root / "shared" / "g02" / "kinematics.urdf"
    kinematics = root / "shared" / "g02" / "kinematics.json"
    if {
        item.relative_to(root / "shared" / "g02")
        for item in (root / "shared" / "g02").rglob("*")
        if item.is_file()
    } != {Path("kinematics.urdf"), Path("kinematics.json")}:
        raise ValueError(f"{root}: unexpected shared layout")
    text = urdf.read_text(encoding="utf-8")
    if any(token in text.lower() for token in ("<mesh", "filename=", "package://")):
        raise ValueError(f"{urdf}: mesh/path reference")
    _reject_nonportable(_json_load(kinematics), "shared/g02/kinematics.json")
    _validate_no_leaks(root)
    return episodes


def _export_episode(
    selected: Selection, root: Path, info: Mapping[str, Any]
) -> dict[str, Any]:
    _, _, pq = _import_pyarrow()
    private_metadata = _read_episode_metadata(selected.task, selected.episode_index)
    fps = float(info["fps"])
    episode = root / selected.task.name / selected.episode_name
    observations = episode / "observations"
    observations.mkdir(parents=True)
    table = _public_trajectory(selected, private_metadata, fps)
    pq.write_table(
        table,
        episode / "trajectory.parquet",
        compression="zstd",
        # The Arrow schema preserves the fixed-size vector widths. Validation
        # scans this metadata too, so it cannot carry private provenance.
        store_schema=True,
    )
    image_references = {}
    for camera in CAMERAS:
        video_path, offset = _video_location(selected.task, private_metadata, camera, fps)
        _write_rgb_png(
            _decode_rgb(video_path, offset + selected.start),
            observations / f"{camera}.png",
        )
        image_references[camera] = f"observations/{camera}.png"
    _write_json(
        episode / "calibration.json",
        _portable_calibration(selected.task, selected.episode_index),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "episode_name": selected.episode_name,
        "episode_index": selected.episode_index,
        "task_name": selected.task.name,
        "num_frames": selected.length,
        "num_transitions": selected.length - 1,
        "seed_local_index": 0,
        "fps": fps,
        "files": {
            "trajectory": "trajectory.parquet",
            "observations": image_references,
            "calibration": "calibration.json",
        },
    }
    _reject_nonportable(metadata, "metadata")
    _write_json(episode / "metadata.json", metadata)
    files = [
        path.relative_to(episode)
        for path in episode.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    ]
    _write_checksum_file(episode, files, episode / "checksums.sha256")
    return validate_episode(episode)


def _write_task(
    root: Path,
    task: TaskSpec,
    info: Mapping[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    task_root = root / task.name
    features = info["features"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": task.name,
        "episodes": [item["episode"] for item in episodes],
        "fps": float(info["fps"]),
        "features": {
            "action": _descriptor(features["action"], ACTION_WIDTH),
            "observation.state": _descriptor(features["observation.state"], STATE_WIDTH),
            "observations": {
                camera: {
                    "dtype": "uint8",
                    "encoding": "png",
                    "color_space": "RGB",
                    "role": "initial_observation",
                }
                for camera in CAMERAS
            },
        },
    }
    _reject_nonportable(payload, f"{task.name}/task.json")
    _write_json(task_root / "task.json", payload)
    _write_json(
        task_root / "schema.json",
        _portable_task_schema(info, float(info["fps"])),
    )


def _replace_directory_atomically(staged: Path, destination: Path) -> None:
    if not destination.exists():
        os.replace(staged, destination)
        return
    backup = destination.with_name(f".{destination.name}.previous-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _build_dataset(
    destination: Path,
    selections: list[Selection],
    infos: Mapping[str, Mapping[str, Any]],
    urdf_path: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        (staged / "README.md").write_text(_dataset_card(), encoding="utf-8")
        _write_json(staged / "schema.json", _root_schema())
        kinematics = _strip_urdf(
            urdf_path, staged / "shared" / "g02" / "kinematics.urdf"
        )
        _write_json(staged / "shared" / "g02" / "kinematics.json", kinematics)
        grouped: dict[str, list[dict[str, Any]]] = {}
        tasks = {}
        for selected in selections:
            tasks[selected.task.name] = selected.task
            grouped.setdefault(selected.task.name, []).append(
                _export_episode(selected, staged, infos[selected.task.name])
            )
        all_episodes = []
        for name in sorted(grouped):
            _write_task(staged, tasks[name], infos[name], grouped[name])
            all_episodes.extend(grouped[name])
        all_episodes.sort(key=lambda item: (item["task"], item["episode"]))
        _write_json(
            staged / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "tasks": sorted(grouped),
                "episodes": all_episodes,
                "kinematics": "shared/g02/kinematics.json",
            },
        )
        root_files = [
            path.relative_to(staged)
            for path in staged.rglob("*")
            if path.is_file() and path != staged / "checksums.sha256"
        ]
        _write_checksum_file(staged, root_files, staged / "checksums.sha256")
        _validate_root(staged)
        _replace_directory_atomically(staged, destination)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def _parse_task_overrides(values: list[str] | None) -> dict[str, TaskSpec]:
    if not values:
        return dict(DEFAULT_TASKS)
    result = {}
    for value in values:
        if "=" in value:
            name, root = value.split("=", 1)
            if name not in DEFAULT_TASKS:
                raise ValueError(f"unknown task {name!r}")
            canonical = DEFAULT_TASKS[name]
            result[name] = TaskSpec(name, Path(root), canonical.dataset_id, canonical.task_id)
        else:
            if value not in DEFAULT_TASKS:
                raise ValueError(f"unknown task {value!r}")
            result[value] = DEFAULT_TASKS[value]
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", action="append", help="NAME or NAME=/private/root")
    parser.add_argument("--urdf-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    if not args.validate_only and not args.dry_run:
        if not args.overwrite:
            parser.error("--overwrite is required")
        if args.urdf_path is None:
            parser.error("--urdf-path is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.output_root.resolve()
    if args.validate_only:
        episodes = _validate_root(destination)
        print(f"[valid] {destination}: {len(episodes)} episodes")
        return 0
    tasks = _parse_task_overrides(args.task)
    selections = discover_selections(args.selection_root.resolve(), tasks, args.limit)
    infos = {}
    for name, task in tasks.items():
        path = task.source_root / "meta" / "info.json"
        infos[name] = _json_load(path)
        _validate_source_info(infos[name], path)
    if args.dry_run:
        for selected in selections:
            _read_episode_metadata(selected.task, selected.episode_index)
        print(f"[ready] {len(selections)} episodes")
        return 0
    _build_dataset(destination, selections, infos, args.urdf_path.resolve())
    print(f"[done] {destination}: {len(selections)} episodes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
