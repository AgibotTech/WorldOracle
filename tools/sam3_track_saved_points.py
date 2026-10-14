#!/usr/bin/env python3
"""Track reviewed SAM3 point prompts and write ACWMBench mask archives.

Inputs are JSON files produced by ``sam3_review_first_frame_points.py``.
Outputs:
  target=prompt:
    <episode>/prompt/masks.npz
  target=multiview:
    <episode>/multiview_masks/<view>/masks.npz

Every archive contains uint8 ``masks`` with shape ``[M, T, H, W]``.
"""

# pylint: disable=import-error,no-name-in-module

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]


DEFAULT_POINTS = Path("/path/to/sam3-points")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FRAME_NUMBER = re.compile(r"(\d+)(?!.*\d)")


def add_sam3_to_path(path: Path | None) -> None:
    candidates = [
        path,
        Path(os.environ["SAM3_ROOT"]) if os.environ.get("SAM3_ROOT") else None,
        Path("/path/to/sam3"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "sam3").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return
    raise SystemExit("SAM3 source not found; pass --sam3-root or set SAM3_ROOT")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def natural_key(path: Path) -> tuple[int, str]:
    match = FRAME_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def list_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )


@dataclass(frozen=True)
class Annotation:
    path: Path
    task: str
    episode: str
    episode_dir: Path
    target: str
    view: str
    resource_path: Path
    objects: dict[int, dict[str, list[Any]]]
    stored_seed: int | None

    @property
    def key(self) -> str:
        return f"{self.task}/{self.episode}/{self.target}/{self.view}"

    @property
    def output_dir(self) -> Path:
        if self.target == "prompt":
            return self.episode_dir / "prompt"
        return self.episode_dir / "multiview_masks" / self.view


def load_annotation(path: Path) -> Annotation:
    data = json.loads(path.read_text(encoding="utf-8"))
    target = str(data.get("target", ""))
    view = str(data.get("view", ""))
    if target not in {"prompt", "multiview"}:
        raise ValueError(f"invalid target {target!r}")
    if data.get("confirmed_first_frame_response") is not True:
        raise ValueError("annotation was not saved after a confirmed SAM3 response")
    if target == "prompt" and view != "top_head":
        raise ValueError("prompt target must use top_head")
    objects: dict[int, dict[str, list[Any]]] = {}
    for object_id, record in (data.get("objects") or {}).items():
        points = [
            [float(point[0]), float(point[1])]
            for point in record.get("points", [])
        ]
        labels = [int(label) for label in record.get("labels", [])]
        if len(points) != len(labels):
            raise ValueError(f"object {object_id}: points/labels length mismatch")
        if points:
            objects[int(object_id)] = {"points": points, "labels": labels}
    if not objects:
        raise ValueError("annotation has no non-empty objects")
    return Annotation(
        path=path,
        task=str(data["task"]),
        episode=str(data["episode"]),
        episode_dir=Path(data["episode_dir"]),
        target=target,
        view=view,
        resource_path=Path(data["resource_path"]),
        objects=objects,
        stored_seed=int(data["seed"]) if data.get("seed") is not None else None,
    )


def discover_annotations(
    points_root: Path,
    tasks: set[str] | None,
    episodes: set[str] | None,
    targets: set[str],
    views: set[str] | None,
) -> tuple[list[Annotation], list[tuple[Path, str]]]:
    annotations: list[Annotation] = []
    errors: list[tuple[Path, str]] = []
    for path in sorted(points_root.glob("*/*/*.json")):
        try:
            annotation = load_annotation(path)
            if tasks and annotation.task not in tasks:
                continue
            if episodes and annotation.episode not in episodes:
                continue
            if annotation.target not in targets:
                continue
            if views and annotation.view not in views:
                continue
            annotations.append(annotation)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append((path, str(error)))
    annotations.sort(
        key=lambda item: (item.task, item.episode, item.target, item.view)
    )
    return annotations, errors


def output_paths(annotation: Annotation) -> tuple[Path, Path, Path]:
    return (
        annotation.output_dir / "masks.npz",
        annotation.output_dir / "masks_meta.json",
        annotation.output_dir / "overlay.mp4",
    )


def response_masks(
    output: dict[str, Any],
) -> Iterator[tuple[int, np.ndarray]]:
    if "out_binary_masks" not in output or "out_obj_ids" not in output:
        return
    masks = output["out_binary_masks"]
    object_ids = output["out_obj_ids"]
    for index in range(min(len(masks), len(object_ids))):
        mask = np.asarray(masks[index])
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        yield int(object_ids[index]), mask


class SavedPointTracker:
    def __init__(
        self,
        checkpoint_path: str,
        version: str,
        use_fa3: bool,
        compile_model: bool,
        seed: int,
        deterministic: bool,
        write_overlay: bool,
        overlay_fps: float,
    ) -> None:
        from sam3.model_builder import (  # type: ignore[import-not-found]
            build_sam3_predictor,
        )
        from sam3.visualization_utils import (  # type: ignore[import-not-found]
            render_masklet_frame,
        )

        self.seed = seed
        self.deterministic = deterministic
        self.write_overlay = write_overlay
        self.overlay_fps = overlay_fps
        self.render_masklet_frame = render_masklet_frame
        seed_everything(seed, deterministic)
        self.predictor = build_sam3_predictor(
            checkpoint_path=checkpoint_path,
            version=version,
            compile=compile_model,
            use_fa3=use_fa3,
        )

    def run(self, annotation: Annotation, overwrite: bool) -> dict[str, Any]:
        npz_path, meta_path, overlay_path = output_paths(annotation)
        if npz_path.is_file() and not overwrite:
            return {"status": "skipped", "path": str(npz_path)}

        frames = list_frames(annotation.resource_path)
        if not frames:
            raise RuntimeError(f"no frames found in {annotation.resource_path}")
        first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"failed to read first frame: {frames[0]}")
        height, width = first.shape[:2]
        frame_count = len(frames)
        object_ids = sorted(annotation.objects)
        object_to_row = {
            object_id: row for row, object_id in enumerate(object_ids)
        }

        annotation.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(self.seed, self.deterministic)
        session = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(annotation.resource_path),
            }
        )
        session_id = session["session_id"]
        memmap_path: Path | None = None
        overlay_writer: cv2.VideoWriter | None = None
        tracked_frames: list[int] = []
        try:
            for object_id in object_ids:
                record = annotation.objects[object_id]
                points = torch.tensor(
                    [
                        [point[0] / width, point[1] / height]
                        for point in record["points"]
                    ],
                    dtype=torch.float32,
                )
                labels = torch.tensor(record["labels"], dtype=torch.int32)
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "points": points,
                        "point_labels": labels,
                        "obj_id": object_id,
                    }
                )

            file_handle, temporary_name = tempfile.mkstemp(
                prefix="sam3_masks_",
                suffix=".uint8",
                dir=annotation.output_dir,
            )
            os.close(file_handle)
            memmap_path = Path(temporary_name)
            masks = np.memmap(
                memmap_path,
                dtype=np.uint8,
                mode="w+",
                shape=(len(object_ids), frame_count, height, width),
            )
            masks[:] = 0

            if self.write_overlay:
                overlay_writer = cv2.VideoWriter(
                    str(overlay_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.overlay_fps,
                    (width, height),
                )

            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with autocast:
                stream = self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "start_frame_index": 0,
                        "max_frame_num_to_track": frame_count,
                    }
                )
                for response in stream:
                    frame_index = int(response["frame_index"])
                    if not 0 <= frame_index < frame_count:
                        continue
                    output = response["outputs"]
                    for object_id, mask in response_masks(output):
                        row = object_to_row.get(object_id)
                        if row is None:
                            continue
                        binary = (mask > 0).astype(np.uint8)
                        if binary.shape != (height, width):
                            binary = cv2.resize(
                                binary,
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        masks[row, frame_index] = binary
                    tracked_frames.append(frame_index)

                    if overlay_writer is not None:
                        frame_bgr = cv2.imread(
                            str(frames[frame_index]), cv2.IMREAD_COLOR
                        )
                        if frame_bgr is not None:
                            frame_rgb = cv2.cvtColor(
                                frame_bgr, cv2.COLOR_BGR2RGB
                            )
                            overlay_rgb = self.render_masklet_frame(
                                frame_rgb,
                                output,
                                frame_idx=frame_index,
                                alpha=0.5,
                            )
                            overlay_writer.write(
                                cv2.cvtColor(
                                    overlay_rgb, cv2.COLOR_RGB2BGR
                                )
                            )
                    if (
                        (frame_index + 1) % 100 == 0
                        or frame_index + 1 == frame_count
                    ):
                        print(
                            f"  {annotation.key}: "
                            f"{frame_index + 1}/{frame_count}",
                            flush=True,
                        )

            masks.flush()
            temporary_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
            np.savez_compressed(temporary_npz, masks=masks)
            temporary_npz.replace(npz_path)
            del masks

            metadata = {
                "schema_version": 1,
                "task": annotation.task,
                "episode": annotation.episode,
                "episode_dir": str(annotation.episode_dir),
                "target": annotation.target,
                "view": annotation.view,
                "resource_path": str(annotation.resource_path),
                "points_path": str(annotation.path),
                "mask_path": str(npz_path),
                "shape": [
                    len(object_ids),
                    frame_count,
                    height,
                    width,
                ],
                "dtype": "uint8",
                "format": "[objects, frames, height, width]",
                "npz_key": "masks",
                "object_ids_in_row_order": object_ids,
                "objects": {
                    str(object_id): annotation.objects[object_id]
                    for object_id in object_ids
                },
                "seed": self.seed,
                "annotation_seed": annotation.stored_seed,
                "deterministic": self.deterministic,
                "tracked_frame_count": len(set(tracked_frames)),
                "overlay_path": (
                    str(overlay_path) if self.write_overlay else None
                ),
            }
            temporary_meta = meta_path.with_suffix(".tmp")
            temporary_meta.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_meta.replace(meta_path)
            return {
                "status": "saved",
                "path": str(npz_path),
                "shape": metadata["shape"],
            }
        finally:
            if overlay_writer is not None:
                overlay_writer.release()
            with contextlib.suppress(Exception):
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            if memmap_path is not None:
                memmap_path.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def validate_annotation(annotation: Annotation) -> list[str]:
    issues: list[str] = []
    frames = list_frames(annotation.resource_path)
    if not annotation.episode_dir.is_dir():
        issues.append(f"episode_dir missing: {annotation.episode_dir}")
    if not frames:
        issues.append(f"no frames: {annotation.resource_path}")
    if annotation.target == "prompt" and annotation.view != "top_head":
        issues.append("prompt target is not top_head")
    for object_id, record in annotation.objects.items():
        if len(record["points"]) != len(record["labels"]):
            issues.append(f"object {object_id}: points/labels mismatch")
        if any(label not in {0, 1} for label in record["labels"]):
            issues.append(f"object {object_id}: labels must be 0/1")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-root", type=Path, default=DEFAULT_POINTS)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("prompt", "multiview"),
        default=["prompt", "multiview"],
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--sam3-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-seed-mismatch",
        action="store_true",
        help="Allow tracking seed to differ from a saved annotation seed.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-fa3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--overlay-fps", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_sam3_to_path(args.sam3_root)
    points_root = args.points_root.expanduser().resolve()
    if not points_root.is_dir():
        raise SystemExit(f"points root not found: {points_root}")
    annotations, parse_errors = discover_annotations(
        points_root,
        set(args.tasks) if args.tasks else None,
        set(args.episodes) if args.episodes else None,
        set(args.targets),
        set(args.views) if args.views else None,
    )
    for path, error in parse_errors:
        print(f"[invalid] {path}: {error}", flush=True)
    if not annotations:
        raise SystemExit("no matching valid annotations found")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        annotations = annotations[: args.limit]

    invalid: list[tuple[Annotation, list[str]]] = []
    for annotation in annotations:
        issues = validate_annotation(annotation)
        if (
            annotation.stored_seed is not None
            and annotation.stored_seed != args.seed
            and not args.allow_seed_mismatch
        ):
            issues.append(
                f"annotation seed {annotation.stored_seed} != "
                f"tracking seed {args.seed}"
            )
        output = output_paths(annotation)[0]
        state = "EXISTS" if output.is_file() else "pending"
        print(
            f"[{state}] {annotation.key}: objects={len(annotation.objects)} "
            f"-> {output}",
            flush=True,
        )
        if issues:
            invalid.append((annotation, issues))
            for issue in issues:
                print(f"  issue: {issue}", flush=True)
    if invalid:
        raise SystemExit(f"{len(invalid)} annotation(s) failed validation")
    if args.dry_run:
        print(f"Dry-run validated {len(annotations)} annotation(s).")
        return

    tracker = SavedPointTracker(
        checkpoint_path=args.checkpoint_path,
        version=args.version,
        use_fa3=not args.no_fa3,
        compile_model=args.compile,
        seed=args.seed,
        deterministic=args.deterministic,
        write_overlay=args.write_overlay,
        overlay_fps=args.overlay_fps,
    )
    saved = skipped = failed = 0
    failures: list[dict[str, str]] = []
    for annotation in annotations:
        print(f"== {annotation.key} ==", flush=True)
        try:
            result = tracker.run(annotation, overwrite=args.overwrite)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] == "saved":
                saved += 1
            else:
                skipped += 1
        except Exception as error:  # pylint: disable=broad-exception-caught
            failed += 1
            failures.append({"key": annotation.key, "error": str(error)})
            traceback.print_exc()
            if not args.continue_on_error:
                raise
    print(
        f"Done: saved={saved} skipped={skipped} failed={failed}",
        flush=True,
    )
    if failures:
        failure_path = points_root / "tracking_failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"tracking failures written to {failure_path}")


if __name__ == "__main__":
    main()
