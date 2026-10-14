#!/usr/bin/env python3
"""Resize ACWMBench episode media and masks to 640x480 (width x height).

For every selected episode, this script resizes in place:

  prompt/masks.npz
  multiview_masks/{hand_left,hand_right,top_head}/masks.npz
  raw_video/{hand_left,hand_right,top_head}.mp4
  video/{hand_left,hand_right,top_head}/frame_*.jpg

Interpolation:
  * masks: nearest-neighbor (preserves labels/binary values)
  * RGB JPG/MP4: bilinear

All outputs are staged in an episode-local temporary directory and validated
before replacing the originals. Without ``--apply`` the script only validates
and prints what it would resize.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


VIEWS = ("hand_left", "hand_right", "top_head")
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
DEFAULT_ROOT = Path(
    "/path/to/task-data"
)


def _required_paths(episode: Path) -> tuple[list[Path], list[Path]]:
    files = [episode / "prompt" / "masks.npz"]
    files += [
        episode / "multiview_masks" / view / "masks.npz" for view in VIEWS
    ]
    files += [episode / "raw_video" / f"{view}.mp4" for view in VIEWS]
    frame_dirs = [episode / "video" / view for view in VIEWS]
    return files, frame_dirs


def _validate_layout(episode: Path) -> None:
    files, frame_dirs = _required_paths(episode)
    missing = [str(path) for path in files if not path.is_file()]
    missing += [str(path) for path in frame_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{episode}: missing required paths:\n  " + "\n  ".join(missing)
        )
    for frame_dir in frame_dirs:
        if not any(frame_dir.glob("frame_*.jpg")):
            raise FileNotFoundError(f"{frame_dir}: no frame_*.jpg files")


def _resize_mask_array(masks: np.ndarray, source: Path) -> np.ndarray:
    if masks.ndim == 4:
        # (M,T,H,W)
        leading = masks.shape[:2]
    elif masks.ndim == 3:
        # Legacy (T,H,W)
        leading = masks.shape[:1]
    else:
        raise ValueError(
            f"{source}: masks must be (M,T,H,W) or (T,H,W), got {masks.shape}"
        )

    flat = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    resized = np.empty(
        (flat.shape[0], TARGET_HEIGHT, TARGET_WIDTH), dtype=masks.dtype
    )
    for idx, frame in enumerate(flat):
        resized[idx] = cv2.resize(
            frame,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
    return resized.reshape((*leading, TARGET_HEIGHT, TARGET_WIDTH))


def _write_resized_npz(source: Path, destination: Path) -> None:
    with np.load(source, allow_pickle=True) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if "masks" not in arrays:
        raise KeyError(f"{source}: missing required 'masks' array")
    old_shape = arrays["masks"].shape
    arrays["masks"] = _resize_mask_array(arrays["masks"], source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"  [mask] {source}: {old_shape} -> {arrays['masks'].shape}")


def _probe_video(path: Path) -> tuple[int, int, int]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            try:
                streams = json.loads(result.stdout).get("streams", [])
                if streams:
                    stream = streams[0]
                    count = str(stream.get("nb_read_frames", ""))
                    if count.isdigit():
                        return (
                            int(stream["width"]),
                            int(stream["height"]),
                            int(count),
                        )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def _resize_video(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to resize raw_video/*.mp4")
    src_w, src_h, src_frames = _probe_video(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-vf",
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=bilinear",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source}:\n{result.stderr.strip()}"
        )
    dst_w, dst_h, dst_frames = _probe_video(destination)
    if (dst_w, dst_h) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise RuntimeError(
            f"{destination}: expected {TARGET_WIDTH}x{TARGET_HEIGHT}, "
            f"got {dst_w}x{dst_h}"
        )
    if src_frames > 0 and dst_frames > 0 and src_frames != dst_frames:
        raise RuntimeError(
            f"{destination}: frame count changed {src_frames} -> {dst_frames}"
        )
    print(
        f"  [video] {source}: {src_w}x{src_h}, {src_frames} frames -> "
        f"{dst_w}x{dst_h}, {dst_frames} frames"
    )


def _frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid frame filename: {path.name}") from exc


def _stage_resized_frames(source_dir: Path, destination_dir: Path) -> int:
    frames = sorted(source_dir.glob("frame_*.jpg"), key=_frame_number)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in frames:
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read image: {source}")
        resized = cv2.resize(
            bgr,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        destination = destination_dir / source.name
        if not cv2.imwrite(str(destination), resized):
            raise RuntimeError(f"failed to write image: {destination}")
    written = sorted(destination_dir.glob("frame_*.jpg"), key=_frame_number)
    if len(written) != len(frames):
        raise RuntimeError(
            f"{destination_dir}: expected {len(frames)} JPGs, wrote {len(written)}"
        )
    for path in written:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            raise RuntimeError(f"{path}: invalid resized image shape")
    print(f"  [frames] {source_dir}: resized {len(frames)} JPGs")
    return len(frames)


def _commit_file(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)


def _commit_directory(staged: Path, target: Path) -> None:
    backup = target.with_name(target.name + ".resize_old")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup)


def resize_episode(episode: Path, *, apply: bool) -> None:
    _validate_layout(episode)
    print(
        f"[episode] {episode}: resize masks, MP4s and JPGs to "
        f"{TARGET_WIDTH}x{TARGET_HEIGHT}"
    )
    if not apply:
        return

    temp_root = Path(tempfile.mkdtemp(prefix=".resize_tmp_", dir=episode))
    try:
        mask_paths = [episode / "prompt" / "masks.npz"] + [
            episode / "multiview_masks" / view / "masks.npz" for view in VIEWS
        ]
        for source in mask_paths:
            relative = source.relative_to(episode)
            _write_resized_npz(source, temp_root / relative)

        for view in VIEWS:
            _resize_video(
                episode / "raw_video" / f"{view}.mp4",
                temp_root / "raw_video" / f"{view}.mp4",
            )
            _stage_resized_frames(
                episode / "video" / view,
                temp_root / "video" / view,
            )

        for source in mask_paths:
            relative = source.relative_to(episode)
            _commit_file(temp_root / relative, source)
        for view in VIEWS:
            _commit_file(
                temp_root / "raw_video" / f"{view}.mp4",
                episode / "raw_video" / f"{view}.mp4",
            )
            _commit_directory(
                temp_root / "video" / view,
                episode / "video" / view,
            )
        print(f"[ok] resized {episode}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Task directory containing ep* folders.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Optional episode names, e.g. ep000000 ep000001. Default: all ep*.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the in-place resize. Without this flag, only preview.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.episodes:
        episodes = [root / name for name in args.episodes]
    else:
        episodes = sorted(path for path in root.glob("ep*") if path.is_dir())
    if not episodes:
        raise RuntimeError(f"no episode directories under {root}")

    for episode in episodes:
        resize_episode(episode, apply=args.apply)
    mode = "applied" if args.apply else "preview only; rerun with --apply"
    print(f"[done] {len(episodes)} episodes, {mode}")


if __name__ == "__main__":
    main()
