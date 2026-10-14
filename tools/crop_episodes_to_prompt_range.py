#!/usr/bin/env python3
"""Crop ACWMBench episodes to the range declared in prompt/prompts.json.

Frame conventions:
  * ``start_frame`` is 0-indexed and inclusive.
  * ``end_frame`` is 1-indexed and inclusive as a frame number.
  * Therefore the corresponding Python slice is exactly ``[start_frame:end_frame]``.
  * Existing segment ``key_frame`` values are 1-indexed.
  * Rewritten segment ``key_frame`` values are 0-indexed relative to the crop.

For each episode this script crops, in place:
  * prompt/masks.npz
  * multiview_masks/{hand_left,hand_right,top_head}/masks.npz
  * raw_video/{hand_left,hand_right,top_head}.mp4
  * video/{hand_left,hand_right,top_head}/frame_*.jpg
  * prompt/prompts.json

The image sequences are renumbered from ``frame_00000.jpg``. Files are first
written under an episode-local temporary directory and validated before the
originals are replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


VIEWS = ("hand_left", "hand_right", "top_head")
DEFAULT_ROOT = Path(
    "/path/to/task-data"
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _frame_range(prompts: dict, path: Path) -> tuple[int, int, int]:
    if "start_frame" not in prompts or "end_frame" not in prompts:
        raise KeyError(f"{path} must contain start_frame and end_frame")
    start = int(prompts["start_frame"])
    end = int(prompts["end_frame"])
    if start < 0 or end <= start:
        raise ValueError(
            f"{path}: invalid range start_frame={start}, end_frame={end}"
        )
    return start, end, end - start


def _crop_mask_array(
    masks: np.ndarray, start: int, end: int, path: Path
) -> np.ndarray:
    # Benchmark masks are (M,T,H,W); tolerate the legacy (T,H,W) layout.
    if masks.ndim == 4:
        time_axis = 1
    elif masks.ndim == 3:
        time_axis = 0
    else:
        raise ValueError(
            f"{path}: masks must be (M,T,H,W) or (T,H,W), got {masks.shape}"
        )
    time_len = masks.shape[time_axis]
    if end > time_len:
        raise ValueError(
            f"{path}: end_frame={end} exceeds mask time length {time_len}"
        )
    slc = [slice(None)] * masks.ndim
    slc[time_axis] = slice(start, end)
    return masks[tuple(slc)]


def _write_cropped_npz(
    source: Path, destination: Path, start: int, end: int
) -> None:
    with np.load(source, allow_pickle=True) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if "masks" not in arrays:
        raise KeyError(f"{source}: missing required 'masks' array")
    arrays["masks"] = _crop_mask_array(arrays["masks"], start, end, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _probe_frame_count(path: Path) -> int:
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
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        value = result.stdout.strip()
        if result.returncode == 0 and value.isdigit():
            return int(value)

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _trim_video(
    source: Path, destination: Path, start: int, end: int, expected: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to crop raw_video/*.mp4")
    source_count = _probe_frame_count(source)
    if end > source_count:
        raise ValueError(
            f"{source}: end_frame={end} exceeds video frame count {source_count}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Frame-exact selection. end is an exclusive 0-indexed boundary.
    vf = f"select=between(n\\,{start}\\,{end - 1}),setpts=N/FRAME_RATE/TB"
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
        vf,
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
    output_count = _probe_frame_count(destination)
    if output_count != expected:
        raise RuntimeError(
            f"{destination}: expected {expected} frames, encoded {output_count}"
        )


def _frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid frame filename: {path.name}") from exc


def _stage_frame_directory(
    source_dir: Path,
    destination_dir: Path,
    start: int,
    end: int,
    expected: int,
) -> None:
    frames = sorted(source_dir.glob("frame_*.jpg"), key=_frame_number)
    if end > len(frames):
        raise ValueError(
            f"{source_dir}: end_frame={end} exceeds JPG count {len(frames)}"
        )
    selected = frames[start:end]
    if len(selected) != expected:
        raise RuntimeError(
            f"{source_dir}: expected {expected} selected JPGs, got {len(selected)}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    digits = max(5, len(str(max(expected - 1, 0))))
    for new_idx, source in enumerate(selected):
        destination = destination_dir / f"frame_{new_idx:0{digits}d}.jpg"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _rewrite_prompts(
    prompts: dict, start: int, end: int, new_n_frames: int
) -> dict:
    updated = dict(prompts)
    updated.pop("start_frame", None)
    updated.pop("end_frame", None)
    updated["n_frames"] = int(new_n_frames)

    rewritten_segments = []
    for segment in prompts.get("segments", []):
        if not isinstance(segment, dict) or "key_frame" not in segment:
            rewritten_segments.append(segment)
            continue
        old_key_1idx = int(segment["key_frame"])
        old_key_0idx = old_key_1idx - 1
        # Keep only keyframes represented by the cropped [start:end] sequence.
        if not (start <= old_key_0idx < end):
            continue
        new_segment = dict(segment)
        new_segment["key_frame"] = old_key_0idx - start
        rewritten_segments.append(new_segment)
    updated["segments"] = rewritten_segments
    return updated


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def _required_paths(episode: Path) -> tuple[list[Path], list[Path]]:
    files = [episode / "prompt" / "prompts.json", episode / "prompt" / "masks.npz"]
    files += [
        episode / "multiview_masks" / view / "masks.npz" for view in VIEWS
    ]
    files += [episode / "raw_video" / f"{view}.mp4" for view in VIEWS]
    dirs = [episode / "video" / view for view in VIEWS]
    return files, dirs


def _validate_episode_layout(episode: Path) -> None:
    files, dirs = _required_paths(episode)
    missing = [str(path) for path in files if not path.is_file()]
    missing += [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"{episode}: missing required paths:\n  " + "\n  ".join(missing)
        )


def _commit_file(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)


def _commit_directory(staged: Path, target: Path) -> None:
    backup = target.with_name(target.name + ".crop_old")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup)


def crop_episode(episode: Path, *, apply: bool) -> None:
    _validate_episode_layout(episode)
    prompts_path = episode / "prompt" / "prompts.json"
    prompts = _read_json(prompts_path)
    start, end, new_t = _frame_range(prompts, prompts_path)
    rewritten = _rewrite_prompts(prompts, start, end, new_t)

    print(
        f"[episode] {episode.name}: [{start}:{end}] -> {new_t} frames, "
        f"segments {len(prompts.get('segments', []))} -> "
        f"{len(rewritten.get('segments', []))}"
    )
    if not apply:
        return

    temp_root = Path(tempfile.mkdtemp(prefix=".crop_tmp_", dir=episode))
    try:
        mask_paths = [episode / "prompt" / "masks.npz"] + [
            episode / "multiview_masks" / view / "masks.npz" for view in VIEWS
        ]
        for source in mask_paths:
            relative = source.relative_to(episode)
            _write_cropped_npz(source, temp_root / relative, start, end)

        for view in VIEWS:
            video_source = episode / "raw_video" / f"{view}.mp4"
            _trim_video(
                video_source,
                temp_root / "raw_video" / f"{view}.mp4",
                start,
                end,
                new_t,
            )
            _stage_frame_directory(
                episode / "video" / view,
                temp_root / "video" / view,
                start,
                end,
                new_t,
            )

        _write_json(temp_root / "prompt" / "prompts.json", rewritten)

        # Commit prompts last so its n_frames marks a fully cropped episode.
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
        _commit_file(temp_root / "prompt" / "prompts.json", prompts_path)
        print(f"[ok] cropped {episode}")
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
        help="Perform the in-place crop. Without this flag, only validate/preview.",
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
        crop_episode(episode, apply=args.apply)
    mode = "applied" if args.apply else "preview only; rerun with --apply"
    print(f"[done] {len(episodes)} episodes, {mode}")


if __name__ == "__main__":
    main()
