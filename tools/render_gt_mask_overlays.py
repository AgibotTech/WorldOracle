#!/usr/bin/env python3
"""Render prompt and robot-arm masks over GT top-head frames.

Example:
    python tools/render_gt_mask_overlays.py \
      --episode yam=/path/to/gt_dataset_yam/hangtowel/ep000000 \
      --episode ego=/path/to/gt_dataset_ego/hangtowel/ep000000 \
      --output-dir /path/to/output
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS_BGR = [
    (64, 96, 255),
    (80, 220, 80),
    (255, 160, 48),
    (220, 80, 220),
    (40, 220, 240),
    (240, 180, 80),
]


def natural_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_episode(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--episode must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--episode must be NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def load_masks(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        key = "masks" if "masks" in archive else archive.files[0]
        masks = np.asarray(archive[key], dtype=np.uint8)
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4:
        raise ValueError(f"Expected masks [N,T,H,W], got {masks.shape}: {path}")
    return masks


def frame_paths(episode_root: Path, view: str) -> list[Path]:
    directory = episode_root / "video" / view
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if not paths:
        raise ValueError(f"No image frames found in {directory}")
    return paths


def start_encoder(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def overlay_masks(
    frame: np.ndarray,
    frame_masks: np.ndarray,
    alpha: float,
    source_label: str,
    frame_index: int,
    fps: float,
) -> np.ndarray:
    output = frame.copy()
    for mask_index, raw_mask in enumerate(frame_masks):
        mask = raw_mask > 0
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color = np.asarray(COLORS_BGR[mask_index % len(COLORS_BGR)])
        output[mask] = (
            output[mask].astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            tuple(int(value) for value in color),
            2,
            cv2.LINE_AA,
        )

    banner_height = 30 + 22 * len(frame_masks)
    shade = output[:banner_height].copy()
    output[:banner_height] = (shade.astype(np.float32) * 0.35).astype(np.uint8)
    cv2.putText(
        output,
        f"{source_label} | frame {frame_index:05d} | {frame_index / fps:.2f}s",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for mask_index in range(len(frame_masks)):
        color = COLORS_BGR[mask_index % len(COLORS_BGR)]
        y = 43 + 22 * mask_index
        cv2.rectangle(output, (10, y - 11), (25, y + 3), color, -1)
        cv2.putText(
            output,
            f"mask {mask_index}",
            (32, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return output


def render_one(
    *,
    frames: list[Path],
    mask_path: Path,
    output_path: Path,
    source_label: str,
    fps: float,
    alpha: float,
    crf: int,
    preset: str,
) -> None:
    print(f"[mask-overlay] loading {mask_path}", flush=True)
    masks = load_masks(mask_path)
    if masks.shape[1] != len(frames):
        raise ValueError(
            f"Frame mismatch for {mask_path}: masks={masks.shape[1]}, "
            f"GT={len(frames)}"
        )
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode {frames[0]}")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    encoder = start_encoder(output_path, width, height, fps, crf, preset)
    if encoder.stdin is None:
        raise RuntimeError("Failed to open ffmpeg input pipe")
    try:
        for index, frame_path in enumerate(frames):
            frame = first if index == 0 else cv2.imread(
                str(frame_path), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError(f"Could not decode {frame_path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)
            rendered = overlay_masks(
                frame,
                masks[:, index],
                alpha,
                source_label,
                index,
                fps,
            )
            encoder.stdin.write(np.ascontiguousarray(rendered).tobytes())
            if (index + 1) % 300 == 0 or index + 1 == len(frames):
                print(
                    f"[mask-overlay] {output_path.name}: "
                    f"{index + 1}/{len(frames)}",
                    flush=True,
                )
    except BaseException:
        encoder.stdin.close()
        encoder.wait()
        output_path.unlink(missing_ok=True)
        raise
    encoder.stdin.close()
    return_code = encoder.wait()
    if return_code:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"[mask-overlay] saved {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        action="append",
        type=parse_episode,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    output_dir = args.output_dir.expanduser().resolve()
    jobs = [
        ("prompt/masks.npz", "object masks", "object_masks"),
        (
            f"multiview_masks/{args.view}/masks.npz",
            "robot-arm masks",
            "robot_arm_masks",
        ),
    ]
    for name, raw_root in args.episode:
        episode_root = raw_root.resolve()
        frames = frame_paths(episode_root, args.view)
        episode_label = f"{name} {episode_root.parent.name}/{episode_root.name}"
        for relative_mask, source_name, output_suffix in jobs:
            render_one(
                frames=frames,
                mask_path=episode_root / relative_mask,
                output_path=output_dir / f"{name}_{output_suffix}.mp4",
                source_label=f"{episode_label} | {source_name}",
                fps=args.fps,
                alpha=args.alpha,
                crf=args.crf,
                preset=args.preset,
            )


if __name__ == "__main__":
    main()
