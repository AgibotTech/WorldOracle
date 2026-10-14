#!/usr/bin/env python3
"""
Combine per-episode camera + action_map videos into a 2x3 grid mp4.

Layout (row-major):
    row 0:  top_head            | hand_left            | hand_right
    row 1:  action_map_top_head | action_map_hand_left | action_map_hand_right

For every directory matching
    <gt-root>/<task>/<ep_XXXXXX>/raw_video/
that contains all six expected .mp4 files, this script writes
    <gt-root>/<task>/<ep_XXXXXX>/raw_video/combined.mp4

Usage:
    python combine_episode_videos.py \
        --gt-root /path/to/main-ground-truth

All six inputs are assumed to have the *same frame count* but may have
different declared fps. To play them in lockstep (frame-by-frame aligned)
we drop the source PTS on every input and re-stamp them at a single output
fps via ``setpts=N/(FPS*TB)``. Frame i of every input is then shown at
``t = i / FPS`` so a 16fps action_map and a 30fps camera video finish at
the same wall-clock time. Default ``--fps 30``.

Add --overwrite to regenerate existing combined.mp4 files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Row-major order — must match the docstring above. Each entry is the
# basename (without extension) of an mp4 expected inside raw_video/.
GRID_ROWS: tuple[tuple[str, ...], ...] = (
    ("top_head", "hand_left", "hand_right"),
    ("action_map_top_head", "action_map_hand_left", "action_map_hand_right"),
)
TILE_NAMES_FLAT: tuple[str, ...] = tuple(name for row in GRID_ROWS for name in row)
N_ROWS = len(GRID_ROWS)
N_COLS = len(GRID_ROWS[0])

OUTPUT_NAME = "combined.mp4"


@dataclass
class EpisodeJob:
    raw_video_dir: Path
    tile_w: int
    tile_h: int
    fps: float
    overwrite: bool


def _iter_episode_dirs(gt_root: Path):
    """Yield every <task>/<ep_*>/raw_video directory under gt_root.

    We don't hardcode the ep_ prefix length: anything whose grandparent is the
    raw_video dir and that contains all six required mp4 files is fair game.
    """
    if not gt_root.is_dir():
        raise FileNotFoundError(f"gt_root not found: {gt_root}")
    for task_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            raw_video = ep_dir / "raw_video"
            if raw_video.is_dir():
                yield raw_video


def _all_inputs_present(raw_video_dir: Path) -> tuple[bool, list[str]]:
    missing = [
        name for name in TILE_NAMES_FLAT
        if not (raw_video_dir / f"{name}.mp4").is_file()
    ]
    return (not missing, missing)


def _build_filtergraph(tile_w: int, tile_h: int, out_fps: float) -> str:
    """Build the ffmpeg -filter_complex string for a 2x3 xstack.

    Each input is:
        1. scaled to fit inside (tile_w, tile_h) preserving aspect ratio,
        2. padded to exactly that size,
        3. SAR reset to 1 so xstack does not complain about mixed SAR,
        4. PTS re-stamped with ``setpts=N/(out_fps*TB)`` so frame i lands at
           time ``i / out_fps``. This makes inputs that share frame counts
           but differ in declared fps play in lockstep.
    """
    n = N_ROWS * N_COLS
    pieces: list[str] = []
    for i in range(n):
        pieces.append(
            f"[{i}:v]"
            f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,setpts=N/({out_fps}*TB),format=yuv420p"
            f"[v{i}]"
        )
    # xstack layout in xN_yN cells — written as pixel offsets per input.
    # For uniform tiles: column j is at x = j*tile_w, row i at y = i*tile_h.
    layout_cells: list[str] = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            layout_cells.append(f"{col * tile_w}_{row * tile_h}")
    layout = "|".join(layout_cells)
    inputs = "".join(f"[v{i}]" for i in range(n))
    pieces.append(f"{inputs}xstack=inputs={n}:layout={layout}:fill=black[out]")
    return ";".join(pieces)


def _ffmpeg_combine(job: EpisodeJob) -> Path:
    out_path = job.raw_video_dir / OUTPUT_NAME
    if out_path.exists() and not job.overwrite:
        print(f"[skip] exists: {out_path} (use --overwrite to regenerate)")
        return out_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    cmd: list[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    for name in TILE_NAMES_FLAT:
        cmd += ["-i", str(job.raw_video_dir / f"{name}.mp4")]

    filtergraph = _build_filtergraph(job.tile_w, job.tile_h, job.fps)
    cmd += [
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-r", str(float(job.fps)),
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or f"exit {r.returncode}").strip()
        raise RuntimeError(f"ffmpeg failed for {job.raw_video_dir}:\n{err}")
    print(f"[ok]   wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/path/to/main-ground-truth"),
        help="Root directory containing <task>/<ep_XXXXXX>/raw_video/ trees.",
    )
    p.add_argument(
        "--tile-width", type=int, default=640,
        help="Per-tile width in pixels (default: 640).",
    )
    p.add_argument(
        "--tile-height", type=int, default=480,
        help="Per-tile height in pixels (default: 480).",
    )
    p.add_argument(
        "--fps", type=float, default=30.0,
        help=(
            "Output FPS used both for PTS re-stamping (per-input setpts) and "
            "the final encoder rate. Inputs are treated as frame sequences and "
            "re-timed so frame i is shown at i/FPS seconds (default: 30)."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing combined.mp4 files.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without invoking ffmpeg.",
    )
    p.add_argument(
        "--task-filter", type=str, default=None,
        help="Only process episodes whose <task> directory name matches this substring.",
    )
    p.add_argument(
        "--episode-filter", type=str, default=None,
        help="Only process episodes whose <ep_*> directory name matches this substring.",
    )
    args = p.parse_args()

    ok = 0
    skipped = 0
    failed = 0
    missing_inputs = 0

    for raw_video_dir in _iter_episode_dirs(args.gt_root):
        ep_dir = raw_video_dir.parent
        task_dir = ep_dir.parent
        if args.task_filter and args.task_filter not in task_dir.name:
            continue
        if args.episode_filter and args.episode_filter not in ep_dir.name:
            continue

        present, missing = _all_inputs_present(raw_video_dir)
        if not present:
            print(f"[warn] missing inputs in {raw_video_dir}: {missing}")
            missing_inputs += 1
            continue

        out_path = raw_video_dir / OUTPUT_NAME
        if args.dry_run:
            action = "OVERWRITE" if out_path.exists() else "WRITE"
            if out_path.exists() and not args.overwrite:
                action = "SKIP (exists)"
            print(f"[dry]  {action}: {out_path}")
            continue

        job = EpisodeJob(
            raw_video_dir=raw_video_dir,
            tile_w=int(args.tile_width),
            tile_h=int(args.tile_height),
            fps=args.fps,
            overwrite=args.overwrite,
        )
        try:
            before = out_path.exists()
            _ffmpeg_combine(job)
            if before and not args.overwrite:
                skipped += 1
            else:
                ok += 1
        except Exception as e:
            print(f"[fail] {raw_video_dir}: {e}", file=sys.stderr)
            failed += 1

    print(
        f"\nDone. ok={ok} skipped={skipped} failed={failed} "
        f"missing_inputs={missing_inputs}"
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
