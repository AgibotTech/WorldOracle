#!/usr/bin/env python3
"""Extract selected LeRobot policy-eval MP4s into the GT-style video tree.

The output contains only:

    <output>/<task>/epXXXXXX/video/<view>/frame_XXXXX.jpg

Selection files contain one integer episode index per line.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VIEWS = ("top_head", "hand_left", "hand_right")
DEFAULT_TASKS = ("hangtowel", "pickfruit", "pourwater")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/path/to/policy-source"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/path/to/policy-ground-truth"
        ),
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("/path/to/selection-output"),
        help="Directory containing <task>.txt selection files.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--jpg-quality",
        type=int,
        choices=(85, 90, 95),
        default=95,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_selection(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"selection file not found: {path}")
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
        if episode < 0:
            raise ValueError(f"{path}:{line_number}: episode must be non-negative")
        if episode in seen:
            raise ValueError(f"{path}:{line_number}: duplicate episode {episode}")
        seen.add(episode)
        episodes.append(episode)
    return episodes


def source_video(source: Path, task: str, episode: int, view: str) -> Path:
    return (
        source
        / task
        / "videos"
        / f"observation.images.{view}"
        / "chunk-000"
        / f"file-{episode:03d}.mp4"
    )


def extract_video(src: Path, destination: Path, jpg_quality: int) -> int:
    destination.mkdir(parents=True)
    qscale = {95: "2", 90: "3", 85: "4"}[jpg_quality]
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-q:v",
        qscale,
        "-fps_mode",
        "passthrough",
        "-start_number",
        "0",
        str(destination / "frame_%05d.jpg"),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src}:\n{result.stderr.strip()}"
        )
    return sum(1 for _ in destination.glob("frame_*.jpg"))


def build_episode(
    source: Path,
    output: Path,
    task: str,
    episode: int,
    jpg_quality: int,
    overwrite: bool,
) -> tuple[str, int, str, dict[str, int]]:
    destination = output / task / f"ep{episode:06d}"
    if destination.exists() and not overwrite:
        return task, episode, "skip-exists", {}

    inputs = {
        view: source_video(source, task, episode, view) for view in VIEWS
    }
    missing = [path for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing source video(s): " + ", ".join(str(path) for path in missing)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".ep{episode:06d}.",
            dir=destination.parent,
        )
    )
    try:
        counts = {
            view: extract_video(
                inputs[view], staging / "video" / view, jpg_quality
            )
            for view in VIEWS
        }
        if not counts[VIEWS[0]] or len(set(counts.values())) != 1:
            raise RuntimeError(
                f"invalid frame counts for {task}/ep{episode:06d}: {counts}"
            )
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return task, episode, "ok", counts
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    selection_dir = args.selection_dir.expanduser().resolve()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    if not source.is_dir():
        raise SystemExit(f"source directory not found: {source}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    jobs: list[tuple[str, int]] = []
    for task in args.tasks:
        episodes = read_selection(selection_dir / f"{task}.txt")
        print(f"[selection] {task}: {len(episodes)} episodes")
        jobs.extend((task, episode) for episode in episodes)

    missing = [
        source_video(source, task, episode, view)
        for task, episode in jobs
        for view in VIEWS
        if not source_video(source, task, episode, view).is_file()
    ]
    if missing:
        for path in missing:
            print(f"[missing] {path}", file=sys.stderr)
        raise SystemExit(f"{len(missing)} source video(s) are missing")

    print(f"[source] {source}")
    print(f"[output] {output}")
    print(f"[episodes] {len(jobs)}")
    if args.dry_run:
        print("[dry-run] validation succeeded; no files written")
        return

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                build_episode,
                source,
                output,
                task,
                episode,
                args.jpg_quality,
                args.overwrite,
            ): (task, episode)
            for task, episode in jobs
        }
        for future in as_completed(futures):
            task, episode = futures[future]
            try:
                _, _, status, counts = future.result()
                detail = f" {counts}" if counts else ""
                print(
                    f"[{status}] {task}/ep{episode:06d}{detail}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                failures += 1
                print(
                    f"[error] {task}/ep{episode:06d}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    if failures:
        raise SystemExit(f"finished with {failures} failed episode(s)")
    print(f"[done] generated {len(jobs)} selected episodes")


if __name__ == "__main__":
    main()
