"""Pre-evaluation completeness check for generated data.

For every generated episode under ``val_base``, verify that the number of frame
images in each camera-view folder matches the ``n_frames`` declared in the
ground-truth ``prompt/prompts.json`` for that same (task, episode).

Layout::

    <gt_path>/<task>/<episode>/prompt/prompts.json      -> {"n_frames": N, ...}
    <val_base>/<task>/<episode>/video/<view>/frame_*.jpg

Views checked: whichever of (``top_head``, ``hand_left``, ``hand_right``) exist
under the episode's ``video/`` directory. If frames live directly under
``video/`` (no view subdir), that single directory is checked instead.

Run it with::

    python evaluate.py --check_data --config_path configs/base.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import IMAGE_EXTS, iter_episode_dirs, load_json

logger = logging.getLogger(__name__)

# Camera views we know about; only the ones that actually exist are checked.
CHECK_VIEWS = ("top_head", "hand_left", "hand_right")


def _count_frames(dir_path: Path) -> int:
    """Number of image files directly inside ``dir_path`` (0 if not a dir)."""
    if not dir_path.is_dir():
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.lower().endswith(IMAGE_EXTS))


def _read_n_frames(prompts_json: Path) -> Optional[int]:
    """Read ``n_frames`` from a GT ``prompts.json`` (None if missing/invalid)."""
    if not prompts_json.is_file():
        return None
    try:
        data = load_json(prompts_json)
    except Exception as e:  # noqa: BLE001 - report as missing, keep checking
        logger.warning("Failed to read %s: %s", prompts_json, e)
        return None
    if not isinstance(data, dict):
        return None
    n = data.get("n_frames")
    try:
        return int(n) if n is not None else None
    except (TypeError, ValueError):
        return None


def _norm_ep(x: str) -> str:
    s = str(x).strip().lower()
    return s[2:] if s.startswith("ep") else s


def check_generated_data(
    gt_path: str | os.PathLike,
    val_base: str | os.PathLike,
    task_filter: Optional[List[str]] = None,
    episode_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check frame-count completeness of every generated episode.

    Returns a dict with ``episodes`` (per-episode records), ``problems`` (the
    subset that is not OK), and a ``summary``.

    Each per-episode record has::

        {
          "task_id", "episode_id",
          "n_frames": <expected int or None>,
          "views": {"top_head": <count>, ...},
          "status": "ok" | "mismatch" | "error",
          "issues": ["...", ...],
        }
    """
    gt_root = Path(gt_path)
    val_root = Path(val_base)

    wanted_tasks = (
        {str(x).strip().lower() for x in task_filter} if task_filter else None
    )
    wanted_eps = {_norm_ep(x) for x in episode_filter} if episode_filter else None

    episodes: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []

    if not val_root.is_dir():
        raise FileNotFoundError(f"val_base does not exist or is not a directory: {val_root}")

    for ep in iter_episode_dirs(val_root):
        task_id = ep["task_id"]
        episode_id = ep["episode_id"]
        if wanted_tasks and task_id.strip().lower() not in wanted_tasks:
            continue
        if wanted_eps and _norm_ep(episode_id) not in wanted_eps:
            continue

        ep_dir = Path(ep["episode_dir"])
        video_dir = ep_dir / "video"
        prompts_json = gt_root / task_id / episode_id / "prompt" / "prompts.json"
        n_frames = _read_n_frames(prompts_json)

        record: Dict[str, Any] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "n_frames": n_frames,
            "views": {},
            "status": "ok",
            "issues": [],
        }

        if n_frames is None:
            record["status"] = "error"
            record["issues"].append(f"missing/invalid n_frames in {prompts_json}")

        if not video_dir.is_dir():
            record["status"] = "error"
            record["issues"].append(f"no video/ directory at {video_dir}")
        else:
            present_views = [v for v in CHECK_VIEWS if (video_dir / v).is_dir()]
            if present_views:
                for view in present_views:
                    count = _count_frames(video_dir / view)
                    record["views"][view] = count
                    if count == 0:
                        record["status"] = "error"
                        record["issues"].append(f"{view}: no frames found")
                    elif n_frames is not None and count != n_frames:
                        if record["status"] != "error":
                            record["status"] = "mismatch"
                        record["issues"].append(
                            f"{view}: {count} frames != n_frames {n_frames} "
                            f"(diff {count - n_frames:+d})"
                        )
            else:
                # No view subdir: frames may live directly under video/.
                count = _count_frames(video_dir)
                record["views"]["video"] = count
                if count == 0:
                    record["status"] = "error"
                    record["issues"].append(
                        f"no view subdir ({'/'.join(CHECK_VIEWS)}) and no frames under {video_dir}"
                    )
                elif n_frames is not None and count != n_frames:
                    if record["status"] != "error":
                        record["status"] = "mismatch"
                    record["issues"].append(
                        f"video: {count} frames != n_frames {n_frames} "
                        f"(diff {count - n_frames:+d})"
                    )

        episodes.append(record)
        if record["status"] != "ok":
            problems.append(record)

    n_total = len(episodes)
    n_ok = n_total - len(problems)
    n_mismatch = sum(1 for r in problems if r["status"] == "mismatch")
    n_error = sum(1 for r in problems if r["status"] == "error")

    summary = {
        "n_episodes": n_total,
        "n_ok": n_ok,
        "n_mismatch": n_mismatch,
        "n_error": n_error,
        "all_ok": len(problems) == 0,
    }

    return {"episodes": episodes, "problems": problems, "summary": summary}


def print_check_report(report: Dict[str, Any], printer=print) -> None:
    """Pretty-print a completeness report produced by ``check_generated_data``."""
    summary = report["summary"]
    problems = report["problems"]

    printer("")
    printer("================ generated data completeness check ================")
    printer(f"  episodes checked : {summary['n_episodes']}")
    printer(f"  OK               : {summary['n_ok']}")
    printer(f"  mismatch (count) : {summary['n_mismatch']}")
    printer(f"  error            : {summary['n_error']}")

    if problems:
        printer("  problems:")
        for r in problems:
            views = ", ".join(f"{k}={v}" for k, v in r["views"].items()) or "-"
            printer(
                f"    [{r['status']}] {r['task_id']}/{r['episode_id']} "
                f"(n_frames={r['n_frames']}; {views})"
            )
            for issue in r["issues"]:
                printer(f"        - {issue}")
    else:
        printer("  all episodes have the expected frame counts. ✅")
    printer("===================================================================")
