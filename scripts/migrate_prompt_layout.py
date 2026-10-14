#!/usr/bin/env python3
"""Migrate the flat prompt layout to the prompt/ subdir layout.

For every episode under ``--data-root`` that still stores its manifest as::

    <data-root>/ep000000/prompt.json

this script creates the ``prompt/`` subdir and copies the file to::

    <data-root>/ep000000/prompt/prompts.json

This is the layout expected by the viewer (lerobot_viewer/vlm_app.py) and the
eval reader (acwmbench/dataset.py). The original ``prompt.json`` is kept by
default (copy); pass ``--move`` to remove it after a successful copy.

Usage::

    python scripts/migrate_prompt_layout.py
    python scripts/migrate_prompt_layout.py \
        --data-root /path/to/task-data
    python scripts/migrate_prompt_layout.py --move --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    "/path/to/task-data"
)


def iter_episode_dirs(data_root: Path):
    """Yield top-level ``ep*`` directories (sorted, non-recursive)."""
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.startswith("ep"):
            yield p


def migrate_episode(ep_dir: Path, *, move: bool, overwrite: bool) -> str:
    """Migrate one episode. Returns a short status string for logging."""
    src = ep_dir / "prompt.json"
    dst = ep_dir / "prompt" / "prompts.json"

    if not src.is_file():
        # Already migrated (only prompt/prompts.json present) or nothing to do.
        if dst.is_file():
            return "skip (already migrated)"
        return "skip (no prompt.json)"

    if dst.is_file() and not overwrite:
        return "skip (dest exists; use --overwrite)"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if move:
        src.unlink()
        return "moved"
    return "copied"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root dir containing ep*/prompt.json (default: %(default)s).",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Remove the original prompt.json after a successful copy (default: keep it).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite prompt/prompts.json if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing anything.",
    )
    args = p.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data-root is not a directory: {data_root}")

    counts: dict[str, int] = {}
    for ep_dir in iter_episode_dirs(data_root):
        if args.dry_run:
            src = ep_dir / "prompt.json"
            dst = ep_dir / "prompt" / "prompts.json"
            if not src.is_file():
                status = "skip (already migrated)" if dst.is_file() else "skip (no prompt.json)"
            elif dst.is_file() and not args.overwrite:
                status = "skip (dest exists; use --overwrite)"
            else:
                status = "would move" if args.move else "would copy"
        else:
            status = migrate_episode(ep_dir, move=args.move, overwrite=args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        print(f"{ep_dir.name}: {status}", flush=True)

    print("\nSummary:", flush=True)
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}", flush=True)
    if not counts:
        print("  (no ep* directories found)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
