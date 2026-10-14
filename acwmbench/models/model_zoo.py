"""Single source of truth for supported VLM checkpoints.

Both download scripts, the smoke/perf tooling, and `evaluate.py` resolve
short aliases through `resolve_model_path()` so swapping checkpoints never
requires editing YAML.

Supported short aliases:
  - "qwen3_vl_8b" / "qwen3-vl-8b"  -> Qwen/Qwen3-VL-8B-Instruct  (default)
  - "qwen3_5_9b"  / "qwen3.5-9b"   -> Qwen/Qwen3.5-9B

You can also pass:
  - a full HF repo id "<owner>/<name>" (resolved against local ckpt/)
  - an absolute / relative path to an existing model directory
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Canonical alias -> HF repo id. Keep flat; do NOT add ordering semantics here.
MODEL_ALIASES: dict[str, str] = {
    "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl8b":   "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_5_9b":  "Qwen/Qwen3.5-9B",
    "qwen3.5-9b":  "Qwen/Qwen3.5-9B",
    "qwen3.5_9b":  "Qwen/Qwen3.5-9B",
    "qwen35-9b":   "Qwen/Qwen3.5-9B",
}

# Used by smoke/perf tools when no explicit alias/path is given.
DEFAULT_SEARCH_ORDER: tuple[str, ...] = (
    "qwen3_vl_8b",
    "qwen3_5_9b",
)


def repo_id_to_local_name(repo_id: str) -> str:
    """`Qwen/Qwen3-VL-8B-Instruct` -> `Qwen3-VL-8B-Instruct`."""
    return repo_id.split("/", 1)[-1]


def resolve_model_path(
    alias_or_path: str,
    ckpt_root: str | os.PathLike[str] = "ckpt",
) -> str:
    """Resolve a short alias / repo id / path to an on-disk model directory.

    Resolution order:
      1. if `alias_or_path` is an existing directory -> return it as-is
      2. if it's a known alias -> map to repo id, then to `ckpt_root/<name>`
      3. if it looks like `owner/name` -> `ckpt_root/<name>`
      4. else: treat as a path (may not exist yet -- caller decides)
    """
    if not alias_or_path:
        raise ValueError("resolve_model_path: empty alias_or_path")

    candidate = Path(alias_or_path)
    if candidate.exists() and candidate.is_dir():
        return str(candidate)

    lookup = alias_or_path.lower()
    if lookup in MODEL_ALIASES:
        repo = MODEL_ALIASES[lookup]
        return str(Path(ckpt_root) / repo_id_to_local_name(repo))

    if "/" in alias_or_path and not alias_or_path.startswith(("/", ".", "~")):
        return str(Path(ckpt_root) / repo_id_to_local_name(alias_or_path))

    return str(candidate)


def auto_pick_default(ckpt_root: str | os.PathLike[str] = "ckpt") -> Optional[str]:
    """Return the first model in DEFAULT_SEARCH_ORDER whose dir exists."""
    for alias in DEFAULT_SEARCH_ORDER:
        path = resolve_model_path(alias, ckpt_root=ckpt_root)
        if Path(path).exists():
            return path
    return None
