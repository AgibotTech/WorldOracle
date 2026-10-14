"""Data discovery for ACWMBench.

Walks the EWMBench-compatible directory layout and produces a list of
`EpisodeTrial` items, each of which represents one (task, episode, trial)
combination plus all the auxiliary metadata required by the metrics
(instruction text, optional task_spec, optional key-step indices, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import json_repair

from .io_utils import iter_episode_dirs, load_json, read_text


@dataclass
class EpisodeTrial:
    task_id: str
    episode_id: str
    trial_id: str
    video_dir: str
    instruction: str = ""
    init_frame: Optional[str] = None
    task_spec: Optional[dict] = None
    keysteps: Optional[List[int]] = None
    prompt_segments: Optional[List[dict]] = None
    grounding_objects: Optional[List[str]] = None
    gt_video_dir: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.task_id}__{self.episode_id}__{self.trial_id}"


def _read_instruction(episode_dir: Path) -> str:
    intro = episode_dir / "prompt" / "introduction.txt"
    if intro.is_file():
        return read_text(intro)
    return ""


def _read_task_spec(task_dir: Path, episode_dir: Path) -> Optional[dict]:
    """Look for a task_spec.json (episode-level overrides task-level)."""
    for path in (episode_dir / "task_spec.json", task_dir / "task_spec.json"):
        if path.is_file():
            return load_json(path)
    return None


def _read_keysteps(episode_dir: Path) -> Optional[List[int]]:
    path = episode_dir / "keysteps.json"
    if not path.is_file():
        return None
    data = load_json(path)
    if isinstance(data, list):
        return [int(x) for x in data]
    if isinstance(data, dict) and "frames" in data:
        return [int(x) for x in data["frames"]]
    return None


def _read_prompt_segments(episode_dir: Path) -> Optional[List[dict]]:
    """Read the per-episode physical-status prompts.

    Prefers the ``prompt/prompts.json`` layout; falls back to the flat
    ``prompt.json`` produced by the worldsim exporter before migration.
    """
    path = episode_dir / "prompt" / "prompts.json"
    if not path.is_file():
        path = episode_dir / "prompt.json"
    if not path.is_file():
        return None

    text = read_text(path)
    try:
        data: Any = json_repair.loads(text)
    except Exception:
        data = load_json(path)

    if not isinstance(data, dict):
        return None
    n_frames = data.get("n_frames")
    segments = data.get("segments")
    if not isinstance(segments, list):
        return None

    out: List[dict] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        # The worldsim generator writes ``key_frame``; older GT manifests use
        # ``end_frame``. Accept either as the frame this segment is judged at.
        frame = seg.get("end_frame", seg.get("key_frame"))
        if frame is None:
            continue
        criteria = seg.get("success_criteria", "")
        if isinstance(criteria, list):
            criteria = [str(x).strip() for x in criteria if str(x).strip()]
        else:
            criteria = str(criteria).strip()
        if not criteria:
            continue
        out.append(
            {
                "segment_id": int(seg.get("segment_id", idx)),
                "end_frame": int(frame),
                "source_n_frames": int(n_frames) if n_frames is not None else None,
                "success_criteria": criteria,
            }
        )
    return out or None


def _read_grounding_objects(episode_dir: Path) -> Optional[List[str]]:
    """Read the per-episode object inventory (``grounding_objects``) if present."""
    path = episode_dir / "prompt" / "prompts.json"
    if not path.is_file():
        path = episode_dir / "prompt.json"
    if not path.is_file():
        return None
    text = read_text(path)
    try:
        data: Any = json_repair.loads(text)
    except Exception:
        data = load_json(path)
    if not isinstance(data, dict):
        return None
    objs = data.get("grounding_objects")
    if not isinstance(objs, list):
        return None
    out = [str(x).strip() for x in objs if str(x).strip()]
    return out or None


def _init_frame(episode_dir: Path) -> Optional[str]:
    cand = episode_dir / "prompt" / "init_frame.png"
    return str(cand) if cand.is_file() else None


def _append_trial(
    trials: List[EpisodeTrial],
    *,
    task_id: str,
    episode_id: str,
    trial_id: str,
    video_dir: Path,
    instruction: str,
    init_frame: Optional[str],
    task_spec: Optional[dict],
    keysteps: Optional[List[int]],
    prompt_segments: Optional[List[dict]],
    grounding_objects: Optional[List[str]],
    gt_video_dir: Optional[str],
    meta: Optional[dict] = None,
) -> None:
    trials.append(
        EpisodeTrial(
            task_id=str(task_id),
            episode_id=str(episode_id),
            trial_id=str(trial_id),
            video_dir=str(video_dir),
            instruction=instruction,
            init_frame=init_frame,
            task_spec=task_spec,
            keysteps=keysteps,
            prompt_segments=prompt_segments,
            grounding_objects=grounding_objects,
            gt_video_dir=gt_video_dir,
            meta=meta or {},
        )
    )


def build_trial_list(data_base: str | os.PathLike, gt_path: str | os.PathLike) -> List[EpisodeTrial]:
    """Enumerate every (task, episode, trial) combination under `data_base`.

    The corresponding GT instruction / task_spec / keysteps are read from
    `gt_path/<task>/<episode>/...` whenever available.
    """
    gt_root = Path(gt_path)
    trials: List[EpisodeTrial] = []

    for ep in iter_episode_dirs(data_base):
        task_id = ep["task_id"]
        episode_id = ep["episode_id"]
        episode_dir = Path(ep["episode_dir"])

        gt_task_dir = gt_root / task_id
        gt_episode_dir = gt_task_dir / episode_id
        instruction = _read_instruction(gt_episode_dir) if gt_episode_dir.is_dir() else ""
        task_spec = _read_task_spec(gt_task_dir, gt_episode_dir) if gt_task_dir.is_dir() else None
        keysteps = _read_keysteps(gt_episode_dir) if gt_episode_dir.is_dir() else None
        prompt_segments = _read_prompt_segments(gt_episode_dir) if gt_episode_dir.is_dir() else None
        grounding_objects = _read_grounding_objects(gt_episode_dir) if gt_episode_dir.is_dir() else None
        init_frame = _init_frame(gt_episode_dir) if gt_episode_dir.is_dir() else None
        gt_video_dir = gt_episode_dir / "video"
        gt_video_dir_str = str(gt_video_dir) if gt_video_dir.is_dir() else None

        # New ACWMBench layout: <test_dataset>/<task>/<episode>/video/<view>/...
        direct_video_dir = episode_dir / "video"
        if direct_video_dir.is_dir():
            meta_path = episode_dir / "meta.json"
            meta = load_json(meta_path) if meta_path.is_file() else {}
            _append_trial(
                trials,
                task_id=str(task_id),
                episode_id=str(episode_id),
                trial_id=str(meta.get("trial_id", "default")),
                video_dir=direct_video_dir,
                instruction=instruction,
                init_frame=init_frame,
                task_spec=task_spec,
                keysteps=keysteps,
                prompt_segments=prompt_segments,
                grounding_objects=grounding_objects,
                gt_video_dir=gt_video_dir_str,
                meta=meta,
            )
            continue

        # Legacy layout: <generated>/<task>/<episode>/<trial_id>/video/...
        for trial_id in sorted(os.listdir(episode_dir)):
            trial_path = episode_dir / trial_id
            if not trial_path.is_dir():
                continue
            video_dir = trial_path / "video"
            if not video_dir.is_dir():
                continue

            meta_path = trial_path / "meta.json"
            meta = load_json(meta_path) if meta_path.is_file() else {}

            _append_trial(
                trials,
                task_id=str(task_id),
                episode_id=str(episode_id),
                trial_id=str(trial_id),
                video_dir=video_dir,
                instruction=instruction,
                init_frame=init_frame,
                task_spec=task_spec,
                keysteps=keysteps,
                prompt_segments=prompt_segments,
                grounding_objects=grounding_objects,
                gt_video_dir=gt_video_dir_str,
                meta=meta,
            )
    return trials
