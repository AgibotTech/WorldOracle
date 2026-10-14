"""Generate PI0.5/GE-Sim closed loops from policy_eval_dataset episodes."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from .io_utils import list_frame_files
from .policy_eval import (
    POLICY_ACTION_LAYOUT,
    WM_ACTION_LAYOUT,
    ACWMPredictedStateProvider,
    ClosedLoopOrchestrator,
    CommandedStateProvider,
    ControlProtocol,
    Observation,
    OpenPIWebsocketPolicyClient,
    PolicyClient,
    RolloutMetadata,
    StateProvider,
    VIEW_NAMES,
    WorldModelClient,
)

CLOSED_LOOP_PROTOCOL_VERSION = 11


@dataclass(frozen=True)
class PolicyEvalEpisode:
    root: Path
    task_id: str
    episode_id: str
    metadata: Mapping[str, Any]
    frames: Mapping[str, tuple[Path, ...]]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _normal_episode_id(value: str) -> str:
    text = str(value).strip().lower()
    return text[2:] if text.startswith("ep") else text


def discover_policy_eval_episodes(
    dataset_root: str | Path,
    *,
    task_filter: Iterable[str] | None = None,
    episode_filter: Iterable[str] | None = None,
    limit_episodes: int | None = None,
    limit_episodes_per_task: int | None = None,
) -> list[PolicyEvalEpisode]:
    """Discover policy-evaluation episodes and their three RGB timelines."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"policy_eval_dataset root does not exist: {root}")
    wanted_tasks = {str(value).strip().lower() for value in task_filter or ()}
    wanted_episodes = {_normal_episode_id(value) for value in episode_filter or ()}
    episodes: list[PolicyEvalEpisode] = []
    task_counts: dict[str, int] = {}
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_id = task_dir.name
        if task_id.startswith(("_", ".")):
            continue
        if wanted_tasks and task_id.lower() not in wanted_tasks:
            continue
        for episode_dir in sorted(
            path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("ep")
        ):
            if wanted_episodes and _normal_episode_id(episode_dir.name) not in wanted_episodes:
                continue
            if (
                limit_episodes_per_task is not None
                and limit_episodes_per_task > 0
                and task_counts.get(task_id, 0) >= limit_episodes_per_task
            ):
                break
            frames = {
                name: tuple(
                    Path(path)
                    for path in list_frame_files(
                        episode_dir / "video" / name
                    )
                )
                for name in VIEW_NAMES
            }
            counts = {name: len(paths) for name, paths in frames.items()}
            if min(counts.values(), default=0) < 2:
                raise FileNotFoundError(
                    f"{episode_dir} needs at least 2 frames in every "
                    f"video/<view> directory; got {counts}"
                )
            if len(set(counts.values())) != 1:
                raise ValueError(
                    f"{episode_dir} view frame counts differ: {counts}"
                )
            frame_count = next(iter(counts.values()))
            prompts_path = episode_dir / "prompt" / "prompts.json"
            prompts = _load_json(prompts_path) if prompts_path.is_file() else {}
            declared_frames = int(prompts.get("n_frames", frame_count))
            if declared_frames != frame_count:
                raise ValueError(
                    f"{episode_dir}: prompts.json declares {declared_frames} "
                    f"frames but video contains {frame_count}"
                )
            meta_path = episode_dir / "meta.json"
            episode_meta = _load_json(meta_path) if meta_path.is_file() else {}
            episodes.append(
                PolicyEvalEpisode(
                    root=episode_dir,
                    task_id=task_id,
                    episode_id=episode_dir.name,
                    metadata={
                        **episode_meta,
                        "fps": float(episode_meta.get("fps", 30.0)),
                        "num_frames": declared_frames,
                    },
                    frames=frames,
                )
            )
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            if limit_episodes is not None and limit_episodes > 0:
                if len(episodes) >= limit_episodes:
                    return episodes
    if not episodes:
        raise ValueError(f"no matching policy-evaluation episodes under {root}")
    return episodes


def load_initial_observation(
    episode: PolicyEvalEpisode, initial_states_path: str | Path
) -> Observation:
    """Load policy_eval_dataset frame zero plus task state/instruction."""

    config_path = Path(initial_states_path)
    with config_path.open("r", encoding="utf-8") as handle:
        all_states = yaml.safe_load(handle) or {}
    task_state = all_states.get(episode.task_id)
    if not isinstance(task_state, Mapping):
        raise KeyError(
            f"{config_path} has no initial state for task {episode.task_id!r}"
        )
    arm_joints = np.asarray(task_state.get("arm_joints"), dtype=np.float32)
    grippers = np.asarray(task_state.get("grippers"), dtype=np.float32)
    if arm_joints.shape != (14,) or grippers.shape != (2,):
        raise ValueError(
            f"{config_path}: {episode.task_id} needs 14 arm_joints and 2 grippers"
        )
    state = np.concatenate((arm_joints, grippers)).astype(np.float32)
    head = np.asarray(task_state.get("head"), dtype=np.float32)
    waist = np.asarray(task_state.get("waist"), dtype=np.float32)
    if head.shape != (3,) or waist.shape != (5,):
        raise ValueError(
            f"{config_path}: {episode.task_id} needs 3 head and 5 waist values"
        )
    instruction = str(task_state.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(
            f"{config_path}: {episode.task_id} needs a non-empty instruction"
        )
    views = {}
    for name in VIEW_NAMES:
        path = episode.frames[name][0]
        with Image.open(path) as image:
            views[name] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Observation(
        top_head=views["top_head"],
        hand_left=views["hand_left"],
        hand_right=views["hand_right"],
        state=state,
        task=instruction,
        robot_context={"head": head, "waist": waist},
    )


def _state_provider(name: str) -> StateProvider:
    normalized = str(name).strip().lower()
    if normalized == "commanded":
        return CommandedStateProvider()
    if normalized == "acwm_predicted":
        return ACWMPredictedStateProvider()
    raise ValueError("state_source must be 'commanded' or 'acwm_predicted'")


def _target_frame_count(episode: PolicyEvalEpisode) -> int:
    return len(episode.frames["top_head"])


def _rollout_complete(path: Path, expected: Mapping[str, Any]) -> bool:
    metadata_path = path / "meta.json"
    if not metadata_path.is_file():
        return False
    try:
        data = _load_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    extra = data.get("extra") or {}
    return extra.get("status") == "complete" and all(
        extra.get(key) == value for key, value in expected.items()
    )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    Image.fromarray(array).save(path, quality=95, subsampling=0)


def _persist_rollout(
    destination: Path,
    *,
    metadata: RolloutMetadata,
    initial: Observation,
    trace: Sequence[Any],
    state_provider: StateProvider,
    target_frames: int,
    overwrite: bool,
) -> None:
    expected = dict(metadata.extra)
    if not overwrite and _rollout_complete(destination, expected):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        actions = np.concatenate(
            [np.asarray(step.consumed_wm_actions, dtype=np.float32) for step in trace],
            axis=0,
        )[: max(0, target_frames - 1)]
        feedback_states = [initial.state]
        for step in trace:
            feedback_states.append(
                state_provider.state(step.consumed_wm_actions, step.output)
            )
        np.save(staging / "actions.npy", actions, allow_pickle=False)
        np.save(
            staging / "states.npy",
            np.asarray(feedback_states, dtype=np.float32),
            allow_pickle=False,
        )
        video_root = staging / "video"
        frame_indices = {name: 0 for name in VIEW_NAMES}
        for name, image in initial.views.items():
            view_dir = video_root / name
            view_dir.mkdir(parents=True)
            _save_rgb(view_dir / "frame_00000.jpg", image)
            frame_indices[name] = 1
        for step in trace:
            for name, sequence in step.output.frames.items():
                for frame in sequence:
                    if frame_indices[name] >= target_frames:
                        break
                    index = frame_indices[name]
                    _save_rgb(
                        video_root / name / f"frame_{index:05d}.jpg",
                        frame,
                    )
                    frame_indices[name] = index + 1
        output_frames = frame_indices["top_head"]
        if output_frames != target_frames:
            raise RuntimeError(
                "rollout top_head produced "
                f"{output_frames} frames, expected {target_frames}"
            )
        payload = metadata.to_dict()
        payload["extra"] = {
            **dict(metadata.extra),
            "status": "complete",
            "output_frames": output_frames,
            "view_frame_counts": frame_indices,
            "policy_rounds_completed": len(trace),
        }
        (staging / "meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_wm_policy_rollouts(
    *,
    dataset_root: str | Path,
    rollout_root: str | Path,
    initial_states_path: str | Path,
    policy: PolicyClient,
    world_model: WorldModelClient,
    policy_id: str,
    policy_checkpoint: str,
    world_model_id: str,
    world_model_checkpoint: str,
    action_horizon: int,
    wm_chunk_size: int,
    compress_actions: bool,
    state_source: str,
    rollout_number: int,
    task_filter: Iterable[str] | None = None,
    episode_filter: Iterable[str] | None = None,
    limit_episodes: int | None = None,
    limit_episodes_per_task: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Run selected WM-only closed loops and atomically persist their artifacts."""

    if rollout_number <= 0:
        raise ValueError("rollout_number must be positive")
    if not compress_actions and action_horizon > wm_chunk_size:
        raise ValueError(
            "the first GE-Sim AR integration requires compress_actions=true when "
            "the PI horizon exceeds the WM chunk"
        )
    episodes = discover_policy_eval_episodes(
        dataset_root,
        task_filter=task_filter,
        episode_filter=episode_filter,
        limit_episodes=limit_episodes,
        limit_episodes_per_task=limit_episodes_per_task,
    )
    provider = _state_provider(state_source)
    target_horizon = wm_chunk_size if compress_actions else action_horizon
    orchestrator = ClosedLoopOrchestrator(
        policy,
        world_model,
        target_horizon=target_horizon,
        policy_horizon=action_horizon,
        state_provider=provider,
    )
    output_paths: list[Path] = []
    for episode in episodes:
        initial = load_initial_observation(episode, initial_states_path)
        target_frames = _target_frame_count(episode)
        policy_rounds = int(
            math.ceil((target_frames - 1) / target_horizon)
        )
        for rollout_index in range(int(rollout_number)):
            seed = 42 + rollout_index
            rollout_id = str(rollout_index)
            destination = (
                Path(rollout_root)
                / world_model_id
                / episode.task_id
                / episode.episode_id
                / rollout_id
            )
            identity = {
                "closed_loop_protocol_version": CLOSED_LOOP_PROTOCOL_VERSION,
                "dataset_root": str(Path(dataset_root).resolve()),
                "initial_states_path": str(Path(initial_states_path).resolve()),
                "world_model_checkpoint": str(world_model_checkpoint),
                "rollout_index": rollout_index,
                "seed": seed,
                "policy_seed": seed,
                "world_model_seed": seed,
                "target_frames": target_frames,
                "policy_rounds": int(policy_rounds),
                "wm_chunk_size": int(wm_chunk_size),
                "action_horizon": int(action_horizon),
                "compress_actions": bool(compress_actions),
                "action_chunk_strategy": "fixed_prefix_trim_output",
                "state_source": str(state_source),
            }
            if not overwrite and _rollout_complete(destination, identity):
                output_paths.append(destination)
                continue
            configure = getattr(world_model, "configure_episode", None)
            if callable(configure):
                configure(episode.root, seed=seed)
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                reset_policy(seed=seed)
            trace = orchestrator.run(initial, policy_rounds)
            metadata = RolloutMetadata(
                policy_id=policy_id,
                task_id=episode.task_id,
                init_id=episode.episode_id,
                domain="wm",
                rollout_id=rollout_id,
                checkpoint=str(policy_checkpoint),
                prompt=initial.task,
                protocol=ControlProtocol(
                    fps=float(episode.metadata.get("fps", 30.0)),
                    prediction_horizon=int(action_horizon),
                    replan_horizon=int(wm_chunk_size),
                    action_layout=WM_ACTION_LAYOUT,
                    state_layout=POLICY_ACTION_LAYOUT,
                    state_source=str(state_source),
                    # The legacy config key enables prefix truncation, not
                    # temporal averaging/compression.
                    action_compression=False,
                ),
                seed=seed,
                world_model_id=world_model_id,
                termination_reason="target_frame_count",
                extra=identity,
            )
            _persist_rollout(
                destination,
                metadata=metadata,
                initial=initial,
                trace=trace,
                state_provider=provider,
                target_frames=target_frames,
                overwrite=overwrite,
            )
            output_paths.append(destination)
    return output_paths


def build_policy_client(config: Mapping[str, Any]) -> OpenPIWebsocketPolicyClient:
    """Construct the production PI0.5 websocket client from YAML settings."""

    server_url = str(config.get("server_url", "ws://127.0.0.1:8000"))
    return OpenPIWebsocketPolicyClient(server_url)
