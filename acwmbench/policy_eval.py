"""Contracts and action adapters for policy/world-model rollouts.

Policy-evaluation rollouts use this directory layout::

    <rollout_root>/<world_model_id>/<task_id>/<init_id>/<rollout_id>/
      meta.json
      actions.npy
      states.npy
      video/<view>/frame_*.jpg

Legacy paired real/WM layouts are also discoverable. ``meta.json`` records the
policy checkpoint, control protocol, world-model seed, and state source.
Offline scoring stores a Physical-Judge-compatible ``policy_eval.json`` in
each episode directory, keyed by rollout ID. Legacy per-rollout
``physical_judge.json`` files remain readable.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence
from urllib import error, request

# The IDE's lightweight lint interpreter may not load project dependencies.
# pylint: disable=import-error
import numpy as np


POLICY_ACTION_LAYOUT = "[L7_arm,R7_arm,L_grip,R_grip]"
WM_ACTION_LAYOUT = "[L7_arm,L_grip,R7_arm,R_grip]"
ENERVERSE_ACTION_LAYOUT = "[L_xyz,L_quat_xyzw,L_grip,R_xyz,R_quat_xyzw,R_grip]"
ROLLOUT_SCHEMA_VERSION = 1
ROLLOUT_DOMAINS = ("real", "wm")
VIEW_NAMES = ("top_head", "hand_left", "hand_right")


def _validate_image(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 image, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have shape (H, W, C), got {arr.shape}")
    return arr


def validate_policy_state(state: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a finite policy state in ``[L7, R7, Lgrip, Rgrip]`` layout."""

    arr = np.asarray(state, dtype=np.float32)
    if arr.shape != (16,):
        raise ValueError(f"expected policy state with shape (16,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("policy state contains NaN or infinity")
    return arr


@dataclass(frozen=True)
class Observation:
    """One policy input with a required head view and optional wrist views."""

    top_head: np.ndarray
    hand_left: np.ndarray | None
    hand_right: np.ndarray | None
    state: np.ndarray
    task: str
    robot_context: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "top_head", _validate_image("top_head", self.top_head)
        )
        for name in ("hand_left", "hand_right"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _validate_image(name, value))
        object.__setattr__(self, "state", validate_policy_state(self.state))
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if self.robot_context is not None:
            context = {
                str(key): np.asarray(value, dtype=np.float32).copy()
                for key, value in self.robot_context.items()
            }
            if context.get("head", np.empty(0)).shape != (3,):
                raise ValueError("robot_context.head must have shape (3,)")
            if context.get("waist", np.empty(0)).shape != (5,):
                raise ValueError("robot_context.waist must have shape (5,)")
            if not all(np.isfinite(value).all() for value in context.values()):
                raise ValueError("robot_context contains NaN or infinity")
            object.__setattr__(self, "robot_context", context)

    @property
    def views(self) -> dict[str, np.ndarray]:
        return {
            name: value
            for name in VIEW_NAMES
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True)
class OpenPIKeys:
    """Names used by a particular OpenPI policy server."""

    # pi05_agibot's AgibotInputs transform consumes these raw sample keys.
    top_head: str = "top_head"
    hand_left: str = "hand_left"
    hand_right: str = "hand_right"
    state: str = "state"
    task: str = "prompt"
    actions: str = "actions"


class PolicyClient(Protocol):
    def reset(self, seed: int = 42) -> None: ...

    def infer(self, observation: Observation) -> np.ndarray: ...


class OpenPIWebsocketPolicyClient:
    """Thin OpenPI websocket adapter with no import-time OpenPI dependency."""

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        keys: OpenPIKeys | None = None,
        client_factory: Callable[[str, int], Any] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port) if port is not None else None
        self.keys = keys or OpenPIKeys()
        self._client_factory = client_factory
        self._client: Any | None = None
        self._seed = 42

    def _connect(self) -> Any:
        if self._client is None:
            factory = self._client_factory
            if factory is None:
                try:
                    module = importlib.import_module(
                        "openpi_client.websocket_client_policy"
                    )
                except ImportError as exc:
                    raise ImportError(
                        "OpenPI policy rollout requires the optional openpi-client package"
                    ) from exc
                factory = module.WebsocketClientPolicy
            self._client = (
                factory(self.host)
                if self.port is None
                else factory(self.host, self.port)
            )
        return self._client

    def infer(self, observation: Observation) -> np.ndarray:
        keys = self.keys
        payload = {
            keys.top_head: observation.top_head,
            keys.state: observation.state,
            keys.task: observation.task,
        }
        if observation.hand_left is not None:
            payload[keys.hand_left] = observation.hand_left
        if observation.hand_right is not None:
            payload[keys.hand_right] = observation.hand_right
        connection = self._connect()
        response = connection.infer(payload)
        if not isinstance(response, Mapping) or keys.actions not in response:
            raise ValueError(f"OpenPI response must contain {keys.actions!r}")
        actions = np.asarray(response[keys.actions], dtype=np.float32)
        # pi05_agibot has model action_dim=32 but only the first 16 dimensions
        # are real robot channels; the transform stack zero-pads the remainder.
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise ValueError(
                "OpenPI actions must have shape (horizon, >=16), "
                f"got {actions.shape}"
            )
        return validate_policy_actions(actions[:, :16])

    def reset(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._connect().reset(seed=self._seed)


def policy_to_wm_layout(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Reorder final dimension from ``[L7,R7,Lg,Rg]`` to ``[L7,Lg,R7,Rg]``."""

    arr = np.asarray(values)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 16:
        raise ValueError(f"expected shape (16,) or (T, 16), got {arr.shape}")
    return arr[..., [*range(7), 14, *range(7, 14), 15]].copy()


def wm_to_policy_layout(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Reorder final dimension from ``[L7,Lg,R7,Rg]`` to ``[L7,R7,Lg,Rg]``."""

    arr = np.asarray(values)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 16:
        raise ValueError(f"expected shape (16,) or (T, 16), got {arr.shape}")
    return arr[..., [*range(7), *range(8, 15), 7, 15]].copy()


def select_policy_action_prefix(
    actions: np.ndarray | Sequence[Sequence[float]], target_horizon: int
) -> np.ndarray:
    """Select the executable prefix for receding-horizon control.

    One policy action corresponds to one generated frame. If the policy predicts
    farther than the world model can consume, execute only the first
    ``target_horizon`` actions and replan from the resulting observation.
    """

    arr = validate_policy_actions(actions)
    target = int(target_horizon)
    if target <= 0:
        raise ValueError("target_horizon must be positive")
    if target > len(arr):
        raise ValueError("action prefix selection cannot increase the horizon")
    if target == len(arr):
        return arr.copy()
    return arr[:target].copy()


# Backward-compatible import name for older callers/config terminology.
compress_policy_actions = select_policy_action_prefix


@dataclass(frozen=True)
class WorldModelOutput:
    """WM output with policy-feedback views and optional persistence frames.

    ``views`` may stay at the model's native resolution while ``frames`` is
    resized to the benchmark dataset resolution. They must describe the same
    cameras and timeline, but their pixel arrays need not be identical.
    """

    views: Mapping[str, np.ndarray]
    predicted_state: np.ndarray | None = None
    frames: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        names = tuple(name for name in VIEW_NAMES if name in self.views)
        unknown = set(self.views) - set(VIEW_NAMES)
        if unknown:
            raise ValueError(f"world-model output has unknown views: {sorted(unknown)}")
        if "top_head" not in names:
            raise ValueError("world-model output must include top_head")
        checked: dict[str, np.ndarray] = {}
        implicit_sequences: dict[str, np.ndarray] = {}
        for name in names:
            value = np.asarray(self.views[name])
            if value.ndim == 4:
                if len(value) == 0:
                    raise ValueError(f"{name} generated frame sequence must not be empty")
                implicit_sequences[name] = value
                value = value[-1]
            checked[name] = _validate_image(name, value)
        object.__setattr__(self, "views", checked)
        if self.frames is None:
            sequences = {
                name: implicit_sequences.get(name, image[None])
                for name, image in checked.items()
            }
            if len({len(sequence) for sequence in sequences.values()}) != 1:
                raise ValueError("world-model views must have equal frame counts")
        else:
            unknown_frames = set(self.frames) - set(VIEW_NAMES)
            if unknown_frames:
                raise ValueError(
                    "world-model frame sequences have unknown views: "
                    f"{sorted(unknown_frames)}"
                )
            missing_frames = set(names) - set(self.frames)
            extra_frames = set(self.frames) - set(names)
            if missing_frames or extra_frames:
                raise ValueError("world-model frames and views must use the same cameras")
            sequences = {}
            lengths = set()
            for name in names:
                sequence = np.asarray(self.frames[name])
                if sequence.dtype != np.uint8:
                    raise ValueError(
                        f"{name} frame sequence must be uint8, got {sequence.dtype}"
                    )
                if sequence.ndim != 4 or sequence.shape[-1] not in (1, 3, 4):
                    raise ValueError(
                        f"{name} frame sequence must be (T,H,W,C), got {sequence.shape}"
                    )
                if len(sequence) == 0:
                    raise ValueError("world-model frame sequences must not be empty")
                sequences[name] = sequence
                lengths.add(len(sequence))
            if len(lengths) != 1:
                raise ValueError("world-model views must have equal frame counts")
        object.__setattr__(self, "frames", sequences)
        if self.predicted_state is not None:
            object.__setattr__(
                self, "predicted_state", validate_policy_state(self.predicted_state)
            )


class WorldModelClient(Protocol):
    def reset(self, observation: Observation) -> WorldModelOutput: ...

    def step(self, actions: np.ndarray) -> WorldModelOutput: ...


class StateProvider(Protocol):
    def state(
        self, consumed_wm_actions: np.ndarray, output: WorldModelOutput
    ) -> np.ndarray: ...


class CommandedStateProvider:
    """Use the final action actually consumed by the WM as the next state."""

    def state(
        self, consumed_wm_actions: np.ndarray, output: WorldModelOutput
    ) -> np.ndarray:
        del output
        actions = np.asarray(consumed_wm_actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) == 0:
            raise ValueError(
                f"expected consumed WM actions with shape (T, 16), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("consumed WM actions contain NaN or infinity")
        return validate_policy_state(wm_to_policy_layout(actions[-1]))


class ACWMPredictedStateProvider:
    """Prefer ACWM's state head, with an explicit commanded-state fallback."""

    def __init__(
        self,
        fallback: StateProvider | None = None,
        *,
        require_prediction: bool = False,
    ) -> None:
        self.fallback = fallback or CommandedStateProvider()
        self.require_prediction = require_prediction

    def state(
        self, consumed_wm_actions: np.ndarray, output: WorldModelOutput
    ) -> np.ndarray:
        if output.predicted_state is not None:
            return validate_policy_state(output.predicted_state)
        if self.require_prediction:
            raise ValueError("ACWM output did not provide a predicted state")
        return self.fallback.state(consumed_wm_actions, output)


def _encode_array(array: np.ndarray) -> dict[str, str]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return {"npy_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def _decode_array(value: Any) -> np.ndarray:
    if isinstance(value, Mapping) and "npy_base64" in value:
        raw = base64.b64decode(value["npy_base64"])
        return np.load(io.BytesIO(raw), allow_pickle=False)
    return np.asarray(value)


class HTTPWorldModelClient:
    """JSON-over-HTTP ACWM client using encoded NumPy arrays."""

    def __init__(
        self,
        base_url: str,
        *,
        reset_path: str = "/reset",
        step_path: str = "/step",
        timeout: float = 120.0,
        wire_action_layout: Literal["policy", "wm"] = "policy",
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.reset_path = reset_path
        self.step_path = step_path
        self.timeout = float(timeout)
        if wire_action_layout not in {"policy", "wm"}:
            raise ValueError("wire_action_layout must be 'policy' or 'wm'")
        self.wire_action_layout = wire_action_layout
        self._urlopen = urlopen or request.urlopen
        self._asset_dir: str | None = None
        self._seed: int | None = None
        self._task_name: str | None = None
        self._episode_index: int | None = None

    def configure_episode(self, asset_dir: str | Path, *, seed: int) -> None:
        """Select the policy_eval_dataset episode used by the next reset."""

        episode_dir = Path(asset_dir).expanduser().resolve()
        self._asset_dir = str(episode_dir)
        self._seed = int(seed)
        self._task_name = episode_dir.parent.name
        public_index = episode_dir.name.removeprefix("ep")
        if not public_index.isdigit():
            raise ValueError(
                f"cannot derive public episode index from {episode_dir.name!r}"
            )
        self._episode_index = int(public_index)

    def _post(self, path: str, payload: Mapping[str, Any]) -> WorldModelOutput:
        req = request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"world-model POST {path} failed with HTTP {exc.code}: "
                f"{details or exc.reason}"
            ) from exc
        if not isinstance(body, Mapping):
            raise ValueError("world-model HTTP response must be a JSON object")
        raw_frames = body.get("frames")
        frame_sequences = (
            {
                name: _decode_array(value)
                for name, value in raw_frames.items()
                if name in VIEW_NAMES
            }
            if isinstance(raw_frames, Mapping)
            else None
        )
        raw_views = body.get("views", body)
        if not isinstance(raw_views, Mapping):
            raise ValueError("world-model response views must be an object")
        decoded_views = {
            name: _decode_array(value)
            for name, value in raw_views.items()
            if name in VIEW_NAMES
        }
        if "top_head" not in decoded_views:
            raise ValueError("world-model response must include top_head")
        if frame_sequences is None and all(
            value.ndim == 4 for value in decoded_views.values()
        ):
            frame_sequences = decoded_views
        # ``views`` is the server's explicit next-policy observation.  It may
        # intentionally stay at the WM's native resolution while ``frames`` is
        # resized for benchmark persistence, so do not replace it with
        # ``frames[-1]``.
        views = dict(decoded_views)
        state = body.get("predicted_state")
        decoded_state = None if state is None else _decode_array(state)
        if decoded_state is not None and decoded_state.ndim == 2:
            decoded_state = decoded_state[-1]
        return WorldModelOutput(
            views=views,
            predicted_state=decoded_state,
            frames=frame_sequences,
        )

    def reset(self, observation: Observation) -> WorldModelOutput:
        return self._post(
            self.reset_path,
            {
                "asset_dir": self._asset_dir,
                "seed": self._seed,
                "task_name": self._task_name,
                "episode_index": self._episode_index,
                "views": {
                    name: _encode_array(image)
                    for name, image in observation.views.items()
                },
                "state": _encode_array(observation.state),
                "robot_context": (
                    {
                        key: _encode_array(value)
                        for key, value in observation.robot_context.items()
                    }
                    if observation.robot_context is not None
                    else None
                ),
                "task": observation.task,
            },
        )

    def step(self, actions: np.ndarray) -> WorldModelOutput:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 16 or len(arr) == 0:
            raise ValueError(f"expected WM actions with shape (T, 16), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("WM actions contain NaN or infinity")
        wire_actions = (
            wm_to_policy_layout(arr)
            if self.wire_action_layout == "policy"
            else arr
        )
        return self._post(
            self.step_path, {"actions": _encode_array(wire_actions)}
        )


@dataclass(frozen=True)
class ClosedLoopStep:
    observation: Observation
    policy_actions: np.ndarray
    consumed_wm_actions: np.ndarray
    output: WorldModelOutput


class ClosedLoopOrchestrator:
    """Minimal closed loop: PI actions -> ACWM views/state -> next PI input."""

    def __init__(
        self,
        policy: PolicyClient,
        world_model: WorldModelClient,
        *,
        target_horizon: int,
        policy_horizon: int | None = None,
        state_provider: StateProvider | None = None,
    ) -> None:
        self.policy = policy
        self.world_model = world_model
        self.target_horizon = int(target_horizon)
        if self.target_horizon <= 0:
            raise ValueError("target_horizon must be positive")
        self.policy_horizon = (
            int(policy_horizon) if policy_horizon is not None else None
        )
        if self.policy_horizon is not None and self.policy_horizon <= 0:
            raise ValueError("policy_horizon must be positive")
        self.state_provider = state_provider or ACWMPredictedStateProvider()

    def run(
        self,
        initial: Observation,
        steps: int,
    ) -> list[ClosedLoopStep]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.world_model.reset(initial)
        observation = initial
        trace: list[ClosedLoopStep] = []
        for _ in range(steps):
            policy_actions = self.policy.infer(observation)
            if (
                self.policy_horizon is not None
                and len(policy_actions) != self.policy_horizon
            ):
                raise ValueError(
                    "policy returned "
                    f"{len(policy_actions)} actions; expected "
                    f"{self.policy_horizon}"
                )
            executable = select_policy_action_prefix(
                policy_actions,
                self.target_horizon,
            )
            consumed = policy_to_wm_layout(executable).astype(
                np.float32,
                copy=False,
            )
            output = self.world_model.step(consumed)
            next_state = self.state_provider.state(consumed, output)
            trace.append(
                ClosedLoopStep(observation, policy_actions, consumed, output)
            )
            observation = Observation(
                top_head=output.views["top_head"],
                hand_left=output.views.get("hand_left"),
                hand_right=output.views.get("hand_right"),
                state=next_state,
                task=initial.task,
                robot_context=initial.robot_context,
            )
        return trace


@dataclass(frozen=True)
class ControlProtocol:
    """Control settings that must be identical in real and WM rollouts."""

    fps: float = 30.0
    prediction_horizon: int = 50
    replan_horizon: int = 8
    action_layout: str = POLICY_ACTION_LAYOUT
    state_layout: str = POLICY_ACTION_LAYOUT
    state_source: str = "measured"
    action_compression: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if not 0 < self.replan_horizon <= self.prediction_horizon:
            raise ValueError("replan_horizon must be in [1, prediction_horizon]")


@dataclass(frozen=True)
class RolloutMetadata:
    """Machine-readable identity and provenance for one policy rollout."""

    policy_id: str
    task_id: str
    init_id: str
    domain: Literal["real", "wm"]
    rollout_id: str
    checkpoint: str
    prompt: str
    protocol: ControlProtocol = field(default_factory=ControlProtocol)
    seed: int | None = None
    world_model_id: str | None = None
    termination_reason: str | None = None
    schema_version: int = ROLLOUT_SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in ROLLOUT_DOMAINS:
            raise ValueError(f"domain must be one of {ROLLOUT_DOMAINS}, got {self.domain!r}")
        for name in ("policy_id", "task_id", "init_id", "rollout_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != ROLLOUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported rollout schema {self.schema_version}; "
                f"expected {ROLLOUT_SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyRollout:
    """Discovered rollout plus optional precomputed judge output."""

    root: Path
    metadata: RolloutMetadata
    video_dir: Path | None
    judge_path: Path | None

    @property
    def uid(self) -> str:
        m = self.metadata
        model_id = m.world_model_id or "real"
        return (
            f"{model_id}__{m.policy_id}__{m.task_id}__{m.init_id}__"
            f"{m.domain}__{m.rollout_id}"
        )


class BatchedForwardKinematics(Protocol):
    """G2 FK callback used by :func:`g2_joints_to_enerverse`.

    Inputs are ``(T, 7)`` left/right joint arrays. Outputs are ``(T, 7)`` poses
    in ``[x, y, z, qx, qy, qz, qw]`` order unless ``quaternion_order`` says
    otherwise.
    """

    def __call__(
        self, left_joints: np.ndarray, right_joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


def validate_policy_actions(actions: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return finite ``(T, 16)`` G2 absolute joint/gripper targets."""

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"expected policy actions with shape (T, 16), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("policy action chunk must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("policy actions contain NaN or infinity")
    return arr


def _quaternion_to_xyzw(poses: np.ndarray, order: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"FK poses must have shape (T, 7), got {poses.shape}")
    if order == "xyzw":
        return poses
    if order == "wxyz":
        out = poses.copy()
        out[:, 3:7] = poses[:, [4, 5, 6, 3]]
        return out
    raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")


def g2_joints_to_enerverse(
    actions: np.ndarray | Sequence[Sequence[float]],
    fk: BatchedForwardKinematics,
    *,
    quaternion_order: Literal["xyzw", "wxyz"] = "xyzw",
    gripper_scale: float = 1.0,
) -> np.ndarray:
    """Convert π0.5 G2 joint targets to the ACWM EnerVerse pose condition.

    π0.5 emits ``[L7, R7, Lg, Rg]`` absolute targets. ACWMBench's action map
    consumes ``[L_xyz, L_quat_xyzw, Lg, R_xyz, R_quat_xyzw, Rg]``. Robot-specific
    kinematics are deliberately injected because GESim's bundled FK is G01-only
    and must not be reused for G2.
    """

    arr = validate_policy_actions(actions)
    left_pose, right_pose = fk(arr[:, 0:7], arr[:, 7:14])
    left_pose = _quaternion_to_xyzw(left_pose, quaternion_order)
    right_pose = _quaternion_to_xyzw(right_pose, quaternion_order)
    if left_pose.shape[0] != arr.shape[0] or right_pose.shape[0] != arr.shape[0]:
        raise ValueError("FK output length must match the action chunk length")
    if not np.isfinite(left_pose).all() or not np.isfinite(right_pose).all():
        raise ValueError("FK output contains NaN or infinity")

    out = np.empty((arr.shape[0], 16), dtype=np.float32)
    out[:, 0:7] = left_pose
    out[:, 7] = arr[:, 14] * float(gripper_scale)
    out[:, 8:15] = right_pose
    out[:, 15] = arr[:, 15] * float(gripper_scale)
    return out


def _load_metadata(
    path: Path,
    identity: tuple[str, str, str, str, str] | None = None,
) -> RolloutMetadata:
    policy_id, task_id, init_id, domain, rollout_id = identity or (
        "",
        "",
        "",
        "wm",
        path.parent.name,
    )
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object")
        data = loaded

    protocol_data = data.pop("protocol", {}) or {}
    protocol = ControlProtocol(**protocol_data)
    declared_extra = data.pop("extra", {}) or {}
    if not isinstance(declared_extra, Mapping):
        raise ValueError(f"{path}: extra must be a JSON object")
    known = {
        "policy_id": str(data.pop("policy_id", policy_id)),
        "task_id": str(data.pop("task_id", task_id)),
        "init_id": str(data.pop("init_id", init_id)),
        "domain": str(data.pop("domain", domain)),
        "rollout_id": str(data.pop("rollout_id", rollout_id)),
        "checkpoint": str(data.pop("checkpoint", "")),
        "prompt": str(data.pop("prompt", "")),
        "protocol": protocol,
        "seed": data.pop("seed", None),
        "world_model_id": data.pop("world_model_id", None),
        "termination_reason": data.pop("termination_reason", None),
        "schema_version": int(data.pop("schema_version", ROLLOUT_SCHEMA_VERSION)),
        "extra": {**dict(declared_extra), **data},
    }
    metadata = RolloutMetadata(**known)
    if identity is not None:
        expected = (policy_id, task_id, init_id, domain, rollout_id)
        actual = (
            metadata.policy_id,
            metadata.task_id,
            metadata.init_id,
            metadata.domain,
            metadata.rollout_id,
        )
        if actual != expected:
            raise ValueError(
                f"metadata identity {actual} disagrees with directory identity "
                f"{expected}"
            )
    return metadata


def discover_policy_rollouts(
    rollout_root: str | Path,
    *,
    task_filter: Iterable[str] | None = None,
    init_filter: Iterable[str] | None = None,
) -> list[PolicyRollout]:
    """Discover metadata-driven rollouts in model-first or legacy layouts."""

    root = Path(rollout_root)
    if not root.is_dir():
        raise FileNotFoundError(f"policy rollout root does not exist: {root}")
    wanted_tasks = {str(x).lower() for x in task_filter or ()}

    def normalize_init(value: Any) -> str:
        text = str(value).strip().lower()
        return text[2:] if text.startswith("ep") else text

    wanted_inits = {normalize_init(x) for x in init_filter or ()}
    found: list[PolicyRollout] = []

    for metadata_path in sorted(root.rglob("meta.json")):
        relative = metadata_path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        rollout_dir = metadata_path.parent
        metadata = _load_metadata(metadata_path)
        if wanted_tasks and metadata.task_id.lower() not in wanted_tasks:
            continue
        if wanted_inits and normalize_init(metadata.init_id) not in wanted_inits:
            continue
        video_dir = rollout_dir / "video"
        video = video_dir if video_dir.is_dir() else None
        episode_judge_path = rollout_dir.parent / "policy_eval.json"
        legacy_judge_path = rollout_dir / "physical_judge.json"
        judge = None
        if episode_judge_path.is_file():
            try:
                with episode_judge_path.open("r", encoding="utf-8") as handle:
                    episode_judges = json.load(handle)
                if isinstance(episode_judges, Mapping) and isinstance(
                    episode_judges.get(metadata.rollout_id)
                    or episode_judges.get(
                        (
                            f"{metadata.world_model_id or 'real'}__"
                            f"{metadata.policy_id}__{metadata.task_id}__"
                            f"{metadata.init_id}__{metadata.domain}__"
                            f"{metadata.rollout_id}"
                        )
                    ),
                    Mapping,
                ):
                    judge = episode_judge_path
            except (OSError, ValueError, json.JSONDecodeError):
                judge = None
        if judge is None and legacy_judge_path.is_file():
            judge = legacy_judge_path
        if video is None and judge is None:
            raise ValueError(
                f"{rollout_dir} needs video/ or physical_judge.json"
            )
        found.append(
            PolicyRollout(
                root=rollout_dir,
                metadata=metadata,
                video_dir=video,
                judge_path=judge,
            )
        )
    if not found:
        raise ValueError(f"no policy rollouts found under {root}")
    return found


def validate_paired_rollouts(rollouts: Sequence[PolicyRollout]) -> None:
    """Require paired domains and matching control timing/action contracts.

    ``state_source`` is intentionally excluded from cross-domain equality:
    real rollouts normally use measured state while WM rollouts may use a pose
    head, a dynamics model, or commanded-state fallback.
    """

    domains: dict[tuple[str, str, str], set[str]] = {}
    protocols: dict[tuple[str, str, str], set[tuple[Any, ...]]] = {}
    for rollout in rollouts:
        m = rollout.metadata
        key = (m.policy_id, m.task_id, m.init_id)
        domains.setdefault(key, set()).add(m.domain)
        p = m.protocol
        protocols.setdefault(key, set()).add(
            (
                p.fps,
                p.prediction_horizon,
                p.replan_horizon,
                p.action_layout,
                p.state_layout,
                p.action_compression,
            )
        )
    missing = {key: sorted(set(ROLLOUT_DOMAINS) - value) for key, value in domains.items()}
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"unpaired policy-eval cells: {missing}")
    mismatched = [key for key, values in protocols.items() if len(values) != 1]
    if mismatched:
        raise ValueError(
            "real and WM control protocols must match; mismatched cells: "
            + ", ".join("/".join(key) for key in mismatched)
        )
