from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from acwmbench.policy_eval import Observation, WorldModelOutput
from acwmbench.policy_rollout import (
    PolicyEvalEpisode,
    _target_frame_count,
    discover_policy_eval_episodes,
    generate_wm_policy_rollouts,
    load_initial_observation,
)


def _observation() -> Observation:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return Observation(image, image, image, np.arange(16, dtype=np.float32), "task")


class _Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_seeds = []

    def reset(self, seed: int = 42) -> None:
        self.reset_seeds.append(seed)

    def infer(self, observation: Observation) -> np.ndarray:
        self.calls += 1
        base = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
        return base + self.calls


class _WorldModel:
    def __init__(self, *, top_only: bool = False) -> None:
        self.configured = []
        self.steps = []
        self.top_only = top_only

    def configure_episode(self, asset_dir: Path, *, seed: int) -> None:
        self.configured.append((Path(asset_dir), seed))

    def reset(self, observation: Observation) -> WorldModelOutput:
        return WorldModelOutput(observation.views)

    def step(self, actions: np.ndarray) -> WorldModelOutput:
        self.steps.append(actions.copy())
        names = ("top_head",) if self.top_only else ("top_head", "hand_left", "hand_right")
        frames = {
            name: np.full((len(actions), 8, 10, 3), len(self.steps), dtype=np.uint8)
            for name in names
        }
        return WorldModelOutput(
            {name: sequence[-1] for name, sequence in frames.items()},
            frames=frames,
        )


class PolicyRolloutTest(unittest.TestCase):
    def _dataset_root(self, root: Path) -> Path:
        dataset = root / "policy_eval_dataset"
        task = dataset / "pickfruit"
        task.mkdir(parents=True)
        for index in range(2):
            episode = task / f"ep{index:06d}"
            for name in ("top_head", "hand_left", "hand_right"):
                view = episode / "video" / name
                view.mkdir(parents=True)
                for frame in range(2):
                    Image.fromarray(
                        np.full((8, 10, 3), frame, dtype=np.uint8)
                    ).save(view / f"frame_{frame:05d}.png")
        return dataset

    def test_discovery_reads_policy_eval_dataset_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset_root(Path(temporary))
            found = discover_policy_eval_episodes(
                dataset, episode_filter=["000001"], limit_episodes_per_task=1
            )
        self.assertEqual([item.episode_id for item in found], ["ep000001"])

    def test_initial_state_and_instruction_come_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "pickfruit" / "ep000000"
            frames = {}
            for name in ("top_head", "hand_left", "hand_right"):
                view = episode / "video" / name
                view.mkdir(parents=True)
                path = view / "frame_00000.png"
                Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(path)
                frames[name] = (path,)
            config = root / "initial_states.yaml"
            config.write_text(
                "pickfruit:\n"
                "  instruction: fixed instruction\n"
                f"  arm_joints: {list(range(14))}\n"
                "  grippers: [14, 15]\n"
                "  head: [0, 0, 0]\n"
                "  waist: [0, 0, 0, 0, 0]\n",
                encoding="utf-8",
            )
            policy_episode = PolicyEvalEpisode(
                episode, "pickfruit", "ep000000", {"num_frames": 1}, frames
            )

            observation = load_initial_observation(policy_episode, config)

        np.testing.assert_array_equal(
            observation.state, np.arange(16, dtype=np.float32)
        )
        self.assertEqual(observation.task, "fixed instruction")
        np.testing.assert_array_equal(
            observation.robot_context["head"], np.zeros(3, dtype=np.float32)
        )

    def test_target_length_comes_from_policy_eval_frames(self) -> None:
        episode = PolicyEvalEpisode(
            Path("/unused/ep000000"),
            "pickfruit",
            "ep000000",
            {"num_frames": 38},
            {
                name: tuple(Path(f"/{name}/{index}") for index in range(38))
                for name in ("top_head", "hand_left", "hand_right")
            },
        )
        self.assertEqual(_target_frame_count(episode), 38)

    def test_rollouts_use_distinct_deterministic_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset_root(root)
            policy = _Policy()
            world_model = _WorldModel()
            fake_episode = PolicyEvalEpisode(
                dataset / "pickfruit" / "ep000000",
                "pickfruit",
                "ep000000",
                {"fps": 30, "num_frames": 2},
                {
                    name: tuple(Path(f"/{name}/{index}") for index in range(2))
                    for name in ("top_head", "hand_left", "hand_right")
                },
            )
            with (
                mock.patch(
                    "acwmbench.policy_rollout.discover_policy_eval_episodes",
                    return_value=[fake_episode],
                ),
                mock.patch(
                    "acwmbench.policy_rollout.load_initial_observation",
                    return_value=_observation(),
                ),
                mock.patch(
                    "acwmbench.policy_rollout._target_frame_count",
                    return_value=2,
                ),
            ):
                paths = generate_wm_policy_rollouts(
                    dataset_root=dataset,
                    rollout_root=root / "rollouts",
                    initial_states_path=root / "initial_states.yaml",
                    policy=policy,
                    world_model=world_model,
                    policy_id="pi05",
                    policy_checkpoint="/ckpt/pi05",
                    world_model_id="ctrlworld",
                    world_model_checkpoint="/ckpt/ctrlworld",
                    action_horizon=50,
                    wm_chunk_size=4,
                    compress_actions=True,
                    state_source="commanded",
                    rollout_number=3,
                )

            self.assertEqual(policy.reset_seeds, [42, 43, 44])
            self.assertEqual(
                [seed for _, seed in world_model.configured], [42, 43, 44]
            )
            self.assertEqual([path.name for path in paths], ["0", "1", "2"])
            self.assertEqual(
                [
                    json.loads((path / "meta.json").read_text())["seed"]
                    for path in paths
                ],
                [42, 43, 44],
            )

    def test_closed_loop_persists_consumed_actions_states_and_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset_root(root)
            rollout_root = root / "rollouts"
            policy = _Policy()
            world_model = _WorldModel(top_only=True)
            fake_episode = PolicyEvalEpisode(
                dataset / "pickfruit" / "ep000000",
                "pickfruit",
                "ep000000",
                {"fps": 30, "num_frames": 38},
                {
                    name: tuple(Path(f"/{name}/{index}") for index in range(38))
                    for name in ("top_head", "hand_left", "hand_right")
                },
            )
            with (
                mock.patch(
                    "acwmbench.policy_rollout.discover_policy_eval_episodes",
                    return_value=[fake_episode],
                ),
                mock.patch(
                    "acwmbench.policy_rollout.load_initial_observation",
                    return_value=_observation(),
                ),
                mock.patch(
                    "acwmbench.policy_rollout._target_frame_count",
                    return_value=38,
                ),
            ):
                paths = generate_wm_policy_rollouts(
                    dataset_root=dataset,
                    rollout_root=rollout_root,
                    initial_states_path=root / "initial_states.yaml",
                    policy=policy,
                    world_model=world_model,
                    policy_id="pi05",
                    policy_checkpoint="/ckpt/pi05",
                    world_model_id="gesim",
                    world_model_checkpoint="/ckpt/gesim",
                    action_horizon=50,
                    wm_chunk_size=25,
                    compress_actions=True,
                    state_source="commanded",
                    rollout_number=1,
                )

            destination = paths[0]
            self.assertEqual(destination.name, "0")
            self.assertEqual(
                destination.relative_to(rollout_root).parts,
                ("gesim", "pickfruit", "ep000000", "0"),
            )
            self.assertEqual(policy.calls, 2)
            self.assertEqual(policy.reset_seeds, [42])
            self.assertEqual(world_model.configured[0][1], 42)
            self.assertEqual(
                [chunk.shape for chunk in world_model.steps],
                [(25, 16), (25, 16)],
            )
            self.assertEqual(np.load(destination / "actions.npy").shape, (37, 16))
            self.assertEqual(np.load(destination / "states.npy").shape, (3, 16))
            self.assertEqual(
                len(list((destination / "video" / "top_head").glob("*.jpg"))),
                38,
            )
            with Image.open(
                destination / "video" / "top_head" / "frame_00037.jpg"
            ) as image:
                self.assertEqual(image.size, (10, 8))
            for name in ("hand_left", "hand_right"):
                self.assertEqual(
                    len(list((destination / "video" / name).glob("*.jpg"))),
                    1,
                )
            metadata = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["extra"]["status"], "complete")
            self.assertEqual(metadata["extra"]["output_frames"], 38)
            self.assertEqual(
                metadata["extra"]["view_frame_counts"],
                {"top_head": 38, "hand_left": 1, "hand_right": 1},
            )
            self.assertEqual(metadata["extra"]["target_frames"], 38)
            self.assertEqual(metadata["seed"], 42)

            # A matching completed rollout resumes without contacting either service.
            with (
                mock.patch(
                    "acwmbench.policy_rollout.discover_policy_eval_episodes",
                    return_value=[fake_episode],
                ),
                mock.patch(
                    "acwmbench.policy_rollout.load_initial_observation",
                    return_value=_observation(),
                ),
                mock.patch(
                    "acwmbench.policy_rollout._target_frame_count",
                    return_value=38,
                ),
            ):
                generate_wm_policy_rollouts(
                    dataset_root=dataset,
                    rollout_root=rollout_root,
                    initial_states_path=root / "initial_states.yaml",
                    policy=policy,
                    world_model=world_model,
                    policy_id="pi05",
                    policy_checkpoint="/ckpt/pi05",
                    world_model_id="gesim",
                    world_model_checkpoint="/ckpt/gesim",
                    action_horizon=50,
                    wm_chunk_size=25,
                    compress_actions=True,
                    state_source="commanded",
                    rollout_number=1,
                )
            self.assertEqual(policy.calls, 2)


if __name__ == "__main__":
    unittest.main()
