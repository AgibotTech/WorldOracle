from __future__ import annotations

# The IDE's lightweight lint interpreter does not load project dependencies.
# pylint: disable=import-error,no-name-in-module

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acwmbench.metrics.policy_eval import compute_policy_evaluation
from acwmbench.policy_eval import (
    ACWMPredictedStateProvider,
    ClosedLoopOrchestrator,
    CommandedStateProvider,
    ControlProtocol,
    HTTPWorldModelClient,
    Observation,
    OpenPIKeys,
    OpenPIWebsocketPolicyClient,
    WorldModelOutput,
    _encode_array,
    compress_policy_actions,
    g2_joints_to_enerverse,
    policy_to_wm_layout,
    wm_to_policy_layout,
)


def _judge(score: int) -> dict:
    return {
        "score": score,
        "key_steps": [
            {
                "segment_id": 0,
                "success_criteria": ["object reaches the target"],
                "aspect_scores": [score],
                "format_valid": True,
            }
        ],
    }


class G2ActionAdapterTest(unittest.TestCase):
    def test_joint_targets_are_packed_as_enerverse_poses(self) -> None:
        actions = np.arange(32, dtype=np.float32).reshape(2, 16)

        def fake_fk(left: np.ndarray, right: np.ndarray):
            left_pose = np.concatenate(
                [left[:, :3], np.tile([[1, 2, 3, 4]], (len(left), 1))], axis=1
            )
            right_pose = np.concatenate(
                [right[:, :3], np.tile([[5, 6, 7, 8]], (len(right), 1))], axis=1
            )
            return left_pose, right_pose

        packed = g2_joints_to_enerverse(
            actions, fake_fk, quaternion_order="wxyz", gripper_scale=120.0
        )
        self.assertEqual(packed.shape, (2, 16))
        np.testing.assert_array_equal(packed[:, 0:3], actions[:, 0:3])
        np.testing.assert_array_equal(packed[0, 3:7], [2, 3, 4, 1])
        np.testing.assert_array_equal(packed[0, 11:15], [6, 7, 8, 5])
        np.testing.assert_allclose(packed[:, 7], actions[:, 14] * 120.0)
        np.testing.assert_allclose(packed[:, 15], actions[:, 15] * 120.0)


def _views(value: int = 0) -> dict[str, np.ndarray]:
    return {
        name: np.full((2, 3, 3), value, dtype=np.uint8)
        for name in ("top_head", "hand_left", "hand_right")
    }


class LightweightRolloutContractsTest(unittest.TestCase):
    def test_observation_validates_images_state_and_task(self) -> None:
        observation = Observation(**_views(), state=np.arange(16), task="fold towel")
        self.assertEqual(observation.state.dtype, np.float32)
        head_only = Observation(
            top_head=_views()["top_head"],
            hand_left=None,
            hand_right=None,
            state=np.arange(16),
            task="fold towel",
        )
        self.assertEqual(set(head_only.views), {"top_head"})
        with self.assertRaisesRegex(ValueError, "uint8"):
            Observation(
                **{**_views(), "top_head": np.zeros((2, 3, 3), dtype=np.float32)},
                state=np.zeros(16),
                task="task",
            )
        with self.assertRaisesRegex(ValueError, r"shape \(16,\)"):
            Observation(**_views(), state=np.zeros(15), task="task")

    def test_policy_and_wm_layouts_round_trip(self) -> None:
        policy = np.arange(32).reshape(2, 16)
        wm = policy_to_wm_layout(policy)
        np.testing.assert_array_equal(
            wm[0], [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]
        )
        np.testing.assert_array_equal(wm_to_policy_layout(wm), policy)

    def test_action_alignment_executes_prefix_then_replans(self) -> None:
        actions = np.zeros((50, 16), dtype=np.float32)
        actions[:, :14] = np.arange(50, dtype=np.float32)[:, None]
        actions[:, 14] = np.arange(50)
        actions[:, 15] = 100 + np.arange(50)
        compressed = compress_policy_actions(actions, 8)
        self.assertEqual(compressed.shape, (8, 16))
        np.testing.assert_array_equal(compressed, actions[:8])

    def test_state_providers_use_prediction_or_command_fallback(self) -> None:
        policy_actions = np.arange(48, dtype=np.float32).reshape(3, 16)
        consumed = policy_to_wm_layout(policy_actions)
        output = WorldModelOutput(_views())
        commanded = CommandedStateProvider().state(consumed, output)
        np.testing.assert_array_equal(commanded, policy_actions[-1])

        predicted = np.full(16, 7, dtype=np.float32)
        predicted_output = WorldModelOutput(_views(), predicted_state=predicted)
        provider = ACWMPredictedStateProvider()
        np.testing.assert_array_equal(provider.state(consumed, predicted_output), predicted)
        np.testing.assert_array_equal(provider.state(consumed, output), policy_actions[-1])
        with self.assertRaisesRegex(ValueError, "did not provide"):
            ACWMPredictedStateProvider(require_prediction=True).state(consumed, output)

    def test_openpi_client_uses_configurable_keys_and_validates_actions(self) -> None:
        calls = []
        reset_seeds = []

        class FakeClient:
            def infer(self, payload):
                calls.append(payload)
                return {"chunk": np.ones((50, 16), dtype=np.float32)}

            def reset(self, seed=42):
                reset_seeds.append(seed)

        factories = []

        def factory(host, port):
            factories.append((host, port))
            return FakeClient()

        client = OpenPIWebsocketPolicyClient(
            "policy.local",
            8000,
            keys=OpenPIKeys(
                top_head="top",
                hand_left="left",
                hand_right="right",
                state="robot",
                task="text",
                actions="chunk",
            ),
            client_factory=factory,
        )
        observation = Observation(**_views(), state=np.zeros(16), task="task")
        client.reset(seed=42)
        self.assertEqual(client.infer(observation).shape, (50, 16))
        self.assertEqual(factories, [("policy.local", 8000)])
        self.assertEqual(reset_seeds, [42])
        self.assertEqual(set(calls[0]), {"top", "left", "right", "robot", "text"})
        client.infer(
            Observation(
                top_head=_views()["top_head"],
                hand_left=None,
                hand_right=None,
                state=np.zeros(16),
                task="task",
            )
        )
        self.assertEqual(set(calls[1]), {"top", "robot", "text"})
        self.assertEqual(reset_seeds, [42])

    def test_http_world_model_preserves_explicit_policy_feedback_view(self) -> None:
        native_view = np.full((2, 3, 3), 7, dtype=np.uint8)
        persisted_frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

            def read(self):
                return json.dumps(
                    {
                        "views": {"top_head": _encode_array(native_view)},
                        "frames": {"top_head": _encode_array(persisted_frames)},
                        "predicted_state": None,
                    }
                ).encode("utf-8")

        client = HTTPWorldModelClient(
            "http://wm.local",
            urlopen=lambda request, timeout: FakeResponse(),
        )
        output = client.step(np.zeros((25, 16), dtype=np.float32))

        np.testing.assert_array_equal(output.views["top_head"], native_view)
        self.assertEqual(output.frames["top_head"].shape, (2, 4, 5, 3))

    def test_mock_closed_loop_feeds_generated_views_and_selected_state(self) -> None:
        class FakePolicy:
            def __init__(self):
                self.observations = []

            def infer(self, observation):
                self.observations.append(observation)
                value = len(self.observations)
                actions = np.full((50, 16), value, dtype=np.float32)
                actions[:, 14:16] = value * 10
                return actions

        class FakeWorldModel:
            def __init__(self):
                self.reset_observation = None
                self.actions = []

            def reset(self, observation):
                self.reset_observation = observation
                return WorldModelOutput(_views())

            def step(self, actions):
                self.actions.append(actions.copy())
                index = len(self.actions)
                state = np.full(16, 20 + index, dtype=np.float32)
                image = _views()["top_head"]
                generated = {"top_head": np.stack([image, image + index])}
                return WorldModelOutput(generated, predicted_state=state)

        initial = Observation(**_views(), state=np.zeros(16), task="pick fruit")
        policy = FakePolicy()
        world_model = FakeWorldModel()
        trace = ClosedLoopOrchestrator(
            policy, world_model, target_horizon=8
        ).run(initial, steps=2)

        self.assertIs(world_model.reset_observation, initial)
        self.assertEqual(len(trace), 2)
        self.assertEqual(world_model.actions[0].shape, (8, 16))
        np.testing.assert_array_equal(policy.observations[1].top_head, _views(1)["top_head"])
        self.assertIsNone(policy.observations[1].hand_left)
        self.assertIsNone(policy.observations[1].hand_right)
        np.testing.assert_array_equal(policy.observations[1].state, np.full(16, 21))
        self.assertEqual(policy.observations[1].task, "pick fruit")

    def test_closed_loop_always_executes_full_action_prefix(self) -> None:
        class FakePolicy:
            def reset(self, seed=42):
                del seed

            def infer(self, observation):
                del observation
                return np.arange(50 * 16, dtype=np.float32).reshape(50, 16)

        class FakeWorldModel:
            def __init__(self):
                self.action_lengths = []

            def reset(self, observation):
                return WorldModelOutput(observation.views)

            def step(self, actions):
                self.action_lengths.append(len(actions))
                image = _views()["top_head"]
                return WorldModelOutput(
                    {"top_head": np.repeat(image[None], len(actions), axis=0)}
                )

        initial = Observation(**_views(), state=np.zeros(16), task="pick fruit")
        world_model = FakeWorldModel()
        trace = ClosedLoopOrchestrator(
            FakePolicy(),
            world_model,
            target_horizon=8,
        ).run(initial, steps=2)

        self.assertEqual(world_model.action_lengths, [8, 8])
        self.assertEqual([len(step.consumed_wm_actions) for step in trace], [8, 8])


class PolicyEvaluationTest(unittest.TestCase):
    def _write_rollout(
        self,
        root: Path,
        policy: str,
        task: str,
        init_id: str,
        domain: str,
        rollout_id: str,
        score: int,
        *,
        horizon: int = 8,
        state_source: str = "measured",
        seed: int = 0,
    ) -> None:
        rollout = root / policy / task / init_id / domain / rollout_id
        rollout.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "policy_id": policy,
            "task_id": task,
            "init_id": init_id,
            "domain": domain,
            "rollout_id": rollout_id,
            "checkpoint": f"/checkpoints/{policy}",
            "prompt": task,
            "seed": seed,
            "protocol": {
                "fps": 30,
                "prediction_horizon": 50,
                "replan_horizon": horizon,
                "state_source": state_source,
            },
        }
        (rollout / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        (rollout / "physical_judge.json").write_text(
            json.dumps(_judge(score)), encoding="utf-8"
        )

    def _build_dataset(self, root: Path, *, invert_wm: bool) -> None:
        values = {
            ("policy_a", "task_a"): 1,
            ("policy_a", "task_b"): 1,
            ("policy_b", "task_a"): 0,
            ("policy_b", "task_b"): 0,
        }
        for index, ((policy, task), real_score) in enumerate(values.items()):
            horizon = 5 if index % 2 == 0 else 8
            state_source = "measured" if index < 2 else "pose_head"
            init_id = f"ep{index:06d}"
            wm_score = 1 - real_score if invert_wm else real_score
            self._write_rollout(
                root,
                policy,
                task,
                init_id,
                "real",
                "r0",
                real_score,
                horizon=horizon,
                state_source="measured",
            )
            self._write_rollout(
                root,
                policy,
                task,
                init_id,
                "wm",
                "r0",
                wm_score,
                horizon=horizon,
                state_source=state_source,
                seed=0,
            )
            self._write_rollout(
                root,
                policy,
                task,
                init_id,
                "wm",
                "r1",
                wm_score,
                horizon=horizon,
                state_source=state_source,
                seed=1,
            )

    def test_gt_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_dataset(root, invert_wm=False)
            result = compute_policy_evaluation(
                rollout_root=root,
                gt_root=root,
                bootstrap_samples=100,
                bootstrap_seed=7,
            )
        calibration = result["summary"]["calibration"]
        correlation = result["summary"]["correlation"]
        self.assertEqual(calibration["brier_score"], 0.0)
        self.assertEqual(calibration["threshold_agreement"], 1.0)
        self.assertEqual(correlation["pearson"], 1.0)
        self.assertEqual(correlation["spearman"], 1.0)
        self.assertEqual(len(result["protocol_ablations"]), 4)
        self.assertEqual(len(result["wm_seed_ablations"]), 2)
        self.assertTrue(
            all(row["brier_score"] == 0.0 for row in result["wm_seed_ablations"])
        )

    def test_shuffled_or_wrong_dynamics_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_dataset(root, invert_wm=True)
            result = compute_policy_evaluation(
                rollout_root=root,
                gt_root=root,
                bootstrap_samples=0,
            )
        calibration = result["summary"]["calibration"]
        correlation = result["summary"]["correlation"]
        self.assertEqual(calibration["brier_score"], 1.0)
        self.assertEqual(calibration["threshold_agreement"], 0.0)
        self.assertEqual(correlation["pearson"], -1.0)
        self.assertEqual(correlation["spearman"], -1.0)

    def test_real_and_wm_protocols_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_rollout(
                root, "policy", "task", "ep0", "real", "r0", 1, horizon=5
            )
            self._write_rollout(
                root, "policy", "task", "ep0", "wm", "r0", 1, horizon=8
            )
            with self.assertRaisesRegex(ValueError, "protocols must match"):
                compute_policy_evaluation(
                    rollout_root=root,
                    gt_root=root,
                    bootstrap_samples=0,
                )

    def test_control_protocol_rejects_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            ControlProtocol(prediction_horizon=50, replan_horizon=51)


if __name__ == "__main__":
    unittest.main()
