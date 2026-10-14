from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acwmbench.metrics.policy_eval import (
    _load_precomputed_judges,
    compute_policy_evaluation,
)
from acwmbench.policy_eval import discover_policy_rollouts


class WorldModelOnlyPolicyMetricTest(unittest.TestCase):
    def test_wm_rollout_mode_reports_imagined_success_without_fake_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = (
                ("ep000000", "0", 0),
                ("ep000000", "1", 1),
                ("ep000001", "0", 0),
                ("ep000001", "1", 0),
            )
            for episode, rollout_id, score in samples:
                rollout = root / "pi05" / "pickfruit" / episode / "wm" / rollout_id
                rollout.mkdir(parents=True)
                (rollout / "meta.json").write_text(
                    json.dumps(
                        {
                            "policy_id": "pi05",
                            "task_id": "pickfruit",
                            "init_id": episode,
                            "domain": "wm",
                            "rollout_id": rollout_id,
                            "checkpoint": "/ckpt/pi05",
                            "prompt": "pick the fruit",
                            "seed": 42,
                            "world_model_id": "gesim",
                            "protocol": {
                                "prediction_horizon": 50,
                                "replan_horizon": 25,
                                "state_source": "commanded",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (rollout / "physical_judge.json").write_text(
                    json.dumps(
                        {
                            "key_steps": [
                                {
                                    "segment_id": 0,
                                    "success_criteria": ["fruit reaches target"],
                                    "aspect_scores": [score],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            result = compute_policy_evaluation(
                rollout_root=root,
                gt_root=root,
                require_paired=False,
            )

        self.assertEqual(result["mode"], "wm_score")
        self.assertEqual(result["summary"]["n_rollouts"], 4)
        self.assertEqual(result["summary"]["n_episodes"], 2)
        self.assertEqual(result["summary"]["mean_wm_success"], 0.5)
        self.assertEqual(result["policy_task"][0]["wm_score"], 0.5)
        self.assertEqual(
            sum(row["selected_best"] for row in result["wm_rollouts"]), 2
        )
        self.assertNotIn("correlation", result["summary"])

    def test_overwrite_marks_cached_judges_for_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rollout = root / "gesim" / "pickfruit" / "ep000000" / "0"
            rollout.mkdir(parents=True)
            (rollout / "meta.json").write_text(
                json.dumps(
                    {
                        "policy_id": "pi05",
                        "task_id": "pickfruit",
                        "init_id": "ep000000",
                        "domain": "wm",
                        "rollout_id": "0",
                        "checkpoint": "/ckpt/pi05",
                        "prompt": "pick the fruit",
                        "world_model_id": "gesim",
                    }
                ),
                encoding="utf-8",
            )
            (rollout / "physical_judge.json").write_text(
                json.dumps({"score": 1.0}),
                encoding="utf-8",
            )
            discovered = discover_policy_rollouts(root)
            cached, missing = _load_precomputed_judges(discovered)
            migrated = json.loads(
                (rollout.parent / "policy_eval.json").read_text(
                    encoding="utf-8"
                )
            )
            overwritten, recompute = _load_precomputed_judges(
                discovered,
                overwrite=True,
            )
        self.assertEqual(len(cached), 1)
        self.assertEqual(migrated["0"]["score"], 1.0)
        self.assertFalse(missing)
        self.assertFalse(overwritten)
        self.assertEqual(recompute, discovered)

    def test_episode_policy_eval_json_is_loaded_by_rollout_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "gesim" / "pickfruit" / "ep000000"
            rollout = episode / "0"
            rollout.mkdir(parents=True)
            (rollout / "meta.json").write_text(
                json.dumps(
                    {
                        "policy_id": "pi05",
                        "task_id": "pickfruit",
                        "init_id": "ep000000",
                        "domain": "wm",
                        "rollout_id": "0",
                        "checkpoint": "/ckpt/pi05",
                        "prompt": "pick the fruit",
                        "world_model_id": "gesim",
                    }
                ),
                encoding="utf-8",
            )
            expected = {
                "status": "complete",
                "score": 1.0,
                "key_steps": [],
            }
            (episode / "policy_eval.json").write_text(
                json.dumps({"0": expected}),
                encoding="utf-8",
            )
            discovered = discover_policy_rollouts(root)
            cached, missing = _load_precomputed_judges(discovered)
        self.assertFalse(missing)
        self.assertEqual(cached[discovered[0].uid], expected)


if __name__ == "__main__":
    unittest.main()
