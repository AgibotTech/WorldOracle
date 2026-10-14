import unittest

from acwmbench.configuration import (
    configure_distribution_paths,
    configure_policy_eval_paths,
)


class DistributionPathConfigTest(unittest.TestCase):
    def test_policy_rollout_selects_gt_and_runtime_paths(self):
        config = {
            "save_path": "./output",
            "data": {
                "gt_path": "/datasets/human",
                "val_base": "/predictions/old",
                "distributions": {
                    "policy_rollout": {
                        "gt_path": "/datasets/policy_rollout",
                    }
                },
            },
        }
        configure_distribution_paths(
            config,
            distribution="policy_rollout",
            val_base="/predictions/ctrlworld",
            output_dir="/results/ctrlworld",
        )
        self.assertEqual(config["data"]["distribution"], "policy_rollout")
        self.assertEqual(config["data"]["gt_path"], "/datasets/policy_rollout")
        self.assertEqual(config["data"]["val_base"], "/predictions/ctrlworld")
        self.assertEqual(config["save_path"], "/results/ctrlworld")

    def test_unknown_distribution_lists_available_profiles(self):
        config = {
            "data": {
                "distributions": {
                    "human_operated": {"gt_path": "/datasets/human"},
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "human_operated"):
            configure_distribution_paths(config, distribution="missing")


class PolicyEvaluationPathConfigTest(unittest.TestCase):
    def test_policy_data_paths_select_offline_wm_score(self):
        config = {
            "data": {
                "policy_val_base": "/rollouts/gesim_g01",
                "policy_gt_path": "/datasets/policy_eval",
            },
            "dimensions": {
                "policy_eval": {
                    "mode": "wm_generate",
                    "rollout_root": "/old/root",
                }
            },
        }
        configure_policy_eval_paths(config, ["policy_eval"])
        policy = config["dimensions"]["policy_eval"]
        self.assertEqual(policy["mode"], "wm_score")
        self.assertEqual(policy["rollout_root"], "/rollouts/gesim_g01")
        self.assertEqual(policy["gt_root"], "/datasets/policy_eval")

    def test_non_policy_runs_are_unchanged(self):
        config = {
            "data": {"policy_val_base": "/rollouts/gesim_g01"},
            "dimensions": {"policy_eval": {"mode": "wm_generate"}},
        }
        configure_policy_eval_paths(config, ["physical_judge"])
        self.assertEqual(
            config["dimensions"]["policy_eval"]["mode"],
            "wm_generate",
        )


if __name__ == "__main__":
    unittest.main()
