import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_acwmbench_scores import (
    keyarea_episode_score,
    summarize_dimension,
)


NORMALIZATION = {
    "psnr": {"min": 8.0, "max": 30.0},
    "lpips": {"min": 0.1, "max": 0.35},
    "vjepa_similarity": {"min": 0.97, "max": 1.0},
}


class ScoreSummaryTest(unittest.TestCase):
    def test_keyarea_normalizes_each_mask_before_averaging(self):
        mask = {
            "psnr": 19.0,
            "lpips": 0.225,
            "vjepa_similarity": 0.985,
        }
        data = {
            "mask_sources": {
                source: {
                    "per_frame": [{"frame": 0, "masks": [mask]}],
                }
                for source in ("object", "robot_arm")
            }
        }
        self.assertAlmostEqual(
            keyarea_episode_score(data, NORMALIZATION),
            50.0,
        )

    def test_dimension_score_is_task_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {
                ("task_a", "ep000000"): 100.0,
                ("task_a", "ep000001"): 0.0,
                ("task_b", "ep000000"): 100.0,
            }
            for (task, episode), score in records.items():
                path = root / task / episode
                path.mkdir(parents=True)
                (path / "interaction_spatial_alignment.json").write_text(
                    json.dumps(
                        {
                            "metric_version": 2,
                            "alignment_score": score,
                        }
                    ),
                    encoding="utf-8",
                )
            summary = summarize_dimension(
                root,
                "interaction_spatial_alignment",
                NORMALIZATION,
                tasks=None,
            )
        self.assertAlmostEqual(summary["score"], 75.0)
        self.assertEqual(summary["num_tasks"], 2)
        self.assertEqual(summary["num_episodes"], 3)


if __name__ == "__main__":
    unittest.main()
