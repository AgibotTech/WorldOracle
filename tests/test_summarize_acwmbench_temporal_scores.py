import unittest

from tools.summarize_acwmbench_temporal_scores import (
    cumulative_windows,
    multiview_window_score,
    physical_window_score,
)


class TemporalScoreSummaryTest(unittest.TestCase):
    def test_default_cumulative_windows_reach_120_seconds(self):
        self.assertEqual(
            cumulative_windows(10.0, 120.0),
            [float(value) for value in range(10, 121, 10)],
        )

    def test_physical_window_uses_strict_frame_end(self):
        data = {
            "trial": {
                "status": "complete",
                "key_steps": [
                    {"frame": 299, "score": 1.0},
                    {"frame": 300, "score": 0.0},
                ],
            }
        }
        self.assertEqual(physical_window_score(data, 300), 100.0)
        self.assertEqual(physical_window_score(data, 301), 50.0)

    def test_multiview_window_is_cumulative(self):
        data = {
            "metric_version": 5,
            "per_frame": [
                {"frame": 10, "multiview_consistency_score": 80.0},
                {"frame": 400, "multiview_consistency_score": 20.0},
            ],
        }
        self.assertEqual(multiview_window_score(data, 300), 80.0)
        self.assertEqual(multiview_window_score(data, 600), 50.0)


if __name__ == "__main__":
    unittest.main()
