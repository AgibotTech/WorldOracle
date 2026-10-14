import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

# The focused helper tests do not parse prompts, but importing EpisodeTrial
# traverses dataset.py, whose optional runtime dependency is absent in the
# lightweight test environment.
if "json_repair" not in sys.modules:
    json_repair = types.ModuleType("json_repair")
    json_repair.repair_json = lambda value, **_kwargs: value
    sys.modules["json_repair"] = json_repair
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from acwmbench.metrics.multiview_consistency import (
    _combine_view_scores,
    _compute_hand_views,
    _interaction_settings,
    interaction_robot_arm_digest,
)


def _interaction(score=80.0):
    return {
        "uid": "task/ep000000",
        "metric_version": 2,
        "evaluation_settings": {
            "eval_view": "top_head",
            "downsample_rate": 10,
            "fps": 30.0,
            "checkpoint": "/tmp/dinov3.pth",
            "dinov3_repo": "/tmp/dinov3",
            "model_name": "dinov3_vitb16",
            "feature_batch_size": 8,
            "fuse_gt_generated": True,
            "min_mask_patches": 2,
            "min_mutual_matches": 3,
            "inlier_radius_px": 32.0,
            "confidence_bias": -2.0,
            "confidence_cosine_weight": 3.0,
            "confidence_margin_weight": 8.0,
            "confidence_mutual_weight": 1.5,
            "confidence_inlier_weight": 1.5,
            "match_support_target_ratio": 0.3,
            "match_support_count_scale": 8.0,
            "match_support_power": 1.0,
            "confidence_bad": 0.5,
            "confidence_good": 0.8,
            "distance_good_px": 30.0,
            "distance_bad_px": 150.0,
            "alignment_confidence_power": 3.0,
        },
        "mask_sources": {
            "robot_arm": {
                "per_frame": [
                    {"frame": 0, "alignment_score": score},
                ]
            }
        },
    }


class MultiViewConsistencyTest(unittest.TestCase):
    def test_always_averages_three_view_scores(self):
        result = _combine_view_scores(
            frame=30,
            view_scores={
                "top_head": 90.0,
                "hand_left": 60.0,
                "hand_right": 30.0,
            },
            fps=30.0,
        )
        self.assertEqual(result["multiview_consistency_score"], 60.0)
        self.assertEqual(result["time_sec"], 1.0)

    def test_unavailable_view_contributes_zero(self):
        result = _combine_view_scores(
            frame=0,
            view_scores={
                "top_head": 90.0,
                "hand_left": None,
                "hand_right": 30.0,
            },
            fps=30.0,
        )
        self.assertEqual(result["multiview_consistency_score"], 40.0)
        self.assertFalse(result["view_available"]["hand_left"])

    def test_settings_are_derived_from_interaction(self):
        settings = _interaction_settings(_interaction())
        self.assertEqual(settings["distance_bad_px"], 150.0)
        self.assertEqual(settings["feature_batch_size"], 8)

    def test_digest_changes_when_consumed_top_score_changes(self):
        self.assertNotEqual(
            interaction_robot_arm_digest(_interaction(80.0)),
            interaction_robot_arm_digest(_interaction(81.0)),
        )

    def test_cross_view_pipeline_is_parallel_equivalent_and_coalesced(self):
        resources = {}
        for view_index, view in enumerate(("hand_left", "hand_right")):
            masks = np.full(
                (1, 3, 4, 5),
                view_index,
                dtype=np.uint8,
            )
            resources[view] = {
                "gt_frames": {
                    frame: Path(f"{view}_gt_{frame}.png")
                    for frame in range(3)
                },
                "generated_frames": {
                    frame: Path(f"{view}_pred_{frame}.png")
                    for frame in range(3)
                },
                "masks": masks,
            }

        batch_sizes = []

        def prepare(_extractor, item_ids, *_args):
            batch_sizes.append(len(item_ids))
            return types.SimpleNamespace(
                frames=list(item_ids),
                gt_batch=None,
                generated_batch=None,
            )

        def extract(_extractor, prepared, **_kwargs):
            features = [
                np.asarray([item_id], dtype=np.float32)
                for item_id in prepared.frames
            ]
            return features, features

        def match(*, frame, masks, **_kwargs):
            return [
                {
                    "frame": frame,
                    "view_marker": int(masks[0, 0, 0, 0]),
                    "valid": True,
                    "alignment_score": float(frame),
                }
            ]

        common = {
            "extractor": object(),
            "selected_frames": [0, 1, 2],
            "resources": resources,
            "fps": 30.0,
            "match_kwargs": {},
        }
        with (
            patch(
                "acwmbench.metrics.multiview_consistency._prepare_frame_batch",
                side_effect=prepare,
            ),
            patch(
                "acwmbench.metrics.multiview_consistency._extract_paired_features",
                side_effect=extract,
            ),
            patch(
                "acwmbench.metrics.multiview_consistency._match_hand_frame",
                side_effect=match,
            ),
        ):
            serial = _compute_hand_views(
                **common,
                execution={
                    "frame_batch_size": 2,
                    "loader_workers": 1,
                    "prefetch_batches": 1,
                    "matcher_workers": 1,
                    "pin_memory": False,
                    "fuse_gt_generated": False,
                },
            )
            serial_batches = list(batch_sizes)
            batch_sizes.clear()
            parallel = _compute_hand_views(
                **common,
                execution={
                    "frame_batch_size": 2,
                    "loader_workers": 2,
                    "prefetch_batches": 2,
                    "matcher_workers": 2,
                    "pin_memory": False,
                    "fuse_gt_generated": False,
                },
            )
        self.assertEqual(parallel, serial)
        self.assertEqual(serial_batches, [2, 2, 2])
        self.assertEqual(batch_sizes, [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
