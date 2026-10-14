from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# pylint: disable=protected-access

SCRIPT = Path(__file__).parents[1] / "tools" / "export_inference_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_inference_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


class ExportInferenceDatasetTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[exporter.TaskSpec, Path, Path]:
        pa, _, pq = exporter._import_pyarrow()
        private = root / "private_robot_data"
        crop_root = root / "private_crop_plan"
        crop = crop_root / "pickfruit" / "ep000007"
        crop.mkdir(parents=True)
        (crop / "slice.json").write_text(
            json.dumps(
                {
                    "source_episode_index": 42,
                    "start_frame": 1,
                    "end_frame": 4,
                    "source_video": "/opt/company/camera.mp4",
                }
            ),
            encoding="utf-8",
        )

        features = {
            "action": {
                "dtype": "float32",
                "shape": [52],
                "field_descriptions": {
                    "joint_position": {"start": 0, "end": 52}
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [208],
                "field_descriptions": {
                    "robot_state": {"start": 0, "end": 208}
                },
            },
            **{
                exporter.CAMERA_FEATURES[camera]: {
                    "dtype": "video",
                    "shape": [3, 8, 10],
                }
                for camera in exporter.CAMERAS
            },
        }
        (private / "meta").mkdir(parents=True)
        (private / "meta" / "info.json").write_text(
            json.dumps({"fps": 20, "features": features}), encoding="utf-8"
        )

        episode_data = {
            "episode_index": [42],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
        for camera in exporter.CAMERAS:
            feature = exporter.CAMERA_FEATURES[camera]
            episode_data[f"videos/{feature}/chunk_index"] = [0]
            episode_data[f"videos/{feature}/file_index"] = [0]
            episode_data[f"videos/{feature}/from_timestamp"] = [0.5]
        metadata_dir = private / "meta" / "episodes" / "chunk-000"
        metadata_dir.mkdir(parents=True)
        pq.write_table(pa.table(episode_data), metadata_dir / "file-000.parquet")

        actions = np.arange(5 * 52, dtype=np.float32).reshape(5, 52) / 10
        states = np.arange(5 * 208, dtype=np.float32).reshape(5, 208) / 20
        data_dir = private / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([42] * 5, type=pa.int64()),
                    "frame_index": pa.array(range(5), type=pa.int64()),
                    "action": pa.FixedSizeListArray.from_arrays(
                        pa.array(actions.reshape(-1), type=pa.float32()), 52
                    ),
                    "observation.state": pa.FixedSizeListArray.from_arrays(
                        pa.array(states.reshape(-1), type=pa.float32()), 208
                    ),
                }
            ),
            data_dir / "file-000.parquet",
        )

        calibration_dir = private / "meta" / "agibot"
        calibration_dir.mkdir(parents=True)
        calibration = {"42": {"camera_resolution": {}}}
        for stem in ("head_front_rgb", "hand_left_rgb", "hand_right_rgb"):
            calibration["42"][f"intrinsic_{stem}"] = [
                [100.0, 0.0, 5.0],
                [0.0, 100.0, 4.0],
                [0.0, 0.0, 1.0],
            ]
            calibration["42"]["camera_resolution"][stem] = [10, 8]
        (calibration_dir / "camera_params.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )

        urdf = root / "private_robot.urdf"
        urdf.write_text(
            """<robot name="private_g02">
  <link name="base">
    <visual><geometry><mesh filename="/opt/company/base.dae"/></geometry></visual>
  </link>
  <link name="wrist">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/><child link="wrist"/>
    <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>
""",
            encoding="utf-8",
        )
        return exporter.TaskSpec("pickfruit", private, 17721, 10451), crop_root, urdf

    def test_complete_export_is_portable_and_preserves_vectors(self) -> None:
        _, _, pq = exporter._import_pyarrow()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, crop_root, urdf = self._inputs(root)
            selections = exporter.discover_selections(
                crop_root, {"pickfruit": task}, None
            )
            info = exporter._json_load(task.source_root / "meta" / "info.json")
            output = root / "public_dataset"
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            rgb[..., 1] = 127
            with mock.patch.object(exporter, "_decode_rgb", return_value=rgb):
                exporter._build_dataset(
                    output, selections, {"pickfruit": info}, urdf
                )

            summaries = exporter._validate_root(output)
            self.assertEqual(summaries[0]["path"], "pickfruit/ep000007")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "README.md",
                    "manifest.json",
                    "schema.json",
                    "checksums.sha256",
                    "shared",
                    "pickfruit",
                },
            )
            episode = output / "pickfruit" / "ep000007"
            table = pq.read_table(episode / "trajectory.parquet")
            self.assertEqual(
                table.column_names, list(exporter.PUBLIC_TRAJECTORY_COLUMNS)
            )
            self.assertEqual(table["frame_index"].to_pylist(), [0, 1, 2])
            np.testing.assert_array_equal(
                np.asarray(table["timestamp"].to_pylist(), dtype=np.float32),
                np.asarray([0.0, 0.05, 0.1], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.stack(table["action"].to_pylist()),
                (np.arange(5 * 52, dtype=np.float32).reshape(5, 52) / 10)[1:4],
            )
            np.testing.assert_array_equal(
                np.stack(table["observation.state"].to_pylist()),
                (np.arange(5 * 208, dtype=np.float32).reshape(5, 208) / 20)[1:4],
            )
            urdf_text = (
                output / "shared" / "g02" / "kinematics.urdf"
            ).read_text(encoding="utf-8")
            self.assertNotIn("mesh", urdf_text.lower())
            self.assertIn("<inertial>", urdf_text.lower())
            all_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file()
                and path.suffix in {".json", ".md", ".urdf", ".sha256"}
            ).lower()
            for forbidden in (
                "/opt/company",
                "source_root",
                "source_video",
                "selection",
                "lerobot",
                "eval_dataset",
            ):
                self.assertNotIn(forbidden, all_text)
            self.assertIn("No future RGB frames", (output / "README.md").read_text())

    def test_private_episode_id_is_not_public_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task, crop_root, _ = self._inputs(Path(temporary))
            selected = exporter.discover_selections(
                crop_root, {"pickfruit": task}, None
            )[0]
            self.assertEqual(selected.episode_name, "ep000007")
            self.assertEqual(selected.episode_index, 42)

    def test_recursive_portability_validator_rejects_leaks(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute path"):
            exporter._reject_nonportable(
                {"nested": [{"artifact": "/opt/company/private.bin"}]}, "test"
            )
        with self.assertRaisesRegex(ValueError, "forbidden text"):
            exporter._reject_nonportable(
                {"nested": {"source_root": "relative/value"}}, "test"
            )

    def test_checksum_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"original")
            checksums = root / "checksums.sha256"
            exporter._write_checksum_file(root, [Path("payload.bin")], checksums)
            exporter._verify_checksum_file(checksums, root)
            payload.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                exporter._verify_checksum_file(checksums, root)

    def test_bad_crop_and_bad_source_width_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "slice.json"
            with self.assertRaisesRegex(ValueError, "invalid half-open range"):
                exporter._slice_bounds(
                    {"start_frame": 8, "end_frame": 8}, path
                )
        features = {
            "action": {"shape": [51]},
            "observation.state": {"shape": [208]},
            **{
                exporter.CAMERA_FEATURES[camera]: {}
                for camera in exporter.CAMERAS
            },
        }
        with self.assertRaisesRegex(ValueError, r"expected \[52\]"):
            exporter._validate_source_info(
                {"fps": 30, "features": features}, Path("info.json")
            )

    def test_overwrite_and_urdf_are_required(self) -> None:
        with self.assertRaises(SystemExit):
            exporter.parse_args([])
        with self.assertRaises(SystemExit):
            exporter.parse_args(["--overwrite"])
        args = exporter.parse_args(
            ["--overwrite", "--urdf-path", "/opt/robot.urdf"]
        )
        self.assertTrue(args.overwrite)
        validate = exporter.parse_args(["--validate-only"])
        self.assertEqual(validate.output_root, exporter.DEFAULT_OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
