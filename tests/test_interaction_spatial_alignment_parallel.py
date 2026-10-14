from __future__ import annotations

import json
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from acwmbench import episode_store
from acwmbench.dataset import EpisodeTrial
from acwmbench.metrics.interaction_spatial_alignment import (
    SUMMARY_METRICS,
    _PreparedFrameBatch,
    _prefetched_frame_batches,
    compute_interaction_spatial_alignment,
)
from acwmbench.runner import _done_uids_from_episode_files
from acwmbench.runtime import (
    GPUInfo,
    resolve_interaction_execution,
)


class _FakeDinoExtractor:
    def __init__(self, *, device: str, **_):
        self.torch = torch
        self.device = torch.device(device)

    def prepare_batch(self, images):
        return torch.stack(
            [
                torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy() / 255.0
                ).permute(2, 0, 1)
                for image in images
            ]
        )

    def extract_prepared(self, batch, *, pin_memory: bool = False):
        del pin_memory
        resized = torch.nn.functional.interpolate(
            batch, size=(4, 5), mode="bilinear", align_corners=False
        )
        features = torch.cat(
            [resized, resized[:, 0:1] * resized[:, 1:2]], dim=1
        )
        features = torch.nn.functional.normalize(features, dim=1)
        array = features.numpy()
        return [array[index] for index in range(len(array))]


def _write_episode(root, episode_id: str, offset: int) -> EpisodeTrial:
    task = "pickfruit"
    pred_video = root / "pred" / task / episode_id / "video" / "top_head"
    gt_episode = root / "gt" / task / episode_id
    gt_video = gt_episode / "video" / "top_head"
    pred_video.mkdir(parents=True)
    gt_video.mkdir(parents=True)
    rng = np.random.default_rng(1000 + offset)
    for frame in range(5):
        gt = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
        pred = np.roll(gt, shift=offset + frame % 2, axis=1)
        Image.fromarray(gt).save(gt_video / f"frame_{frame:05d}.png")
        Image.fromarray(pred).save(pred_video / f"frame_{frame:05d}.png")
    object_masks = np.zeros((2, 5, 32, 40), dtype=np.uint8)
    object_masks[0, :, 3:18, 4:20] = 1
    object_masks[1, :, 14:29, 18:36] = 1
    arm_masks = np.zeros((1, 5, 32, 40), dtype=np.uint8)
    arm_masks[0, :, 8:28, 10:32] = 1
    prompt_dir = gt_episode / "prompt"
    arm_dir = gt_episode / "multiview_masks" / "top_head"
    prompt_dir.mkdir(parents=True)
    arm_dir.mkdir(parents=True)
    np.savez_compressed(prompt_dir / "masks.npz", masks=object_masks)
    np.savez_compressed(arm_dir / "masks.npz", masks=arm_masks)
    return EpisodeTrial(
        task_id=task,
        episode_id=episode_id,
        trial_id="default",
        video_dir=str(pred_video.parent),
        gt_video_dir=str(gt_video.parent),
    )


def _compute(trials, tmp_path, execution, *, save_root=None):
    return compute_interaction_spatial_alignment(
        episodes=trials,
        gt_root=tmp_path / "gt",
        device="cpu",
        eval_view="top_head",
        downsample_rate=1,
        fps=30.0,
        checkpoint="fake.pt",
        dinov3_repo="fake-repo",
        model_name="fake-dino",
        batch_size=2,
        min_mask_patches=1,
        min_mutual_matches=1,
        save_root=save_root,
        progress=False,
        execution=execution,
    )


def test_serial_pipeline_fusion_equivalence_and_save(
    tmp_path, monkeypatch
) -> None:
    import acwmbench.metrics.interaction_spatial_alignment as module

    monkeypatch.setattr(module, "DinoExtractor", _FakeDinoExtractor)
    trials = [
        _write_episode(tmp_path, "ep000001", 1),
        _write_episode(tmp_path, "ep000002", 2),
    ]
    serial = _compute(
        trials,
        tmp_path,
        {"mode": "serial", "frame_batch_size": 2},
    )
    pipeline = _compute(
        trials,
        tmp_path,
        {
            "mode": "pipeline",
            "frame_batch_size": 3,
            "loader_workers": 2,
            "prefetch_batches": 2,
            "matcher_workers": 2,
            "pin_memory": False,
            "fuse_gt_generated": True,
        },
        save_root=tmp_path / "pred",
    )
    serial_values = json.loads(json.dumps(serial))
    pipeline_values = json.loads(json.dumps(pipeline))
    for payload in (serial_values, pipeline_values):
        payload["aggregated"].pop("evaluation_settings", None)
        for episode in payload["per_episode"]:
            episode.pop("evaluation_settings", None)
    assert json.dumps(pipeline_values, sort_keys=True) == json.dumps(
        serial_values, sort_keys=True
    )
    for trial in trials:
        result_path = (
            tmp_path
            / "pred"
            / trial.task_id
            / trial.episode_id
            / "interaction_spatial_alignment.json"
        )
        assert result_path.is_file()


def test_prefetch_is_bounded_and_ordered() -> None:
    frames = list(range(12))
    active = 0
    max_active = 0

    def prepare(batch):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.002 * (4 - batch[0] % 4))
        active -= 1
        return _PreparedFrameBatch(list(batch), None, None)

    output = list(
        _prefetched_frame_batches(
            frames,
            frame_batch_size=2,
            loader_workers=3,
            prefetch_batches=2,
            prepare=prepare,
        )
    )
    assert [frame for batch in output for frame in batch.frames] == frames
    assert max_active <= 2


def test_bad_episode_is_isolated(tmp_path, monkeypatch) -> None:
    import acwmbench.metrics.interaction_spatial_alignment as module

    monkeypatch.setattr(module, "DinoExtractor", _FakeDinoExtractor)
    bad = _write_episode(tmp_path, "ep000001", 1)
    good = _write_episode(tmp_path, "ep000002", 2)
    bad_frame = (
        tmp_path
        / "pred"
        / bad.task_id
        / bad.episode_id
        / "video"
        / "top_head"
        / "frame_00000.png"
    )
    bad_frame.write_bytes(b"not an image")
    result = _compute(
        [bad, good],
        tmp_path,
        {
            "mode": "pipeline",
            "frame_batch_size": 2,
            "loader_workers": 2,
            "prefetch_batches": 2,
            "matcher_workers": 2,
            "pin_memory": False,
            "fuse_gt_generated": True,
        },
    )
    assert [episode["uid"] for episode in result["per_episode"]] == [good.uid]


def test_4090_interaction_profile_and_serial_mode() -> None:
    gpu = GPUInfo(True, 1, 0, "NVIDIA GeForce RTX 4090", 23.99)
    with patch("acwmbench.runtime.detect_gpu", return_value=gpu):
        pipeline = resolve_interaction_execution(
            {"gpu_profile": "rtx_4090", "auto_detect": True},
            {"mode": "pipeline", "frame_batch_size": "auto"},
            device="cuda",
        )
        serial = resolve_interaction_execution(
            {"gpu_profile": "rtx_4090", "auto_detect": True},
            {
                "mode": "serial",
                "frame_batch_size": 4,
                "loader_workers": 8,
                "matcher_workers": 8,
                "fuse_gt_generated": True,
            },
            device="cuda",
        )
    assert pipeline.frame_batch_size == 4
    assert pipeline.fuse_gt_generated is False
    assert pipeline.loader_workers == 4
    assert serial.frame_batch_size == 4
    assert serial.loader_workers == serial.matcher_workers == 1
    assert serial.fuse_gt_generated is False


def test_resume_checks_numerically_relevant_extraction_settings(tmp_path) -> None:
    trial = _write_episode(tmp_path, "ep000001", 1)
    settings = {
        "eval_view": "top_head",
        "downsample_rate": 10,
        "fps": 30.0,
        "checkpoint": "fake.pt",
        "dinov3_repo": "fake-repo",
        "model_name": "fake-dino",
        "feature_batch_size": 4,
        "fuse_gt_generated": False,
        "min_mask_patches": 2,
    }
    episode_store.save_metric(
        tmp_path / "pred",
        trial.task_id,
        trial.episode_id,
        "interaction_spatial_alignment",
        {
            "uid": trial.uid,
            "metric_version": 2,
            "metrics": list(SUMMARY_METRICS),
            "evaluation_settings": {**settings, "batch_size": 4},
        },
    )
    assert _done_uids_from_episode_files(
        tmp_path / "pred",
        [trial],
        "interaction_spatial_alignment",
        expected_metric_settings=settings,
    ) == {trial.uid}

    changed = dict(settings, feature_batch_size=8)
    assert _done_uids_from_episode_files(
        tmp_path / "pred",
        [trial],
        "interaction_spatial_alignment",
        expected_metric_settings=changed,
    ) == set()


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_execution_batch_is_rejected(
    tmp_path, monkeypatch, value
) -> None:
    import acwmbench.metrics.interaction_spatial_alignment as module

    monkeypatch.setattr(module, "DinoExtractor", _FakeDinoExtractor)
    with pytest.raises(ValueError, match="frame_batch_size"):
        _compute(
            [],
            tmp_path,
            {"mode": "pipeline", "frame_batch_size": value},
        )
