from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from acwmbench.metrics.keyarea_visual_quality import (
    RESULT_METRICS,
    _compute_masked_metrics,
    _prepared_episode_groups,
    compute_keyarea_visual_quality,
)
from acwmbench.dataset import EpisodeTrial
from acwmbench.runtime import GPUInfo, resolve_keyarea_execution
from acwmbench.runner import _done_uids_from_episode_files


class _FakeLPIPS(torch.nn.Module):
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return (pred - gt).abs().mean(dim=(1, 2, 3), keepdim=True)


class _FakeJEPA(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Return [batch, tokens, features], matching the real encoder contract.
        features = batch.squeeze(2).mean(dim=(2, 3))
        return features.unsqueeze(1)


def _fake_transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((16, 16)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 256, size=(6, 48, 64, 3), dtype=np.uint8)
    gt = rng.integers(0, 256, size=(6, 48, 64, 3), dtype=np.uint8)
    masks = np.zeros((2, 6, 48, 64), dtype=np.uint8)
    # Two exact crop sizes plus moving masks exercise LPIPS shape bucketing.
    for frame in range(6):
        masks[0, frame, 4 + frame : 24 + frame, 5:30] = 1
        masks[1, frame, 15:40, 30 - frame : 55 - frame] = 1
    return pred, gt, masks


def test_batched_masked_metrics_match_serial() -> None:
    pred, gt, masks = _fixture_arrays()
    common = dict(
        pred=pred,
        gt=gt,
        masks=masks,
        device="cpu",
        downsample_rate=1,
        crop_to_bbox=True,
        bbox_pad=3,
        lpips_min_crop_size=16,
        mask_dilation_pixels=2,
        mask_aware_psnr=True,
        lpips_fn=_FakeLPIPS(),
        jepa_model=_FakeJEPA(),
        jepa_transform=_fake_transform,
    )
    serial = _compute_masked_metrics(
        **common, lpips_batch_size=1, vjepa_batch_size=1
    )
    batched = _compute_masked_metrics(
        **common, lpips_batch_size=4, vjepa_batch_size=5
    )

    assert [(x["frame"], x["mask_index"]) for x in serial] == [
        (x["frame"], x["mask_index"]) for x in batched
    ]
    for expected, actual in zip(serial, batched):
        assert actual["psnr"] == expected["psnr"]
        assert actual["lpips"] == pytest.approx(expected["lpips"], abs=1e-7)
        assert actual["vjepa_similarity"] == pytest.approx(
            expected["vjepa_similarity"], abs=1e-7
        )


def test_prefetch_groups_preserve_episode_order_and_isolate_failure() -> None:
    episodes = list(range(7))

    def prepare(value: int):
        time.sleep(0.001 * (7 - value))
        if value == 3:
            return None
        return value

    groups = list(
        _prepared_episode_groups(
            episodes,
            prepare,
            workers=3,
            max_in_flight=2,
            prefetch_depth=1,
        )
    )
    assert [value for group in groups for value in group] == [0, 1, 2, 4, 5, 6]
    assert all(len(group) <= 2 for group in groups)


def test_4090_profile_and_explicit_overrides() -> None:
    gpu = GPUInfo(True, 1, 0, "NVIDIA GeForce RTX 4090", 23.99)
    with patch("acwmbench.runtime.detect_gpu", return_value=gpu):
        config = resolve_keyarea_execution(
            {
                "gpu_profile": "rtx_4090",
                "auto_detect": True,
                "memory_utilization_target": 0.85,
            },
            {
                "mode": "pipeline",
                "episode_workers": "auto",
                "max_in_flight_episodes": 3,
                "vjepa_batch_size": "auto",
            },
            device="cuda",
        )
    assert config.gpu_profile == "rtx_4090"
    assert config.episode_workers == 2
    assert config.max_in_flight_episodes == 3
    assert config.vjepa_batch_size == 32
    assert config.lpips_batch_size == 8


def test_serial_mode_forces_singletons() -> None:
    with patch(
        "acwmbench.runtime.detect_gpu",
        return_value=GPUInfo(False, 0, None, "cpu", 0.0),
    ):
        config = resolve_keyarea_execution(
            {},
            {
                "mode": "serial",
                "episode_workers": 8,
                "vjepa_batch_size": 64,
            },
            device="cpu",
        )
    assert config.episode_workers == 1
    assert config.max_in_flight_episodes == 1
    assert config.vjepa_batch_size == 1
    assert config.lpips_batch_size == 1


def test_invalid_memory_target_is_rejected() -> None:
    with patch(
        "acwmbench.runtime.detect_gpu",
        return_value=GPUInfo(False, 0, None, "cpu", 0.0),
    ):
        with pytest.raises(ValueError, match="memory_utilization_target"):
            resolve_keyarea_execution(
                {"memory_utilization_target": 1.5},
                {},
                device="cpu",
            )


class _FakeLPIPSModule:
    @staticmethod
    def LPIPS(net: str):
        assert net == "alex"
        return _FakeLPIPS()


def _write_episode(root, task: str, episode: str, offset: int) -> EpisodeTrial:
    pred_video = root / "pred" / task / episode / "video"
    gt_episode = root / "gt" / task / episode
    gt_video = gt_episode / "video"
    for base in (pred_video, gt_video):
        (base / "top_head").mkdir(parents=True)
    rng = np.random.default_rng(100 + offset)
    for index in range(4):
        gt = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
        pred = np.clip(gt.astype(np.int16) + offset + index, 0, 255).astype(
            np.uint8
        )
        Image.fromarray(gt).save(
            gt_video / "top_head" / f"frame_{index:05d}.jpg"
        )
        Image.fromarray(pred).save(
            pred_video / "top_head" / f"frame_{index:05d}.jpg"
        )
    masks = np.zeros((1, 4, 32, 40), dtype=np.uint8)
    masks[:, :, 5:27, 7:34] = 1
    prompt = gt_episode / "prompt"
    secondary = gt_episode / "multiview_masks" / "top_head"
    prompt.mkdir(parents=True)
    secondary.mkdir(parents=True)
    np.savez_compressed(prompt / "masks.npz", masks=masks)
    np.savez_compressed(secondary / "masks.npz", masks=masks)
    return EpisodeTrial(
        task_id=task,
        episode_id=episode,
        trial_id="default",
        video_dir=str(pred_video),
        gt_video_dir=str(gt_video),
    )


def test_pipeline_full_episode_save_visualize_and_equivalence(
    tmp_path, monkeypatch
) -> None:
    import acwmbench.metrics.keyarea_visual_quality as module

    monkeypatch.setattr(module, "ensure_lpips", lambda: _FakeLPIPSModule())
    monkeypatch.setattr(
        module, "ensure_jepa", lambda *args, **kwargs: (_FakeJEPA(), _fake_transform)
    )
    trials = [
        _write_episode(tmp_path, "pickfruit", "ep000001", 1),
        _write_episode(tmp_path, "pickfruit", "ep000002", 2),
    ]
    common = dict(
        episodes=trials,
        gt_root=tmp_path / "gt",
        device="cpu",
        eval_view="top_head",
        downsample_rate=1,
        crop_to_bbox=True,
        bbox_pad=2,
        lpips_min_crop_size=16,
        mask_dilation_pixels=1,
        mask_aware_psnr=True,
        vjepa_model_path=None,
        vjepa_model_name="fake",
        secondary_mask_view="top_head",
        progress=False,
    )
    serial = compute_keyarea_visual_quality(
        **common, execution={"mode": "serial"}
    )
    pipeline = compute_keyarea_visual_quality(
        **common,
        visualize=True,
        visualize_max_frames=1,
        save_root=tmp_path / "pred",
        execution={
            "mode": "pipeline",
            "episode_workers": 2,
            "max_in_flight_episodes": 2,
            "prefetch_depth": 1,
            "lpips_batch_size": 4,
            "vjepa_batch_size": 4,
            "pin_memory": False,
        },
    )
    assert len(pipeline["per_episode"]) == 2
    for expected, actual in zip(
        serial["per_episode"], pipeline["per_episode"]
    ):
        for metric in RESULT_METRICS:
            assert actual[metric] == pytest.approx(expected[metric], abs=1e-7)
        assert len(actual["per_frame"]) == len(expected["per_frame"])
        for expected_frame, actual_frame in zip(
            expected["per_frame"], actual["per_frame"]
        ):
            assert actual_frame["frame"] == expected_frame["frame"]
            for metric in RESULT_METRICS:
                assert actual_frame[metric] == pytest.approx(
                    expected_frame[metric], abs=1e-7
                )
        episode_dir = (
            tmp_path / "pred" / actual["task_id"] / actual["episode_id"]
        )
        assert (episode_dir / "keyarea_visual_quality.json").is_file()
        assert (
            episode_dir
            / "visualize"
            / "keyarea_visual_quality_manifest.json"
        ).is_file()


def test_resume_cache_checks_all_numerical_settings(tmp_path) -> None:
    from acwmbench import episode_store

    trial = _write_episode(tmp_path, "pickfruit", "ep000001", 1)
    settings = {
        "downsample_rate": 2,
        "fps": 30.0,
        "secondary_mask_view": "top_head",
        "eval_view": "top_head",
        "crop_to_bbox": True,
        "bbox_pad": 10,
        "lpips_min_crop_size": 64,
        "mask_dilation_pixels": 1,
        "mask_aware_psnr": True,
        "vjepa_model_name": "fake",
        "vjepa_model_path": None,
    }
    episode_store.save_metric(
        tmp_path / "pred",
        trial.task_id,
        trial.episode_id,
        "keyarea_visual_quality",
        {
            "uid": trial.uid,
            "metrics": list(RESULT_METRICS),
            "metric_version": 12,
            **settings,
        },
    )
    assert _done_uids_from_episode_files(
        tmp_path / "pred",
        [trial],
        "keyarea_visual_quality",
        expected_metric_settings=settings,
    ) == {trial.uid}

    changed = dict(settings, eval_view="hand_left")
    assert _done_uids_from_episode_files(
        tmp_path / "pred",
        [trial],
        "keyarea_visual_quality",
        expected_metric_settings=changed,
    ) == set()
    assert (
        episode_store.load_metric(
            tmp_path / "pred",
            trial.task_id,
            trial.episode_id,
            "keyarea_visual_quality",
        )
        is None
    )

