#!/usr/bin/env python3
"""Measure spatial alignment of generated frames with dense DINOv3 features."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCES = ("object", "robot_arm")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/keyarea_results.yaml")
    parser.add_argument(
        "--gt-root",
        default="/path/to/main-ground-truth",
    )
    parser.add_argument("--task", default="hangtowel")
    parser.add_argument("--episode", default="ep000000")
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--checkpoint",
        default="./ckpt/dinov3/"
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
    parser.add_argument("--dinov3-repo")
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--visualize-every", type=int, default=5)
    parser.add_argument("--min-mask-patches", type=int, default=2)
    parser.add_argument("--min-mutual-matches", type=int, default=3)
    parser.add_argument("--inlier-radius-px", type=float, default=32.0)
    parser.add_argument("--confidence-bias", type=float, default=-2.0)
    parser.add_argument("--confidence-cosine-weight", type=float, default=3.0)
    parser.add_argument("--confidence-margin-weight", type=float, default=8.0)
    parser.add_argument("--confidence-mutual-weight", type=float, default=1.5)
    parser.add_argument("--confidence-inlier-weight", type=float, default=1.5)
    parser.add_argument("--match-support-target-ratio", type=float, default=0.3)
    parser.add_argument("--match-support-count-scale", type=float, default=8.0)
    parser.add_argument("--match-support-power", type=float, default=1.0)
    parser.add_argument("--confidence-bad", type=float, default=0.5)
    parser.add_argument("--confidence-good", type=float, default=0.8)
    parser.add_argument("--distance-good-px", type=float, default=30.0)
    parser.add_argument("--distance-bad-px", type=float, default=150.0)
    parser.add_argument("--alignment-confidence-power", type=float, default=3.0)
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def json_safe(value: Any) -> Any:
    """Recursively convert metadata to strict JSON values without NaN/Infinity."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-value))


def normalized_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def match_feature_maps(
    gt_features: np.ndarray,
    generated_features: np.ndarray,
    mask: np.ndarray,
    *,
    min_mask_patches: int = 2,
    min_mutual_matches: int = 3,
    inlier_radius_px: float = 32.0,
    confidence_bias: float = -2.0,
    confidence_cosine_weight: float = 3.0,
    confidence_margin_weight: float = 8.0,
    confidence_mutual_weight: float = 1.5,
    confidence_inlier_weight: float = 1.5,
    match_support_target_ratio: float = 0.3,
    match_support_count_scale: float = 8.0,
    match_support_power: float = 1.0,
    confidence_bad: float = 0.5,
    confidence_good: float = 0.8,
    distance_good_px: float = 30.0,
    distance_bad_px: float = 150.0,
    alignment_confidence_power: float = 3.0,
) -> dict[str, Any]:
    """Match masked GT patch tokens to generated tokens using cosine MNNs."""
    result: dict[str, Any] = {
        "valid": False,
        "status": "invalid",
        "num_mask_patches": 0,
        "num_matches": 0,
        "num_mutual_matches": 0,
        "used_mutual": False,
        "match_gt_xy_json": "[]",
        "match_generated_xy_json": "[]",
        "match_cosine_json": "[]",
        "gt_grid_height": 0,
        "gt_grid_width": 0,
        "generated_grid_height": 0,
        "generated_grid_width": 0,
        **{
            name: float("nan")
            for name in (
                "top_cosine", "mean_cosine", "top1_top2_margin", "mutual_ratio",
                "median_dx_px", "median_dy_px", "mean_displacement_px",
                "median_displacement_px", "p95_displacement_px",
                "normalized_displacement", "inlier_ratio", "estimated_center_x",
                "estimated_center_y", "gt_center_x", "gt_center_y", "mask_area_px",
                "match_coverage_score", "sample_support_score",
                "match_support_score", "feature_confidence_score",
                "confidence_score", "confidence_quality_score",
                "distance_score", "alignment_score",
            )
        },
    }
    gt = np.asarray(gt_features, np.float32)
    gen = np.asarray(generated_features, np.float32)
    mask = np.asarray(mask) > 0
    if gt.ndim != 3 or gen.ndim != 3 or mask.ndim != 2:
        result["status"] = "bad_shape"
        return result
    if gt.shape[0] != gen.shape[0] or min(gt.shape[1:]) < 1 or min(gen.shape[1:]) < 1:
        result["status"] = "incompatible_features"
        return result
    height, width = mask.shape
    if not mask.any():
        result["status"] = "empty_mask"
        return result

    mask_small = cv2.resize(
        mask.astype(np.uint8), (gt.shape[2], gt.shape[1]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    gt_y, gt_x = np.nonzero(mask_small)
    result["num_mask_patches"] = int(len(gt_x))
    if len(gt_x) < min_mask_patches:
        result["status"] = "tiny_mask"
        return result

    gt_tokens = normalized_rows(gt[:, gt_y, gt_x].T)
    gen_tokens = normalized_rows(gen.reshape(gen.shape[0], -1).T)
    similarity = gt_tokens @ gen_tokens.T
    if not np.isfinite(similarity).all():
        similarity = np.nan_to_num(similarity, nan=-1.0, posinf=1.0, neginf=-1.0)
    top1_indices = np.argmax(similarity, axis=1)
    top1_values = similarity[np.arange(len(gt_tokens)), top1_indices]
    if similarity.shape[1] > 1:
        top2_values = np.partition(similarity, -2, axis=1)[:, -2]
    else:
        top2_values = np.full_like(top1_values, -1.0)
    reverse = np.argmax(similarity, axis=0)
    mutual = reverse[top1_indices] == np.arange(len(gt_tokens))
    num_mutual_matches = int(mutual.sum())
    mutual_ratio = float(mutual.mean())
    selected = np.flatnonzero(mutual)
    used_mutual = len(selected) >= min_mutual_matches
    if not used_mutual:
        selected = np.arange(len(gt_tokens))

    gen_y, gen_x = np.divmod(top1_indices[selected], gen.shape[2])
    gt_px_x = (gt_x[selected] + 0.5) * width / gt.shape[2]
    gt_px_y = (gt_y[selected] + 0.5) * height / gt.shape[1]
    gen_px_x = (gen_x + 0.5) * width / gen.shape[2]
    gen_px_y = (gen_y + 0.5) * height / gen.shape[1]
    match_cosines = top1_values[selected]
    dx = gen_px_x - gt_px_x
    dy = gen_px_y - gt_px_y
    displacement = np.hypot(dx, dy)
    median_dx, median_dy = float(np.median(dx)), float(np.median(dy))
    residual = np.hypot(dx - median_dx, dy - median_dy)
    inlier_ratio = float(np.mean(residual <= max(inlier_radius_px, 1e-6)))
    mask_y, mask_x = np.nonzero(mask)
    center_x, center_y = float(mask_x.mean()), float(mask_y.mean())
    area = float(mask.sum())
    median_displacement = float(np.median(displacement))
    normalized_displacement = median_displacement / max(math.sqrt(area), 1.0)
    mean_cosine = float(np.mean(match_cosines))
    top_cosine = float(np.max(match_cosines))
    margin = float(np.mean(match_cosines - top2_values[selected]))
    confidence_logit = (
        confidence_bias
        + confidence_cosine_weight * mean_cosine
        + confidence_margin_weight * margin
        + confidence_mutual_weight * mutual_ratio
        + confidence_inlier_weight * inlier_ratio
    )
    feature_confidence = sigmoid(confidence_logit)
    coverage_score = min(
        mutual_ratio / max(match_support_target_ratio, 1e-8),
        1.0,
    )
    sample_support = 1.0 - math.exp(
        -num_mutual_matches / max(match_support_count_scale, 1e-8)
    )
    match_support = math.sqrt(max(coverage_score * sample_support, 0.0))
    confidence = feature_confidence * match_support ** max(
        match_support_power,
        0.0,
    )
    confidence_quality = float(np.clip(
        (confidence - confidence_bad) / max(confidence_good - confidence_bad, 1e-8),
        0.0,
        1.0,
    ))
    distance_score = float(np.clip(
        (distance_bad_px - median_displacement)
        / max(distance_bad_px - distance_good_px, 1e-8),
        0.0,
        1.0,
    ))
    alignment = 100.0 * (
        distance_score
        * confidence_quality ** max(alignment_confidence_power, 0.0)
    )
    values = {
        "top_cosine": top_cosine,
        "mean_cosine": mean_cosine,
        "top1_top2_margin": margin,
        "mutual_ratio": mutual_ratio,
        "median_dx_px": median_dx,
        "median_dy_px": median_dy,
        "mean_displacement_px": float(np.mean(displacement)),
        "median_displacement_px": median_displacement,
        "p95_displacement_px": float(np.percentile(displacement, 95)),
        "normalized_displacement": normalized_displacement,
        "inlier_ratio": inlier_ratio,
        "estimated_center_x": center_x + median_dx,
        "estimated_center_y": center_y + median_dy,
        "gt_center_x": center_x,
        "gt_center_y": center_y,
        "mask_area_px": area,
        "match_coverage_score": coverage_score,
        "sample_support_score": sample_support,
        "match_support_score": match_support,
        "feature_confidence_score": feature_confidence,
        "confidence_score": confidence,
        "confidence_quality_score": confidence_quality,
        "distance_score": distance_score,
        "alignment_score": alignment,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        result["status"] = "nonfinite_result"
        return result
    result.update(
        valid=True,
        status="ok",
        num_matches=int(len(selected)),
        num_mutual_matches=num_mutual_matches,
        used_mutual=used_mutual,
        match_gt_xy_json=json.dumps(
            np.round(np.column_stack((gt_px_x, gt_px_y)), 2).tolist(),
            separators=(",", ":"),
        ),
        match_generated_xy_json=json.dumps(
            np.round(np.column_stack((gen_px_x, gen_px_y)), 2).tolist(),
            separators=(",", ":"),
        ),
        match_cosine_json=json.dumps(
            np.round(match_cosines, 4).tolist(), separators=(",", ":")
        ),
        gt_grid_height=int(gt.shape[1]),
        gt_grid_width=int(gt.shape[2]),
        generated_grid_height=int(gen.shape[1]),
        generated_grid_width=int(gen.shape[2]),
        **values,
    )
    return result


def synthetic_test() -> None:
    rng = np.random.default_rng(7)
    channels, height, width = 96, 12, 16
    shift_y, shift_x = 2, -3
    gt = normalized_rows(rng.normal(size=(height * width, channels))).T.reshape(
        channels, height, width
    )
    generated = normalized_rows(rng.normal(size=(height * width, channels))).T.reshape(
        channels, height, width
    )
    for y in range(height):
        for x in range(width):
            yy, xx = y + shift_y, x + shift_x
            if 0 <= yy < height and 0 <= xx < width:
                generated[:, yy, xx] = gt[:, y, x]
    mask = np.zeros((height * PATCH_SIZE, width * PATCH_SIZE), np.uint8)
    mask[2 * PATCH_SIZE : 8 * PATCH_SIZE, 4 * PATCH_SIZE : 13 * PATCH_SIZE] = 1
    result = match_feature_maps(
        gt, generated, mask, min_mutual_matches=3, inlier_radius_px=PATCH_SIZE * 0.75
    )
    expected_dx, expected_dy = shift_x * PATCH_SIZE, shift_y * PATCH_SIZE
    assert result["valid"], result
    assert result["num_mutual_matches"] >= 3, result
    assert result["match_support_score"] > 0.0, result
    assert 0.0 <= result["alignment_score"] <= 100.0, result
    assert abs(result["median_dx_px"] - expected_dx) <= PATCH_SIZE, result
    assert abs(result["median_dy_px"] - expected_dy) <= PATCH_SIZE, result
    assert len(json.loads(result["match_cosine_json"])) == result["num_matches"], result
    print(
        "[synthetic-test] PASS: recovered "
        f"dx={result['median_dx_px']:.1f}, dy={result['median_dy_px']:.1f} px"
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or len(baselines) != 4:
        raise ValueError(f"{path}: expected exactly four baselines")
    return config


def frame_number(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def discover_frames(episode_dir: Path, view: str) -> dict[int, Path]:
    for directory in (episode_dir / "video" / view, episode_dir / "video"):
        if not directory.is_dir():
            continue
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            continue
        parsed: dict[int, Path] = {}
        for ordinal, path in enumerate(files):
            index = frame_number(path)
            if index is None:
                index = ordinal
            if index in parsed:
                raise ValueError(f"Duplicate frame index {index}: {parsed[index]} and {path}")
            parsed[index] = path
        return parsed
    raise FileNotFoundError(f"No image frames under {episode_dir / 'video'}")


def load_masks(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing masks: {path}")
    with np.load(path) as archive:
        key = "masks" if "masks" in archive else list(archive.keys())[0]
        masks = np.asarray(archive[key])
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4:
        raise ValueError(f"{path}: expected M,T,H,W masks, got {masks.shape}")
    return masks


def read_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def model_size(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    grid_h = max(1, height // PATCH_SIZE)
    grid_w = max(1, min(width // PATCH_SIZE, round(grid_h * ratio)))
    grid_h = max(1, min(height // PATCH_SIZE, round(grid_w / ratio)))
    return grid_w * PATCH_SIZE, grid_h * PATCH_SIZE


class DinoExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.torch = torch
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        repo = args.dinov3_repo or "facebookresearch/dinov3"
        source = "local" if args.dinov3_repo else "github"
        print(f"[model] Loading {args.model_name} from {repo} on {self.device}...", flush=True)
        self.model = torch.hub.load(
            repo, args.model_name, source=source, weights=args.checkpoint
        ).eval().to(self.device)

    def _unwrap(self, value: Any) -> Any:
        while isinstance(value, (tuple, list)):
            if not value:
                raise RuntimeError("DINOv3 returned an empty intermediate-layer tuple")
            tensors = [item for item in value if self.torch.is_tensor(item)]
            value = tensors[0] if tensors else value[0]
        return value

    def extract_batch(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        torch = self.torch
        tensors = []
        for image in images:
            image = image.resize(model_size(*image.size), Image.Resampling.BICUBIC)
            array = np.asarray(image, np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            tensors.append((tensor - mean) / std)
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("All images in a feature batch must have the same dimensions")
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            output = self.model.get_intermediate_layers(
                batch, n=1, reshape=True, norm=True
            )
            features = self._unwrap(output)
        if features.ndim != 4:
            raise RuntimeError(f"Expected B,C,H,W feature map, got {tuple(features.shape)}")
        features = torch.nn.functional.normalize(features.float(), dim=1)
        return [item.cpu().numpy() for item in features]


def quality_lookup(path: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    lookup: dict[tuple[str, int, int], dict[str, float]] = {}
    for source in SOURCES:
        source_data = (data.get("mask_sources") or {}).get(source) or {}
        records = source_data.get("per_mask") or []
        if not records:
            records = [
                {**mask_record, "frame": frame_record.get("frame")}
                for frame_record in source_data.get("per_frame") or []
                for mask_record in frame_record.get("masks") or []
            ]
        for record in records:
            try:
                key = (source, int(record["frame"]), int(record["mask_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            lookup[key] = {
                "psnr": finite(record.get("psnr")),
                "lpips": finite(record.get("lpips")),
            }
    return lookup


def aggregate(per_mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = per_mask[per_mask["valid"]].copy()
    metrics = [
        "top_cosine", "mean_cosine", "top1_top2_margin", "mutual_ratio",
        "num_mask_patches", "num_matches", "num_mutual_matches",
        "match_coverage_score", "sample_support_score", "match_support_score",
        "feature_confidence_score",
        "median_dx_px", "median_dy_px", "mean_displacement_px",
        "median_displacement_px", "p95_displacement_px", "normalized_displacement",
        "inlier_ratio", "confidence_score", "confidence_quality_score",
        "distance_score", "alignment_score",
        "psnr", "lpips",
    ]
    rows = []
    for (baseline, source, frame), group in valid.groupby(
        ["baseline", "source", "frame"], sort=True
    ):
        row: dict[str, Any] = {
            "baseline": baseline, "source": source, "frame": int(frame),
            "time_sec": float(group["time_sec"].iloc[0]), "num_masks": len(group),
        }
        for metric in metrics:
            row[metric] = finite(group[metric].mean()) if metric in group else float("nan")
        rows.append(row)
    per_frame = pd.DataFrame(rows)
    if per_frame.empty:
        return per_frame, pd.DataFrame()

    consistency: dict[tuple[str, int], float] = {}
    for (baseline, frame), group in per_frame.groupby(["baseline", "frame"]):
        directions = group.set_index("source")[["median_dx_px", "median_dy_px"]]
        if set(SOURCES) <= set(directions.index):
            delta = directions.loc["object"].to_numpy() - directions.loc["robot_arm"].to_numpy()
            consistency[(baseline, int(frame))] = float(np.linalg.norm(delta))
    per_frame["relative_object_robot_displacement_px"] = [
        consistency.get((row.baseline, int(row.frame)), float("nan"))
        for row in per_frame.itertuples()
    ]

    summaries = []
    for baseline, group in per_frame.groupby("baseline", sort=False):
        row = {"baseline": baseline, "num_source_frames": len(group)}
        for metric in metrics:
            row[metric] = finite(group[metric].mean())
        for source in SOURCES:
            subset = group[group["source"] == source]
            for metric in ("alignment_score", "confidence_score",
                           "confidence_quality_score", "median_displacement_px",
                           "normalized_displacement", "mean_cosine", "mutual_ratio",
                           "num_mutual_matches", "match_support_score",
                           "feature_confidence_score", "psnr", "lpips"):
                row[f"{source}_{metric}"] = finite(subset[metric].mean())
        row["relative_object_robot_displacement_px"] = finite(
            group["relative_object_robot_displacement_px"].mean()
        )
        summaries.append(row)
    return per_frame, pd.DataFrame(summaries)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
    head_size: float,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    if math.hypot(end[0] - start[0], end[1] - start[1]) < head_size:
        return
    left = (
        end[0] - head_size * math.cos(angle - math.pi / 6),
        end[1] - head_size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_size * math.cos(angle + math.pi / 6),
        end[1] - head_size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=color)


def decode_correspondences(row: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode compact CSV correspondence fields, tolerating malformed rows."""
    try:
        gt_xy = np.asarray(json.loads(row.match_gt_xy_json), dtype=np.float32).reshape(-1, 2)
        generated_xy = np.asarray(
            json.loads(row.match_generated_xy_json), dtype=np.float32
        ).reshape(-1, 2)
        cosine = np.asarray(json.loads(row.match_cosine_json), dtype=np.float32).reshape(-1)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    count = min(len(gt_xy), len(generated_xy), len(cosine))
    if count == 0:
        return gt_xy[:0], generated_xy[:0], cosine[:0]
    valid = (
        np.isfinite(gt_xy[:count]).all(axis=1)
        & np.isfinite(generated_xy[:count]).all(axis=1)
        & np.isfinite(cosine[:count])
    )
    return gt_xy[:count][valid], generated_xy[:count][valid], cosine[:count][valid]


def add_patch_matches(
    image: Image.Image,
    rows: pd.DataFrame,
    size: tuple[int, int],
    max_arrows_per_mask: int = 64,
) -> Image.Image:
    """Add generated-patch cosine heatmap and a readable match-arrow sample."""
    width, height = size
    grid_h = max(
        (int(row.generated_grid_height) for row in rows.itertuples() if row.valid),
        default=0,
    )
    grid_w = max(
        (int(row.generated_grid_width) for row in rows.itertuples() if row.valid),
        default=0,
    )
    if grid_h <= 0 or grid_w <= 0:
        return image

    heat = np.zeros((grid_h, grid_w), dtype=np.float32)
    occupied = np.zeros((grid_h, grid_w), dtype=np.uint8)
    decoded: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray]] = []
    for row in rows.itertuples():
        if not bool(row.valid):
            continue
        gt_xy, generated_xy, cosine = decode_correspondences(row)
        decoded.append((row, gt_xy, generated_xy, cosine))
        if not len(cosine):
            continue
        x_index = np.clip((generated_xy[:, 0] * grid_w / width).astype(int), 0, grid_w - 1)
        y_index = np.clip((generated_xy[:, 1] * grid_h / height).astype(int), 0, grid_h - 1)
        confidence = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)
        np.maximum.at(heat, (y_index, x_index), confidence)
        occupied[y_index, x_index] = 1

    rgb = np.asarray(image, dtype=np.uint8).copy()
    heat_u8 = np.round(heat * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    colored_rgb = colored_bgr[:, :, ::-1]
    colored_rgb = cv2.resize(colored_rgb, size, interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(occupied, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    alpha = (0.38 * alpha)[..., None]
    rgb = np.round(rgb * (1.0 - alpha) + colored_rgb * alpha).astype(np.uint8)
    result = Image.fromarray(rgb)
    draw = ImageDraw.Draw(result, "RGBA")
    for row, gt_xy, generated_xy, cosine in decoded:
        count = len(cosine)
        if count == 0:
            continue
        if count > max_arrows_per_mask:
            selected = np.linspace(0, count - 1, max_arrows_per_mask, dtype=int)
        else:
            selected = np.arange(count)
        for index in selected:
            confidence = float(np.clip((cosine[index] + 1.0) * 0.5, 0.0, 1.0))
            color = (
                int(255 * (1.0 - confidence)),
                int(255 * confidence),
                70,
                190,
            )
            start = tuple(float(value) for value in gt_xy[index])
            end = tuple(float(value) for value in generated_xy[index])
            draw_arrow(draw, start, end, color, width=1, head_size=4.0)
            radius = 2
            draw.ellipse(
                (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
                fill=color,
            )
    return result


def overlay_image(
    path: Path,
    size: tuple[int, int],
    rows: pd.DataFrame,
    *,
    gt: bool,
    title: str,
    masks_by_index: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    image = read_rgb(path, size)
    if gt and masks_by_index:
        combined = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        source = str(rows["source"].iloc[0]) if not rows.empty else "object"
        color = (255, 80, 60) if source == "object" else (40, 220, 255)
        for mask in masks_by_index.values():
            active = np.asarray(mask, bool)
            combined[active, :3] = color
            combined[active, 3] = 85
        image = Image.alpha_composite(
            image.convert("RGBA"), Image.fromarray(combined)
        ).convert("RGB")
    if not gt:
        image = add_patch_matches(image, rows, size)
    draw = ImageDraw.Draw(image, "RGBA")
    for row in rows.itertuples():
        color = (255, 80, 60, 230) if row.source == "object" else (40, 220, 255, 230)
        if bool(row.valid) and not gt:
            start = (row.gt_center_x, row.gt_center_y)
            end = (row.estimated_center_x, row.estimated_center_y)
            draw_arrow(draw, start, end, color, width=3, head_size=10.0)
            radius = 5
            draw.ellipse((end[0]-radius, end[1]-radius, end[0]+radius, end[1]+radius),
                         outline=color, width=3)
    canvas = np.asarray(image)[:, :, ::-1].copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    valid = rows[rows["valid"]]
    detail = "no valid masks"
    if not valid.empty and not gt:
        detail = (
            f"A {valid.alignment_score.mean():.1f}/100  "
            f"C {valid.confidence_score.mean():.2f}  "
            f"Sup {valid.match_support_score.mean():.2f}  "
            f"M {valid.num_mutual_matches.mean():.1f}  "
            f"D {valid.median_displacement_px.mean():.1f}px"
        )
    cv2.putText(canvas, detail, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def make_panel(
    frame: int,
    source: str,
    per_mask: pd.DataFrame,
    gt_frames: Mapping[int, Path],
    baseline_frames: Mapping[str, Mapping[int, Path]],
    masks: np.ndarray,
    output: Path,
) -> np.ndarray:
    height, width = masks.shape[-2:]
    size = (width, height)
    pieces = []
    frame_rows = per_mask[(per_mask.frame == frame) & (per_mask.source == source)].copy()
    gt_rows = frame_rows.drop_duplicates("mask_index").copy()
    gt_masks = {
        int(index): masks[int(index), frame] > 0 for index in gt_rows.mask_index
    }
    pieces.append(
        overlay_image(
            gt_frames[frame],
            size,
            gt_rows,
            gt=True,
            title=f"GT | {source}",
            masks_by_index=gt_masks,
        )
    )
    for baseline, files in baseline_frames.items():
        rows = frame_rows[frame_rows.baseline == baseline]
        pieces.append(overlay_image(files[frame], size, rows, gt=False, title=baseline))
    target_h = 360
    pieces = [cv2.resize(piece, (round(piece.shape[1] * target_h / piece.shape[0]), target_h))
              for piece in pieces]
    panel = np.concatenate(pieces, axis=1)
    cv2.putText(panel, f"frame {frame}", (10, panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), panel)
    return panel


def write_video(frames: Iterable[np.ndarray], output: Path, fps: float) -> None:
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".cv2.mp4")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 0.1), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temporary}")
    writer.write(first)
    for frame in iterator:
        writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        converted = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
            check=False,
        )
        if converted.returncode == 0:
            temporary.unlink(missing_ok=True)
            return
    os.replace(temporary, output)


def save_plots(
    per_mask: pd.DataFrame, per_frame: pd.DataFrame, summary: pd.DataFrame, plots: Path
) -> list[str]:
    plots.mkdir(parents=True, exist_ok=True)
    made = []
    if per_frame.empty:
        return made
    for metric, filename, ylabel in (
        ("alignment_score", "temporal_alignment.png", "Alignment score (0-100)"),
        ("median_displacement_px", "temporal_displacement.png", "Median displacement (px)"),
        ("confidence_score", "temporal_confidence.png", "Confidence"),
        ("match_support_score", "temporal_match_support.png", "Match support"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        for (baseline, source), group in per_frame.groupby(["baseline", "source"]):
            ax.plot(group.time_sec, group[metric], label=f"{baseline} / {source}", alpha=.85)
        ax.set(xlabel="Time (s)", ylabel=ylabel)
        if metric == "alignment_score":
            ax.set_ylim(0, 100)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=150)
        plt.close(fig)
        made.append(filename)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(summary))
    for ax, metric, label in zip(
        axes,
        (
            "alignment_score",
            "confidence_score",
            "match_support_score",
            "median_displacement_px",
        ),
        ("Alignment", "Confidence", "Match support", "Displacement (px)"),
    ):
        ax.bar(x, summary[metric])
        ax.set_xticks(x, summary.baseline, rotation=25, ha="right")
        ax.set_title(label)
        if metric == "alignment_score":
            ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(plots / "baseline_summary.png", dpi=150)
    plt.close(fig)
    made.append("baseline_summary.png")
    scatter = per_mask[per_mask.valid & (per_mask.psnr.notna() | per_mask.lpips.notna())]
    if not scatter.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for baseline, group in scatter.groupby("baseline"):
            axes[0].scatter(group.psnr, group.alignment_score, s=16, alpha=.55, label=baseline)
            axes[1].scatter(group.lpips, group.alignment_score, s=16, alpha=.55, label=baseline)
        axes[0].set(xlabel="KeyArea PSNR", ylabel="Spatial alignment (0-100)")
        axes[1].set(xlabel="KeyArea LPIPS", ylabel="Spatial alignment (0-100)")
        axes[0].set_ylim(0, 100)
        axes[1].set_ylim(0, 100)
        axes[0].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(plots / "keyarea_vs_spatial.png", dpi=150)
        plt.close(fig)
        made.append("keyarea_vs_spatial.png")
    return made


def write_html(
    output: Path, plots: Sequence[str], panel_paths: Sequence[Path], failures: pd.DataFrame
) -> None:
    plot_html = "".join(
        f'<a href="plots/{html.escape(name)}"><img src="plots/{html.escape(name)}"></a>'
        for name in plots
    )
    panel_html = "".join(
        f'<a href="{path.relative_to(output)}">{html.escape(path.stem)}</a> '
        for path in panel_paths
    )
    failure_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (row.baseline, row.source, row.frame, row.mask_index,
                          f"{row.alignment_score:.1f}", f"{row.median_displacement_px:.1f}")
        ) + "</tr>"
        for row in failures.itertuples()
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>DINOv3 spatial alignment</title><style>
body{{font:14px system-ui;margin:24px;max-width:1400px}} img{{max-width:100%;margin:8px 0}}
video{{width:100%;max-width:1200px}} table{{border-collapse:collapse}}
td,th{{padding:6px 10px;border:1px solid #ccc}} a{{margin-right:10px}}
</style></head><body><h1>DINOv3 spatial alignment</h1>
<p><a href="per_mask.csv">per_mask.csv</a><a href="per_frame.csv">per_frame.csv</a>
<a href="baseline_summary.csv">baseline_summary.csv</a><a href="metadata.json">metadata.json</a></p>
<h2>Videos</h2><h3>Object</h3><video controls src="videos/object.mp4"></video>
<h3>Robot arm</h3><video controls src="videos/robot_arm.mp4"></video>
<h2>Plots</h2>{plot_html}<h2>Panels</h2><p>{panel_html}</p>
<h2>Top failure cases</h2><table><tr><th>Baseline</th><th>Source</th><th>Frame</th>
<th>Mask</th><th>Alignment (0-100)</th><th>Displacement (px)</th></tr>{failure_rows}</table>
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.synthetic_test:
        synthetic_test()
        return
    if args.stride <= 0 or args.fps <= 0 or args.batch_size <= 0:
        raise ValueError("--stride, --fps, and --batch-size must be positive")
    if args.match_support_target_ratio <= 0 or args.match_support_count_scale <= 0:
        raise ValueError(
            "--match-support-target-ratio and --match-support-count-scale "
            "must be positive"
        )
    if args.confidence_good <= args.confidence_bad:
        raise ValueError("--confidence-good must be greater than --confidence-bad")
    if args.distance_bad_px <= args.distance_good_px:
        raise ValueError("--distance-bad-px must be greater than --distance-good-px")
    config_path = Path(args.config)
    config = load_config(config_path)
    output = (
        Path(args.output_dir) if args.output_dir
        else Path("output/dinov3_spatial_alignment") / f"{args.task}__{args.episode}"
    )
    for directory in (output, output / "panels", output / "plots", output / "videos"):
        directory.mkdir(parents=True, exist_ok=True)

    gt_episode = Path(args.gt_root) / args.task / args.episode
    gt_frames = discover_frames(gt_episode, args.view)
    masks = {
        "object": load_masks(gt_episode / "prompt" / "masks.npz"),
        "robot_arm": load_masks(
            gt_episode / "multiview_masks" / args.view / "masks.npz"
        ),
    }
    mask_sizes = {tuple(array.shape[-2:]) for array in masks.values()}
    if len(mask_sizes) != 1:
        raise ValueError(f"Object and robot-arm mask resolutions differ: {mask_sizes}")
    baseline_frames, qualities, episode_dirs = {}, {}, {}
    for baseline, root in config["baselines"].items():
        episode_dir = Path(root) / args.task / args.episode
        episode_dirs[baseline] = episode_dir
        baseline_frames[baseline] = discover_frames(episode_dir, args.view)
        qualities[baseline] = quality_lookup(episode_dir / "keyarea_visual_quality.json")
    common = set(gt_frames)
    for files in baseline_frames.values():
        common &= set(files)
    mask_limit = min(array.shape[1] for array in masks.values())
    selected_frames = sorted(frame for frame in common if frame < mask_limit and frame % args.stride == 0)
    if not selected_frames:
        raise ValueError("No stride-selected common frames across GT, baselines, and masks")
    print(f"[data] {len(selected_frames)} common frames; output={output}", flush=True)

    extractor = DinoExtractor(args)
    kwargs = {
        name: getattr(args, name)
        for name in (
            "min_mask_patches", "min_mutual_matches", "inlier_radius_px",
            "confidence_bias", "confidence_cosine_weight", "confidence_margin_weight",
            "confidence_mutual_weight", "confidence_inlier_weight",
            "confidence_bad", "confidence_good", "distance_good_px", "distance_bad_px",
            "match_support_target_ratio", "match_support_count_scale",
            "match_support_power", "alignment_confidence_power",
        )
    }
    rows: list[dict[str, Any]] = []
    height, width = next(iter(masks.values())).shape[-2:]
    size = (width, height)
    for position, frame in enumerate(selected_frames, 1):
        print(f"[features] frame {frame} ({position}/{len(selected_frames)})", flush=True)
        gt_image = read_rgb(gt_frames[frame], size)
        gt_features = extractor.extract_batch([gt_image])[0]
        baseline_names = list(baseline_frames)
        generated_features: dict[str, np.ndarray] = {}
        for start in range(0, len(baseline_names), args.batch_size):
            names = baseline_names[start : start + args.batch_size]
            images = [read_rgb(baseline_frames[name][frame], size) for name in names]
            for name, features in zip(names, extractor.extract_batch(images)):
                generated_features[name] = features
        for baseline in baseline_names:
            for source, source_masks in masks.items():
                for mask_index in range(source_masks.shape[0]):
                    result = match_feature_maps(
                        gt_features, generated_features[baseline],
                        source_masks[mask_index, frame], **kwargs
                    )
                    quality = qualities[baseline].get(
                        (source, frame, mask_index), {"psnr": float("nan"), "lpips": float("nan")}
                    )
                    rows.append({
                        "baseline": baseline, "source": source, "frame": frame,
                        "time_sec": frame / args.fps, "mask_index": mask_index,
                        "gt_frame_path": str(gt_frames[frame]),
                        "generated_frame_path": str(baseline_frames[baseline][frame]),
                        **result, **quality,
                    })
        del gt_features, generated_features

    per_mask = pd.DataFrame(rows)
    per_frame, baseline_summary = aggregate(per_mask)
    per_mask.to_csv(output / "per_mask.csv", index=False)
    per_frame.to_csv(output / "per_frame.csv", index=False)
    baseline_summary.to_csv(output / "baseline_summary.csv", index=False)
    print("[visualize] Writing panels, videos, plots, and HTML...", flush=True)
    panel_paths: list[Path] = []

    def panel_stream(source: str) -> Iterable[np.ndarray]:
        for ordinal, frame in enumerate(selected_frames):
            panel_path = output / "panels" / f"{source}_frame_{frame:06d}.png"
            panel = make_panel(
                frame, source, per_mask, gt_frames, baseline_frames, masks[source], panel_path
            )
            if ordinal % max(args.visualize_every, 1) == 0:
                panel_paths.append(panel_path)
            else:
                panel_path.unlink(missing_ok=True)
            yield panel

    playback_fps = args.fps / args.stride
    for source in SOURCES:
        write_video(
            panel_stream(source), output / "videos" / f"{source}.mp4", playback_fps
        )
    plots = save_plots(per_mask, per_frame, baseline_summary, output / "plots")
    failures = per_mask[per_mask.valid].nsmallest(20, "alignment_score")
    metadata = {
        "task": args.task, "episode": args.episode, "view": args.view,
        "stride": args.stride, "fps": args.fps, "selected_frames": selected_frames,
        "baselines": {name: str(path) for name, path in episode_dirs.items()},
        "gt_episode": str(gt_episode), "checkpoint": args.checkpoint,
        "model_name": args.model_name, "dinov3_repo": args.dinov3_repo,
        "device": str(extractor.device), "score_calibration": kwargs,
        "alignment_score_range": [0.0, 100.0],
        "alignment_score_formula": (
            "100 * distance_score * "
            "confidence_quality_score ** alignment_confidence_power"
        ),
        "feature_patch_size": PATCH_SIZE,
        "correspondence_storage": {
            "coordinates": "mask-resolution pixel centers",
            "fields": [
                "match_gt_xy_json",
                "match_generated_xy_json",
                "match_cosine_json",
            ],
            "selection": "all mutual matches, or all GT top-1 matches on fallback",
        },
        "quality_json_available": {
            name: bool(lookup) for name, lookup in qualities.items()
        },
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, indent=2, allow_nan=False)
    write_html(output, plots, panel_paths, failures)
    print(f"[done] Results written to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
