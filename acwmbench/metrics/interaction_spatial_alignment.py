"""DINOv3 interaction spatial-alignment metric.

GT object and robot-arm masks select template patch tokens in a DINOv3 feature
map. Mutual nearest-neighbour matches locate the corresponding tokens in each
generated frame. The metric combines normalized displacement, feature
confidence, and match support into an alignment score.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..dataset import EpisodeTrial

logger = logging.getLogger(__name__)

PATCH_SIZE = 16
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SOURCES = ("object", "robot_arm")
SUMMARY_METRICS = (
    "alignment_score",
    "confidence_score",
    "confidence_quality_score",
    "distance_score",
    "normalized_displacement",
    "median_displacement_px",
    "mean_displacement_px",
    "p95_displacement_px",
    "mean_cosine",
    "top1_top2_margin",
    "mutual_ratio",
    "inlier_ratio",
    "num_mutual_matches",
    "match_support_score",
    "feature_confidence_score",
)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-value))


def _normalized_rows(array: np.ndarray) -> np.ndarray:
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
) -> Dict[str, Any]:
    """Match masked GT patch tokens to generated tokens using cosine MNNs."""
    result: Dict[str, Any] = {
        "valid": False,
        "status": "invalid",
        "num_mask_patches": 0,
        "num_matches": 0,
        "num_mutual_matches": 0,
        "used_mutual": False,
        "match_gt_xy": [],
        "match_generated_xy": [],
        "match_cosine": [],
        "gt_grid_height": 0,
        "gt_grid_width": 0,
        "generated_grid_height": 0,
        "generated_grid_width": 0,
    }
    gt = np.asarray(gt_features, np.float32)
    generated = np.asarray(generated_features, np.float32)
    mask_bool = np.asarray(mask) > 0
    if gt.ndim != 3 or generated.ndim != 3 or mask_bool.ndim != 2:
        result["status"] = "bad_shape"
        return result
    if (
        gt.shape[0] != generated.shape[0]
        or min(gt.shape[1:]) < 1
        or min(generated.shape[1:]) < 1
    ):
        result["status"] = "incompatible_features"
        return result
    height, width = mask_bool.shape
    if not mask_bool.any():
        result["status"] = "empty_mask"
        return result

    mask_small = cv2.resize(
        mask_bool.astype(np.uint8),
        (gt.shape[2], gt.shape[1]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    gt_y, gt_x = np.nonzero(mask_small)
    result["num_mask_patches"] = int(len(gt_x))
    if len(gt_x) < min_mask_patches:
        result["status"] = "tiny_mask"
        return result

    gt_tokens = _normalized_rows(gt[:, gt_y, gt_x].T)
    generated_tokens = _normalized_rows(
        generated.reshape(generated.shape[0], -1).T
    )
    similarity = gt_tokens @ generated_tokens.T
    if not np.isfinite(similarity).all():
        similarity = np.nan_to_num(
            similarity,
            nan=-1.0,
            posinf=1.0,
            neginf=-1.0,
        )
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

    generated_y, generated_x = np.divmod(
        top1_indices[selected],
        generated.shape[2],
    )
    gt_px_x = (gt_x[selected] + 0.5) * width / gt.shape[2]
    gt_px_y = (gt_y[selected] + 0.5) * height / gt.shape[1]
    generated_px_x = (
        (generated_x + 0.5) * width / generated.shape[2]
    )
    generated_px_y = (
        (generated_y + 0.5) * height / generated.shape[1]
    )
    match_cosines = top1_values[selected]
    dx = generated_px_x - gt_px_x
    dy = generated_px_y - gt_px_y
    displacement = np.hypot(dx, dy)
    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    residual = np.hypot(dx - median_dx, dy - median_dy)
    inlier_ratio = float(
        np.mean(residual <= max(inlier_radius_px, 1e-6))
    )
    mask_y, mask_x = np.nonzero(mask_bool)
    center_x = float(mask_x.mean())
    center_y = float(mask_y.mean())
    area = float(mask_bool.sum())
    median_displacement = float(np.median(displacement))
    normalized_displacement = median_displacement / max(math.sqrt(area), 1.0)
    mean_cosine = float(np.mean(match_cosines))
    top_cosine = float(np.max(match_cosines))
    margin = float(np.mean(match_cosines - top2_values[selected]))
    feature_confidence = _sigmoid(
        confidence_bias
        + confidence_cosine_weight * mean_cosine
        + confidence_margin_weight * margin
        + confidence_mutual_weight * mutual_ratio
        + confidence_inlier_weight * inlier_ratio
    )
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
    confidence_quality = float(
        np.clip(
            (confidence - confidence_bad)
            / max(confidence_good - confidence_bad, 1e-8),
            0.0,
            1.0,
        )
    )
    distance_score = float(
        np.clip(
            (distance_bad_px - median_displacement)
            / max(distance_bad_px - distance_good_px, 1e-8),
            0.0,
            1.0,
        )
    )
    alignment_score = 100.0 * (
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
        "alignment_score": alignment_score,
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
        match_gt_xy=np.round(
            np.column_stack((gt_px_x, gt_px_y)),
            2,
        ).tolist(),
        match_generated_xy=np.round(
            np.column_stack((generated_px_x, generated_px_y)),
            2,
        ).tolist(),
        match_cosine=np.round(match_cosines, 4).tolist(),
        gt_grid_height=int(gt.shape[1]),
        gt_grid_width=int(gt.shape[2]),
        generated_grid_height=int(generated.shape[1]),
        generated_grid_width=int(generated.shape[2]),
        **values,
    )
    return result


def _frame_number(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def _discover_frames(root: str | os.PathLike, view: str) -> Dict[int, Path]:
    root = Path(root)
    candidates = (
        root / view,
        root,
        root / "video" / view,
        root / "video",
    )
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            continue
        parsed: Dict[int, Path] = {}
        for ordinal, path in enumerate(files):
            index = _frame_number(path)
            index = ordinal if index is None else index
            if index in parsed:
                raise ValueError(
                    f"Duplicate frame index {index}: {parsed[index]} and {path}"
                )
            parsed[index] = path
        return parsed
    raise FileNotFoundError(f"No image frames found under {root}")


def _load_masks(path: Path) -> np.ndarray:
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


def _read_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def _model_size(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    grid_h = max(1, height // PATCH_SIZE)
    grid_w = max(1, min(width // PATCH_SIZE, round(grid_h * ratio)))
    grid_h = max(1, min(height // PATCH_SIZE, round(grid_w / ratio)))
    return grid_w * PATCH_SIZE, grid_h * PATCH_SIZE


class DinoExtractor:
    """Load one DINOv3 encoder and return normalized dense feature maps."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint: str,
        dinov3_repo: str | None,
        device: str,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        repo = dinov3_repo or "facebookresearch/dinov3"
        source = "local" if dinov3_repo else "github"
        logger.info(
            "Loading %s from %s on %s",
            model_name,
            repo,
            self.device,
        )
        self.model = torch.hub.load(
            repo,
            model_name,
            source=source,
            weights=checkpoint,
        ).eval().to(self.device)
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406), dtype=torch.float32
        )[:, None, None]
        self.std = torch.tensor(
            (0.229, 0.224, 0.225), dtype=torch.float32
        )[:, None, None]

    def _unwrap(self, value: Any) -> Any:
        while isinstance(value, (tuple, list)):
            if not value:
                raise RuntimeError(
                    "DINOv3 returned an empty intermediate-layer tuple"
                )
            tensors = [
                item for item in value if self.torch.is_tensor(item)
            ]
            value = tensors[0] if tensors else value[0]
        return value

    def prepare_batch(
        self,
        images: Sequence[Image.Image],
    ):
        """CPU image preprocessing, safe to call from loader threads."""

        torch = self.torch
        tensors = []
        for image in images:
            target_size = _model_size(*image.size)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BICUBIC)
            array = np.asarray(image, np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - self.mean) / self.std)
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError(
                "All images in a feature batch must have the same dimensions"
            )
        if not tensors:
            raise ValueError("Cannot extract an empty feature batch")
        return torch.stack(tensors)

    def extract_prepared(
        self,
        batch,
        *,
        pin_memory: bool = False,
    ) -> List[np.ndarray]:
        """Run the sole GPU-owner forward on a preprocessed CPU batch."""

        torch = self.torch
        use_pin = pin_memory and self.device.type == "cuda"
        if use_pin and not batch.is_pinned():
            batch = batch.pin_memory()
        batch = batch.to(self.device, non_blocking=use_pin)
        with torch.inference_mode():
            output = self.model.get_intermediate_layers(
                batch,
                n=1,
                reshape=True,
                norm=True,
            )
            features = self._unwrap(output)
        if features.ndim != 4:
            raise RuntimeError(
                f"Expected B,C,H,W feature map, got {tuple(features.shape)}"
            )
        features = torch.nn.functional.normalize(
            features.float(),
            dim=1,
        )
        array = features.detach().cpu().numpy()
        return [array[index] for index in range(len(array))]

    def extract_batch(
        self,
        images: Sequence[Image.Image],
    ) -> List[np.ndarray]:
        """Compatibility wrapper for callers outside the pipeline."""

        return self.extract_prepared(self.prepare_batch(images))


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
        and math.isfinite(float(record[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _source_frames(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        if record.get("valid"):
            grouped.setdefault(int(record["frame"]), []).append(record)
    output = []
    for frame, masks in sorted(grouped.items()):
        output.append(
            {
                "frame": frame,
                "time_sec": masks[0].get("time_sec"),
                "num_masks": len(masks),
                **{metric: _mean(masks, metric) for metric in SUMMARY_METRICS},
                "masks": masks,
            }
        )
    return output


def _combine_sources(
    source_frames: Mapping[str, Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    by_source = {
        source: {int(record["frame"]): record for record in records}
        for source, records in source_frames.items()
    }
    common = sorted(
        set(by_source["object"]) & set(by_source["robot_arm"])
    )
    output = []
    for frame in common:
        records = [by_source[source][frame] for source in SOURCES]
        object_dx = records[0].get("median_dx_px")
        object_dy = records[0].get("median_dy_px")
        arm_dx = records[1].get("median_dx_px")
        arm_dy = records[1].get("median_dy_px")
        relative = float("nan")
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (object_dx, object_dy, arm_dx, arm_dy)
        ):
            relative = float(
                np.hypot(object_dx - arm_dx, object_dy - arm_dy)
            )
        output.append(
            {
                "frame": frame,
                "time_sec": records[0].get("time_sec"),
                **{
                    metric: _mean(records, metric)
                    for metric in SUMMARY_METRICS
                },
                "relative_object_robot_displacement_px": relative,
                "object": records[0],
                "robot_arm": records[1],
            }
        )
    return output


@dataclass
class _PreparedFrameBatch:
    frames: List[int]
    gt_batch: Any
    generated_batch: Any


def _prepare_frame_batch(
    extractor: DinoExtractor,
    frame_batch: Sequence[int],
    gt_frames: Mapping[int, Path],
    generated_frames: Mapping[int, Path],
    size: tuple[int, int],
) -> _PreparedFrameBatch:
    """Read and normalize a paired batch without touching CUDA."""

    frames = list(frame_batch)
    gt_images = [_read_rgb(gt_frames[frame], size) for frame in frames]
    generated_images = [
        _read_rgb(generated_frames[frame], size) for frame in frames
    ]
    return _PreparedFrameBatch(
        frames=frames,
        gt_batch=extractor.prepare_batch(gt_images),
        generated_batch=extractor.prepare_batch(generated_images),
    )


def _prefetched_frame_batches(
    selected_frames: Sequence[int],
    *,
    frame_batch_size: int,
    loader_workers: int,
    prefetch_batches: int,
    prepare: Callable[[Sequence[int]], _PreparedFrameBatch],
) -> Iterator[_PreparedFrameBatch]:
    """Yield prepared batches in frame order with strictly bounded look-ahead."""

    batches = [
        selected_frames[start : start + frame_batch_size]
        for start in range(0, len(selected_frames), frame_batch_size)
    ]
    if loader_workers <= 1:
        for batch in batches:
            yield prepare(batch)
        return

    pending: deque[Future[_PreparedFrameBatch]] = deque()
    source = iter(batches)
    capacity = max(1, prefetch_batches)
    with ThreadPoolExecutor(
        max_workers=loader_workers, thread_name_prefix="isa-loader"
    ) as pool:
        for _ in range(min(capacity, len(batches))):
            pending.append(pool.submit(prepare, next(source)))
        exhausted = len(pending) >= len(batches)
        while pending:
            future = pending.popleft()
            prepared = future.result()
            if not exhausted:
                try:
                    pending.append(pool.submit(prepare, next(source)))
                except StopIteration:
                    exhausted = True
            yield prepared


def _extract_paired_features(
    extractor: DinoExtractor,
    prepared: _PreparedFrameBatch,
    *,
    fuse_gt_generated: bool,
    pin_memory: bool,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Use one GPU owner; optionally fuse the two image domains into one call."""

    if fuse_gt_generated:
        torch = extractor.torch
        pair_count = len(prepared.frames)
        features = extractor.extract_prepared(
            torch.cat((prepared.gt_batch, prepared.generated_batch), dim=0),
            pin_memory=pin_memory,
        )
        return features[:pair_count], features[pair_count:]
    return (
        extractor.extract_prepared(
            prepared.gt_batch, pin_memory=pin_memory
        ),
        extractor.extract_prepared(
            prepared.generated_batch, pin_memory=pin_memory
        ),
    )


def _match_frame(
    frame: int,
    gt_feature: np.ndarray,
    generated_feature: np.ndarray,
    masks: Mapping[str, np.ndarray],
    fps: float,
    match_kwargs: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """CPU-only matching for one frame, preserving source/mask order."""

    records = {source: [] for source in SOURCES}
    for source in SOURCES:
        source_masks = masks[source]
        for mask_index in range(source_masks.shape[0]):
            match = match_feature_maps(
                gt_feature,
                generated_feature,
                source_masks[mask_index, frame],
                **match_kwargs,
            )
            records[source].append(
                {
                    "frame": int(frame),
                    "time_sec": float(frame / fps),
                    "mask_index": int(mask_index),
                    "mask_source": source,
                    **match,
                }
            )
    return records


def _append_frame_records(
    per_mask: Dict[str, List[Dict[str, Any]]],
    records: Mapping[str, Sequence[Dict[str, Any]]],
) -> None:
    for source in SOURCES:
        per_mask[source].extend(records[source])


def _compute_episode_matches(
    *,
    extractor: DinoExtractor,
    selected_frames: Sequence[int],
    gt_frames: Mapping[int, Path],
    generated_frames: Mapping[int, Path],
    masks: Mapping[str, np.ndarray],
    size: tuple[int, int],
    fps: float,
    match_kwargs: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Bounded loader -> GPU -> matcher pipeline for one episode."""

    frame_batch_size = int(execution["frame_batch_size"])
    loader_workers = int(execution["loader_workers"])
    prefetch_batches = int(execution["prefetch_batches"])
    matcher_workers = int(execution["matcher_workers"])
    pin_memory = bool(execution["pin_memory"])
    fuse = bool(execution["fuse_gt_generated"])
    per_mask: Dict[str, List[Dict[str, Any]]] = {
        source: [] for source in SOURCES
    }

    def prepare(frames: Sequence[int]) -> _PreparedFrameBatch:
        return _prepare_frame_batch(
            extractor,
            frames,
            gt_frames,
            generated_frames,
            size,
        )

    prepared_batches = _prefetched_frame_batches(
        selected_frames,
        frame_batch_size=frame_batch_size,
        loader_workers=loader_workers,
        prefetch_batches=prefetch_batches,
        prepare=prepare,
    )
    if matcher_workers <= 1:
        for prepared in prepared_batches:
            gt_features, generated_features = _extract_paired_features(
                extractor,
                prepared,
                fuse_gt_generated=fuse,
                pin_memory=pin_memory,
            )
            for frame, gt_feature, generated_feature in zip(
                prepared.frames, gt_features, generated_features
            ):
                _append_frame_records(
                    per_mask,
                    _match_frame(
                        frame,
                        gt_feature,
                        generated_feature,
                        masks,
                        fps,
                        match_kwargs,
                    ),
                )
        return per_mask

    # Keep only a small feature backlog while CPU matching overlaps the next
    # DINO forward. Futures are drained FIFO, so output order is deterministic.
    backlog_limit = max(
        frame_batch_size, frame_batch_size * prefetch_batches
    )
    pending: deque[Future[Dict[str, List[Dict[str, Any]]]]] = deque()
    with ThreadPoolExecutor(
        max_workers=matcher_workers, thread_name_prefix="isa-matcher"
    ) as matcher_pool:
        for prepared in prepared_batches:
            gt_features, generated_features = _extract_paired_features(
                extractor,
                prepared,
                fuse_gt_generated=fuse,
                pin_memory=pin_memory,
            )
            for frame, gt_feature, generated_feature in zip(
                prepared.frames, gt_features, generated_features
            ):
                pending.append(
                    matcher_pool.submit(
                        _match_frame,
                        frame,
                        gt_feature,
                        generated_feature,
                        masks,
                        fps,
                        match_kwargs,
                    )
                )
                if len(pending) >= backlog_limit:
                    _append_frame_records(per_mask, pending.popleft().result())
        while pending:
            _append_frame_records(per_mask, pending.popleft().result())
    return per_mask


def compute_interaction_spatial_alignment(
    episodes: List[EpisodeTrial],
    gt_root: str | os.PathLike,
    *,
    device: str,
    eval_view: str = "top_head",
    downsample_rate: int = 10,
    fps: float = 30.0,
    checkpoint: str = (
        "./ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ),
    dinov3_repo: str | None = None,
    model_name: str = "dinov3_vitb16",
    batch_size: int = 4,
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
    save_root: str | os.PathLike | None = None,
    progress: bool = True,
    execution: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate DINOv3 alignment and save one JSON per episode."""
    if downsample_rate <= 0 or fps <= 0 or batch_size <= 0:
        raise ValueError(
            "downsample_rate, fps, and batch_size must be positive"
        )
    if match_support_target_ratio <= 0 or match_support_count_scale <= 0:
        raise ValueError(
            "match-support target ratio and count scale must be positive"
        )
    if confidence_good <= confidence_bad:
        raise ValueError("confidence_good must be greater than confidence_bad")
    if distance_bad_px <= distance_good_px:
        raise ValueError("distance_bad_px must be greater than distance_good_px")
    execution = dict(execution or {})
    execution.setdefault("mode", "serial")
    execution.setdefault("frame_batch_size", int(batch_size))
    execution.setdefault("loader_workers", 1)
    execution.setdefault("prefetch_batches", 1)
    execution.setdefault("matcher_workers", 1)
    execution.setdefault("pin_memory", False)
    execution.setdefault("fuse_gt_generated", False)
    for key in (
        "frame_batch_size",
        "loader_workers",
        "prefetch_batches",
        "matcher_workers",
    ):
        execution[key] = int(execution[key])
        if execution[key] < 1:
            raise ValueError(f"execution.{key} must be >= 1")
    execution["mode"] = str(execution["mode"]).lower()
    if execution["mode"] not in {"serial", "pipeline"}:
        raise ValueError("execution.mode must be 'serial' or 'pipeline'")
    if execution["mode"] == "serial":
        execution.update(
            loader_workers=1,
            prefetch_batches=1,
            matcher_workers=1,
            pin_memory=False,
            fuse_gt_generated=False,
        )
    extractor = DinoExtractor(
        model_name=model_name,
        checkpoint=checkpoint,
        dinov3_repo=dinov3_repo,
        device=device,
    )
    match_kwargs = {
        "min_mask_patches": min_mask_patches,
        "min_mutual_matches": min_mutual_matches,
        "inlier_radius_px": inlier_radius_px,
        "confidence_bias": confidence_bias,
        "confidence_cosine_weight": confidence_cosine_weight,
        "confidence_margin_weight": confidence_margin_weight,
        "confidence_mutual_weight": confidence_mutual_weight,
        "confidence_inlier_weight": confidence_inlier_weight,
        "match_support_target_ratio": match_support_target_ratio,
        "match_support_count_scale": match_support_count_scale,
        "match_support_power": match_support_power,
        "confidence_bad": confidence_bad,
        "confidence_good": confidence_good,
        "distance_good_px": distance_good_px,
        "distance_bad_px": distance_bad_px,
        "alignment_confidence_power": alignment_confidence_power,
    }
    settings = {
        "eval_view": eval_view,
        "downsample_rate": int(downsample_rate),
        "fps": float(fps),
        "checkpoint": str(checkpoint),
        "dinov3_repo": str(dinov3_repo) if dinov3_repo else None,
        "model_name": model_name,
        # DINO kernels can change near-tied patch matches across batch shapes;
        # these two execution choices are therefore part of result identity.
        "feature_batch_size": int(execution["frame_batch_size"]),
        "fuse_gt_generated": bool(execution["fuse_gt_generated"]),
        **match_kwargs,
    }
    results = []
    iterator = (
        tqdm(episodes, desc="Interaction Spatial Alignment")
        if progress
        else episodes
    )
    gt_root = Path(gt_root)
    for episode in iterator:
        gt_episode = (
            Path(episode.gt_video_dir).parent
            if episode.gt_video_dir
            else gt_root / episode.task_id / episode.episode_id
        )
        try:
            gt_frames = _discover_frames(
                episode.gt_video_dir or gt_episode / "video",
                eval_view,
            )
            generated_frames = _discover_frames(
                episode.video_dir,
                eval_view,
            )
            masks = {
                "object": _load_masks(
                    gt_episode / "prompt" / "masks.npz"
                ),
                "robot_arm": _load_masks(
                    gt_episode
                    / "multiview_masks"
                    / eval_view
                    / "masks.npz"
                ),
            }
            mask_sizes = {
                tuple(array.shape[-2:]) for array in masks.values()
            }
            if len(mask_sizes) != 1:
                raise ValueError(
                    "Object and robot-arm mask resolutions differ: "
                    f"{mask_sizes}"
                )
            mask_limit = min(array.shape[1] for array in masks.values())
            selected_frames = sorted(
                frame
                for frame in set(gt_frames) & set(generated_frames)
                if frame < mask_limit and frame % downsample_rate == 0
            )
            if not selected_frames:
                raise ValueError(
                    "No stride-selected common frames across GT, "
                    "generated video, and masks"
                )
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("Skipping %s: %s", episode.uid, exc)
            continue

        height, width = next(iter(masks.values())).shape[-2:]
        size = (width, height)
        try:
            per_mask = _compute_episode_matches(
                extractor=extractor,
                selected_frames=selected_frames,
                gt_frames=gt_frames,
                generated_frames=generated_frames,
                masks=masks,
                size=size,
                fps=fps,
                match_kwargs=match_kwargs,
                execution=execution,
            )
        except Exception as exc:  # noqa: BLE001
            cuda_oom = getattr(
                getattr(extractor.torch, "cuda", None),
                "OutOfMemoryError",
                (),
            )
            if cuda_oom and isinstance(exc, cuda_oom):
                raise
            logger.exception("Skipping %s after pipeline failure: %s", episode.uid, exc)
            continue

        source_frames = {
            source: _source_frames(per_mask[source])
            for source in SOURCES
        }
        combined_frames = _combine_sources(source_frames)
        all_source_frames = [
            record
            for source in SOURCES
            for record in source_frames[source]
        ]
        source_blocks = {}
        for source in SOURCES:
            source_blocks[source] = {
                **{
                    metric: _mean(source_frames[source], metric)
                    for metric in SUMMARY_METRICS
                },
                "num_valid_frames": len(source_frames[source]),
                "per_frame": source_frames[source],
                "per_mask": per_mask[source],
            }
        result = {
            "uid": episode.uid,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "metrics": list(SUMMARY_METRICS),
            "metric_version": 2,
            "score_range": [0.0, 100.0],
            "score_formula": (
                "100 * distance_score * "
                "confidence_quality_score ** alignment_confidence_power"
            ),
            **{
                metric: _mean(all_source_frames, metric)
                for metric in SUMMARY_METRICS
            },
            "relative_object_robot_displacement_px": _mean(
                combined_frames,
                "relative_object_robot_displacement_px",
            ),
            "num_selected_frames": len(selected_frames),
            "num_valid_combined_frames": len(combined_frames),
            "selected_frames": selected_frames,
            "evaluation_settings": settings,
            "mask_sources": source_blocks,
            "per_frame": combined_frames,
        }
        if save_root is not None:
            from .. import episode_store

            path = episode_store.save_metric(
                save_root,
                episode.task_id,
                episode.episode_id,
                "interaction_spatial_alignment",
                result,
            )
            logger.info(
                "Saved %s interaction alignment to %s",
                episode.uid,
                path,
            )
        results.append(result)

    aggregated = {
        metric: _mean(results, metric) for metric in SUMMARY_METRICS
    }
    aggregated["relative_object_robot_displacement_px"] = _mean(
        results,
        "relative_object_robot_displacement_px",
    )
    aggregated["num_episodes"] = len(results)
    aggregated["evaluation_settings"] = settings
    return {"per_episode": results, "aggregated": aggregated}
