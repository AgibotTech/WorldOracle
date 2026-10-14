"""KeyArea Visual Quality Metric.

Computes masked PSNR, LPIPS, and V-JEPA similarity between generated and GT
frames, only on the
key-area regions defined by masks (mask value > 0).
Each generated frame is resized to the mask resolution, cropped to the mask
bounding box, and its background zeroed out (留白) before comparison with GT.

Metrics:
  - PSNR on masked pixels
  - LPIPS and V-JEPA cosine similarity on the cropped, black-background masked patch

Each episode has:
  - prompt/masks.npz : key "masks", shape (M, T, H, W) uint8, M = num mask
                       sequences (a legacy prompt/masks.npy raw array also works)
  - video/            : generated frames (T frames)
  - gt_video_dir      : GT frames (T frames)

Final output: PSNR, LPIPS, and V-JEPA similarity averaged over the dataset.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..dataset import EpisodeTrial
from ..io_utils import load_frames_as_array, save_json
from .visual_dependencies import ensure_jepa, ensure_lpips

logger = logging.getLogger(__name__)

RESULT_METRICS = ("psnr", "lpips", "vjepa_similarity")

def _expand_interval(start: int, end: int, limit: int, min_size: int) -> Tuple[int, int]:
    """Expand a half-open interval around its center, clamped to ``[0, limit)``."""
    target = min(limit, max(end - start, min_size))
    center = (start + end) / 2.0
    new_start = int(round(center - target / 2.0))
    new_start = max(0, min(new_start, limit - target))
    return new_start, new_start + target


def _mask_bbox(
    mask_bool: np.ndarray,
    pad: int,
    min_size: int = 64,
) -> Tuple[int, int, int, int] | None:
    """Return a padded bbox guaranteed large enough for LPIPS/AlexNet."""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask_bool.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad + 1)
    x1 = min(w, x1 + pad + 1)
    y0, y1 = _expand_interval(y0, y1, h, max(1, int(min_size)))
    x0, x1 = _expand_interval(x0, x1, w, max(1, int(min_size)))
    return y0, y1, x0, x1


def _masked_psnr(p: np.ndarray, g: np.ndarray, mask_bool: np.ndarray) -> float:
    """PSNR over masked pixels only (p, g in [0, 1] (H, W, 3); mask_bool (H, W))."""
    if mask_bool.sum() == 0:
        return float("nan")
    diff = (p[mask_bool] - g[mask_bool]).astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _dilate_mask(mask_bool: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a boolean mask outwards by approximately ``pixels``."""
    pixels = max(0, int(pixels))
    if pixels == 0:
        return mask_bool
    kernel_size = 2 * pixels + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1)
    return dilated > 0


def _psnr(p: np.ndarray, g: np.ndarray) -> float:
    """PSNR over all pixels in two [0, 1] RGB arrays."""
    mse = float(np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _draw_bbox(img_u8: np.ndarray, bbox, color=(255, 0, 0), width: int = 2) -> np.ndarray:
    """Return a copy of ``img_u8`` (H, W, 3) with the bbox rectangle drawn on it."""
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    im = _Image.fromarray(np.ascontiguousarray(img_u8).astype(np.uint8))
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        _ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return np.array(im)


def _make_montage(
    tiles: List[Tuple[str, np.ndarray, bool]],
    title: str | None = None,
    tile_h: int = 256,
    pad: int = 8,
    label_h: int = 18,
    title_h: int = 20,
    bg=(18, 18, 18),
):
    """Compose labelled tiles side-by-side into one RGB PIL image.

    Each tile is ``(label, array, is_mask)``; ``is_mask`` uses nearest-neighbour
    resize so mask edges stay crisp. Returns a ``PIL.Image``.
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    ims = []
    for label, arr, is_mask in tiles:
        a = arr
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        im = _Image.fromarray(a.astype(np.uint8))
        if im.height != tile_h:
            w = max(1, int(round(im.width * tile_h / im.height)))
            im = im.resize((w, tile_h), _Image.NEAREST if is_mask else _Image.BILINEAR)
        ims.append((label, im))

    t_h = title_h if title else 0
    total_w = pad + sum(im.width + pad for _, im in ims)
    total_h = pad + t_h + label_h + tile_h + pad
    canvas = _Image.new("RGB", (total_w, total_h), bg)
    draw = _ImageDraw.Draw(canvas)
    if title:
        draw.text((pad, max(0, (t_h - 10) // 2)), title, fill=(255, 255, 0))
    x = pad
    y_lab = pad + t_h
    for label, im in ims:
        draw.text((x, y_lab), label, fill=(210, 210, 210))
        canvas.paste(im, (x, y_lab + label_h))
        x += im.width + pad
    return canvas


def _save_visualization_contact_sheets(
    viz_dir: Path,
    uid: str,
    columns: int = 2,
    thumbnail_size: Tuple[int, int] = (900, 260),
) -> Dict[str, Any]:
    """Build compact episode-level indexes for the per-mask debug panels."""
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    manifest: Dict[str, Any] = {
        "directory": str(viz_dir),
        "sources": {},
    }
    for source in ("object", "robot_arm"):
        paths = sorted(
            viz_dir.glob(
                f"keyarea_visual_quality_{source}_frame*_mask*.png"
            )
        )
        source_manifest: Dict[str, Any] = {
            "num_panels": len(paths),
            "panels": [path.name for path in paths],
            "contact_sheet": None,
        }
        if not paths:
            manifest["sources"][source] = source_manifest
            continue

        thumbnails = []
        for path in paths:
            with _Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail(thumbnail_size, _Image.Resampling.LANCZOS)
                thumbnails.append((path.name, thumb.copy()))

        columns_used = max(1, min(int(columns), len(thumbnails)))
        rows = int(np.ceil(len(thumbnails) / columns_used))
        label_h = 18
        pad = 8
        cell_w = max(image.width for _, image in thumbnails)
        cell_h = max(image.height for _, image in thumbnails) + label_h
        title_h = 24
        canvas = _Image.new(
            "RGB",
            (
                pad + columns_used * (cell_w + pad),
                title_h + pad + rows * (cell_h + pad),
            ),
            (18, 18, 18),
        )
        draw = _ImageDraw.Draw(canvas)
        draw.text(
            (pad, 5),
            f"{uid} — {source} debug panels ({len(paths)})",
            fill=(255, 255, 0),
        )
        for index, (name, image) in enumerate(thumbnails):
            row, column = divmod(index, columns_used)
            x = pad + column * (cell_w + pad)
            y = title_h + pad + row * (cell_h + pad)
            draw.text((x, y), name, fill=(210, 210, 210))
            canvas.paste(image, (x, y + label_h))

        output_path = (
            viz_dir / f"keyarea_visual_quality_contact_sheet_{source}.png"
        )
        canvas.save(output_path)
        source_manifest["contact_sheet"] = output_path.name
        manifest["sources"][source] = source_manifest
    return manifest


def _mean_metric(records: List[Dict[str, Any]], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
    ]
    return float(np.nanmean(values)) if values else float("nan")


def _average_masks_per_frame(
    per_mask: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Average all valid masks from one mask source independently per frame."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in per_mask:
        grouped.setdefault(int(record["frame"]), []).append(record)

    per_frame: List[Dict[str, Any]] = []
    for frame in sorted(grouped):
        records = grouped[frame]
        result: Dict[str, Any] = {
            "frame": frame,
            "num_masks": len(records),
            **{
                metric: _mean_metric(records, metric)
                for metric in RESULT_METRICS
            },
            "masks": [
                {
                    "mask_index": int(record["mask_index"]),
                    **{
                        metric: record[metric]
                        for metric in RESULT_METRICS
                    },
                }
                for record in sorted(
                    records,
                    key=lambda item: int(item["mask_index"]),
                )
            ],
        }
        if "time_sec" in records[0]:
            result["time_sec"] = records[0]["time_sec"]
        per_frame.append(result)
    return per_frame


def _combine_sources_per_frame(
    object_per_frame: List[Dict[str, Any]],
    robot_arm_per_frame: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Average object-mask and robot-arm-mask frame means with equal weight."""
    object_by_frame = {int(record["frame"]): record for record in object_per_frame}
    robot_arm_by_frame = {
        int(record["frame"]): record for record in robot_arm_per_frame
    }
    combined: List[Dict[str, Any]] = []
    for frame in sorted(set(object_by_frame) | set(robot_arm_by_frame)):
        object_result = object_by_frame.get(frame)
        robot_arm_result = robot_arm_by_frame.get(frame)
        sources = [
            record
            for record in (object_result, robot_arm_result)
            if record is not None
        ]
        result: Dict[str, Any] = {
            "frame": frame,
            "num_sources": len(sources),
            **{
                metric: _mean_metric(sources, metric)
                for metric in RESULT_METRICS
            },
            "object": object_result,
            "robot_arm": robot_arm_result,
        }
        for record in sources:
            if "time_sec" in record:
                result["time_sec"] = record["time_sec"]
                break
        combined.append(result)
    return combined


@dataclass
class _MaskedSample:
    """One numerically independent frame/mask comparison."""

    frame: int
    mask_index: int
    mask_source: str
    psnr: float
    pred_masked: np.ndarray
    gt_masked: np.ndarray
    downsampled_index: int
    bbox: Tuple[int, int, int, int] | None = None
    pred_full_u8: np.ndarray | None = None
    gt_full_u8: np.ndarray | None = None
    mask_full: np.ndarray | None = None
    lpips: float | None = None
    vjepa_similarity: float | None = None


def _resize_frames(
    frames: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    from PIL import Image as PILImage

    return np.stack(
        [
            np.asarray(
                PILImage.fromarray(frame).resize(
                    (target_w, target_h), PILImage.BICUBIC
                )
            )
            for frame in frames
        ],
        axis=0,
    )


def _prepare_masked_samples(
    pred: np.ndarray,
    gt: np.ndarray,
    masks: np.ndarray,
    *,
    downsample_rate: int,
    crop_to_bbox: bool,
    bbox_pad: int,
    lpips_min_crop_size: int,
    mask_dilation_pixels: int,
    mask_aware_psnr: bool,
    visualize: bool,
    visualize_max_frames: int,
    mask_source: str,
) -> List[_MaskedSample]:
    """CPU-only preprocessing for one mask source."""

    downsample_rate = max(1, int(downsample_rate or 1))
    if masks.ndim == 3:
        masks = masks[None]
    M, T_mask, H_mask, W_mask = masks.shape
    common_t = min(len(pred), len(gt), T_mask)
    pred = pred[:common_t:downsample_rate]
    gt = gt[:common_t:downsample_rate]
    masks = masks[:, :common_t:downsample_rate]
    T = len(pred)
    if downsample_rate > 1:
        logger.info(
            "Downsample x%d: kept %d / %d frames.",
            downsample_rate,
            T,
            common_t,
        )
    if (
        pred.shape[1:3] != (H_mask, W_mask)
        or gt.shape[1:3] != (H_mask, W_mask)
    ):
        logger.warning(
            "Resolution mismatch: pred/gt=%s vs masks=(%d, %d). Resizing.",
            pred.shape[1:3],
            H_mask,
            W_mask,
        )
        pred = _resize_frames(pred, H_mask, W_mask)
        gt = _resize_frames(gt, H_mask, W_mask)
    assert pred.shape == gt.shape == (T, H_mask, W_mask, 3)

    viz_frames: set[int] = set()
    if visualize and T:
        n = max(1, int(visualize_max_frames))
        if T <= n:
            viz_frames = set(range(T))
        else:
            step = T / float(n)
            viz_frames = {int(i * step) for i in range(n)}

    pred_float = pred.astype(np.float32) / 255.0
    gt_float = gt.astype(np.float32) / 255.0
    samples: List[_MaskedSample] = []
    for m in range(M):
        for t in range(T):
            raw_mask = masks[m, t]
            if raw_mask.sum() < 10:
                continue
            mask_bool = _dilate_mask(raw_mask > 0, mask_dilation_pixels)
            mask_full = mask_bool.copy()
            p = pred_float[t]
            g = gt_float[t]
            bbox = None
            if crop_to_bbox:
                bbox = _mask_bbox(
                    mask_bool, bbox_pad, min_size=lpips_min_crop_size
                )
                if bbox is not None:
                    y0, y1, x0, x1 = bbox
                    p = p[y0:y1, x0:x1]
                    g = g[y0:y1, x0:x1]
                    mask_bool = mask_bool[y0:y1, x0:x1]
            mask_float = mask_bool.astype(np.float32)[..., None]
            pred_masked = np.ascontiguousarray(p * mask_float)
            gt_masked = np.ascontiguousarray(g * mask_float)
            psnr = (
                _masked_psnr(p, g, mask_bool)
                if mask_aware_psnr
                else _psnr(pred_masked, gt_masked)
            )
            keep_viz = t in viz_frames
            samples.append(
                _MaskedSample(
                    frame=int(t * downsample_rate),
                    mask_index=int(m),
                    mask_source=mask_source,
                    psnr=psnr,
                    pred_masked=pred_masked,
                    gt_masked=gt_masked,
                    downsampled_index=t,
                    bbox=bbox,
                    pred_full_u8=pred[t].astype(np.uint8) if keep_viz else None,
                    gt_full_u8=gt[t].astype(np.uint8) if keep_viz else None,
                    mask_full=mask_full if keep_viz else None,
                )
            )
    return samples


def _to_device_batch(
    tensors: Sequence[torch.Tensor],
    device: str,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    batch = torch.stack(list(tensors), dim=0)
    use_pin = pin_memory and torch.device(device).type == "cuda"
    if use_pin:
        batch = batch.pin_memory()
    return batch.to(device, non_blocking=use_pin)


def _infer_masked_samples(
    samples: Sequence[_MaskedSample],
    *,
    device: str,
    lpips_fn,
    jepa_model,
    jepa_transform,
    lpips_batch_size: int,
    vjepa_batch_size: int,
    pin_memory: bool,
) -> None:
    """Single-owner GPU inference; mutates samples with neural metric values."""

    if not samples:
        return
    # LPIPS is shape-sensitive. Exact HxW buckets preserve the serial inputs.
    buckets: dict[tuple[int, int], list[_MaskedSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.pred_masked.shape[:2]].append(sample)
    with torch.inference_mode():
        for bucket in buckets.values():
            for start in range(0, len(bucket), max(1, lpips_batch_size)):
                chunk = bucket[start : start + max(1, lpips_batch_size)]
                pred_batch = _to_device_batch(
                    [
                        torch.from_numpy(s.pred_masked).permute(2, 0, 1)
                        for s in chunk
                    ],
                    device,
                    pin_memory=pin_memory,
                )
                gt_batch = _to_device_batch(
                    [
                        torch.from_numpy(s.gt_masked).permute(2, 0, 1)
                        for s in chunk
                    ],
                    device,
                    pin_memory=pin_memory,
                )
                values = (
                    lpips_fn(pred_batch * 2.0 - 1.0, gt_batch * 2.0 - 1.0)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                for sample, value in zip(chunk, values):
                    sample.lpips = float(value)

        from PIL import Image as PILImage

        for start in range(0, len(samples), max(1, vjepa_batch_size)):
            chunk = samples[start : start + max(1, vjepa_batch_size)]
            transformed: list[torch.Tensor] = []
            for sample in chunk:
                for image in (sample.pred_masked, sample.gt_masked):
                    pil = PILImage.fromarray(
                        (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
                    )
                    transformed.append(jepa_transform(pil))
            batch = _to_device_batch(
                transformed, device, pin_memory=pin_memory
            ).unsqueeze(2)
            features = jepa_model(batch)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = features.mean(dim=1)
            values = (
                torch.nn.functional.cosine_similarity(
                    features[0::2], features[1::2], dim=1
                )
                .clamp(-1.0, 1.0)
                .detach()
                .cpu()
                .tolist()
            )
            for sample, value in zip(chunk, values):
                sample.vjepa_similarity = float(value)


def _samples_to_records(
    samples: Sequence[_MaskedSample],
    *,
    fps: float | None,
    viz_dir: Path | None,
    viz_uid: str | None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if sample.lpips is None or sample.vjepa_similarity is None:
            raise RuntimeError("Masked sample is missing GPU inference results")
        record: Dict[str, Any] = {
            "frame": sample.frame,
            "mask_index": sample.mask_index,
            "mask_source": sample.mask_source,
            "psnr": sample.psnr,
            "lpips": sample.lpips,
            "vjepa_similarity": sample.vjepa_similarity,
        }
        if fps:
            record["time_sec"] = sample.frame / float(fps)
        records.append(record)
        if (
            viz_dir is None
            or sample.pred_full_u8 is None
            or sample.gt_full_u8 is None
            or sample.mask_full is None
        ):
            continue
        try:
            mask_vis = np.stack(
                [sample.mask_full.astype(np.uint8) * 255] * 3, axis=-1
            )
            pred_crop = (
                np.clip(sample.pred_masked, 0.0, 1.0) * 255
            ).astype(np.uint8)
            gt_crop = (
                np.clip(sample.gt_masked, 0.0, 1.0) * 255
            ).astype(np.uint8)
            title = (
                f"{viz_uid}  source={sample.mask_source}  "
                f"frame={sample.frame}  mask={sample.mask_index}  "
                f"psnr={sample.psnr:.2f}  lpips={sample.lpips:.3f}  "
                f"vjepa={sample.vjepa_similarity:.3f}"
            )
            montage = _make_montage(
                [
                    (
                        "gt_frame",
                        _draw_bbox(sample.gt_full_u8, sample.bbox),
                        False,
                    ),
                    (
                        "pred_frame",
                        _draw_bbox(sample.pred_full_u8, sample.bbox),
                        False,
                    ),
                    (
                        "mask",
                        _draw_bbox(mask_vis, sample.bbox, color=(0, 255, 0)),
                        True,
                    ),
                    ("gt masked crop", gt_crop, False),
                    ("pred masked crop", pred_crop, False),
                ],
                title=title,
            )
            montage.save(
                viz_dir
                / (
                    "keyarea_visual_quality_"
                    f"{sample.mask_source}_frame{sample.frame:06d}_"
                    f"mask{sample.mask_index}.png"
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Visualization failed for %s frame=%d mask=%d: %s",
                viz_uid,
                sample.frame,
                sample.mask_index,
                exc,
            )
    records.sort(key=lambda r: (r["frame"], r["mask_index"]))
    return records


def _compute_masked_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    masks: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    downsample_rate: int = 1,
    fps: float | None = None,
    crop_to_bbox: bool = True,
    bbox_pad: int = 10,
    lpips_min_crop_size: int = 64,
    mask_dilation_pixels: int = 0,
    mask_aware_psnr: bool = True,
    visualize: bool = False,
    viz_dir: "Path | None" = None,
    viz_uid: str | None = None,
    visualize_max_frames: int = 8,
    mask_source: str = "object",
    lpips_fn=None,
    jepa_model=None,
    jepa_transform=None,
    lpips_batch_size: int = 1,
    vjepa_batch_size: int = 1,
    pin_memory: bool = False,
) -> List[Dict[str, Any]]:
    """Compute records with CPU preparation and exact-input GPU batching."""

    if lpips_fn is None:
        lpips_fn = ensure_lpips().LPIPS(net="alex").to(device).eval()
    if jepa_model is None or jepa_transform is None:
        raise ValueError("V-JEPA model and transform are required")
    samples = _prepare_masked_samples(
        pred,
        gt,
        masks,
        downsample_rate=downsample_rate,
        crop_to_bbox=crop_to_bbox,
        bbox_pad=bbox_pad,
        lpips_min_crop_size=lpips_min_crop_size,
        mask_dilation_pixels=mask_dilation_pixels,
        mask_aware_psnr=mask_aware_psnr,
        visualize=visualize,
        visualize_max_frames=visualize_max_frames,
        mask_source=mask_source,
    )
    _infer_masked_samples(
        samples,
        device=device,
        lpips_fn=lpips_fn,
        jepa_model=jepa_model,
        jepa_transform=jepa_transform,
        lpips_batch_size=lpips_batch_size,
        vjepa_batch_size=vjepa_batch_size,
        pin_memory=pin_memory,
    )
    return _samples_to_records(
        samples,
        fps=fps,
        viz_dir=Path(viz_dir) if visualize and viz_dir is not None else None,
        viz_uid=viz_uid,
    )


def _load_npz_masks(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        key = "masks" if "masks" in npz else list(npz.keys())[0]
        return npz[key]


@dataclass
class _PreparedEpisode:
    trial: EpisodeTrial
    prompt_masks_path: Path
    secondary_masks_path: Path
    prompt_mask_count: int
    secondary_mask_count: int
    object_samples: List[_MaskedSample]
    robot_arm_samples: List[_MaskedSample]
    viz_dir: Path | None

    @property
    def all_samples(self) -> List[_MaskedSample]:
        return self.object_samples + self.robot_arm_samples


def _prepare_episode(
    ep: EpisodeTrial,
    *,
    gt_root: Path,
    eval_view: str,
    secondary_mask_view: str,
    downsample_rate: int,
    crop_to_bbox: bool,
    bbox_pad: int,
    lpips_min_crop_size: int,
    mask_dilation_pixels: int,
    mask_aware_psnr: bool,
    visualize: bool,
    save_root: str | os.PathLike | None,
    visualize_max_frames: int,
) -> _PreparedEpisode | None:
    """Load and preprocess an episode without touching CUDA."""

    gt_episode_dir = (
        Path(ep.gt_video_dir).parent
        if ep.gt_video_dir
        else gt_root / ep.task_id / ep.episode_id
    )
    prompt_npz = gt_episode_dir / "prompt" / "masks.npz"
    prompt_npy = gt_episode_dir / "prompt" / "masks.npy"
    try:
        if prompt_npz.exists():
            prompt_masks = _load_npz_masks(prompt_npz)
            prompt_path = prompt_npz
        elif prompt_npy.exists():
            prompt_masks = np.load(prompt_npy)
            prompt_path = prompt_npy
        else:
            logger.warning(
                "No masks.npz/masks.npy under %s for %s; skipping.",
                gt_episode_dir / "prompt",
                ep.uid,
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load prompt masks for %s: %s", ep.uid, exc)
        return None

    secondary_path = (
        gt_episode_dir / "multiview_masks" / secondary_mask_view / "masks.npz"
    )
    if not secondary_path.exists():
        logger.warning("No secondary masks at %s for %s; skipping.", secondary_path, ep.uid)
        return None
    try:
        secondary_masks = _load_npz_masks(secondary_path)
        pred_frames = load_frames_as_array(ep.video_dir, preferred_view=eval_view)
        if ep.gt_video_dir:
            gt_frames = load_frames_as_array(
                ep.gt_video_dir, preferred_view=eval_view
            )
        else:
            logger.warning(
                "No gt_video_dir for %s; using prediction frames as GT.", ep.uid
            )
            gt_frames = pred_frames.copy()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to prepare frames/masks for %s: %s", ep.uid, exc)
        return None

    viz_dir = None
    if visualize and save_root is not None:
        from .. import episode_store

        viz_dir = episode_store.visualize_dir(
            save_root, ep.task_id, ep.episode_id
        )
    common = dict(
        downsample_rate=downsample_rate,
        crop_to_bbox=crop_to_bbox,
        bbox_pad=bbox_pad,
        lpips_min_crop_size=lpips_min_crop_size,
        mask_dilation_pixels=mask_dilation_pixels,
        mask_aware_psnr=mask_aware_psnr,
        visualize=visualize and viz_dir is not None,
        visualize_max_frames=visualize_max_frames,
    )
    object_samples = _prepare_masked_samples(
        pred_frames,
        gt_frames,
        prompt_masks,
        mask_source="object",
        **common,
    )
    robot_arm_samples = _prepare_masked_samples(
        pred_frames,
        gt_frames,
        secondary_masks,
        mask_source="robot_arm",
        **common,
    )
    logger.info(
        "Prepared %s: object=%d robot_arm=%d samples",
        ep.uid,
        len(object_samples),
        len(robot_arm_samples),
    )
    return _PreparedEpisode(
        trial=ep,
        prompt_masks_path=prompt_path,
        secondary_masks_path=secondary_path,
        prompt_mask_count=int(prompt_masks.shape[0]) if prompt_masks.ndim == 4 else 1,
        secondary_mask_count=(
            int(secondary_masks.shape[0]) if secondary_masks.ndim == 4 else 1
        ),
        object_samples=object_samples,
        robot_arm_samples=robot_arm_samples,
        viz_dir=viz_dir,
    )


def _prepared_episode_groups(
    episodes: Sequence[EpisodeTrial],
    prepare,
    *,
    workers: int,
    max_in_flight: int,
    prefetch_depth: int,
) -> Iterator[List[_PreparedEpisode]]:
    """Bounded deterministic prefetch; each yielded group can share GPU batches."""

    if workers <= 1:
        for start in range(0, len(episodes), max_in_flight):
            group = [
                prepared
                for prepared in (prepare(ep) for ep in episodes[start : start + max_in_flight])
                if prepared is not None
            ]
            if group:
                yield group
        return

    capacity = max(max_in_flight, workers + prefetch_depth)
    source = iter(episodes)
    pending: deque[Future[_PreparedEpisode | None]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="keyarea-prepare"
    ) as pool:
        for _ in range(min(capacity, len(episodes))):
            pending.append(pool.submit(prepare, next(source)))
        exhausted = len(pending) >= len(episodes)
        while pending:
            group: List[_PreparedEpisode] = []
            while pending and len(group) < max_in_flight:
                future = pending.popleft()
                try:
                    prepared = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("KeyArea episode preparation failed: %s", exc)
                    prepared = None
                if prepared is not None:
                    group.append(prepared)
                if not exhausted:
                    try:
                        pending.append(pool.submit(prepare, next(source)))
                    except StopIteration:
                        exhausted = True
            if group:
                yield group


def _finalize_prepared_episode(
    prepared: _PreparedEpisode,
    *,
    fps: float | None,
    eval_view: str,
    downsample_rate: int,
    crop_to_bbox: bool,
    bbox_pad: int,
    lpips_min_crop_size: int,
    mask_dilation_pixels: int,
    mask_aware_psnr: bool,
    vjepa_model_name: str,
    vjepa_model_path: str | None,
    vjepa_repo: str | None,
    secondary_mask_view: str,
    save_root: str | os.PathLike | None,
) -> Dict[str, Any]:
    ep = prepared.trial
    object_per_mask = _samples_to_records(
        prepared.object_samples,
        fps=fps,
        viz_dir=prepared.viz_dir,
        viz_uid=ep.uid,
    )
    robot_arm_per_mask = _samples_to_records(
        prepared.robot_arm_samples,
        fps=fps,
        viz_dir=prepared.viz_dir,
        viz_uid=ep.uid,
    )
    visualization_manifest = None
    if prepared.viz_dir is not None:
        visualization_manifest = _save_visualization_contact_sheets(
            prepared.viz_dir, ep.uid
        )
        save_json(
            visualization_manifest,
            prepared.viz_dir / "keyarea_visual_quality_manifest.json",
        )

    object_per_frame = _average_masks_per_frame(object_per_mask)
    robot_arm_per_frame = _average_masks_per_frame(robot_arm_per_mask)
    per_frame = _combine_sources_per_frame(
        object_per_frame, robot_arm_per_frame
    )
    object_summary = {
        metric: _mean_metric(object_per_frame, metric)
        for metric in RESULT_METRICS
    }
    robot_arm_summary = {
        metric: _mean_metric(robot_arm_per_frame, metric)
        for metric in RESULT_METRICS
    }
    episode_summary = {
        metric: _mean_metric(per_frame, metric) for metric in RESULT_METRICS
    }
    result = {
        "uid": ep.uid,
        "task_id": ep.task_id,
        "episode_id": ep.episode_id,
        "metrics": list(RESULT_METRICS),
        "metric_version": 12,
        **episode_summary,
        "downsample_rate": int(downsample_rate),
        "fps": float(fps) if fps is not None else None,
        "eval_view": str(eval_view),
        "secondary_mask_view": str(secondary_mask_view),
        "crop_to_bbox": bool(crop_to_bbox),
        "bbox_pad": int(bbox_pad),
        "lpips_min_crop_size": int(lpips_min_crop_size),
        "mask_dilation_pixels": int(mask_dilation_pixels),
        "mask_aware_psnr": bool(mask_aware_psnr),
        "vjepa_model_name": str(vjepa_model_name),
        "vjepa_model_path": (
            str(vjepa_model_path) if vjepa_model_path is not None else None
        ),
        "vjepa_repo": str(vjepa_repo) if vjepa_repo is not None else None,
        "visualization": visualization_manifest,
        "mask_sources": {
            "object": {
                "path": str(prepared.prompt_masks_path),
                "num_mask_sequences": prepared.prompt_mask_count,
                **object_summary,
                "per_frame": object_per_frame,
                "per_mask": object_per_mask,
            },
            "robot_arm": {
                "path": str(prepared.secondary_masks_path),
                "num_mask_sequences": prepared.secondary_mask_count,
                **robot_arm_summary,
                "per_frame": robot_arm_per_frame,
                "per_mask": robot_arm_per_mask,
            },
        },
        "per_frame": per_frame,
    }
    if save_root is not None:
        from .. import episode_store

        path = episode_store.save_metric(
            save_root, ep.task_id, ep.episode_id, "keyarea_visual_quality", result
        )
        logger.info("Saved %s keyarea result to %s", ep.uid, path)
    return result


def compute_keyarea_visual_quality(
    episodes: List[EpisodeTrial],
    gt_root: str | os.PathLike,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_view: str = "top_head",
    downsample_rate: int = 1,
    fps: float | None = None,
    crop_to_bbox: bool = True,
    bbox_pad: int = 10,
    lpips_min_crop_size: int = 64,
    mask_dilation_pixels: int = 0,
    mask_aware_psnr: bool = True,
    vjepa_model_path: str | None = None,
    vjepa_model_name: str = "vjepa2_1_vit_base_384",
    vjepa_repo: str | None = None,
    secondary_mask_view: str = "top_head",
    visualize: bool = False,
    save_root: "str | os.PathLike | None" = None,
    visualize_max_frames: int = 8,
    progress: bool = True,
    execution: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Main entry point for KeyArea Visual Quality metric.

    Args:
        downsample_rate: time-axis stride applied to frames + masks before
            metric computation (1 = every frame, 4 = every 4th, etc.).
        fps: optional frame rate; when set, per-frame records carry ``time_sec``.
        crop_to_bbox: crop each frame + mask to the mask bounding box (+bbox_pad)
            before masking, so the black background no longer dominates.
        bbox_pad: padding (px) around the mask bbox when cropping.
        lpips_min_crop_size: minimum crop height and width. Small mask boxes are
            expanded to this size so AlexNet pooling in LPIPS cannot collapse a
            spatial dimension to zero.
        mask_dilation_pixels: expand each object and robot-arm mask outwards by
            this many pixels before cropping, masking, PSNR, and LPIPS.
        mask_aware_psnr: compute PSNR over masked pixels only (pixel-level).
        vjepa_model_path: optional local V-JEPA checkpoint.
        vjepa_model_name: torch.hub V-JEPA encoder architecture.
        secondary_mask_view: also evaluate masks from
            ``multiview_masks/<view>/masks.npz`` and combine its per-frame mean
            equally with the prompt-mask per-frame mean.
        visualize: when True, dump per-(frame, mask) panels showing gt/pred
            frames, the mask, and the cropped-masked pair fed to the metrics.
        save_root: generated-dataset root (val_base). When set, each episode's
            result is written to ``<save_root>/<task>/<ep>/keyarea_visual_quality.json``
            as soon as it finishes, and visualizations go under
            ``<save_root>/<task>/<ep>/visualize/``.
        visualize_max_frames: number of frames per episode to visualize.

    Returns a dict with per-episode PSNR/LPIPS/V-JEPA results and dataset-level
    averages. Each per-episode record also carries a ``per_frame`` list with
    the raw metric values per frame/mask.
    """
    if int(mask_dilation_pixels) < 0:
        raise ValueError("mask_dilation_pixels must be >= 0")
    mask_dilation_pixels = int(mask_dilation_pixels)
    execution = dict(execution or {})
    mode = str(execution.get("mode", "serial"))
    episode_workers = int(execution.get("episode_workers", 1))
    max_in_flight = int(execution.get("max_in_flight_episodes", 1))
    prefetch_depth = int(execution.get("prefetch_depth", 1))
    lpips_batch_size = int(execution.get("lpips_batch_size", 1))
    vjepa_batch_size = int(execution.get("vjepa_batch_size", 1))
    pin_memory = bool(execution.get("pin_memory", False))
    if mode == "serial":
        episode_workers = max_in_flight = prefetch_depth = 1
        lpips_batch_size = vjepa_batch_size = 1
    for name, value in (
        ("episode_workers", episode_workers),
        ("max_in_flight_episodes", max_in_flight),
        ("prefetch_depth", prefetch_depth),
        ("lpips_batch_size", lpips_batch_size),
        ("vjepa_batch_size", vjepa_batch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    lpips_fn = ensure_lpips().LPIPS(net="alex").to(device).eval()
    jepa_model, jepa_transform = ensure_jepa(
        device,
        model_path=vjepa_model_path,
        model_name=vjepa_model_name,
        repo_or_dir=vjepa_repo,
    )
    if jepa_model is None or jepa_transform is None:
        raise RuntimeError(
            "V-JEPA is required for KeyArea Visual Quality but failed to load "
            f"(model_name={vjepa_model_name}, model_path={vjepa_model_path})."
        )

    gt_root = Path(gt_root)
    prepare_kwargs = dict(
        gt_root=gt_root,
        eval_view=eval_view,
        secondary_mask_view=secondary_mask_view,
        downsample_rate=downsample_rate,
        crop_to_bbox=crop_to_bbox,
        bbox_pad=bbox_pad,
        lpips_min_crop_size=lpips_min_crop_size,
        mask_dilation_pixels=mask_dilation_pixels,
        mask_aware_psnr=mask_aware_psnr,
        visualize=visualize,
        save_root=save_root,
        visualize_max_frames=visualize_max_frames,
    )

    def prepare(ep: EpisodeTrial) -> _PreparedEpisode | None:
        return _prepare_episode(ep, **prepare_kwargs)

    results: List[Dict[str, Any]] = []
    groups = _prepared_episode_groups(
        episodes,
        prepare,
        workers=episode_workers,
        max_in_flight=max_in_flight,
        prefetch_depth=prefetch_depth,
    )
    progress_bar = (
        tqdm(total=len(episodes), desc="KeyArea Visual Quality")
        if progress
        else None
    )
    try:
        for group in groups:
            # This is the only CUDA-owner path. Samples from several episodes and
            # both mask sources share the same exact-input batches.
            group_samples = [
                sample for prepared in group for sample in prepared.all_samples
            ]
            _infer_masked_samples(
                group_samples,
                device=device,
                lpips_fn=lpips_fn,
                jepa_model=jepa_model,
                jepa_transform=jepa_transform,
                lpips_batch_size=lpips_batch_size,
                vjepa_batch_size=vjepa_batch_size,
                pin_memory=pin_memory,
            )
            for prepared in group:
                try:
                    results.append(
                        _finalize_prepared_episode(
                            prepared,
                            fps=fps,
                            eval_view=eval_view,
                            downsample_rate=downsample_rate,
                            crop_to_bbox=crop_to_bbox,
                            bbox_pad=bbox_pad,
                            lpips_min_crop_size=lpips_min_crop_size,
                            mask_dilation_pixels=mask_dilation_pixels,
                            mask_aware_psnr=mask_aware_psnr,
                            vjepa_model_name=vjepa_model_name,
                            vjepa_model_path=vjepa_model_path,
                            vjepa_repo=vjepa_repo,
                            secondary_mask_view=secondary_mask_view,
                            save_root=save_root,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Failed to finalize KeyArea result for %s: %s",
                        prepared.trial.uid,
                        exc,
                    )
                finally:
                    if progress_bar is not None:
                        progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    results.sort(key=lambda r: (r["task_id"], r["episode_id"], r["uid"]))
    all_metrics = {
        metric: [result[metric] for result in results]
        for metric in RESULT_METRICS
    }
    all_source_metrics = {
        source: {
            metric: [
                result["mask_sources"][source][metric] for result in results
            ]
            for metric in RESULT_METRICS
        }
        for source in ("object", "robot_arm")
    }

    agg = {
        **{
            metric: (
                float(np.nanmean(all_metrics[metric]))
                if all_metrics[metric]
                else 0.0
            )
            for metric in RESULT_METRICS
        },
        "mask_sources": {
            source: {
                metric: (
                    float(np.nanmean(all_source_metrics[source][metric]))
                    if all_source_metrics[source][metric]
                    else 0.0
                )
                for metric in RESULT_METRICS
            }
            for source in ("object", "robot_arm")
        },
        "num_episodes": len(results),
    }

    return {
        "per_episode": results,
        "aggregated": agg,
    }
