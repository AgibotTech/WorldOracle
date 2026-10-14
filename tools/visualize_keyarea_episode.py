#!/usr/bin/env python3
"""Visualize one common episode across multiple KeyArea baselines.

The tool reads the evaluated frame indices from each result JSON, preserves the
original episode timeline when playing downsampled results, creates temporal
plots, and renders synchronized videos. Two additional comparison videos show
the object and robot-arm masks frame by frame: each column is GT or one
baseline, with the mask overlay above and the masked content below.
Each lower row is the dilated, bbox-cropped, black-background input actually
passed to PSNR/LPIPS/V-JEPA, annotated with scores loaded from the result JSON.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image as PILImage


PSNR_METRICS = ("psnr",)
RAW_METRICS = ("psnr", "lpips", "vjepa_similarity")
SCORE_METRICS = (
    "psnr_score",
    "lpips_score",
    "vjepa_score",
    "quality_score",
)
ALL_METRICS = RAW_METRICS + SCORE_METRICS
FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one complete episode across KeyArea baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis/keyarea_downsampling.yaml",
        help="YAML containing baselines and normalization anchors.",
    )
    parser.add_argument("--task", required=True, help="Common task id, e.g. hangtowel.")
    parser.add_argument(
        "--episode", required=True, help="Common episode id, e.g. ep000000."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional original-frame cap. By default, visualize the whole episode.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--view",
        default="top_head",
        help="Generated-frame view used when video/<view>/ exists.",
    )
    parser.add_argument(
        "--gt-root",
        default="data/acwmbench_dataset",
        help="GT dataset root containing <task>/<episode>/video/<view>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <analysis output_dir>/episodes/<task>__<episode>.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or len(baselines) < 2:
        raise ValueError("config.baselines must contain at least two baselines")
    normalization = config.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("config.normalization is required")
    for metric in RAW_METRICS:
        anchors = normalization.get(metric)
        if not isinstance(anchors, dict) or "min" not in anchors or "max" not in anchors:
            raise ValueError(f"normalization.{metric} must define min and max")
        anchors["min"] = float(anchors["min"])
        anchors["max"] = float(anchors["max"])
        if anchors["max"] <= anchors["min"]:
            raise ValueError(f"normalization.{metric}.max must exceed min")
    return config


def finite(value: object, metric: str, psnr_inf_cap: float) -> float:
    if not isinstance(value, (int, float)):
        return float("nan")
    number = float(value)
    if math.isfinite(number):
        return number
    if metric in PSNR_METRICS and number == float("inf"):
        return psnr_inf_cap
    return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def normalize_scores(
    metrics: Mapping[str, float],
    normalization: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise TypeError(
            "normalize_scores.metrics must be a metric-name mapping, "
            f"got {type(metrics).__name__}"
        )
    psnr_cfg = normalization["psnr"]
    lpips_cfg = normalization["lpips"]
    vjepa_cfg = normalization["vjepa_similarity"]
    psnr_unit = float(
        np.clip(
            (metrics["psnr"] - psnr_cfg["min"])
            / (psnr_cfg["max"] - psnr_cfg["min"]),
            0.0,
            1.0,
        )
    )
    lpips_unit = float(
        np.clip(
            (lpips_cfg["max"] - metrics["lpips"])
            / (lpips_cfg["max"] - lpips_cfg["min"]),
            0.0,
            1.0,
        )
    )
    vjepa_unit = float(
        np.clip(
            (metrics["vjepa_similarity"] - vjepa_cfg["min"])
            / (vjepa_cfg["max"] - vjepa_cfg["min"]),
            0.0,
            1.0,
        )
    )
    return {
        "psnr_score": 100.0 * psnr_unit,
        "lpips_score": 100.0 * lpips_unit,
        "vjepa_score": 100.0 * vjepa_unit,
        "quality_score": 100.0 * (psnr_unit * lpips_unit * vjepa_unit) ** (1.0 / 3.0),
    }


def load_episode_frames(
    result_path: Path,
    baseline: str,
    max_frames: int | None,
    fps: float,
    normalization: Mapping[str, Mapping[str, float]],
    psnr_inf_cap: float,
) -> tuple[pd.DataFrame, dict, Dict[str, Dict[int, dict]]]:
    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    downsample_rate = int(data.get("downsample_rate", -1))
    if downsample_rate < 1:
        raise ValueError(f"{result_path}: invalid downsample_rate")
    if int(data.get("metric_version", -1)) < 12:
        raise ValueError(
            f"{result_path}: metric_version must be >= 12 for compatible "
            "PSNR/LPIPS/V-JEPA records"
        )

    source_lookup: Dict[str, Dict[int, dict]] = {}
    for source in ("object", "robot_arm"):
        source_data = (data.get("mask_sources") or {}).get(source)
        records = source_data.get("per_frame") if isinstance(source_data, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"{result_path}: missing mask_sources.{source}.per_frame")
        fallback_masks: Dict[int, list[dict]] = {}
        per_mask = source_data.get("per_mask") if isinstance(source_data, dict) else None
        if isinstance(per_mask, list):
            for mask_record in per_mask:
                if not isinstance(mask_record, dict) or "frame" not in mask_record:
                    continue
                fallback_masks.setdefault(int(mask_record["frame"]), []).append(
                    mask_record
                )
        parsed = {}
        for record in records:
            frame = int(record["frame"])
            if frame < 0 or (max_frames is not None and frame >= max_frames):
                continue
            mask_records = record.get("masks")
            if not isinstance(mask_records, list):
                mask_records = fallback_masks.get(frame, [])
            parsed_masks = []
            for mask_record in mask_records:
                if not isinstance(mask_record, dict):
                    continue
                mask_metrics = {
                    metric: finite(
                        mask_record.get(metric),
                        metric,
                        psnr_inf_cap,
                    )
                    for metric in RAW_METRICS
                }
                if not all(
                    math.isfinite(value) for value in mask_metrics.values()
                ):
                    continue
                parsed_masks.append(
                    {
                        "mask_index": int(mask_record["mask_index"]),
                        **mask_metrics,
                        **normalize_scores(
                            mask_metrics,
                            normalization,
                        ),
                    }
                )
            if not parsed_masks:
                continue
            parsed[frame] = {
                **{
                    metric: finite_mean(
                        mask_record[metric]
                        for mask_record in parsed_masks
                    )
                    for metric in ALL_METRICS
                },
                "masks": sorted(
                    parsed_masks,
                    key=lambda item: item["mask_index"],
                ),
            }
        source_lookup[source] = parsed

    rows = []
    common_frames = sorted(
        set(source_lookup["object"]) | set(source_lookup["robot_arm"])
    )
    for frame in common_frames:
        sources = [
            source_lookup[source][frame]
            for source in ("object", "robot_arm")
            if frame in source_lookup[source]
        ]
        rows.append(
            {
                "baseline": baseline,
                "frame": frame,
                "time_sec": frame / fps,
                **{
                    metric: finite_mean(record[metric] for record in sources)
                    for metric in ALL_METRICS
                },
            }
        )
    frame_scores = pd.DataFrame(rows).sort_values("frame")
    if frame_scores.empty:
        raise ValueError(f"{result_path}: no valid records in requested frame range")

    summary = {
        "baseline": baseline,
        "downsample_rate": downsample_rate,
        "metric_version": int(data.get("metric_version", -1)),
        "crop_to_bbox": bool(data.get("crop_to_bbox", True)),
        "bbox_pad": int(data.get("bbox_pad", 10)),
        "lpips_min_crop_size": int(data.get("lpips_min_crop_size", 64)),
        "mask_dilation_pixels": int(data.get("mask_dilation_pixels", 0)),
        "mask_aware_psnr": bool(data.get("mask_aware_psnr", True)),
        "num_frames": int(len(frame_scores)),
        "first_frame": int(frame_scores["frame"].min()),
        "last_frame": int(frame_scores["frame"].max()),
        "duration_sec": float((frame_scores["frame"].max() + 1) / fps),
        **{
            metric: float(frame_scores[metric].mean())
            for metric in ALL_METRICS
        },
    }
    for source in ("object", "robot_arm"):
        records = list(source_lookup[source].values())
        for metric in SCORE_METRICS:
            summary[f"{source}_{metric}"] = finite_mean(
                record[metric] for record in records
            )
    for metric in SCORE_METRICS:
        summary[metric] = finite_mean(
            summary[f"{source}_{metric}"]
            for source in ("object", "robot_arm")
        )
    return frame_scores, summary, source_lookup


def frame_files(episode_dir: Path, view: str) -> list[Path]:
    candidates = [
        episode_dir / "video" / view,
        episode_dir / "video",
    ]
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS
        )
        if files:
            return files
    raise FileNotFoundError(
        f"No generated frames found under {episode_dir / 'video'}"
    )


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return stem or "baseline"


def write_video(
    files: Sequence[Path],
    output_path: Path,
    baseline: str,
    frame_scores: pd.DataFrame | None,
    frame_indices: Sequence[int],
    source_fps: float,
    playback_fps: float,
) -> int:
    selected = [
        (frame, files[frame])
        for frame in frame_indices
        if 0 <= frame < len(files)
    ]
    if not selected:
        raise ValueError(f"No frames available for {baseline}")
    first = cv2.imread(str(selected[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode {selected[0]}")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        playback_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {temporary_path}")
    score_lookup = (
        frame_scores.set_index("frame").to_dict(orient="index")
        if frame_scores is not None
        else {}
    )
    written = 0
    try:
        for frame, path in selected:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            score = score_lookup.get(frame)
            labels = [
                f"{baseline}  frame={frame}  t={frame / source_fps:.2f}s"
            ]
            if score is not None:
                labels.extend(
                    [
                        (
                            f"PSNR={score['psnr']:.2f}  "
                            f"LPIPS={score['lpips']:.3f}"
                        ),
                        f"V-JEPA={score['vjepa_similarity']:.3f}",
                        (
                            f"P={score['psnr_score']:.1f}  "
                            f"L={score['lpips_score']:.1f}"
                        ),
                        (
                            f"V={score['vjepa_score']:.1f}  "
                            f"Quality={score['quality_score']:.1f}"
                        ),
                    ]
                )
            overlay_height = 14 + 30 * len(labels)
            cv2.rectangle(image, (8, 8), (min(width - 8, 570), overlay_height), (0, 0, 0), -1)
            for line_index, label in enumerate(labels):
                cv2.putText(
                    image,
                    label,
                    (16, 34 + line_index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(image)
            written += 1
    finally:
        writer.release()
    if written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No decodable frames for {baseline}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(temporary_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
            temporary_path.unlink(missing_ok=True)
        except (OSError, subprocess.CalledProcessError):
            os.replace(temporary_path, output_path)
    else:
        os.replace(temporary_path, output_path)
    return written


def load_mask_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    if path.suffix == ".npy":
        masks = np.load(path)
    else:
        with np.load(path) as archive:
            key = "masks" if "masks" in archive else list(archive.keys())[0]
            masks = archive[key]
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4:
        raise ValueError(f"{path}: expected masks with shape (M,T,H,W)")
    return masks


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if pixels <= 0:
        return mask_u8
    kernel_size = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask_u8, kernel, iterations=1)


def masks_for_frame(
    masks: np.ndarray,
    frame: int,
    dilation_pixels: int,
) -> list[np.ndarray]:
    if frame < 0 or frame >= masks.shape[1]:
        return []
    return [
        dilate_mask(masks[index, frame], dilation_pixels)
        for index in range(masks.shape[0])
    ]


def expand_interval(start: int, end: int, limit: int, min_size: int) -> tuple[int, int]:
    target = min(limit, max(end - start, min_size))
    center = (start + end) / 2.0
    new_start = int(round(center - target / 2.0))
    new_start = max(0, min(new_start, limit - target))
    return new_start, new_start + target


def mask_bbox(
    mask_bool: np.ndarray,
    pad: int,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    height, width = mask_bool.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(height, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(width, int(xs.max()) + pad + 1)
    y0, y1 = expand_interval(y0, y1, height, max(1, min_size))
    x0, x1 = expand_interval(x0, x1, width, max(1, min_size))
    return y0, y1, x0, x1


def resize_like_evaluator(
    image_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Match KeyArea's PIL bicubic resize before converting back to BGR."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = PILImage.fromarray(rgb).resize(
        (width, height),
        PILImage.BICUBIC,
    )
    return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)


def visualize_mask_on_frame(
    image: np.ndarray,
    frame_masks: Sequence[np.ndarray],
    crop_to_bbox: bool,
    bbox_pad: int,
    min_crop_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return full-frame overlay and exact masked crops used by the metrics."""
    colors = (
        (60, 220, 60),
        (40, 200, 255),
        (220, 80, 220),
        (255, 120, 40),
        (80, 160, 255),
    )
    mask_h, mask_w = frame_masks[0].shape if frame_masks else image.shape[:2]
    image = resize_like_evaluator(image, mask_w, mask_h)
    overlay = image.copy()
    masked_images = []
    for index, mask in enumerate(frame_masks):
        mask_bool = mask > 0
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        overlay[mask_bool] = (
            0.55 * overlay[mask_bool].astype(np.float32) + 0.45 * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            colors[index % len(colors)],
            2,
        )
        bbox = (
            mask_bbox(mask_bool, bbox_pad, min_crop_size)
            if crop_to_bbox
            else None
        )
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            cv2.rectangle(
                overlay,
                (x0, y0),
                (max(x0, x1 - 1), max(y0, y1 - 1)),
                colors[index % len(colors)],
                2,
            )
            crop = image[y0:y1, x0:x1]
            crop_mask = mask_bool[y0:y1, x0:x1]
        else:
            crop = image
            crop_mask = mask_bool
        masked = np.zeros_like(crop)
        masked[crop_mask] = crop[crop_mask]
        masked_images.append(masked)
    return overlay, masked_images


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def comparison_cell(
    image: np.ndarray,
    frame_masks: Sequence[np.ndarray],
    label: str,
    score: Mapping[str, float] | None,
    panel_width: int,
    crop_to_bbox: bool,
    bbox_pad: int,
    min_crop_size: int,
) -> np.ndarray:
    mask_h, mask_w = frame_masks[0].shape if frame_masks else image.shape[:2]
    overlay, masked_images = visualize_mask_on_frame(
        image,
        frame_masks,
        crop_to_bbox,
        bbox_pad,
        min_crop_size,
    )
    panel_height = max(1, int(round(mask_h * panel_width / mask_w)))
    overlay = fit_panel(overlay, panel_width, panel_height)
    masked_images = [
        fit_panel(masked, panel_width, panel_height)
        for masked in masked_images
    ]
    header_h = 58
    mask_text_h = 50
    cell = np.zeros(
        (
            header_h
            + panel_height
            + len(masked_images) * (4 + mask_text_h + panel_height),
            panel_width,
            3,
        ),
        dtype=np.uint8,
    )
    cv2.putText(
        cell,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    score_text = "GT reference"
    if score is not None:
        score_text = (
            f"PSNR {score['psnr']:.1f} | LPIPS {score['lpips']:.3f} | "
            f"V-JEPA {score['vjepa_similarity']:.3f} | "
            f"Q {score['quality_score']:.1f}"
        )
    cv2.putText(
        cell,
        score_text,
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cell[header_h : header_h + panel_height] = overlay
    y = header_h + panel_height + 4
    mask_scores = {
        int(record["mask_index"]): record
        for record in (score.get("masks", []) if score is not None else [])
        if isinstance(record, Mapping) and "mask_index" in record
    }
    for mask_index, masked in enumerate(masked_images):
        record = mask_scores.get(mask_index)
        detail_lines = [f"mask {mask_index} | GT metric input"]
        if record is not None:
            detail_lines = [
                (
                    f"mask {mask_index} | PSNR {record['psnr']:.1f} | "
                    f"LPIPS {record['lpips']:.3f} | "
                    f"V-JEPA {record['vjepa_similarity']:.3f}"
                ),
                (
                    f"Pscore {record['psnr_score']:.1f} | "
                    f"Lscore {record['lpips_score']:.1f} | "
                    f"Vscore {record['vjepa_score']:.1f} | "
                    f"Q {record['quality_score']:.1f}"
                ),
            ]
        for line_index, detail in enumerate(detail_lines):
            cv2.putText(
                cell,
                detail,
                (6, y + 18 + 20 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        image_y = y + mask_text_h
        cell[image_y : image_y + panel_height] = masked
        y = image_y + panel_height + 4
    return cell


def write_mask_comparison_video(
    streams: Mapping[str, Sequence[Path]],
    masks: np.ndarray,
    source: str,
    score_lookups: Mapping[str, Mapping[int, Mapping[str, float]]],
    frame_indices: Sequence[int],
    output_path: Path,
    source_fps: float,
    playback_fps: float,
    dilation_pixels: int,
    crop_to_bbox: bool,
    bbox_pad: int,
    min_crop_size: int,
    panel_width: int = 320,
) -> int:
    labels = list(streams)
    valid_indices = [
        frame
        for frame in frame_indices
        if frame < masks.shape[1]
        and all(frame < len(streams[label]) for label in labels)
    ]
    if not valid_indices:
        raise ValueError(f"No common frames available for {source} mask video")
    first_masks = masks_for_frame(masks, valid_indices[0], dilation_pixels)
    mask_h, mask_w = masks.shape[2:]
    if not first_masks:
        first_masks = [np.zeros((mask_h, mask_w), dtype=np.uint8)]
    placeholder = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    first_cell = comparison_cell(
        placeholder,
        first_masks,
        labels[0],
        None,
        panel_width,
        crop_to_bbox,
        bbox_pad,
        min_crop_size,
    )
    header_h = 46
    height = header_h + first_cell.shape[0]
    width = first_cell.shape[1] * len(labels)
    width -= width % 2
    height -= height % 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        playback_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {temporary_path}")

    written = 0
    try:
        for frame in valid_indices:
            current_masks = masks_for_frame(masks, frame, dilation_pixels)
            if not current_masks:
                current_masks = [
                    np.zeros((mask_h, mask_w), dtype=np.uint8)
                ]
            cells = []
            for label in labels:
                image = cv2.imread(
                    str(streams[label][frame]),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    image = placeholder.copy()
                score = (
                    score_lookups.get(label, {}).get(frame)
                    if label != "GT"
                    else None
                )
                cells.append(
                    comparison_cell(
                        image,
                        current_masks,
                        label,
                        score,
                        panel_width,
                        crop_to_bbox,
                        bbox_pad,
                        min_crop_size,
                    )
                )
            body = np.concatenate(cells, axis=1)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            title = (
                f"{source} masks | frame={frame} | "
                f"t={frame / source_fps:.2f}s | dilation={dilation_pixels}px | "
                f"crop={'bbox' if crop_to_bbox else 'full'} "
                f"(pad={bbox_pad}, min={min_crop_size})"
            )
            cv2.putText(
                canvas,
                title,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            canvas[header_h : header_h + body.shape[0], : body.shape[1]] = body[
                : height - header_h,
                :width,
            ]
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
    if written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No mask-comparison frames written for {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(temporary_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
            temporary_path.unlink(missing_ok=True)
        except (OSError, subprocess.CalledProcessError):
            os.replace(temporary_path, output_path)
    else:
        os.replace(temporary_path, output_path)
    return written


def pairwise_temporal_correlations(frame_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = sorted(frame_scores["baseline"].unique())
    for metric in ALL_METRICS:
        pivot = frame_scores.pivot(index="frame", columns="baseline", values=metric)
        for left, right in combinations(baselines, 2):
            paired = pivot[[left, right]].dropna()
            rows.append(
                {
                    "metric": metric,
                    "baseline_a": left,
                    "baseline_b": right,
                    "num_frames": len(paired),
                    "pearson": float(paired[left].corr(paired[right], method="pearson")),
                    "spearman": float(paired[left].corr(paired[right], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def save_timeseries_plot(
    frame_scores: pd.DataFrame,
    metrics: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    metrics = tuple(metrics)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.2 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        for baseline, group in frame_scores.groupby("baseline"):
            group = group.sort_values("time_sec")
            axis.plot(
                group["time_sec"],
                group[metric],
                linewidth=1.0,
                alpha=0.85,
                label=baseline,
            )
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    x = np.arange(len(summary))
    width = 0.26
    axes[0].bar(x, summary["psnr"])
    axes[0].set_xticks(x, summary["baseline"], rotation=15, ha="right")
    axes[0].set_title("PSNR over selected window")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].bar(x, summary["lpips"])
    axes[1].set_xticks(x, summary["baseline"], rotation=15, ha="right")
    axes[1].set_title("LPIPS over selected window")
    axes[1].set_ylabel("LPIPS")
    axes[2].bar(x, summary["vjepa_similarity"])
    axes[2].set_xticks(x, summary["baseline"], rotation=15, ha="right")
    axes[2].set_title("V-JEPA similarity over selected window")
    axes[2].set_ylabel("V-JEPA cosine")
    quality_metrics = (
        "quality_score",
        "object_quality_score",
        "robot_arm_quality_score",
    )
    for offset, metric in zip((-width, 0.0, width), quality_metrics):
        axes[3].bar(x + offset, summary[metric], width, label=metric)
    axes[3].set_xticks(x, summary["baseline"], rotation=15, ha="right")
    axes[3].set_ylim(0, 100)
    axes[3].set_title("Overall and source quality scores")
    axes[3].set_ylabel("Score (0–100)")
    axes[3].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmaps(correlations: pd.DataFrame, output_path: Path) -> None:
    metrics = tuple(ALL_METRICS)
    columns = 3
    rows_count = int(math.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(14, 4 * rows_count),
        squeeze=False,
    )
    for axis, metric in zip(axes.flat, metrics):
        rows = correlations[correlations["metric"] == metric]
        names = sorted(set(rows["baseline_a"]) | set(rows["baseline_b"]))
        matrix = np.eye(len(names), dtype=np.float64)
        for _, row in rows.iterrows():
            left = names.index(row["baseline_a"])
            right = names.index(row["baseline_b"])
            matrix[left, right] = matrix[right, left] = row["pearson"]
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
        axis.set_yticks(range(len(names)), names, fontsize=8)
        axis.set_title(f"{metric}: temporal Pearson")
        for row in range(len(names)):
            for column in range(len(names)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    for axis in axes.flat[len(metrics) :]:
        axis.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Pearson correlation")
    fig.suptitle("Same-frame temporal correlation between baseline metric curves")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_dashboard(
    output_path: Path,
    task: str,
    episode: str,
    fps: float,
    max_frames: int,
    downsample_rate: int,
    playback_fps: float,
    summaries: Sequence[dict],
    videos: Mapping[str, str],
) -> None:
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['baseline'])}</td>"
        f"<td>{row['num_frames']}</td>"
        f"<td>{row['psnr']:.4f}</td>"
        f"<td>{row['lpips']:.4f}</td>"
        f"<td>{row['vjepa_similarity']:.4f}</td>"
        f"<td>{row['psnr_score']:.2f}</td>"
        f"<td>{row['lpips_score']:.2f}</td>"
        f"<td>{row['vjepa_score']:.2f}</td>"
        f"<td>{row['quality_score']:.2f}</td>"
        f"<td>{row['object_quality_score']:.2f}</td>"
        f"<td>{row['robot_arm_quality_score']:.2f}</td>"
        "</tr>"
        for row in summaries
    )
    video_cards = "\n".join(
        (
            "<section class='video-card comparison'>"
            if "comparison" in baseline
            else "<section class='video-card'>"
        )
        + f"<h3>{html.escape(baseline)}</h3>"
        + f"<video preload='metadata' src='{html.escape(relative_path)}'></video>"
        + "</section>"
        for baseline, relative_path in videos.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KeyArea episode comparison</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
  body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: #111; color: #eee; }}
  h1, h2, h3 {{ margin: 0 0 10px; }}
  .muted {{ color: #aaa; }}
  .controls {{ position: sticky; top: 0; z-index: 2; padding: 12px; background: #1b1b1b; border: 1px solid #333; }}
  .controls input {{ width: min(720px, 70vw); }}
  .videos {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 18px 0 26px; }}
  .video-card {{ border: 1px solid #333; padding: 10px; background: #181818; }}
  .video-card.comparison {{ grid-column: 1 / -1; }}
  video {{ width: 100%; display: block; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 28px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #333; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  img {{ width: 100%; margin: 8px 0 24px; background: white; }}
  @media (max-width: 900px) {{ .videos {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>KeyArea: {html.escape(task)} / {html.escape(episode)}</h1>
<p class="muted">Original frames 0–{max_frames - 1} · source {fps:g} FPS ·
stride {downsample_rate} · playback {playback_fps:g} FPS ·
timeline duration {max_frames / fps:.2f} s</p>
<div class="controls">
  <button id="play">Play all</button>
  <button id="pause">Pause all</button>
  <label> Time <input id="time" type="range" min="0" max="{max_frames / fps:.4f}" step="{1 / fps:.6f}" value="0"></label>
  <span id="clock">0.00 s</span>
</div>
<div class="videos">{video_cards}</div>
<h2>Window summary</h2>
<table>
<thead><tr><th>Baseline</th><th>Frames</th><th>PSNR</th><th>LPIPS</th><th>V-JEPA</th><th>PSNR score</th><th>LPIPS score</th><th>V-JEPA score</th><th>Overall quality</th><th>Object quality</th><th>Robot-arm quality</th></tr></thead>
<tbody>{summary_rows}</tbody>
</table>
<h2>Baseline summary</h2><img src="plots/baseline_summary.png" alt="Baseline summary">
<h2>Raw metric timelines</h2><img src="plots/raw_timeseries.png" alt="Raw metric timelines">
<h2>Normalized score timelines</h2><img src="plots/score_timeseries.png" alt="Normalized score timelines">
<h2>Temporal correlations</h2><img src="plots/temporal_correlations.png" alt="Temporal correlation heatmaps">
<script>
const videos = [...document.querySelectorAll("video")];
const slider = document.getElementById("time");
const clock = document.getElementById("clock");
const seek = value => {{
  const t = Number(value);
  videos.forEach(video => {{ if (Math.abs(video.currentTime - t) > 0.05) video.currentTime = t; }});
  clock.textContent = `${{t.toFixed(2)}} s`;
}};
slider.addEventListener("input", event => seek(event.target.value));
document.getElementById("play").addEventListener("click", () => videos.forEach(video => video.play()));
document.getElementById("pause").addEventListener("click", () => videos.forEach(video => video.pause()));
videos[0]?.addEventListener("timeupdate", () => {{
  if (!videos[0].paused) {{
    slider.value = videos[0].currentTime;
    seek(videos[0].currentTime);
  }}
}});
</script>
</body>
</html>"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    config = load_config(args.config)
    normalization = config["normalization"]
    psnr_inf_cap = float(config.get("psnr_inf_cap", 100.0))
    default_root = Path(
        config.get("output_dir", "output/keyarea_downsampling_analysis")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_root / "episodes" / f"{args.task}__{args.episode}"
    )
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_frame_scores, summaries = [], []
    baseline_frames: Dict[str, list[Path]] = {}
    source_score_lookups: Dict[str, Dict[str, Dict[int, dict]]] = {
        "object": {},
        "robot_arm": {},
    }
    evaluated_frame_sets = []
    gt_episode_dir = Path(args.gt_root) / args.task / args.episode
    for baseline, root_value in config["baselines"].items():
        root = Path(root_value)
        episode_dir = root / args.task / args.episode
        result_path = episode_dir / "keyarea_visual_quality.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result for {baseline}: {result_path}")
        frame_scores, summary, source_lookup = load_episode_frames(
            result_path,
            baseline,
            args.max_frames,
            args.fps,
            normalization,
            psnr_inf_cap,
        )
        evaluated_frame_sets.append(set(frame_scores["frame"].astype(int)))
        baseline_frames[baseline] = frame_files(episode_dir, args.view)
        for source in ("object", "robot_arm"):
            source_score_lookups[source][baseline] = source_lookup[source]
        summaries.append(summary)
        all_frame_scores.append(frame_scores)

    downsample_rates = {int(row["downsample_rate"]) for row in summaries}
    if len(downsample_rates) != 1:
        raise ValueError(
            f"Baselines use inconsistent downsample rates: {downsample_rates}"
        )
    downsample_rate = downsample_rates.pop()
    dilation_values = {int(row["mask_dilation_pixels"]) for row in summaries}
    if len(dilation_values) != 1:
        raise ValueError(
            f"Baselines use inconsistent mask dilation: {dilation_values}"
        )
    mask_dilation_pixels = dilation_values.pop()
    preprocessing_fields = (
        "crop_to_bbox",
        "bbox_pad",
        "lpips_min_crop_size",
        "mask_aware_psnr",
    )
    preprocessing = {}
    for field in preprocessing_fields:
        values = {row[field] for row in summaries}
        if len(values) != 1:
            raise ValueError(
                f"Baselines use inconsistent {field}: {values}"
            )
        preprocessing[field] = values.pop()
    frame_indices = sorted(set.intersection(*evaluated_frame_sets))
    frame_indices = [frame for frame in frame_indices if frame >= 0]
    if args.max_frames is not None:
        frame_indices = [
            frame for frame in frame_indices if frame < args.max_frames
        ]
    if not frame_indices:
        raise ValueError("No common evaluated frame indices across baselines")
    playback_fps = args.fps / downsample_rate

    gt_frames = frame_files(gt_episode_dir, args.view)
    streams: Dict[str, Sequence[Path]] = {"GT": gt_frames, **baseline_frames}
    timeline_frame_count = min(len(files) for files in streams.values())
    if args.max_frames is not None:
        timeline_frame_count = min(timeline_frame_count, args.max_frames)
    frame_indices = [
        frame for frame in frame_indices if frame < timeline_frame_count
    ]
    if not frame_indices:
        raise ValueError("No common evaluated frames exist in all video streams")
    video_paths = {}
    gt_video = videos_dir / "GT.mp4"
    print(f"[visualize] Writing GT video ({len(frame_indices)} frames)...", flush=True)
    gt_video_frames = write_video(
        gt_frames,
        gt_video,
        "GT",
        None,
        frame_indices,
        args.fps,
        playback_fps,
    )
    video_paths["GT"] = f"videos/{gt_video.name}"
    for summary in summaries:
        baseline = summary["baseline"]
        frame_scores = next(
            scores
            for scores in all_frame_scores
            if scores["baseline"].iloc[0] == baseline
        )
        output_video = videos_dir / f"{safe_stem(baseline)}.mp4"
        print(f"[visualize] Writing {baseline} video...", flush=True)
        video_frames = write_video(
            baseline_frames[baseline],
            output_video,
            baseline,
            frame_scores,
            frame_indices,
            args.fps,
            playback_fps,
        )
        summary["video_frames"] = video_frames
        summary["video_duration_sec"] = video_frames / playback_fps
        video_paths[baseline] = f"videos/{output_video.name}"

    object_mask_path = gt_episode_dir / "prompt" / "masks.npz"
    if not object_mask_path.is_file():
        object_mask_path = gt_episode_dir / "prompt" / "masks.npy"
    mask_paths = {
        "object": object_mask_path,
        "robot_arm": (
            gt_episode_dir
            / "multiview_masks"
            / args.view
            / "masks.npz"
        ),
    }
    comparison_paths = {}
    for source, mask_path in mask_paths.items():
        masks = load_mask_array(mask_path)
        output_video = videos_dir / f"{source}_mask_comparison.mp4"
        print(
            f"[visualize] Writing {source} mask comparison from JSON scores...",
            flush=True,
        )
        write_mask_comparison_video(
            streams=streams,
            masks=masks,
            source=source,
            score_lookups=source_score_lookups[source],
            frame_indices=frame_indices,
            output_path=output_video,
            source_fps=args.fps,
            playback_fps=playback_fps,
            dilation_pixels=mask_dilation_pixels,
            crop_to_bbox=bool(preprocessing["crop_to_bbox"]),
            bbox_pad=int(preprocessing["bbox_pad"]),
            min_crop_size=int(preprocessing["lpips_min_crop_size"]),
        )
        comparison_paths[f"{source} mask comparison"] = (
            f"videos/{output_video.name}"
        )
    video_paths = {**comparison_paths, **video_paths}

    frame_scores = pd.concat(all_frame_scores, ignore_index=True)
    print("[visualize] Writing plots and dashboard...", flush=True)
    summary_frame = pd.DataFrame(summaries)
    correlations = pairwise_temporal_correlations(frame_scores)
    frame_scores.to_csv(output_dir / "frame_scores.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    correlations.to_csv(output_dir / "temporal_correlations.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "task": args.task,
                "episode": args.episode,
                "fps": args.fps,
                "downsample_rate": downsample_rate,
                "playback_fps": playback_fps,
                "max_frames": args.max_frames,
                "timeline_frame_count": timeline_frame_count,
                "evaluated_frame_indices": frame_indices,
                "mask_dilation_pixels": mask_dilation_pixels,
                "preprocessing": preprocessing,
                "mask_paths": {
                    source: str(path.resolve())
                    for source, path in mask_paths.items()
                },
                "normalization": normalization,
                "psnr_inf_cap": psnr_inf_cap,
                "gt_root": str(Path(args.gt_root).resolve()),
                "gt_video_frames": gt_video_frames,
                "summaries": summaries,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_summary_plot(summary_frame, plots_dir / "baseline_summary.png")
    save_timeseries_plot(
        frame_scores,
        RAW_METRICS,
        plots_dir / "raw_timeseries.png",
        f"{args.task}/{args.episode}: raw KeyArea metrics",
    )
    save_timeseries_plot(
        frame_scores,
        SCORE_METRICS,
        plots_dir / "score_timeseries.png",
        f"{args.task}/{args.episode}: normalized KeyArea scores",
    )
    save_correlation_heatmaps(
        correlations,
        plots_dir / "temporal_correlations.png",
    )
    render_dashboard(
        output_dir / "index.html",
        args.task,
        args.episode,
        args.fps,
        timeline_frame_count,
        downsample_rate,
        playback_fps,
        summaries,
        video_paths,
    )
    print(f"Saved episode comparison dashboard to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
