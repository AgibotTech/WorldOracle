#!/usr/bin/env python3
"""Visualize DINOv3 Multi-View Consistency results offline.

This tool never runs DINOv3. It reads completed ``multiview_consistency.json``
files, renders stored robot-arm correspondences for all three views, and plots
the unconditional three-view averages used by the evaluator.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml


VIEWS = ("top_head", "hand_left", "hand_right")
COLORS = (
    (255, 82, 82),
    (64, 196, 255),
    (255, 196, 64),
    (156, 110, 255),
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/multiview_consistency.yaml",
        help="YAML containing a baselines name-to-root mapping.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--display-width", type=int, default=480)
    parser.add_argument(
        "--max-correspondences",
        type=int,
        default=40,
        help="Maximum match lines shown per mask/view.",
    )
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=10,
        help="Save one standalone PNG every N evaluated frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional limit; default visualizes the complete evaluated episode.",
    )
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError(f"{path}: baselines must be a non-empty mapping")
    return config


def _frame_number(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def discover_frames(episode_dir: Path, view: str) -> dict[int, Path]:
    for directory in (
        episode_dir / "video" / view,
        episode_dir / "video",
        episode_dir / view,
        episode_dir,
    ):
        if not directory.is_dir():
            continue
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            continue
        result = {}
        for ordinal, path in enumerate(files):
            frame = _frame_number(path)
            result[ordinal if frame is None else frame] = path
        return result
    raise FileNotFoundError(
        f"No frames for view={view!r} under {episode_dir}"
    )


def load_masks(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        key = "masks" if "masks" in archive else list(archive.keys())[0]
        masks = np.asarray(archive[key])
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4:
        raise ValueError(f"{path}: expected M,T,H,W masks, got {masks.shape}")
    return masks


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if int(result.get("metric_version", -1)) < 5:
        raise ValueError(
            f"{path}: metric_version >= 5 is required for cross-view results"
        )
    if not isinstance(result.get("per_view"), dict):
        raise ValueError(f"{path}: missing per_view records")
    return result


def finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def average_view_scores(
    scores: Mapping[str, float | None],
    views: Sequence[str] = VIEWS,
) -> float:
    values = [finite(scores.get(view)) for view in views]
    return float(
        np.mean([value if value is not None else 0.0 for value in values])
    )


def result_lookups(result: Mapping[str, Any]) -> tuple[dict, dict]:
    per_view = {
        view: {
            int(record["frame"]): record
            for record in (
                (result.get("per_view") or {}).get(view, {}).get(
                    "per_frame",
                    [],
                )
            )
        }
        for view in VIEWS
    }
    combined = {
        int(record["frame"]): record
        for record in result.get("per_frame", [])
    }
    return per_view, combined


def frame_table(
    baseline: str,
    result: Mapping[str, Any],
) -> pd.DataFrame:
    per_view, combined = result_lookups(result)
    frames = sorted(
        set(combined)
        | set().union(*(set(records) for records in per_view.values()))
    )
    rows = []
    for frame in frames:
        scores = {
            view: finite(
                (per_view[view].get(frame) or {}).get("alignment_score")
            )
            for view in VIEWS
        }
        stored = combined.get(frame) or {}
        rows.append(
            {
                "baseline": baseline,
                "frame": frame,
                "time_sec": finite(stored.get("time_sec")),
                **{
                    f"{view}_alignment_score": scores[view]
                    for view in VIEWS
                },
                "min_view_alignment_score": (
                    min(value for value in scores.values() if value is not None)
                    if all(value is not None for value in scores.values())
                    else np.nan
                ),
                "mean_view_alignment_score": (
                    float(np.mean(list(scores.values())))
                    if all(value is not None for value in scores.values())
                    else np.nan
                ),
                "stored_multiview_consistency_score": finite(
                    stored.get("multiview_consistency_score")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_scores(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline, group in frames.groupby("baseline", sort=False):
        summary = {
            "baseline": baseline,
            "num_frames": len(group),
            "multiview_consistency_score": float(
                group["stored_multiview_consistency_score"].mean()
            ),
        }
        for view in VIEWS:
            summary[f"{view}_mean_alignment"] = float(
                group[f"{view}_alignment_score"].mean()
            )
        rows.append(summary)
    return pd.DataFrame(rows)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def read_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def overlay_masks(
    image: Image.Image,
    masks: np.ndarray,
    frame: int,
) -> Image.Image:
    array = np.asarray(image, dtype=np.float32).copy()
    for mask_index in range(masks.shape[0]):
        if frame >= masks.shape[1]:
            continue
        mask = masks[mask_index, frame] > 0
        if not mask.any():
            continue
        if mask.shape != array.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                image.size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color = np.asarray(
            COLORS[mask_index % len(COLORS)],
            dtype=np.float32,
        )
        array[mask] = 0.58 * array[mask] + 0.42 * color
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _coordinates(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        gt = np.asarray(record.get("match_gt_xy", []), np.float32).reshape(
            -1,
            2,
        )
        generated = np.asarray(
            record.get("match_generated_xy", []),
            np.float32,
        ).reshape(-1, 2)
        cosine = np.asarray(record.get("match_cosine", []), np.float32)
    except (TypeError, ValueError):
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            np.empty(0, np.float32),
        )
    count = min(len(gt), len(generated), len(cosine))
    return gt[:count], generated[:count], cosine[:count]


def draw_view_row(
    *,
    gt_path: Path,
    generated_path: Path,
    masks: np.ndarray,
    frame: int,
    view: str,
    view_record: Mapping[str, Any] | None,
    display_width: int,
    max_correspondences: int,
) -> Image.Image:
    height, width = masks.shape[-2:]
    display_height = max(1, round(display_width * height / width))
    size = (display_width, display_height)
    gt = overlay_masks(read_rgb(gt_path, size), masks, frame)
    generated = read_rgb(generated_path, size)
    label_height = 38
    gap = 18
    canvas = Image.new(
        "RGB",
        (2 * display_width + gap, display_height + label_height),
        "white",
    )
    canvas.paste(gt, (0, label_height))
    canvas.paste(generated, (display_width + gap, label_height))
    draw = ImageDraw.Draw(canvas, "RGBA")
    score = finite((view_record or {}).get("alignment_score"))
    draw.text(
        (8, 8),
        (
            f"{view} | alignment="
            f"{score:.1f}" if score is not None else f"{view} | unavailable"
        ),
        fill=(20, 140, 55) if score is not None else (210, 45, 45),
        font=_font(18),
    )
    draw.text(
        (display_width + gap + 8, 8),
        "Generated DINO matches",
        fill=(25, 25, 25),
        font=_font(16),
    )
    scale_x = display_width / width
    scale_y = display_height / height
    for mask_index, record in enumerate(
        (view_record or {}).get("masks", [])
    ):
        gt_xy, generated_xy, cosine = _coordinates(record)
        if not len(cosine):
            continue
        order = np.argsort(cosine)[::-1][:max_correspondences]
        color = COLORS[mask_index % len(COLORS)]
        for index in order:
            gx = float(gt_xy[index, 0] * scale_x)
            gy = float(gt_xy[index, 1] * scale_y + label_height)
            px = float(
                generated_xy[index, 0] * scale_x
                + display_width
                + gap
            )
            py = float(
                generated_xy[index, 1] * scale_y + label_height
            )
            alpha = int(
                50 + 190 * np.clip((cosine[index] + 1.0) / 2.0, 0, 1)
            )
            draw.line((gx, gy, px, py), fill=(*color, alpha), width=1)
            radius = 2
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=(*color, 245),
            )
    return canvas


def make_panel(
    *,
    baseline: str,
    frame: int,
    per_view: Mapping[str, Mapping[int, Mapping[str, Any]]],
    gt_frames: Mapping[str, Mapping[int, Path]],
    generated_frames: Mapping[str, Mapping[int, Path]],
    masks: Mapping[str, np.ndarray],
    display_width: int,
    max_correspondences: int,
) -> Image.Image:
    records = {view: per_view[view].get(frame) for view in VIEWS}
    score = average_view_scores(
        {
            view: finite((records[view] or {}).get("alignment_score"))
            for view in VIEWS
        },
    )
    rows = [
        draw_view_row(
            gt_path=gt_frames[view][frame],
            generated_path=generated_frames[view][frame],
            masks=masks[view],
            frame=frame,
            view=view,
            view_record=records[view],
            display_width=display_width,
            max_correspondences=max_correspondences,
        )
        for view in VIEWS
    ]
    header_height = 54
    width = max(row.width for row in rows)
    panel = Image.new(
        "RGB",
        (width, header_height + sum(row.height for row in rows)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (10, 7),
        f"{baseline} | frame={frame} | MultiView mean={score:.1f}",
        fill=(20, 80, 160),
        font=_font(20),
    )
    y = header_height
    for row in rows:
        panel.paste(row, (0, y))
        y += row.height
    return panel


def write_video(
    panels: Iterable[Image.Image],
    path: Path,
    fps: float,
) -> None:
    writer = None
    try:
        for panel in panels:
            rgb = np.asarray(panel, dtype=np.uint8)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (rgb.shape[1], rgb.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()


def save_plots(
    frames: pd.DataFrame,
    output: Path,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    made = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, view in zip(axes, VIEWS):
        values = [
            group[f"{view}_alignment_score"].dropna().to_numpy()
            for _, group in frames.groupby("baseline", sort=False)
        ]
        labels = [
            baseline
            for baseline, _ in frames.groupby("baseline", sort=False)
        ]
        axis.boxplot(values, labels=labels, showfliers=False)
        axis.set_title(view)
        axis.set_ylabel("DINO alignment score")
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Per-view robot-arm alignment distributions")
    fig.tight_layout()
    fig.savefig(output / "view_score_distributions.png", dpi=160)
    plt.close(fig)
    made.append("view_score_distributions.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for baseline, group in frames.groupby("baseline", sort=False):
        scores = group["stored_multiview_consistency_score"]
        ax.plot(group.time_sec, scores, label=baseline, alpha=0.85)
    ax.set(
        xlabel="Time (s)",
        ylabel="Three-view mean MultiView score",
        ylim=(0, 100),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "temporal_multiview_score.png", dpi=160)
    plt.close(fig)
    made.append("temporal_multiview_score.png")
    return made


def write_html(
    output: Path,
    summary: pd.DataFrame,
    plots: Sequence[str],
    videos: Sequence[tuple[str, Path]],
    stills: Sequence[tuple[str, Path]],
) -> None:
    summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.baseline))}</td>"
        f"<td>{int(row.num_frames)}</td>"
        f"<td>{row.multiview_consistency_score:.2f}</td>"
        f"<td>{row.top_head_mean_alignment:.2f}</td>"
        f"<td>{row.hand_left_mean_alignment:.2f}</td>"
        f"<td>{row.hand_right_mean_alignment:.2f}</td>"
        "</tr>"
        for row in summary.itertuples()
    )
    plot_html = "".join(
        f'<a href="plots/{html.escape(name)}">'
        f'<img src="plots/{html.escape(name)}"></a>'
        for name in plots
    )
    video_html = "".join(
        f"<h3>{html.escape(baseline)}</h3>"
        f'<video controls preload="metadata" src="{path.relative_to(output)}"></video>'
        for baseline, path in videos
    )
    still_html = "".join(
        f'<a href="{path.relative_to(output)}">'
        f"{html.escape(baseline)} / {html.escape(path.stem)}</a> "
        for baseline, path in stills
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MultiView consistency analysis</title>
<style>
body{{font:14px system-ui;margin:24px;max-width:1500px}}
table{{border-collapse:collapse}} td,th{{padding:7px 10px;border:1px solid #ccc}}
img{{max-width:100%;margin:8px 0}} video{{width:100%;max-width:1100px}}
a{{margin-right:10px}}
</style></head><body>
<h1>MultiView consistency analysis</h1>
<p>Each frame score is the unconditional mean of top-head, left-hand, and
right-hand DINO alignment scores.</p>
<p><a href="frame_scores.csv">frame_scores.csv</a>
<a href="summary.csv">summary.csv</a></p>
<h2>Summary</h2><table>
<thead><tr><th>Baseline</th><th>Frames</th><th>MultiView score</th>
<th>Top</th><th>Left</th><th>Right</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
<h2>Plots</h2>{plot_html}
<h2>Videos</h2>{video_html}
<h2>Sampled full-resolution panels</h2><p>{still_html}</p>
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.display_width <= 0:
        raise ValueError("--fps and --display-width must be positive")
    config = load_config(Path(args.config))
    gt_episode = Path(args.gt_root) / args.task / args.episode
    masks = {
        view: load_masks(
            gt_episode / "multiview_masks" / view / "masks.npz"
        )
        for view in VIEWS
    }
    gt_frames = {
        view: discover_frames(gt_episode, view) for view in VIEWS
    }
    loaded = {}
    frame_tables = []
    for baseline, root in config["baselines"].items():
        episode_dir = Path(root) / args.task / args.episode
        result_path = episode_dir / "multiview_consistency.json"
        result = load_result(result_path)
        per_view, combined = result_lookups(result)
        loaded[baseline] = {
            "episode_dir": episode_dir,
            "result": result,
            "per_view": per_view,
            "combined": combined,
            "frames": {
                view: discover_frames(episode_dir, view)
                for view in VIEWS
            },
        }
        frame_tables.append(frame_table(baseline, result))
    frames = pd.concat(frame_tables, ignore_index=True)
    if frames.empty:
        raise ValueError("No MultiView frame records found")

    output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            config.get(
                "output_dir",
                "output/multiview_consistency_analysis",
            )
        )
        / f"{args.task}__{args.episode}"
    )
    for directory in (
        output,
        output / "plots",
        output / "videos",
        output / "panels",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    summary = summarize_scores(frames)
    frames.to_csv(output / "frame_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    plots = save_plots(frames, output / "plots")

    videos = []
    stills = []
    for baseline, data in loaded.items():
        frame_ids = sorted(
            set(data["combined"]).intersection(
                *(
                    set(data["frames"][view]) & set(gt_frames[view])
                    for view in VIEWS
                )
            )
        )
        if args.max_frames is not None:
            frame_ids = frame_ids[: max(0, args.max_frames)]
        if not frame_ids:
            continue

        def panels(
            frame_ids=frame_ids,
            baseline=baseline,
            data=data,
        ):
            for ordinal, frame in enumerate(frame_ids):
                panel = make_panel(
                    baseline=baseline,
                    frame=frame,
                    per_view=data["per_view"],
                    gt_frames=gt_frames,
                    generated_frames=data["frames"],
                    masks=masks,
                    display_width=args.display_width,
                    max_correspondences=args.max_correspondences,
                )
                if ordinal % max(1, args.visualize_every) == 0:
                    path = (
                        output
                        / "panels"
                        / f"{baseline}_frame_{frame:06d}.png"
                    )
                    panel.save(path)
                    stills.append((baseline, path))
                yield panel

        if args.no_video:
            list(panels())
        else:
            evaluation_settings = (
                data["result"].get("evaluation_settings") or {}
            )
            timing_settings = evaluation_settings.get(
                "interaction_evaluation_settings",
                evaluation_settings,
            )
            stride = int(
                timing_settings.get("downsample_rate", 1)
            )
            video_fps = (
                args.video_fps
                if args.video_fps is not None
                else max(1.0, args.fps / max(1, stride))
            )
            video_path = output / "videos" / f"{baseline}.mp4"
            write_video(panels(), video_path, video_fps)
            videos.append((baseline, video_path))

    write_html(
        output,
        summary,
        plots,
        videos,
        stills,
    )
    print(f"MultiView dashboard: {(output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
