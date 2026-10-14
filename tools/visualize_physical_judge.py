#!/usr/bin/env python3
"""Build an auditable HTML report from per-episode Physical Judge results."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every keyframe and judgment in physical_judge.json."
    )
    parser.add_argument(
        "--config_path",
        default="configs/base.yaml",
        help="Model override config; configs/base.yaml is merged first.",
    )
    parser.add_argument(
        "--base_config",
        default="configs/base.yaml",
        help="Base benchmark configuration.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data.val_base containing per-episode result files.",
    )
    parser.add_argument(
        "--gt-root",
        default=None,
        help="Override data.gt_path for optional GT keyframe comparison.",
    )
    parser.add_argument("--task", default=None, help="Optional task filter.")
    parser.add_argument("--episode", default=None, help="Optional episode filter.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <save_path>/physical_judge_visualization.",
    )
    parser.add_argument("--max-image-long-side", type=int, default=960)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_merged_config(base_path: str | Path, override_path: str | Path) -> dict:
    base = load_yaml(base_path)
    if Path(base_path).resolve() == Path(override_path).resolve():
        return base
    return deep_merge(base, load_yaml(override_path))


def safe_stem(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return result or "item"


def discover_results(
    data_root: Path,
    task_filter: str | None,
    episode_filter: str | None,
) -> list[tuple[str, str, Path]]:
    results = []
    for path in sorted(data_root.glob("*/*/physical_judge.json")):
        relative = path.relative_to(data_root)
        task_id, episode_id = relative.parts[:2]
        if task_filter and task_id != task_filter:
            continue
        if episode_filter and episode_id != episode_filter:
            continue
        results.append((task_id, episode_id, path))
    if not results:
        raise FileNotFoundError(
            f"No matching physical_judge.json files under {data_root}"
        )
    return results


def frame_files(directory: str | Path | None) -> list[Path]:
    if not directory:
        return []
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_generated_frames(
    data_root: Path,
    task_id: str,
    episode_id: str,
    trial: Mapping[str, Any],
) -> list[Path]:
    stored = frame_files(trial.get("video_dir"))
    if stored:
        return stored
    view = str(trial.get("eval_view", "top_head"))
    episode_dir = data_root / task_id / episode_id
    for candidate in (
        episode_dir / "video" / view,
        episode_dir / "video",
    ):
        files = frame_files(candidate)
        if files:
            return files
    return []


def resolve_gt_frames(
    gt_root: Path | None,
    task_id: str,
    episode_id: str,
    trial: Mapping[str, Any],
) -> list[Path]:
    if gt_root is None:
        return []
    view = str(trial.get("eval_view", "top_head"))
    for candidate in (
        gt_root / task_id / episode_id / "video" / view,
        gt_root / task_id / episode_id / "video",
    ):
        files = frame_files(candidate)
        if files:
            return files
    return []


def copy_preview(source: Path, destination: Path, max_long_side: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=90, optimize=True)


def image_at(files: Sequence[Path], index: object) -> Path | None:
    try:
        numeric = int(index)
    except (TypeError, ValueError):
        return None
    return files[numeric] if 0 <= numeric < len(files) else None


def criterion_records(step: Mapping[str, Any]) -> list[dict]:
    constraints = step.get("constraints")
    if isinstance(constraints, list) and constraints:
        return [record for record in constraints if isinstance(record, dict)]
    criteria = step.get("success_criteria")
    if isinstance(criteria, str):
        criteria = [criteria]
    if not isinstance(criteria, list):
        criteria = []
    scores = step.get("aspect_scores")
    if not isinstance(scores, list):
        scores = []
    reasoning = step.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = []
    return [
        {
            "name": f"criterion_{index + 1}",
            "spec": str(criterion),
            "score": scores[index] if index < len(scores) else None,
            "satisfied": (
                bool(scores[index]) if index < len(scores) else None
            ),
            "evidence": str(reasoning[index]) if index < len(reasoning) else "",
        }
        for index, criterion in enumerate(criteria)
    ]


def score_number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def pass_label(value: object) -> tuple[str, str]:
    if value in (True, 1, 1.0, "1"):
        return "PASS", "pass"
    if value in (False, 0, 0.0, "0"):
        return "FAIL", "fail"
    return "N/A", "unknown"


def render_step(
    step: Mapping[str, Any],
    step_index: int,
    generated_image: str | None,
    gt_image: str | None,
) -> str:
    score = score_number(step.get("score"))
    satisfied = step.get("satisfied")
    label, css_class = pass_label(satisfied)
    frame = step.get("frame", "?")
    source_frame = step.get("source_frame", frame)
    segment = step.get("segment_id", step_index)
    criteria = criterion_records(step)
    criterion_rows = []
    for index, record in enumerate(criteria):
        criterion_label, criterion_class = pass_label(
            record.get("score", record.get("satisfied"))
        )
        criterion_rows.append(
            "<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{html.escape(str(record.get('spec', record.get('name', ''))))}</td>"
            f"<td><span class='badge {criterion_class}'>{criterion_label}</span></td>"
            f"<td>{html.escape(str(record.get('evidence', '')))}</td>"
            "</tr>"
        )
    images = []
    if generated_image:
        images.append(
            f"<figure><img src='{html.escape(generated_image)}' "
            "alt='Generated keyframe'><figcaption>Generated frame "
            f"{html.escape(str(frame))}</figcaption></figure>"
        )
    if gt_image:
        images.append(
            f"<figure><img src='{html.escape(gt_image)}' alt='GT keyframe'>"
            f"<figcaption>GT source frame {html.escape(str(source_frame))}</figcaption></figure>"
        )
    raw = step.get("raw", "")
    raw_block = (
        "<details><summary>Raw VLM response</summary>"
        f"<pre>{html.escape(str(raw))}</pre></details>"
        if raw
        else ""
    )
    warning = step.get("warning")
    warning_block = (
        f"<p class='warning'>{html.escape(str(warning))}</p>" if warning else ""
    )
    format_valid = step.get("format_valid")
    return f"""
<section class="step {css_class}">
  <div class="step-title">
    <h2>Segment {html.escape(str(segment))} · frame {html.escape(str(frame))}</h2>
    <span class="badge {css_class}">{label}</span>
    <span>score={score:.3f}</span>
  </div>
  <p class="meta">GT source frame: {html.escape(str(source_frame))} ·
    final: {bool(step.get("is_final"))} · format valid: {html.escape(str(format_valid))} ·
    format retries: {html.escape(str(step.get("format_retry_count", 0)))}</p>
  <div class="images">{''.join(images) if images else '<p>Keyframe image unavailable.</p>'}</div>
  {warning_block}
  <p><strong>Explanation:</strong> {html.escape(str(step.get("explanation", "")))}</p>
  <table>
    <thead><tr><th>#</th><th>Success criterion</th><th>Result</th><th>Evidence</th></tr></thead>
    <tbody>{''.join(criterion_rows) if criterion_rows else '<tr><td colspan="4">No structured criteria.</td></tr>'}</tbody>
  </table>
  {raw_block}
</section>"""


def page_styles() -> str:
    return """
:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
body { max-width: 1500px; margin: 0 auto; padding: 24px; background: #111; color: #eee; }
a { color: #80bfff; }
.meta { color: #aaa; }
.summary { display: flex; gap: 20px; flex-wrap: wrap; margin: 18px 0; }
.stat { background: #1c1c1c; border: 1px solid #333; padding: 12px 16px; min-width: 150px; }
.step { border: 1px solid #333; border-left-width: 5px; margin: 20px 0; padding: 16px; background: #181818; }
.step.pass { border-left-color: #4cae6b; }
.step.fail { border-left-color: #db6464; }
.step.unknown { border-left-color: #888; }
.step-title { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.step-title h2 { margin: 0; }
.badge { border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; }
.badge.pass { background: #245d36; }
.badge.fail { background: #702d2d; }
.badge.unknown { background: #444; }
.images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; }
img { width: 100%; max-height: 600px; object-fit: contain; background: #000; }
figcaption { color: #aaa; padding-top: 5px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #333; vertical-align: top; }
pre { white-space: pre-wrap; word-break: break-word; background: #0d0d0d; padding: 12px; }
.warning { color: #ffca70; }
.overview { width: 100%; max-height: none; background: white; }
@media (max-width: 800px) { .images { grid-template-columns: 1fr; } }
"""


def render_trial_page(
    output_path: Path,
    task_id: str,
    episode_id: str,
    trial_id: str,
    trial: Mapping[str, Any],
    generated_files: Sequence[Path],
    gt_files: Sequence[Path],
    assets_dir: Path,
    max_long_side: int,
) -> tuple[dict, list[dict]]:
    steps = trial.get("key_steps")
    if not isinstance(steps, list):
        steps = []
    step_sections = []
    criterion_rows = []
    format_failures = 0
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        frame = step.get("frame")
        source_frame = step.get("source_frame", frame)
        generated_source = image_at(generated_files, frame)
        gt_source = image_at(gt_files, source_frame)
        generated_relative = None
        gt_relative = None
        prefix = f"{safe_stem(trial_id)}_step{step_index:03d}"
        if generated_source:
            destination = assets_dir / f"{prefix}_pred.jpg"
            copy_preview(generated_source, destination, max_long_side)
            generated_relative = f"assets/{destination.name}"
        if gt_source:
            destination = assets_dir / f"{prefix}_gt.jpg"
            copy_preview(gt_source, destination, max_long_side)
            gt_relative = f"assets/{destination.name}"
        step_sections.append(
            render_step(
                step,
                step_index,
                generated_relative,
                gt_relative,
            )
        )
        if step.get("format_valid") is False:
            format_failures += 1
        for criterion_index, record in enumerate(criterion_records(step)):
            raw_score = record.get("score", record.get("satisfied"))
            numeric = (
                int(bool(raw_score))
                if raw_score in (True, False, 0, 1, 0.0, 1.0)
                else None
            )
            criterion_rows.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "trial_id": trial_id,
                    "segment_id": step.get("segment_id", step_index),
                    "frame": frame,
                    "source_frame": source_frame,
                    "criterion_index": criterion_index + 1,
                    "criterion": record.get("spec", record.get("name", "")),
                    "score": numeric,
                    "evidence": record.get("evidence", ""),
                    "format_valid": step.get("format_valid"),
                }
            )

    score = score_number(trial.get("score"))
    status = str(trial.get("status", "legacy"))
    n_satisfied = sum(row["score"] == 1 for row in criterion_rows)
    n_criteria = sum(row["score"] is not None for row in criterion_rows)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(task_id)} / {html.escape(episode_id)} / {html.escape(trial_id)}</title>
<style>{page_styles()}</style></head><body>
<p><a href="../../index.html">← Dataset overview</a></p>
<h1>{html.escape(task_id)} / {html.escape(episode_id)} / {html.escape(trial_id)}</h1>
<p>{html.escape(str(trial.get("instruction", "")))}</p>
<div class="summary">
  <div class="stat"><strong>Trial score</strong><br>{score:.3f}</div>
  <div class="stat"><strong>Criteria satisfied</strong><br>{n_satisfied}/{n_criteria}</div>
  <div class="stat"><strong>Keyframes</strong><br>{len(steps)}</div>
  <div class="stat"><strong>Status</strong><br>{html.escape(status)}</div>
  <div class="stat"><strong>Format failures</strong><br>{format_failures}</div>
</div>
{''.join(step_sections) if step_sections else '<p>No key-step results.</p>'}
</body></html>"""
    output_path.write_text(document, encoding="utf-8")
    summary = {
        "task_id": task_id,
        "episode_id": episode_id,
        "trial_id": trial_id,
        "score": score,
        "status": status,
        "num_keyframes": len(steps),
        "num_criteria": n_criteria,
        "num_satisfied": n_satisfied,
        "criterion_accuracy": n_satisfied / n_criteria if n_criteria else float("nan"),
        "format_failures": format_failures,
        "page": f"episodes/{output_path.parent.name}/{output_path.name}",
    }
    return summary, criterion_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    valid = [row for row in rows if math.isfinite(score_number(row["score"]))]
    task_scores: Dict[str, list[float]] = defaultdict(list)
    task_accuracies: Dict[str, list[float]] = defaultdict(list)
    for row in valid:
        task_scores[str(row["task_id"])].append(float(row["score"]))
        accuracy = score_number(row["criterion_accuracy"])
        if math.isfinite(accuracy):
            task_accuracies[str(row["task_id"])].append(accuracy)
    tasks = sorted(task_scores)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(tasks, [np.mean(task_scores[task]) for task in tasks])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean Physical Judge score")
    axes[0].set_ylabel("Score (0–1)")
    axes[1].bar(
        tasks,
        [
            np.mean(task_accuracies[task])
            if task_accuracies[task]
            else float("nan")
            for task in tasks
        ],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Criterion satisfaction rate")
    axes[1].set_ylabel("Satisfied fraction")
    axes[2].hist([float(row["score"]) for row in valid], bins=10, range=(0, 1))
    axes[2].set_title("Trial-score distribution")
    axes[2].set_xlabel("Physical Judge score")
    axes[2].set_ylabel("Number of trials")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_index(
    output_path: Path,
    data_root: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    rows = []
    for summary in sorted(
        summaries,
        key=lambda row: (str(row["task_id"]), str(row["episode_id"]), str(row["trial_id"])),
    ):
        score = score_number(summary["score"])
        accuracy = score_number(summary["criterion_accuracy"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(summary['task_id']))}</td>"
            f"<td>{html.escape(str(summary['episode_id']))}</td>"
            f"<td>{html.escape(str(summary['trial_id']))}</td>"
            f"<td>{score:.3f}</td>"
            f"<td>{accuracy:.3f}</td>"
            f"<td>{summary['num_keyframes']}</td>"
            f"<td>{summary['format_failures']}</td>"
            f"<td><a href='{html.escape(str(summary['page']))}'>Inspect</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Physical Judge visualization</title><style>{page_styles()}</style></head><body>
<h1>Physical Judge visualization</h1>
<p class="meta">Source: {html.escape(str(data_root))} · {len(summaries)} trials</p>
<img class="overview" src="overview.png" alt="Physical Judge overview">
<table><thead><tr><th>Task</th><th>Episode</th><th>Trial</th><th>Score</th>
<th>Criterion accuracy</th><th>Keyframes</th><th>Format failures</th><th>Page</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.base_config, args.config_path)
    data_cfg = config.get("data") or {}
    data_root = Path(args.data_root or data_cfg.get("val_base", ""))
    if not data_root.is_dir():
        raise FileNotFoundError(f"Prediction data root does not exist: {data_root}")
    gt_value = args.gt_root or data_cfg.get("gt_path")
    gt_root = Path(gt_value) if gt_value else None
    if gt_root is not None and not gt_root.is_dir():
        raise FileNotFoundError(f"GT data root does not exist: {gt_root}")
    output_dir = Path(
        args.output_dir
        or Path(config.get("save_path", "output")) / "physical_judge_visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_root = output_dir / "episodes"
    episode_root.mkdir(parents=True, exist_ok=True)

    summaries, criteria = [], []
    load_errors = []
    for task_id, episode_id, result_path in discover_results(
        data_root,
        args.task,
        args.episode,
    ):
        try:
            with open(result_path, "r", encoding="utf-8") as file:
                episode_result = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            load_errors.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "path": str(result_path),
                    "error": str(error),
                }
            )
            continue
        if not isinstance(episode_result, dict):
            continue
        page_dir = episode_root / f"{safe_stem(task_id)}__{safe_stem(episode_id)}"
        assets_dir = page_dir / "assets"
        page_dir.mkdir(parents=True, exist_ok=True)
        for trial_id, trial in episode_result.items():
            if not isinstance(trial, dict) or "key_steps" not in trial:
                continue
            generated = resolve_generated_frames(
                data_root,
                task_id,
                episode_id,
                trial,
            )
            gt = resolve_gt_frames(gt_root, task_id, episode_id, trial)
            page_path = page_dir / f"{safe_stem(str(trial_id))}.html"
            summary, trial_criteria = render_trial_page(
                page_path,
                task_id,
                episode_id,
                str(trial_id),
                trial,
                generated,
                gt,
                assets_dir,
                args.max_image_long_side,
            )
            summaries.append(summary)
            criteria.extend(trial_criteria)

    if not summaries:
        raise ValueError("No valid Physical Judge trial results were found")
    write_csv(output_dir / "trial_summary.csv", summaries)
    write_csv(output_dir / "criterion_results.csv", criteria)
    write_csv(output_dir / "load_errors.csv", load_errors)
    plot_overview(summaries, output_dir / "overview.png")
    render_index(output_dir / "index.html", data_root, summaries)
    print(f"Saved Physical Judge visualization to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
