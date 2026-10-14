#!/usr/bin/env python3
"""Threshold-finding experiment for the Multi-View Consistency metrics.

Idea
----
Each segment ``key_frame`` in ``prompt/prompts.json`` is used as an ANCHOR,
treated as the "ground-truth" frame. We then take frames at increasing temporal
offsets (+5, +10, +20, ..., +100) as synthetic "wrong" predictions and measure
masked SSIM / PSNR / LPIPS / V-JEPA similarity between the anchor frame and the
offset frame, restricted to each hand key-area mask AT THE ANCHOR FRAME.

The further the offset, the more the scene has drifted, so the metric-vs-offset
curve tells us how each metric behaves when consistency is good (small offset)
vs bad (large offset) — exactly what we need to pick a threshold.

Conventions
-----------
- masks.npz has key ``masks`` of shape (M, T, H, W); index 0 = LEFT hand key
  area, index 1 = RIGHT hand key area. The two masks are kept SEPARATE and are
  never averaged.
- All three views (top_head, hand_left, hand_right) are processed.
- The same metric code as the benchmark is reused (acwmbench.metrics.
  multiview_consistency), so the numbers match the real evaluation exactly.

Outputs (under --output_dir)
----------------------------
- raw_metrics.csv              : one row per (view, mask, key_frame, offset)
- summary_by_offset.csv        : mean / std per (view, mask, offset)
- trends.png                   : metric value vs offset (4 subplots, 6 lines each)
- qualitative/<view>_<mask>_kf<idx>.png : anchor (left) vs offset frames (right)

Example
-------
    python scripts/threshold_experiment_multiview.py \
        --config_path configs/base.yaml \
        --task pickfruit --episode ep000001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from acwmbench.io_utils import list_frame_files, load_frames_as_array  # noqa: E402
from acwmbench.metrics.keyarea_visual_quality import (  # noqa: E402
    _ensure_lpips,
    _ensure_torchmetrics,
)
from acwmbench.metrics.multiview_consistency import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    METRIC_KEYS,
    _build_metric_objs,
    _compute_mask_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("threshold_experiment")

# mask index -> human-readable name
MASK_NAMES = {0: "left", 1: "right"}

# Default temporal offsets (in frames) from the anchor key frame. 0 = the anchor
# itself (perfect-consistency baseline); large offsets approximate "wrong" frames.
# Extended out to 500 because the masked metrics saturate quickly (~30-40 frames),
# so we need bigger drifts to probe the "clearly inconsistent" end.
DEFAULT_OFFSETS = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]

DEFAULT_VIEWS = ["top_head", "hand_left", "hand_right"]

# Per-metric: True if a HIGHER value means more similar.
_HIGHER_BETTER = {"ssim": True, "psnr": True, "lpips": False, "vjepa_similarity": True}

# Plot styling.
_VIEW_COLORS = {"top_head": "#1f77b4", "hand_left": "#ff7f0e", "hand_right": "#2ca02c"}
_MASK_STYLES = {"left": "-", "right": "--"}
_MASK_MARKERS = {"left": "o", "right": "s"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config_path", default="configs/base.yaml",
                   help="YAML config (used for data.gt_path + vjepa model).")
    p.add_argument("--task", default="pickfruit", help="Subtask folder name.")
    p.add_argument("--episode", default="ep000001",
                   help="Episode folder name ('ep000001' or '000001').")
    p.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    p.add_argument("--offsets", nargs="+", type=int, default=DEFAULT_OFFSETS,
                   help="Frame offsets from each key frame.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write CSVs + figures "
                        "(default: output/threshold_experiment/<task>_<episode>).")
    p.add_argument("--gt_path", default=None,
                   help="Override data.gt_path from the config.")
    p.add_argument("--max_keyframes", type=int, default=None,
                   help="Only use the first N key frames (for quick runs).")
    p.add_argument("--viz_keyframes", type=int, default=3,
                   help="How many key frames to render qualitative strips for.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_vjepa", action="store_true",
                   help="Skip V-JEPA similarity (faster; SSIM/PSNR/LPIPS only).")
    p.add_argument(
        "--crop_to_bbox", action=argparse.BooleanOptionalAction, default=True,
        help="Crop both frames + mask to the mask bounding box before computing "
             "metrics (removes the shared black-background dilution; also feeds "
             "V-JEPA a real hand patch). Use --no-crop_to_bbox for the old "
             "full-frame zero-fill behavior.",
    )
    p.add_argument("--bbox_pad", type=int, default=10,
                   help="Padding (px) added around the mask bbox when cropping.")
    p.add_argument(
        "--mask_aware", action="store_true",
        help="Compute PSNR over the masked pixels ONLY (ignores the zero-filled "
             "background even inside the bbox). Affects PSNR; SSIM/LPIPS/V-JEPA "
             "still use the (cropped) masked patch.",
    )
    p.add_argument(
        "--neg_episodes", nargs="+", default=None,
        help="Other episode id(s) to use as cross-episode HARD NEGATIVES, e.g. "
             "--neg_episodes ep000000 ep000002. For each anchor mask we compare "
             "against sampled frames of these episodes (same view), giving a "
             "'definitely inconsistent' reference band.",
    )
    p.add_argument("--neg_frames", type=int, default=5,
                   help="How many frames to sample from each negative episode.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def norm_episode(ep: str) -> str:
    ep = str(ep).strip()
    return ep if ep.startswith("ep") else f"ep{ep}"


def read_keyframes(prompts_json: Path) -> List[int]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments") or []
    kfs: List[int] = []
    for seg in segs:
        kf = seg.get("key_frame", seg.get("end_frame"))
        if kf is not None:
            kfs.append(int(kf))
    # de-duplicate while preserving order
    seen = set()
    out = []
    for kf in kfs:
        if kf not in seen:
            seen.add(kf)
            out.append(kf)
    return out


def resize_to(img: np.ndarray, hw: tuple) -> np.ndarray:
    if img.shape[:2] == hw:
        return img
    return np.array(Image.fromarray(img).resize((hw[1], hw[0]), Image.BICUBIC))


def load_needed_frames(
    video_dir: Path, view: str, wanted: set, target_hw: tuple
) -> Dict[int, np.ndarray]:
    """Load only the requested frame indices for a view, resized to mask size.

    Returns {frame_index: (H, W, 3) uint8}. Out-of-range indices are dropped.
    """
    files = list_frame_files(video_dir, preferred_view=view)
    n = len(files)
    valid = sorted(i for i in wanted if 0 <= i < n)
    if not valid:
        return {}
    arr = load_frames_as_array(video_dir, indices=valid, preferred_view=view)
    return {idx: resize_to(arr[k], target_hw) for k, idx in enumerate(valid)}


def _masked_psnr(p: np.ndarray, g: np.ndarray, msk_bool: np.ndarray) -> float:
    """PSNR computed over masked pixels only (p, g in [0, 1], msk_bool (H, W))."""
    if msk_bool.sum() == 0:
        return float("nan")
    diff = (p[msk_bool] - g[msk_bool]).astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def masked_metrics(
    anchor: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    metric_objs: dict,
    *,
    crop_to_bbox: bool = True,
    bbox_pad: int = 10,
    mask_aware: bool = False,
) -> Dict[str, float]:
    """Masked SSIM/PSNR/LPIPS/V-JEPA between anchor (gt) and err (pred).

    crop_to_bbox: restrict to the mask bounding box first, so the shared black
        background no longer dominates the metrics (and V-JEPA sees a real patch).
    mask_aware: additionally compute PSNR over masked pixels only.
    """
    msk_bool = mask > 0
    a, e = anchor, err
    if crop_to_bbox:
        bbox = mask_bbox(mask, bbox_pad)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            a = anchor[y0:y1, x0:x1]
            e = err[y0:y1, x0:x1]
            msk_bool = msk_bool[y0:y1, x0:x1]

    g = a.astype(np.float32) / 255.0
    p = e.astype(np.float32) / 255.0
    mskf = msk_bool.astype(np.float32)[..., None]

    sims = _compute_mask_metrics(
        p * mskf,
        g * mskf,
        device=metric_objs["device"],
        ssim_metric=metric_objs["ssim"],
        psnr_metric=metric_objs["psnr"],
        lpips_fn=metric_objs["lpips"],
        jepa_model=metric_objs["jepa_model"],
        jepa_transform=metric_objs["jepa_transform"],
        use_jepa=metric_objs["use_jepa"],
    )
    if mask_aware:
        sims = dict(sims)
        sims["psnr"] = _masked_psnr(p, g, msk_bool)
    return sims


def mask_bbox(mask: np.ndarray, pad: int = 20) -> Optional[tuple]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad + 1)
    x1 = min(w, x1 + pad + 1)
    return y0, y1, x0, x1


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_trends(rows: List[dict], offsets: List[int], out_path: Path) -> None:
    """4 subplots (one per metric), x=offset, one line per (view, mask).

    Cross-episode hard negatives (kind == 'cross_episode') are drawn as a shaded
    horizontal band per (view, mask) to mark the 'clearly inconsistent' level.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()

    off_rows = [r for r in rows if r.get("kind", "offset") == "offset"]
    neg_rows = [r for r in rows if r.get("kind") == "cross_episode"]

    groups: Dict[tuple, Dict[int, List[float]]] = {}
    for r in off_rows:
        key = (r["view"], r["mask"])
        for k in METRIC_KEYS:
            groups.setdefault((key, k), {}).setdefault(r["offset"], []).append(r[k])

    neg_groups: Dict[tuple, List[float]] = {}
    for r in neg_rows:
        for k in METRIC_KEYS:
            neg_groups.setdefault(((r["view"], r["mask"]), k), []).append(r[k])

    for ax, metric in zip(axes, METRIC_KEYS):
        for view in sorted({r["view"] for r in off_rows}):
            for mask in ("left", "right"):
                series = groups.get(((view, mask), metric))
                if not series:
                    continue
                xs = sorted(series.keys())
                means, lo, hi = [], [], []
                for x in xs:
                    vals = np.asarray(series[x], dtype=np.float64)
                    vals = vals[np.isfinite(vals)]  # drop inf (offset 0 PSNR)
                    if vals.size == 0:
                        means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                        continue
                    means.append(float(vals.mean()))
                    lo.append(float(vals.min()))
                    hi.append(float(vals.max()))
                means = np.asarray(means)
                color = _VIEW_COLORS.get(view, "gray")
                ax.plot(
                    xs, means,
                    color=color,
                    linestyle=_MASK_STYLES[mask],
                    marker=_MASK_MARKERS[mask],
                    markersize=4,
                    label=f"{view}-{mask}",
                )
                ax.fill_between(xs, lo, hi, color=color, alpha=0.08)

                # cross-episode negative level for this (view, mask)
                neg_vals = neg_groups.get(((view, mask), metric))
                if neg_vals:
                    nv = np.asarray(neg_vals, dtype=np.float64)
                    nv = nv[np.isfinite(nv)]
                    if nv.size:
                        ax.axhline(float(nv.mean()), color=color,
                                   linestyle=_MASK_STYLES[mask], linewidth=0.9,
                                   alpha=0.45)

        thr = DEFAULT_THRESHOLDS.get(metric)
        if thr is not None:
            ax.axhline(thr, color="red", linestyle=":", linewidth=1.2,
                       label=f"current thr={thr}")
        if neg_rows:
            ax.plot([], [], color="black", linewidth=0.9, alpha=0.45,
                    label="cross-ep neg (mean)")
        direction = "higher = more similar" if _HIGHER_BETTER[metric] else "lower = more similar"
        ax.set_title(f"{metric}  ({direction})")
        ax.set_xlabel("offset from key frame (#frames)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Multi-View Consistency: metric vs temporal offset (anchor = key frame)\n"
        "small offset = good consistency, large offset = bad consistency",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved trend figure -> {out_path}")


def plot_qualitative(
    view: str,
    mask_name: str,
    kf: int,
    anchor_img: np.ndarray,
    mask: np.ndarray,
    frame_map: Dict[int, np.ndarray],
    offsets: List[int],
    metric_lookup: Dict[tuple, Dict[str, float]],
    out_path: Path,
) -> None:
    """Grid (2 rows): anchor first, then offset frame crops, wrapped over 2 rows."""
    bbox = mask_bbox(mask)
    if bbox is None:
        return
    y0, y1, x0, x1 = bbox
    mask_crop = mask[y0:y1, x0:x1]

    pos_offsets = [o for o in offsets if o != 0 and (kf + o) in frame_map]
    cells = [None] + pos_offsets  # None == the anchor cell
    if len(cells) < 2:
        return

    nrows = 2
    ncols = (len(cells) + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 6.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, cell in zip(axes, cells):
        if cell is None:
            ax.imshow(anchor_img[y0:y1, x0:x1])
            ax.contour(mask_crop > 0, levels=[0.5], colors="lime", linewidths=1.0)
            ax.set_title(f"ANCHOR\nkf={kf}", fontsize=9, fontweight="bold")
        else:
            off = cell
            ax.imshow(frame_map[kf + off][y0:y1, x0:x1])
            ax.contour(mask_crop > 0, levels=[0.5], colors="lime", linewidths=1.0)
            m = metric_lookup.get((mask_name, kf, off), {})
            psnr = m.get("psnr", float("nan"))
            psnr_str = "inf" if np.isinf(psnr) else f"{psnr:.1f}"
            ax.set_title(
                f"+{off}\n"
                f"ssim {m.get('ssim', float('nan')):.3f}\n"
                f"psnr {psnr_str}\n"
                f"lpips {m.get('lpips', float('nan')):.3f}\n"
                f"vjepa {m.get('vjepa_similarity', float('nan')):.3f}",
                fontsize=8,
            )
        ax.axis("off")

    # blank any leftover axes
    for ax in axes[len(cells):]:
        ax.axis("off")

    fig.suptitle(f"{view} / {mask_name} hand  —  key frame {kf}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path)

    gt_path = args.gt_path or (cfg.get("data", {}) or {}).get("gt_path")
    if not gt_path:
        raise ValueError("Need data.gt_path in config or --gt_path.")
    gt_path = Path(gt_path)

    episode = norm_episode(args.episode)
    episode_dir = gt_path / args.task / episode
    prompts_json = episode_dir / "prompt" / "prompts.json"
    video_dir = episode_dir / "video"
    mm_dir = episode_dir / "multiview_masks"

    if not prompts_json.is_file():
        raise FileNotFoundError(f"prompts.json not found: {prompts_json}")
    if not mm_dir.is_dir():
        raise FileNotFoundError(f"multiview_masks/ not found: {mm_dir}")

    keyframes = read_keyframes(prompts_json)
    if args.max_keyframes:
        keyframes = keyframes[: args.max_keyframes]
    if not keyframes:
        raise ValueError(f"No key frames parsed from {prompts_json}")
    logger.info(f"Key frames ({len(keyframes)}): {keyframes}")

    # Encode the measurement mode in the output dir so different methods don't
    # overwrite each other (e.g. the earlier full-frame zero-fill run).
    mode_tag = "crop" if args.crop_to_bbox else "fullframe"
    if args.mask_aware:
        mode_tag += "_maskaware"
    out_dir = Path(args.output_dir or (REPO_ROOT / "output" / "threshold_experiment"
                                       / f"{args.task}_{episode}_{mode_tag}"))
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Measurement mode: crop_to_bbox={args.crop_to_bbox} "
                f"(pad={args.bbox_pad}), mask_aware={args.mask_aware}")

    # V-JEPA / LPIPS / torchmetrics objects (loaded once).
    dim_cfg = (cfg.get("dimensions", {}) or {}).get("multiview_consistency", {}) or {}
    vjepa_cfg = dim_cfg.get("vjepa", {}) or {}
    if args.no_vjepa:
        # Build SSIM/PSNR/LPIPS only — avoid touching V-JEPA (which would
        # otherwise try to download weights when no local path is configured).
        tm = _ensure_torchmetrics()
        metric_objs = {
            "device": args.device,
            "ssim": tm.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device),
            "psnr": tm.image.PeakSignalNoiseRatio(data_range=1.0).to(args.device),
            "lpips": _ensure_lpips().LPIPS(net="alex").to(args.device).eval(),
            "jepa_model": None,
            "jepa_transform": None,
            "use_jepa": False,
        }
    else:
        metric_objs = _build_metric_objs(
            args.device,
            vjepa_cfg.get("model_path"),
            vjepa_cfg.get("model_name", "vjepa2_1_vit_base_384"),
        )
    logger.info(f"V-JEPA enabled: {metric_objs['use_jepa']}")

    offsets = sorted(set(int(o) for o in args.offsets))

    rows: List[dict] = []
    # which key frames to render qualitatively (evenly sampled)
    if args.viz_keyframes >= len(keyframes):
        viz_kfs = list(keyframes)
    else:
        idx = np.linspace(0, len(keyframes) - 1, args.viz_keyframes).round().astype(int)
        viz_kfs = [keyframes[i] for i in sorted(set(idx.tolist()))]

    for view in args.views:
        masks_path = mm_dir / view / "masks.npz"
        if not masks_path.exists():
            logger.warning(f"Missing {masks_path}, skipping view {view}.")
            continue

        with np.load(masks_path) as npz:
            if "masks" not in npz:
                logger.warning(f"{masks_path} has no 'masks' key, skipping.")
                continue
            masks = npz["masks"]
            if masks.ndim == 3:
                masks = masks[None]
            T_mask, H, W = masks.shape[1], masks.shape[2], masks.shape[3]
            anchor_masks = {
                kf: masks[:, kf].copy() for kf in keyframes if kf < T_mask
            }
        del masks

        target_hw = (H, W)
        wanted = set()
        for kf in keyframes:
            wanted.add(kf)
            for off in offsets:
                wanted.add(kf + off)
        frame_map = load_needed_frames(video_dir, view, wanted, target_hw)
        if not frame_map:
            logger.warning(f"No frames loaded for view {view}, skipping.")
            continue
        logger.info(f"[{view}] loaded {len(frame_map)} frames (resized to {H}x{W}).")

        # Cross-episode negatives: sample frames from each negative episode's
        # video for THIS view (anchor masks come from the current episode).
        neg_frame_maps: Dict[str, Dict[int, np.ndarray]] = {}
        for neg_ep in (args.neg_episodes or []):
            neg_ep_norm = norm_episode(neg_ep)
            neg_video_dir = gt_path / args.task / neg_ep_norm / "video"
            neg_files = list_frame_files(neg_video_dir, preferred_view=view)
            if not neg_files:
                logger.warning(f"No frames for neg episode {neg_ep_norm} view {view}.")
                continue
            n_neg = len(neg_files)
            sample_idx = sorted(set(
                np.linspace(0, n_neg - 1, args.neg_frames).round().astype(int).tolist()
            ))
            neg_frame_maps[neg_ep_norm] = load_needed_frames(
                neg_video_dir, view, set(sample_idx), target_hw
            )
            logger.info(
                f"[{view}] neg {neg_ep_norm}: {len(neg_frame_maps[neg_ep_norm])} "
                f"frames sampled."
            )

        # lookup for qualitative annotations: (mask_name, kf, off) -> metrics
        view_lookup: Dict[tuple, Dict[str, float]] = {}

        for kf in keyframes:
            if kf not in anchor_masks or kf not in frame_map:
                continue
            anchor_img = frame_map[kf]
            for m_idx, m_name in MASK_NAMES.items():
                if m_idx >= anchor_masks[kf].shape[0]:
                    continue
                mask = anchor_masks[kf][m_idx]
                area = int((mask > 0).sum())
                if area == 0:
                    logger.info(f"[{view}] kf={kf} {m_name}: empty mask, skipping.")
                    continue
                for off in offsets:
                    eidx = kf + off
                    if eidx not in frame_map:
                        continue
                    sims = masked_metrics(
                        anchor_img, frame_map[eidx], mask, metric_objs,
                        crop_to_bbox=args.crop_to_bbox,
                        bbox_pad=args.bbox_pad,
                        mask_aware=args.mask_aware,
                    )
                    row = {
                        "view": view,
                        "mask": m_name,
                        "mask_index": m_idx,
                        "key_frame": kf,
                        "kind": "offset",
                        "offset": off,
                        "error_frame": eidx,
                        "mask_area": area,
                        **{k: sims[k] for k in METRIC_KEYS},
                    }
                    rows.append(row)
                    view_lookup[(m_name, kf, off)] = sims

                # ---- cross-episode HARD NEGATIVES (same view, anchor's mask) ----
                for neg_ep, neg_frames in neg_frame_maps.items():
                    for neg_idx, neg_img in neg_frames.items():
                        sims = masked_metrics(
                            anchor_img, neg_img, mask, metric_objs,
                            crop_to_bbox=args.crop_to_bbox,
                            bbox_pad=args.bbox_pad,
                            mask_aware=args.mask_aware,
                        )
                        rows.append({
                            "view": view,
                            "mask": m_name,
                            "mask_index": m_idx,
                            "key_frame": kf,
                            "kind": "cross_episode",
                            "offset": -1,
                            "error_frame": neg_idx,
                            "neg_episode": neg_ep,
                            "mask_area": area,
                            **{k: sims[k] for k in METRIC_KEYS},
                        })

        # qualitative strips for this view
        for kf in viz_kfs:
            if kf not in anchor_masks or kf not in frame_map:
                continue
            for m_idx, m_name in MASK_NAMES.items():
                if m_idx >= anchor_masks[kf].shape[0]:
                    continue
                mask = anchor_masks[kf][m_idx]
                if int((mask > 0).sum()) == 0:
                    continue
                out_path = qual_dir / f"{view}_{m_name}_kf{kf}.png"
                plot_qualitative(
                    view, m_name, kf, frame_map[kf], mask, frame_map,
                    offsets, view_lookup, out_path,
                )

    if not rows:
        raise RuntimeError("No metric rows computed — check paths / masks.")

    # ---- Save raw CSV (pandas optional; fall back to csv) ----
    raw_csv = out_dir / "raw_metrics.csv"
    summary_csv = out_dir / "summary_by_offset.csv"
    _write_csvs(rows, offsets, raw_csv, summary_csv)

    # ---- Trend figure ----
    plot_trends(rows, offsets, out_dir / "trends.png")

    # ---- Console summary ----
    _print_console_summary(rows, offsets)

    logger.info(f"Done. Outputs in: {out_dir}")


def _write_csvs(rows, offsets, raw_csv: Path, summary_csv: Path) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(raw_csv, index=False)

        clean = df.copy()
        clean["psnr"] = clean["psnr"].replace([np.inf, -np.inf], np.nan)
        if "kind" not in clean.columns:
            clean["kind"] = "offset"
        summary = (
            clean.groupby(["view", "mask", "kind", "offset"])[list(METRIC_KEYS)]
            .agg(["mean", "std", "min", "max", "count"])
        )
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.reset_index().to_csv(summary_csv, index=False)
    except ImportError:
        import csv

        fieldnames = [
            "view", "mask", "mask_index", "key_frame", "kind", "offset",
            "error_frame", "neg_episode", "mask_area", *METRIC_KEYS,
        ]
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    logger.info(f"Saved raw metrics  -> {raw_csv}")
    logger.info(f"Saved summary      -> {summary_csv}")


def _print_console_summary(rows, offsets) -> None:
    by = {}
    neg = {}
    for r in rows:
        key = (r["view"], r["mask"])
        if r.get("kind") == "cross_episode":
            neg.setdefault(key, []).append(r)
        else:
            by.setdefault(key, {}).setdefault(r["offset"], []).append(r)

    def _mean(group, key):
        vals = np.asarray([g[key] for g in group], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    print("\n================ metric means by offset (per view / mask) ================")
    for (view, mask) in sorted(by):
        print(f"\n[{view} / {mask}]")
        print(f"  {'offset':>7} | {'ssim':>7} {'psnr':>7} {'lpips':>7} {'vjepa':>7}")
        for off in offsets:
            group = by[(view, mask)].get(off)
            if not group:
                continue

            def _m(key):
                return _mean(group, key)

            print(f"  {off:>7} | {_m('ssim'):>7.3f} {_m('psnr'):>7.1f} "
                  f"{_m('lpips'):>7.3f} {_m('vjepa_similarity'):>7.3f}")
        ngroup = neg.get((view, mask))
        if ngroup:
            print(f"  {'NEG':>7} | {_mean(ngroup, 'ssim'):>7.3f} "
                  f"{_mean(ngroup, 'psnr'):>7.1f} {_mean(ngroup, 'lpips'):>7.3f} "
                  f"{_mean(ngroup, 'vjepa_similarity'):>7.3f}   <- cross-episode")
    print("\n=========================================================================")


if __name__ == "__main__":
    main()
