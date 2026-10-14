#!/usr/bin/env python3
"""Generate random binary masks.npy for each episode in the gt_dataset.

For every episode that has prompt/prompts.json and prompt/cur_top_head.png,
this script:
  - Reads H, W from cur_top_head.png
  - Reads n_frames from prompts.json
  - Randomly picks m in [1, 5]
  - Generates a (m, n, H, W) uint8 binary array (values 0 or 1)
  - Saves it as prompt/masks.npy
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("/path/to/main-ground-truth")


def create_masks_for_episode(episode_dir: Path) -> None:
    prompt_dir = episode_dir / "prompt"
    json_path = prompt_dir / "prompts.json"
    png_path = prompt_dir / "cur_top_head.png"
    out_path = prompt_dir / "masks.npy"

    if not json_path.exists() or not png_path.exists():
        print(f"  [SKIP] Missing prompts.json or cur_top_head.png in {episode_dir}")
        return

    # Read image shape (H, W)
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [ERROR] Cannot read {png_path}")
        return
    h, w = img.shape[:2]

    # Read n_frames
    with open(json_path) as f:
        meta = json.load(f)
    n = meta.get("n_frames")
    if n is None:
        print(f"  [ERROR] 'n_frames' not found in {json_path}")
        return

    # Random number of mask sequences
    m = random.randint(1, 5)

    # Generate random binary masks: (m, n, h, w), uint8, values in {0, 1}
    masks = np.random.randint(0, 2, size=(m, n, h, w), dtype=np.uint8)

    np.save(out_path, masks)
    print(f"  [OK] {episode_dir.name}: m={m}, n={n}, H={h}, W={w} -> {out_path.name}")


def main():
    random.seed(42)  # for reproducibility; remove if you want truly random each run

    print(f"Scanning dataset under {DATASET_ROOT} ...")
    episodes = sorted(DATASET_ROOT.glob("*/*/"))  # e.g. borrow_flame_from_candle/ep000000

    for ep_dir in episodes:
        if not (ep_dir / "prompt").exists():
            continue
        print(f"Processing {ep_dir.relative_to(DATASET_ROOT)} ...")
        create_masks_for_episode(ep_dir)

    print("Done.")


if __name__ == "__main__":
    main()
