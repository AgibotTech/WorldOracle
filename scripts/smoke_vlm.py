"""Minimal Qwen3.5 VLM smoke test with GPU profiling.

What this script does:
    1. Loads Qwen3.5-9B via our `acwmbench.models.vlm.TransformersVLMBackend`.
    2. Sends a few image+text questions about a real frame from the dataset.
    3. Reports for each call:
         - wall-clock time
         - generated tokens & tokens/s
         - peak GPU memory delta during the call
    4. Prints a final summary.

Usage:
    conda activate acwmbench
    python scripts/smoke_vlm.py \\
        [--model-path ckpt/Qwen3.5-9B] \\
        [--image <path>] \\
        [--queries q1 q2 ...]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acwmbench.models.vlm import TransformersVLMBackend, VLMConfig  # noqa: E402
from acwmbench.models.model_zoo import (  # noqa: E402
    MODEL_ALIASES,
    auto_pick_default,
    resolve_model_path,
)

_CKPT_ROOT = REPO_ROOT / "ckpt"


def _resolve_default_model() -> Path:
    env_path = os.environ.get("ACWMBENCH_VLM_PATH") or os.environ.get(
        "ACWMBENCH_VLM_MODEL"
    )
    if env_path:
        return Path(resolve_model_path(env_path, ckpt_root=_CKPT_ROOT))
    picked = auto_pick_default(ckpt_root=_CKPT_ROOT)
    if picked:
        return Path(picked)
    return Path(resolve_model_path("qwen3_vl_8b", ckpt_root=_CKPT_ROOT))


DEFAULT_MODEL = _resolve_default_model()

# Real frames from a locally available ACWMBench episode.
DEFAULT_IMAGE_CANDIDATES = [
    "/path/to/example-episode/video/top_head/frame_00000.jpg",
    "/path/to/example-episode/video/hand_left/frame_00000.jpg",
]

DEFAULT_QUERIES = [
    "Describe what you see in this image in one sentence.",
    "List every distinct object you can see, and for each one give the count, color, and approximate location (left/center/right, top/middle/bottom).",
    (
        "Answer with ONE JSON object: "
        '{"task": "Hang the towel.", '
        '"satisfied": true|false, '
        '"score": <float in [0,1]>, '
        '"explanation": "<one sentence>"}'
    ),
]


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def gpu_mem_mb() -> dict:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved":  torch.cuda.memory_reserved() / 1024 ** 2,
        "peak":      torch.cuda.max_memory_allocated() / 1024 ** 2,
    }


def fmt_mem(d: dict) -> str:
    return f"alloc={d['allocated']:7.1f} MB  reserved={d['reserved']:7.1f} MB  peak={d['peak']:7.1f} MB"


def print_header(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * (60 - len(title))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_default_image() -> str:
    for cand in DEFAULT_IMAGE_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    raise SystemExit(
        "No default image found; pass --image <path>. "
        f"Tried: {DEFAULT_IMAGE_CANDIDATES}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL),
        help="Local HF model directory (resolved literally).",
    )
    p.add_argument(
        "--vlm-model",
        default=None,
        help=(
            "Short alias / repo id / path (takes precedence over --model-path). "
            f"Known aliases: {sorted(set(MODEL_ALIASES))}."
        ),
    )
    p.add_argument("--image", default=None, help="Path to a test image (PNG/JPG)")
    p.add_argument("--queries", nargs="*", default=None, help="Override the default queries")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-image-long-side", type=int, default=768)
    p.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking=False to the chat template (default: True).",
    )
    p.add_argument(
        "--attn",
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],
        help="attn_implementation override (default: leave to transformers)",
    )
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                   help="Run one short warmup call before timing (default: True).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.vlm_model:
        args.model_path = resolve_model_path(args.vlm_model, ckpt_root=_CKPT_ROOT)
    image_path = args.image or pick_default_image()
    queries: List[str] = args.queries or DEFAULT_QUERIES

    print_header("System")
    print(f"  python:       {sys.version.split()[0]}")
    print(f"  torch:        {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"  gpu[{dev}]:       {torch.cuda.get_device_name(dev)}")
        total = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
        print(f"  total VRAM:   {total:.2f} GB")
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        print("  transformers: NOT INSTALLED")
        raise SystemExit(1)

    print_header("Loading model")
    print(f"  path:  {args.model_path}")
    print(f"  image: {image_path}")
    print(f"  before: {fmt_mem(gpu_mem_mb())}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    cfg = VLMConfig(
        backend="transformers",
        model_path=args.model_path,
        device_map="auto",
        dtype="auto",
        attn_implementation=args.attn,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        disable_thinking=args.disable_thinking,
        max_image_long_side=args.max_image_long_side,
    )
    backend = TransformersVLMBackend(cfg)
    load_time = time.perf_counter() - t0
    print(f"  load_time: {load_time:7.2f} s")
    print(f"  after:  {fmt_mem(gpu_mem_mb())}")

    # ------------------------------------------------------------------
    # Build messages
    # ------------------------------------------------------------------
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print(f"  image size (original): {image.size}")

    def make_messages(q: str):
        return [
            {"role": "system", "content": "You are a helpful assistant for analyzing robot manipulation images."},
            {"role": "user", "content": [
                {"type": "image", "image": image.copy()},
                {"type": "text", "text": q},
            ]},
        ]

    # Optional warmup so the first timed call doesn't include CUDA graph init etc.
    if args.warmup:
        print_header("Warmup")
        t0 = time.perf_counter()
        _ = backend.chat(make_messages("Reply with just the word 'ok'."))
        print(f"  warmup time: {time.perf_counter() - t0:.2f} s")

    # ------------------------------------------------------------------
    # Profile each query
    # ------------------------------------------------------------------
    rows = []
    for i, q in enumerate(queries):
        print_header(f"Query {i + 1} / {len(queries)}")
        print(f"  Q: {q}")

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        before = gpu_mem_mb()
        t0 = time.perf_counter()
        reply = backend.chat(make_messages(q))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        after = gpu_mem_mb()

        # Rough token count via the processor's tokenizer (assistant reply only).
        try:
            n_tokens = len(backend.processor.tokenizer.encode(reply))
        except Exception:
            n_tokens = -1
        tok_s = n_tokens / elapsed if n_tokens > 0 and elapsed > 0 else float("nan")

        print(f"  A: {reply}")
        print(f"  -- time={elapsed:6.2f}s  tokens={n_tokens}  tok/s={tok_s:6.2f}")
        print(f"  -- mem before: {fmt_mem(before)}")
        print(f"  -- mem after:  {fmt_mem(after)}")
        rows.append({
            "i": i + 1,
            "elapsed": elapsed,
            "tokens": n_tokens,
            "tok_s": tok_s,
            "peak_mb": after["peak"],
            "delta_mb": after["allocated"] - before["allocated"],
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("Summary")
    print(f"  {'#':>2}  {'time(s)':>8}  {'tokens':>7}  {'tok/s':>7}  {'peak_MB':>9}  {'delta_MB':>9}")
    for r in rows:
        print(f"  {r['i']:>2}  {r['elapsed']:8.2f}  {r['tokens']:>7}  {r['tok_s']:7.2f}  {r['peak_mb']:9.1f}  {r['delta_mb']:9.1f}")
    if rows:
        tot_time = sum(r["elapsed"] for r in rows)
        tot_tokens = sum(max(r["tokens"], 0) for r in rows)
        print(f"  -- total {tot_time:.2f}s, {tot_tokens} tokens, {tot_tokens / tot_time:.2f} tok/s avg")
        print(f"  -- max peak GPU memory: {max(r['peak_mb'] for r in rows):.1f} MB")
    print(f"  -- model load time:     {load_time:.2f} s")


if __name__ == "__main__":
    main()
