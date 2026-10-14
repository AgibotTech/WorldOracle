#!/usr/bin/env python3
"""Reverse grounding: infer the FIRST frame's success_criteria + grounding_objects.

This is the inverse of ``fill_success_criteria.py``. Instead of looking at the
first frame to *generate* the per-key-frame criteria, this script does the
opposite:

  * It reads the success_criteria already written across a prompt.json's key
    frames (these collectively name the full object inventory of the episode —
    every fruit by color/type and every container by color/style/location).
  * Given the FIRST frame image + that observed criteria set, it asks the VLM to
    work backward and produce:
      1. ``grounding_objects``        — the full object inventory (short labels),
      2. the FIRST frame's ``success_criteria`` — where each fruit is in the
         INITIAL scene (stored as a segment with ``key_frame: 0``).

Both are written back into the same prompt.json (``grounding_objects`` is placed
between ``n_frames`` and ``segments``; the frame-0 segment is inserted first).

Layout (matches the worldsim exporter; both flat and prompt/ subdir work)::

    <data-root>/ep000000/prompt/prompts.json   (or .../prompt.json)
    <data-root>/ep000000/video/<view>/frame_00000.jpg   # first frame

Usage::

    python scripts/infer_first_frame_grounding.py \
        --prompt-json data/acwm_dataset/pickfruit/ep000011/prompt.json \
        --model gpt-5.4 --overwrite

    python scripts/infer_first_frame_grounding.py \
        --data-root data/acwm_dataset/pickfruit --model gpt-5.4 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the shared infrastructure (backend builder, parsing, path/key helpers)
# from the sibling forward script. ``scripts/`` is not a package, so add this
# dir to sys.path and import by module name.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fill_success_criteria as fsc  # noqa: E402

logger = logging.getLogger("infer_first_frame_grounding")


# =============================================================================
# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                         PROMPT ENGINEERING                                ║
# ║  Reverse grounding prompts. Edit freely.                                  ║
# ║  - SYSTEM_PROMPT: role + the reverse task + output contract.              ║
# ║  - USER_PROMPT_TEMPLATE: per-episode message. The first-frame image is    ║
# ║    attached separately. Placeholder: {criteria_block}.                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert annotator for a robot fruit pick-and-place benchmark.

SETTING: A robot with two grippers moves fruits between a table and various \
baskets / fruit plates.

YOU ARE GIVEN:
1. The episode's FIRST frame image: a clear, mostly unoccluded view of the \
INITIAL scene.
2. A list of success_criteria collected from the episode's LATER key frames. \
Across the whole episode these criteria reference the COMPLETE set of objects in \
the scene — every fruit (by color/type) and every container (basket / fruit \
plate, by color, style, and location).

USE THE CRITERIA AS THE AUTHORITATIVE INVENTORY (reverse grounding): the later \
criteria tell you WHICH objects exist and their canonical NAMES. Keep names \
consistent with them (e.g. if they say "the yellow lemon", call it "the yellow \
lemon"). Do NOT invent fruits or containers that the criteria never mention, and \
do not rename them. Match objects in the first frame PRIMARILY BY COLOR.

YOUR JOB — produce two things from the FIRST frame:
1. grounding_objects: the full inventory of objects implied by the criteria, as \
short labels (fruits then containers), e.g. ["yellow lemon", "red apple", \
"brown wicker basket on the top left"].
2. first_frame_success_criteria: a list describing where each fruit is in the \
FIRST frame (the INITIAL scene), using the SAME identity names. Follow these rules:
   - Each criterion has a FRUIT as its subject (named by color/type from the \
inventory). One short, self-contained English sentence.
   - In the first frame each fruit is in exactly ONE state: on the table, in a \
specific basket / fruit plate, or not visible.
   - When a fruit is in a container, name the container by its color, style, and \
location (e.g. "the red apple is in the brown wicker basket on the top left").
   - ONLY write a criterion you are VISUALLY CERTAIN about from the first frame; \
if a fruit's state is ambiguous, OMIT it rather than guess.

OUTPUT CONTRACT: Respond with STRICTLY one JSON object and nothing else (no \
markdown, no code fences):
{"grounding_objects": ["...", "..."], "first_frame_success_criteria": ["...", "..."]}\
"""

USER_PROMPT_TEMPLATE = """\
The attached image is the episode's FIRST frame (the initial scene).

success_criteria observed across the episode's later key frames (use these as the \
object inventory / identity reference):
{criteria_block}

Using these to fix the identity of every fruit and container, output the JSON \
object with `grounding_objects` (full inventory) and `first_frame_success_criteria` \
(where each fruit is in THIS first frame). Respond with ONLY the JSON object.\
"""

# =============================================================================
# ╚══════════════════════ END PROMPT ENGINEERING ════════════════════════════╝
# =============================================================================


def _episode_dir(prompt_path: Path) -> Path:
    """Episode dir holding video/ — file's parent, or grandparent under prompt/."""
    ep = prompt_path.parent
    return ep.parent if ep.name == "prompt" else ep


def _collect_existing_criteria(data: dict) -> list[str]:
    """Union of all non-empty success_criteria across segments, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for seg in data.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        sc = seg.get("success_criteria") or []
        if isinstance(sc, str):
            sc = [sc]
        for c in sc:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def build_messages(first_image, observed_criteria: list[str]) -> list[dict]:
    criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(observed_criteria))
    user_text = USER_PROMPT_TEMPLATE.format(criteria_block=criteria_block)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first_image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _parse_reverse(text: str) -> tuple[list[str], list[str]] | None:
    """Parse {"grounding_objects": [...], "first_frame_success_criteria": [...]}."""
    obj = fsc._loads_lenient(text)
    if not isinstance(obj, dict):
        return None
    objs = obj.get("grounding_objects") or obj.get("objects") or []
    crits = (
        obj.get("first_frame_success_criteria")
        or obj.get("success_criteria")
        or obj.get("criteria")
        or []
    )
    if not isinstance(objs, list):
        objs = [str(objs)]
    if not isinstance(crits, list):
        crits = [str(crits)]
    objs = [fsc._strip_article(str(x)) for x in objs if str(x).strip()]
    crits = [str(x).strip() for x in crits if str(x).strip()]
    # de-dupe grounding_objects, preserve order
    seen: set[str] = set()
    objs = [o for o in objs if not (o in seen or seen.add(o))]
    if not objs and not crits:
        return None
    return objs, crits


def infer_one(
    backend,
    prompt_path: Path,
    *,
    view_priority: tuple[str, ...],
    format_retries: int,
) -> tuple[list[str], list[str], str | None]:
    """Returns (grounding_objects, first_frame_criteria, error)."""
    from PIL import Image

    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    observed = _collect_existing_criteria(data)
    if not observed:
        return [], [], "no existing success_criteria to reverse from"

    ep_dir = _episode_dir(prompt_path)
    first_img_path = fsc.resolve_image_path(ep_dir, 0, view_priority)
    if first_img_path is None:
        return [], [], f"no first frame (frame 0) under {ep_dir}/video/<view>/"

    first_image = Image.open(first_img_path).convert("RGB")
    for attempt in range(format_retries + 1):
        reply = backend.chat(build_messages(first_image.copy(), observed))
        parsed = _parse_reverse(reply)
        if parsed:
            return parsed[0], parsed[1], None
        logger.warning(
            "ep=%s: empty/unparseable reply (attempt %d/%d): %r",
            ep_dir.name, attempt + 1, format_retries + 1, (reply or "")[:200],
        )
    return [], [], "could not parse a reverse-grounding reply"


def _write_back(
    prompt_path: Path,
    grounding: list[str],
    first_frame_criteria: list[str],
    *,
    overwrite: bool,
) -> None:
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if grounding:
        data["grounding_objects"] = grounding

    if first_frame_criteria:
        segments = data.get("segments") or []
        existing0 = next(
            (s for s in segments if isinstance(s, dict) and int(s.get("key_frame", -1)) == 0),
            None,
        )
        if existing0 is not None:
            if overwrite or not (existing0.get("success_criteria") or []):
                existing0["success_criteria"] = first_frame_criteria
        else:
            segments.insert(0, {"key_frame": 0, "success_criteria": first_frame_criteria})
        data["segments"] = segments

    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(fsc._ordered_prompt(data), f, ensure_ascii=False, indent=4)
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-root", type=Path, default=None, help="Root dir; every ep*/prompt[/prompts].json is processed.")
    src.add_argument("--prompt-json", type=Path, default=None, help="Process a single prompt.json / prompts.json.")
    p.add_argument("--model", default="gpt-4.1", help="OpenAI model, e.g. gpt-4.1 or gpt-5.4")
    p.add_argument("--view", default=None, help="Camera view folder (default: try top_head, hand_right, hand_left).")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-image-long-side", type=int, default=1024)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--format-retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing frame-0 success_criteria (grounding_objects is always rewritten).",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N prompt files (debug).")
    p.add_argument("--dry-run", action="store_true", help="Query the VLM and print, but do not write files.")
    args = p.parse_args()

    view_priority = (args.view,) if args.view else fsc.DEFAULT_VIEW_PRIORITY

    if args.prompt_json:
        prompt_files = [args.prompt_json.resolve()]
    else:
        if args.data_root is None:
            raise SystemExit("Provide --data-root or --prompt-json.")
        data_root = args.data_root.resolve()
        if not data_root.is_dir():
            raise SystemExit(f"data-root is not a directory: {data_root}")
        prompt_files = fsc.discover_prompt_files(data_root)
    if args.limit is not None:
        prompt_files = prompt_files[: args.limit]
    if not prompt_files:
        raise SystemExit("No prompt files found.")

    logger.info("Found %d prompt file(s).", len(prompt_files))

    backend = fsc.build_backend(
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_image_long_side=args.max_image_long_side,
        request_timeout=args.request_timeout,
    )

    try:
        from tqdm import tqdm
    except ImportError:  # noqa: BLE001
        tqdm = None  # type: ignore[assignment]

    def _work(fp: Path) -> tuple[Path, list[str], list[str], str | None]:
        try:
            g, c, err = infer_one(
                backend, fp, view_priority=view_priority, format_retries=args.format_retries
            )
            return fp, g, c, err
        except Exception as exc:  # noqa: BLE001
            return fp, [], [], f"{type(exc).__name__}: {exc}"

    results: list[tuple[Path, list[str], list[str], str | None]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_work, fp) for fp in prompt_files]
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="reverse grounding")
        for fut in iterator:
            fp, g, c, err = fut.result()
            results.append((fp, g, c, err))
            ep = _episode_dir(fp).name
            if err is not None:
                logger.error("ep=%s failed: %s", ep, err)
            else:
                logger.info("ep=%s grounding_objects=%s first_frame_criteria=%s", ep, g, c)

    if args.dry_run:
        ok = sum(1 for _, _, _, e in results if e is None)
        logger.info("[dry-run] not writing. %d/%d ok.", ok, len(results))
        return

    written = 0
    for fp, g, c, err in results:
        if err is not None or (not g and not c):
            continue
        _write_back(fp, g, c, overwrite=args.overwrite)
        written += 1

    failures = sum(1 for _, _, _, e in results if e is not None)
    logger.info("Done. Updated %d file(s); %d failed.", written, failures)


if __name__ == "__main__":
    main()
