"""Metric 1: Physical Status Judge.

Asks a VLM to look at the *key frames* of a rollout (final frame at minimum,
plus any key-steps the task spec defines) and decide whether the final /
intermediate physical state satisfies the natural-language instruction
attached to the episode.

Output schema (per trial):

```json
{
  "final": {
    "success": true,
    "score": 1.0,
    "constraints": [{"name": "...", "satisfied": true, "evidence": "..."}],
    "raw": "<raw VLM text>"
  },
  "key_steps": [
    {"frame": 60, "success": true, "score": 1.0, ...},
    ...
  ],
  "score": 1.0   // mean over (final + key_steps)
}
```
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json_repair
from PIL import Image
from tqdm import tqdm

from .. import episode_store
from ..dataset import EpisodeTrial
from ..io_utils import list_frame_files, load_frames_as_pil, uniform_indices
from ..models import VLMBackend
from ..tasks import TaskSpec, load_task_spec

logger = logging.getLogger(__name__)


# =============================================================================
# ╔════════════════════════════ PROMPT ENGINEERING ══════════════════════════╗
# ║  >>> EDIT THE JUDGE PROMPTS HERE — ONE PROMPT SET PER SUBTASK <<<          ║
# ║                                                                           ║
# ║  There is one PromptSet per subtask (= the dataset subfolder name /        ║
# ║  trial.task_id): pickfruit, hangtowel, pourwater, pulldrawer.             ║
# ║  The judge picks the set by task_id (see get_prompt_set); unknown tasks   ║
# ║  fall back to DEFAULT_PROMPT_SET.                                          ║
# ║                                                                           ║
# ║  Each set's SYSTEM_PROMPT + CRITERIA_HEADER are the main things to tune.   ║
# ║  The JSON schema / rules / footer are shared (COMMON_* below) but can be   ║
# ║  overridden per task via the PromptSet fields.                            ║
# ║                                                                           ║
# ║  Quick iteration (one subtask at a time, e.g. point val_base/gt_path at   ║
# ║  .../acwm_dataset and use --top-n, or at .../acwm_dataset/<subtask>):      ║
# ║    python evaluate.py --dimension physical_judge \                        ║
# ║        --config_path configs/base.yaml --overwrite --top-n 15             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


@dataclass(frozen=True)
class PromptSet:
    """All editable prompt text for ONE subtask's physical-status judge."""

    system_prompt: str
    criteria_header: str
    instruction_line: str = "Task instruction: {instruction}"
    checkpoint_line: str = "Checkpoint to evaluate: {step_description}"
    schema_and_rules_template: str = ""  # filled from COMMON_SCHEMA_AND_RULES below
    footer: str = (
        "Do not output anything other than the JSON. "
        "Do not wrap it in markdown code fences."
    )


# ---- COMMON output contract (shared JSON schema + scoring rules) ------------
# NOTE: literal braces in the JSON example are doubled ({{ }}) because this is a
# str.format template; {n_criteria} is the only real placeholder.
COMMON_SCHEMA_AND_RULES = """\
Return exactly this JSON schema:
{{
  "aspect_scores": [0 or 1, ...],
  "reasoning": ["<brief visible evidence for aspect 1>", "..."],
  "explanation": "<one sentence summary>"
}}

Rules:
- The aspect_scores array MUST contain exactly {n_criteria} integer(s), one per success criterion, in the same order.
- Use 1 when the image clearly satisfies the criterion (judge leniently on object identity, strictly on the asserted state / location).
- Use 0 only for a clear contradiction with what is visible.
- Do not compute the average score yourself; the evaluator code will average the 0/1 aspect scores.
- You may return just a bare JSON array like [1, 0] if and only if every entry is the aspect score.\
"""


# ===========================================================================
# pickfruit  (DEVELOPED — judge fruit location by COLOR, not shape/type)
# ===========================================================================
# Aligned with the released benchmark success-criterion definitions:
# each criterion names a FRUIT (identity fixed from a clean reference frame) and
# asserts its LOCATION — on the table, in a specific basket / fruit plate, grasped
# by the gripper (optionally hovering above the table or a named container), or
# not visible.
PICKFRUIT_PROMPT_SET = PromptSet(
    system_prompt=(
        "You are an expert physical-status judge for a robot FRUIT pick-and-place "
        "benchmark. A robot moves fruits between a table and various baskets / fruit "
        "plates. You are shown ONE key frame plus a list of success criteria. Each "
        "criterion names a fruit and asserts WHERE it is: on the table, in a specific "
        "basket / fruit plate (named by color / style / location), grasped by the "
        "gripper (optionally hovering above the table or above a named container), or "
        "not visible in the frame.\n\n"
        "Your job: for each criterion, return a binary 1 (clearly satisfied) or 0 "
        "(clearly not satisfied).\n\n"
        "IDENTITY — DISAMBIGUATE AGAINST THE OBJECT INVENTORY (very important):\n"
        "- You may be given TWO images: IMAGE 1 is the episode's FIRST frame (a clean "
        "reference of the initial scene — use it to recall what each fruit and basket / "
        "fruit plate looks like, their colors and positions); IMAGE 2 is the frame to "
        "JUDGE. If only one image is present, it is the frame to judge.\n"
        "- The user message gives you the episode's OBJECT INVENTORY (grounding_objects): "
        "the COMPLETE set of fruits and containers in the scene. Every object in the "
        "frame is ONE of these.\n"
        "- Do NOT judge identity in a vacuum and do NOT demand a textbook shape/type "
        "match — occlusion, motion blur, the gripper, and viewpoint distort shape and "
        "exact type. Instead, match each frame object to the CLOSEST inventory item, "
        "primarily by COLOR and then by rough shape / size.\n"
        "- A criterion is SATISFIED (1) when the object in the stated location best "
        "matches the criterion's named object AMONG the inventory — i.e. it is closer "
        "to that inventory item than to any other inventory item.\n"
        "- Score 0 when the object in that location best matches a DIFFERENT inventory "
        "item (e.g. the criterion says 'yellow lemon' but what is there is closest to "
        "the 'red apple' in the inventory), or when no matching object is there at all.\n"
        "- Example: inventory = [yellow lemon, red apple, green pear]; criterion 'the "
        "yellow lemon is in the basket'. A yellow object in that basket is closest to "
        "'yellow lemon' (not the red apple or green pear) -> score 1, even if it does "
        "not look like a textbook lemon. A red object there is closest to 'red apple' "
        "-> score 0.\n"
        "- Disambiguate CONTAINERS the same way: pick the basket / fruit plate that "
        "best matches the criterion's named container (by color, style, location) "
        "against the inventory of containers.\n"
        "- If no inventory is provided, fall back to matching primarily by color.\n\n"
        "WHAT TO ACTUALLY VERIFY:\n"
        "- The fruit's LOCATION: on the table vs in the NAMED container vs grasped by "
        "the gripper (and, if stated, hovering above the table or above a named "
        "container) vs not visible. Use the container's color / location to pick the "
        "right basket / plate.\n"
        "- For grasped / hovering criteria: score 1 when the named fruit is clearly "
        "held by the gripper AND, if a hover location is stated, that location matches "
        "(above the table vs above the named basket / plate).\n"
        "- For 'not in the view' / 'not visible' criteria: score 1 when no object that "
        "best matches that fruit in the inventory is visible in the frame.\n\n"
        "Prefer 1 when the closest-inventory match for the object in the right place IS "
        "the criterion's named object. Reserve 0 for clear contradictions: the object "
        "best matches a different inventory item, wrong container, wrong grasp/hover "
        "location, present when it should be absent, or absent when it should be "
        "present. A shape/type mismatch alone (when color and inventory still point to "
        "the named object) is NOT a reason to fail. Return only valid JSON and no "
        "markdown."
    ),
    criteria_header=(
        "Success criteria for this frame. For each, judge 1/0 whether the named fruit "
        "is in the described place (on the table / in a named container / grasped and "
        "hovering / not in the view). Match the object to the CLOSEST item in the "
        "OBJECT INVENTORY above (by color, then rough shape/size); it counts as 1 only "
        "if it is closest to the criterion's named object, not a different inventory "
        "item:"
    ),
)


# ===========================================================================
# hangtowel  (DEVELOPED — judge TOWEL location by COLOR (+ position), mirroring
# pickfruit: each criterion names a towel and asserts WHERE it is)
# ===========================================================================
# Each criterion names a TOWEL (identity fixed
# from a clean reference frame, by COLOR and, when needed, position) and asserts its
# LOCATION — on the table, hanging over the rod, grasped by the gripper near the
# table or near the rod, or not in the view.
HANGTOWEL_PROMPT_SET = PromptSet(
    system_prompt=(
        "You are an expert physical-status judge for a robot TOWEL hang / unhang "
        "benchmark. A robot picks a towel up from the table and hangs / drapes it over "
        "a horizontal hanging rod (a clothes-drying bar / pole), or takes a towel off "
        "the rod and puts it down on the table. You are shown ONE key frame plus a list "
        "of success criteria. Each criterion names a towel (by COLOR, and position when "
        "needed) and asserts WHERE it is: on the table, hanging over the rod, grasped "
        "by the gripper near the table, grasped by the gripper near the rod, or not "
        "visible in the frame.\n\n"
        "Your job: for each criterion, return a binary 1 (clearly satisfied) or 0 "
        "(clearly not satisfied).\n\n"
        "IDENTITY — DISAMBIGUATE AGAINST THE OBJECT INVENTORY (very important):\n"
        "- You may be given TWO images: IMAGE 1 is the episode's FIRST frame (a clean "
        "reference of the initial scene — use it to recall what each towel looks like, "
        "its color and starting position); IMAGE 2 is the frame to JUDGE. If only one "
        "image is present, it is the frame to judge.\n"
        "- The user message gives you the episode's OBJECT INVENTORY (grounding_objects): "
        "the COMPLETE set of towels in the scene (e.g. 'red towel', 'green towel'). "
        "Every towel in the frame is ONE of these.\n"
        "- Do NOT judge identity in a vacuum and do NOT demand a textbook shape — "
        "folds, draping, occlusion, the gripper, and viewpoint distort a towel's shape. "
        "Match each towel in the frame to the CLOSEST inventory item, primarily by "
        "COLOR and then by position.\n"
        "- A criterion is SATISFIED (1) when the towel in the stated location best "
        "matches the criterion's named towel AMONG the inventory — i.e. it is closer to "
        "that inventory towel than to any other inventory towel.\n"
        "- Score 0 when the towel in that location best matches a DIFFERENT inventory "
        "towel (e.g. the criterion says 'red towel' but what is there is closest to the "
        "'green towel'), or when no matching towel is there at all.\n"
        "- When two towels share a color, use their relative position to tell them "
        "apart; otherwise match by color alone.\n"
        "- If no inventory is provided, fall back to matching primarily by color.\n\n"
        "WHAT TO ACTUALLY VERIFY:\n"
        "- The towel's LOCATION: on the table vs hanging / draped over the rod vs "
        "grasped by the gripper near the table vs grasped by the gripper near the rod "
        "vs not visible.\n"
        "- RELATIVE POSITION: both 'on the table' and 'hanging over the rod' criteria "
        "may include a position (e.g. 'on the table on the left', 'hanging over the rod "
        "on the right'). Interpret that position RELATIVE TO THE OTHER towels in the "
        "SAME place — the left-most towel on the table is 'on the table on the left'; "
        "the left-most towel hung on the rod is 'on the rod on the left' — not as "
        "absolute pixels.\n"
        "- 'Hanging over the rod' means the towel is supported by / draped over the "
        "horizontal hanging rod (bar / pole), not lying flat on the table. If multiple "
        "rods / bars exist, use the named position to pick the right one.\n"
        "- For grasped criteria that say 'near the table' / 'near the rod': score 1 "
        "when the towel is clearly held by the gripper AND that proximity matches "
        "(closer to the table surface vs closer to the hanging rod).\n"
        "- For 'not in the view' / 'not visible' criteria: score 1 when no towel that "
        "best matches that inventory towel is visible in the frame.\n\n"
        "Prefer 1 when the closest-inventory match for the towel in the right place IS "
        "the criterion's named towel. Reserve 0 for clear contradictions: the towel "
        "best matches a different inventory towel, wrong location (on the table vs on "
        "the rod vs grasp near table/rod), present when it should be absent, or absent "
        "when it should be present. A shape / fold mismatch alone (when color and "
        "inventory still point to the named towel) is NOT a reason to fail. Return "
        "only valid JSON and no markdown."
    ),
    criteria_header=(
        "Success criteria for this frame. For each, judge 1/0 whether the named towel "
        "is in the described place (on the table / hanging over the rod / grasped near "
        "the table or rod / not in the view). Match the towel to the CLOSEST item in "
        "the OBJECT INVENTORY above (by color, then position); it counts as 1 only if "
        "it is closest to the criterion's named towel, not a different inventory towel:"
    ),
)


# ===========================================================================
# pourwater  (DEVELOPED — disambiguate CUPS by COLOR + LOCATION, judge the
# grasp / pour-pose / water-transfer STATE)
# ===========================================================================
# The robot arm grasps a cup holding water and pours it into another cup. The
# scene's objects are CUPS, which often share the same shape, so each criterion
# names a cup by its COLOR and its LOCATION / POSITION, and asserts that cup's
# state: grasped by the gripper, tilted / pouring over another cup, upright on
# the table, or whether water has been poured into it.
POURWATER_PROMPT_SET = PromptSet(
    system_prompt=(
        "You are an expert physical-status judge for a robot WATER-POURING "
        "benchmark. A robot arm grasps a cup that holds colored water and pours the "
        "water into another cup. You are shown ONE key frame plus a list of success "
        "criteria. A criterion typically describes BOTH a cup's POSE / POSITION "
        "(standing upright on the table, grasped by the gripper, or tilted / pouring "
        "into another cup) AND the WATER inside that cup (is empty, or contains "
        "<color> water), e.g. 'The clear cup on the right is standing upright on the "
        "table and contains red water.' Cups are named by COLOR + LOCATION; water is "
        "named by COLOR (e.g. 'red water').\n\n"
        "Your job: for each criterion, return a binary 1 (clearly satisfied) or 0 "
        "(clearly not satisfied). When a criterion asserts MULTIPLE parts (a cup pose "
        "AND a water content), score 1 only if EVERY part clearly holds; if any part "
        "is contradicted by the frame, score 0.\n\n"
        "IDENTITY — DISAMBIGUATE AGAINST THE OBJECT INVENTORY (very important):\n"
        "- You may be given TWO images: IMAGE 1 is the episode's FIRST frame (a clean "
        "reference of the initial scene — use it ONLY to recall each cup's COLOR and "
        "the COLOR of the water); IMAGE 2 is the frame to JUDGE. If only one image is "
        "present, it is the frame to judge.\n"
        "- POSITIONS ARE DYNAMIC AND RELATIVE: the robot moves cups, so a cup's "
        "position in IMAGE 2 may DIFFER from IMAGE 1. A criterion's position word "
        "refers to where the cup is in IMAGE 2 (the frame being judged), NOT its "
        "starting position in IMAGE 1. Use IMAGE 1 only for color identity; judge "
        "position and state from IMAGE 2. A position word describes a cup RELATIVE TO "
        "THE OTHER CUPS, not absolute pixels — e.g. 'the left cup' is the leftmost cup "
        "even if it sits near the image center. Positions may use left / middle / "
        "right and depth terms (front / behind / front-left / back-right, etc.); "
        "interpret them by comparing the cups to each other.\n"
        "- The user message gives you the episode's OBJECT INVENTORY (grounding_objects): "
        "the COMPLETE set of cups AND water color(s) in the scene (e.g. 'red water'). "
        "Every cup in the frame is ONE of the inventory cups, and any liquid is one of "
        "the inventory water colors.\n"
        "- BE LENIENT ON CUP COLOR. A cup's apparent color is UNRELIABLE — viewpoint, "
        "lighting, shadows, the gripper, and especially colored liquid inside can make "
        "a cup look like a different color than it really is. Do NOT score 0 just "
        "because the cup's color seems not to match the criterion. Identify which cup a "
        "criterion refers to PRIMARILY by its RELATIVE POSITION (and the pour roles), "
        "and treat cup color only as a weak, secondary hint.\n"
        "- THE MAIN THING to verify is the WATER (its presence and color) and the cup's "
        "POSE (grasped / upright / tilted-pouring) — NOT the cup's exact color. A "
        "criterion fails mainly when the water state CLEARLY contradicts the criterion "
        "under the asymmetric empty/contains-water bias below, or the pose / pour "
        "relation is wrong — not because the cup color looks off.\n"
        "- Cups often share the SAME shape, so do NOT rely on shape either. Match the "
        "cup by its RELATIVE position among the cups; for cups named by position alone "
        "(e.g. 'the cup on the left'), no cup-color match is required at all.\n"
        "- A criterion is SATISFIED (1) when the cup at the stated relative position / "
        "role has the asserted pose AND water state (using the bias below); reserve 0 "
        "for clear contradictions in the POSE, the POUR relation, or the WATER. A "
        "cup-color mismatch ALONE is NOT a reason to score 0.\n"
        "- For water color, when a liquid body is clearly visible, match it to the "
        "CLOSEST inventory water color; a clearly wrong liquid color fails a 'contains "
        "<color> water' part. If liquid presence is ambiguous, follow the asymmetric "
        "bias below rather than over-penalizing color.\n"
        "- If no inventory is provided, identify cups by relative position and judge "
        "water by its visible color.\n\n"
        "WHAT TO ACTUALLY VERIFY (the asserted STATE / spatial relation):\n"
        "- GRASP: which cup the gripper / robot hand is holding. The named cup is "
        "grasped when the gripper clearly contacts and holds that cup.\n"
        "- POUR POSE / RELATION: the source cup is tilted and held ABOVE and OVER the "
        "rim of the target cup (its opening angled down toward the target). Verify both "
        "the tilt and that it is positioned over the correct target cup.\n"
        "- EMPTY vs CONTAINS WATER — FOLLOW THE CRITERION WITH AN ASYMMETRIC BIAS. "
        "Treat empty / contains-water as BINARY, but the default depends on what the "
        "criterion asserts:\n"
        "  * If the criterion says the cup 'is empty': DEFAULT TO EMPTY. Score the "
        "empty part as 1 unless water is ESPECIALLY OBVIOUS (a large, unmistakable "
        "colored liquid body / fill). If water is faint, ambiguous, hard to see, only "
        "droplets / sheen / residue, lighting that only looks like liquid, or you are "
        "unsure — judge EMPTY.\n"
        "  * If the criterion says the cup 'contains <color> water': DEFAULT TO "
        "CONTAINS WATER. Score the contains-water part as 1 unless the cup is CLEARLY "
        "EMPTY (no plausible liquid fill at all). If water is ambiguous, hard to see, "
        "partially occluded, or you are unsure whether liquid is present — still "
        "judge CONTAINS WATER (with the criterion's stated color), NOT empty. Only "
        "score 0 for the water part when emptiness is clear / unmistakable.\n"
        "- WATER COLOR: when judging a 'contains <color> water' criterion under the "
        "rule above, require the liquid color to match only if a liquid body is "
        "clearly visible; if the presence of liquid is ambiguous but the cup is not "
        "clearly empty, still prefer contains-water with the stated color. For an "
        "'is empty' criterion, do not upgrade ambiguous color cues into 'contains "
        "water'.\n"
        "- WATER TRANSFER / POUR: when a criterion asserts pouring, judge from the "
        "STRONGEST visible cue: a colored liquid body appearing in the target, a "
        "tilt/pour pose, a colored water stream or droplets, or a clearly changed "
        "water level. Prefer the asserted pour/water outcome unless the contrary "
        "state is clear.\n"
        "- CUP LOCATION / POSE: on the table vs held by the gripper vs upright vs "
        "tilted, using the cup's color + location to pick the right cup.\n"
        "- SPILL: some criteria are about the TABLE rather than a cup, e.g. '<color> "
        "water is spilled on the table.' Score 1 only when there is a CLEAR colored "
        "spill / puddle of the stated color on the table surface; score 0 when the "
        "table is dry, the spill is a different color, or there are only ambiguous "
        "droplets / wet sheen.\n\n"
        "Judge leniently on a cup's exact shade / appearance (lighting and tilt change "
        "how a cup looks) and STRICTLY on the asserted state / spatial relation. Prefer "
        "1 when the closest-inventory cup is in the asserted state; reserve 0 for clear "
        "contradictions: wrong grasp, not tilted / not over the target when a pour is "
        "asserted, or a water state that clearly contradicts the criterion under the "
        "asymmetric empty/contains-water bias above. Return only valid JSON and no "
        "markdown."
    ),
    criteria_header=(
        "Success criteria for this frame. Each names a cup (by color + location) and "
        "usually asserts BOTH its pose/position (upright on the table / grasped / "
        "tilted-pouring into another cup) AND its water content (empty / contains "
        "<color> water). Match each cup PRIMARILY by relative position (cup color is "
        "only a weak hint). For water content, use the asymmetric bias: if the "
        "criterion says empty, default to empty unless water is especially obvious; "
        "if it says contains water, default to contains water unless the cup is "
        "clearly empty. Score 1 only if the named cup's pose AND water content both "
        "hold under that rule; otherwise score 0:"
    ),
)


# ===========================================================================
# pulldrawer  (DEVELOPED — judge drawer OPEN/CLOSED strictly; object-inside
# match leniently by color + rough shape)
# ===========================================================================
PULLDRAWER_PROMPT_SET = PromptSet(
    system_prompt=(
        "You are an expert physical-status judge for a robot DRAWER open/close "
        "benchmark. The scene has one vertical column of 1x3 drawers (top/middle/"
        "bottom), and each drawer has a different color. The robot opens (pulls out) "
        "or closes (pushes in) drawers. An open drawer may contain an object "
        "(typically a fruit).\n\n"
        "You are shown ONE key frame plus success criteria. Criteria should identify "
        "drawers by COLOR only (no position words such as top/middle/bottom). When an "
        "open drawer contains an object, that is usually represented as TWO criteria: "
        "one for drawer-open state and one for object-inside state.\n\n"
        "Your job: for each criterion, return a binary 1 (clearly satisfied) or 0 "
        "(clearly not satisfied).\n\n"
        "IDENTITY / DISAMBIGUATION:\n"
        "- You may be given TWO images: IMAGE 1 is the episode's FIRST frame "
        "(reference), IMAGE 2 is the frame to JUDGE. If only one image is present, "
        "it is the frame to judge.\n"
        "- The user message may include an OBJECT INVENTORY (grounding_objects), such "
        "as ['yellow drawer', 'green fruit']. Use it to align naming, especially for "
        "the object inside a drawer.\n"
        "- For the object inside a drawer, be LENIENT: color and rough shape are "
        "enough. Do not require exact object category (e.g. green fruit can still "
        "match if it looks like a similar green fruit-like object).\n\n"
        "WHAT TO VERIFY:\n"
        "- Drawer OPEN/CLOSED state is the main signal and should be judged STRICTLY.\n"
        "- IMPORTANT: bias against false 'open'. Mark a drawer as OPEN only when there "
        "is clear geometric evidence that the drawer body is pulled out (e.g. visible "
        "inner cavity/interior volume, clearly exposed side rails/gap, or the drawer "
        "front protrudes forward from the cabinet frame).\n"
        "- If those strong cues are absent, treat the drawer as CLOSED. A visible knob/"
        "handle, shadows, slight perspective tilt, or gripper proximity/contact alone "
        "is NOT enough evidence of OPEN.\n"
        "- For 'All drawers are closed', return 1 whenever no drawer has strong open "
        "evidence. If uncertain/ambiguous, prefer CLOSED rather than OPEN.\n"
        "- A criterion like 'The yellow drawer is open.' scores 1 only if that drawer "
        "is clearly open.\n"
        "- For split criteria, evaluate each independently. Example:\n"
        "  * 'The yellow drawer is open.'\n"
        "  * 'A green fruit is inside the yellow drawer.'\n"
        "  The first checks drawer state only; the second checks inside-object only.\n"
        "- For an inside-object criterion (e.g. 'A green fruit is inside the yellow "
        "drawer.'), score 1 only when that same drawer is visibly OPEN and the object "
        "inside matches by color + rough shape. If the drawer appears closed or opening "
        "state is unclear, score 0 for the inside-object criterion.\n"
        "- A criterion like 'All drawers are closed.' scores 1 only if no drawer is "
        "open.\n"
        "- Reserve 0 for clear contradictions in drawer state (wrong drawer open, "
        "drawer closed when it should be open, any drawer open when all should be "
        "closed) or a clearly mismatched / missing inside object when object-inside is "
        "asserted.\n\n"
        "Return only valid JSON and no markdown."
    ),
    criteria_header=(
        "Success criteria for this frame. For each criterion, judge 1/0 whether it is "
        "visibly true. Judge drawer open/closed state strictly; judge object-inside "
        "leniently by color plus rough shape. Count a drawer as OPEN only with strong "
        "visual evidence (pulled-out geometry / visible interior cavity). For 'All "
        "drawers are closed', score 1 when no drawer has strong open evidence:"
    ),
)


# ---- Registry: subtask name (trial.task_id, lowercased) -> PromptSet --------
PROMPT_SETS: Dict[str, PromptSet] = {
    "pickfruit": PICKFRUIT_PROMPT_SET,
    "hangtowel": HANGTOWEL_PROMPT_SET,
    "pourwater": POURWATER_PROMPT_SET,
    "pulldrawer": PULLDRAWER_PROMPT_SET,
}

# Used when the subtask is unknown (keeps the eval running with a generic prompt).
DEFAULT_PROMPT_SET = PICKFRUIT_PROMPT_SET


def get_prompt_set(task_id: str | None) -> PromptSet:
    """Resolve the PromptSet for a subtask by its task_id (dataset subfolder name)."""
    if task_id:
        key = str(task_id).strip().lower()
        if key in PROMPT_SETS:
            return PROMPT_SETS[key]
        # Tolerate ids like "pickfruit_v2" / "10451-pickfruit" by substring match.
        for name, pset in PROMPT_SETS.items():
            if name in key:
                return pset
    return DEFAULT_PROMPT_SET


# =============================================================================
# ╚══════════════════════════ END PROMPT ENGINEERING ════════════════════════╝
# =============================================================================


def _render_user_prompt(
    spec: TaskSpec,
    step_description: str,
    prompt_set: PromptSet,
    success_criteria: Optional[str | List[str]] = None,
    constraints_hint: Optional[List[Dict[str, Any]]] = None,
    grounding_objects: Optional[List[str]] = None,
) -> str:
    """Assemble the user-side text from the given subtask PromptSet."""
    instruction = spec.instruction.strip() or "(no instruction provided)"
    criteria = _criteria_list(success_criteria)
    schema_tpl = prompt_set.schema_and_rules_template or COMMON_SCHEMA_AND_RULES

    parts = [
        prompt_set.instruction_line.format(instruction=instruction),
        prompt_set.checkpoint_line.format(step_description=step_description),
    ]
    # Episode object inventory (grounding_objects) — used to disambiguate which
    # known object a frame object is closest to, instead of guessing in a vacuum.
    inventory = [str(o).strip() for o in (grounding_objects or []) if str(o).strip()]
    if inventory:
        inv_text = "\n".join(f"- {o}" for o in inventory)
        parts.append(
            "OBJECT INVENTORY for this episode — the COMPLETE set of objects in the "
            "scene. Every fruit / container you see in the frame is ONE of these. "
            "When checking a criterion, match the object in the frame to the CLOSEST "
            "item in this inventory (by color, then rough shape/size) and confirm it "
            "is the criterion's named object rather than a different inventory item:\n"
            f"{inv_text}"
        )
    if success_criteria:
        criteria_text = "\n".join(
            f"{i + 1}. {criterion}" for i, criterion in enumerate(criteria)
        )
        parts.append(f"{prompt_set.criteria_header}\n{criteria_text}")
    if spec.has_structured_schema():
        parts.append(
            "Structured judging schema (use it to ground 'constraints'):\n"
            + json.dumps(spec.judging_schema, ensure_ascii=False, indent=2)
        )
    if constraints_hint:
        parts.append(
            "You MUST report a result for each of these named constraints:\n"
            + json.dumps(constraints_hint, ensure_ascii=False, indent=2)
        )
    parts.append(schema_tpl.format(n_criteria=len(criteria) or 1))
    parts.append(prompt_set.footer)
    return "\n\n".join(parts)


def _named_constraints(spec: TaskSpec) -> Optional[List[Dict[str, str]]]:
    """Extract a list of constraint names from a structured schema, if any."""
    if not spec.has_structured_schema():
        return None
    names: List[Dict[str, str]] = []
    schema = spec.judging_schema
    for obj in schema.get("objects", []) or []:
        if isinstance(obj, dict) and obj.get("name"):
            names.append(
                {
                    "name": f"object:{obj['name']}",
                    "spec": json.dumps(obj, ensure_ascii=False),
                }
            )
    for rel in schema.get("spatial_relations", []) or []:
        names.append({"name": f"relation:{rel}", "spec": json.dumps(rel, ensure_ascii=False)})
    return names or None


def _select_indices(
    n_frames: int,
    keysteps: Optional[List[int]],
    n_fallback: int,
) -> List[Dict[str, Any]]:
    """Return a list of `{frame_idx, description}` checkpoints to judge.

    The very last frame is always treated as the final checkpoint. If the
    episode has user-supplied key steps we respect them; otherwise we fall
    back to uniform sampling.
    """
    if n_frames <= 0:
        return []

    if keysteps:
        idxs = sorted({max(0, min(int(k), n_frames - 1)) for k in keysteps})
        if (n_frames - 1) not in idxs:
            idxs.append(n_frames - 1)
        return [{"frame": idx, "is_final": idx == n_frames - 1} for idx in idxs]

    if n_fallback <= 1:
        return [{"frame": n_frames - 1, "is_final": True}]
    sampled = uniform_indices(n_frames, n_fallback)
    if sampled[-1] != n_frames - 1:
        sampled[-1] = n_frames - 1
    return [
        {"frame": idx, "is_final": idx == n_frames - 1} for idx in sampled
    ]


def _criteria_to_constraints(
    success_criteria: Optional[str | List[str]],
) -> Optional[List[Dict[str, str]]]:
    if not success_criteria:
        return None
    if isinstance(success_criteria, list):
        return [
            {"name": f"criterion_{idx + 1}", "spec": str(criterion)}
            for idx, criterion in enumerate(success_criteria)
            if str(criterion).strip()
        ] or None
    return [{"name": "success_criteria", "spec": str(success_criteria)}]


def _criteria_list(success_criteria: Optional[str | List[str]]) -> List[str]:
    if not success_criteria:
        return []
    if isinstance(success_criteria, list):
        return [str(criterion).strip() for criterion in success_criteria if str(criterion).strip()]
    criterion = str(success_criteria).strip()
    return [criterion] if criterion else []


def _resolve_view_dir(video_dir: str, eval_view: str) -> str:
    """Return the concrete directory containing frame images for the requested view."""
    root = Path(video_dir)
    if list_frame_files(root):
        return str(root)
    view_dir = root / eval_view
    if view_dir.is_dir():
        return str(view_dir)
    # Backward compatibility with old docs that used `head` instead of `top_head`.
    aliases = {"top_head": "head", "head": "top_head"}
    alias = aliases.get(eval_view)
    if alias and (root / alias).is_dir():
        return str(root / alias)
    return str(view_dir)


def _map_prompt_frame_to_generated(
    end_frame: int,
    generated_n_frames: int,
    source_n_frames: Optional[int],
) -> int:
    """Map a GT prompt frame index to the generated sequence index."""
    if generated_n_frames <= 0:
        return 0
    if source_n_frames and source_n_frames > 1 and source_n_frames != generated_n_frames:
        ratio = end_frame / float(source_n_frames - 1)
        return max(0, min(int(round(ratio * (generated_n_frames - 1))), generated_n_frames - 1))
    return max(0, min(int(end_frame), generated_n_frames - 1))


def _select_prompt_segments(
    prompt_segments: List[Dict[str, Any]],
    n_frames: int,
) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for idx, segment in enumerate(prompt_segments):
        frame = _map_prompt_frame_to_generated(
            end_frame=int(segment["end_frame"]),
            generated_n_frames=n_frames,
            source_n_frames=segment.get("source_n_frames"),
        )
        steps.append(
            {
                "frame": frame,
                "source_frame": int(segment["end_frame"]),
                "segment_id": int(segment.get("segment_id", idx)),
                "success_criteria": segment.get("success_criteria", ""),
                "is_final": idx == len(prompt_segments) - 1,
            }
        )
    return steps


def _binary_score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "satisfied", "success", "pass"}:
            return 1
        if text in {"0", "false", "no", "not satisfied", "failure", "fail"}:
            return 0
    return None


def _extract_aspect_scores(parsed: Any) -> Optional[List[int]]:
    raw_scores: Any = None
    if isinstance(parsed, list):
        raw_scores = parsed
    elif isinstance(parsed, dict):
        for key in ("aspect_scores", "judgments", "judgements", "scores", "binary_scores"):
            if key in parsed:
                raw_scores = parsed[key]
                break
    if not isinstance(raw_scores, list):
        return None
    scores: List[int] = []
    for item in raw_scores:
        score = _binary_score(item)
        if score is None:
            return None
        scores.append(score)
    return scores


def _normalize_binary_judgment(
    parsed: Any,
    raw: str,
    success_criteria: Optional[str | List[str]] = None,
) -> Dict[str, Any]:
    criteria = _criteria_list(success_criteria)
    aspect_scores = _extract_aspect_scores(parsed)
    used_fallback_score = False

    if aspect_scores is None:
        if isinstance(parsed, dict):
            satisfied = bool(parsed.get("satisfied", parsed.get("success", False)))
            aspect_scores = [1 if satisfied else 0]
            used_fallback_score = True
        else:
            aspect_scores = []

    score = sum(aspect_scores) / len(aspect_scores) if aspect_scores else 0.0
    satisfied = bool(aspect_scores) and all(x == 1 for x in aspect_scores)

    if isinstance(parsed, dict):
        result = dict(parsed)
    else:
        result = {}

    reasoning = result.get("reasoning", result.get("thinking_process", []))
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = []

    constraints: List[Dict[str, Any]] = []
    for idx, aspect_score in enumerate(aspect_scores):
        criterion = criteria[idx] if idx < len(criteria) else f"criterion_{idx + 1}"
        evidence = str(reasoning[idx]) if idx < len(reasoning) else ""
        constraints.append(
            {
                "name": f"criterion_{idx + 1}",
                "satisfied": bool(aspect_score),
                "score": int(aspect_score),
                "spec": criterion,
                "evidence": evidence,
            }
        )

    result["raw"] = raw
    result["aspect_scores"] = aspect_scores
    result["score"] = score
    result["satisfied"] = satisfied
    result["format_valid"] = bool(aspect_scores) and (
        not criteria or (len(aspect_scores) == len(criteria) and not used_fallback_score)
    )
    result["constraints"] = constraints
    result.setdefault("reasoning", reasoning)
    result.setdefault(
        "explanation",
        f"{sum(aspect_scores)}/{len(aspect_scores)} success criteria satisfied."
        if aspect_scores
        else "No valid binary aspect scores returned.",
    )
    if criteria and len(aspect_scores) != len(criteria):
        result["warning"] = (
            f"Expected {len(criteria)} aspect scores, got {len(aspect_scores)}."
        )
    elif criteria and used_fallback_score:
        result["warning"] = (
            "Expected explicit aspect_scores array, but only found a legacy satisfied/success field."
        )
    return result


def _parse_vlm_json(
    raw: str,
    success_criteria: Optional[str | List[str]] = None,
) -> Dict[str, Any]:
    """Robustly parse the VLM's JSON reply."""
    if not raw:
        return {
            "satisfied": False,
            "score": 0.0,
            "aspect_scores": [],
            "format_valid": False,
            "explanation": "empty response",
            "raw": raw,
        }
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # remove a possible leading language tag like "json\n"
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        parsed = json_repair.loads(text)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to parse VLM JSON: %s; raw=%r", e, raw[:200])
        return {
            "satisfied": False,
            "score": 0.0,
            "aspect_scores": [],
            "format_valid": False,
            "explanation": f"parse error: {e}",
            "raw": raw,
        }
    if not isinstance(parsed, (dict, list)):
        return {
            "satisfied": False,
            "score": 0.0,
            "aspect_scores": [],
            "format_valid": False,
            "explanation": "VLM returned non-JSON-object/list output",
            "raw": raw,
        }
    return _normalize_binary_judgment(parsed, raw, success_criteria=success_criteria)


def _format_retry_instruction(
    previous_result: Dict[str, Any],
    success_criteria: Optional[str | List[str]],
) -> str:
    criteria = _criteria_list(success_criteria)
    expected = len(criteria) or 1
    reason = previous_result.get("warning") or previous_result.get("explanation") or "invalid format"
    return (
        "\n\nYour previous answer had invalid format and cannot be scored: "
        f"{reason}\n"
        "Retry now. Return ONLY valid JSON with this exact shape:\n"
        f'{{"aspect_scores": [<exactly {expected} integers, each 0 or 1>], '
        '"reasoning": ["brief evidence per aspect"], '
        '"explanation": "short summary"}}\n'
        "Do not return prose, markdown, booleans, floats, or a different number of scores."
    )


def _chat_with_retry(
    vlm: VLMBackend,
    messages: List[Dict[str, Any]],
    *,
    api_retries: int,
    api_retry_backoff: float,
) -> str:
    """Call ``vlm.chat`` with retries for transient network/API failures."""
    attempts = max(0, int(api_retries)) + 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return vlm.chat(messages)
        except Exception as exc:  # noqa: BLE001 - retry transient backend/network issues
            last_exc = exc
            if i < attempts - 1:
                delay = max(0.0, float(api_retry_backoff)) * (2 ** i)
                logger.warning(
                    "VLM request failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    i + 1,
                    attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _judge_step(
    vlm: VLMBackend,
    images: List[Image.Image],
    spec: TaskSpec,
    step_description: str,
    prompt_set: PromptSet,
    success_criteria: Optional[str | List[str]] = None,
    format_retries: int = 2,
    api_retries: int = 5,
    api_retry_backoff: float = 2.0,
    grounding_objects: Optional[List[str]] = None,
    reference_image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    constraints_hint = None if success_criteria else _named_constraints(spec)
    user_text = _render_user_prompt(
        spec,
        step_description,
        prompt_set,
        success_criteria=success_criteria,
        constraints_hint=constraints_hint,
        grounding_objects=grounding_objects,
    )

    last_result: Dict[str, Any] = {}
    retry_note = ""
    for attempt in range(max(0, format_retries) + 1):
        # backend.chat may resize PIL images in place, so pass fresh copies.
        content: List[Dict[str, Any]] = []
        if reference_image is not None:
            content.append({
                "type": "text",
                "text": (
                    "IMAGE 1 — REFERENCE: the episode's FIRST frame (initial scene). "
                    "Use it to recall the exact appearance and color of each fruit and "
                    "each basket / fruit plate in the inventory, then judge IMAGE 2."
                ),
            })
            content.append({"type": "image", "image": reference_image.copy()})
            content.append({"type": "text", "text": "IMAGE 2 — FRAME TO JUDGE:"})
        else:
            content.append({"type": "text", "text": "FRAME TO JUDGE:"})
        content.extend({"type": "image", "image": img} for img in images)
        content.append({"type": "text", "text": user_text + retry_note})
        messages = [
            {"role": "system", "content": prompt_set.system_prompt},
            {"role": "user", "content": content},
        ]
        raw = _chat_with_retry(
            vlm,
            messages,
            api_retries=api_retries,
            api_retry_backoff=api_retry_backoff,
        )
        result = _parse_vlm_json(raw, success_criteria=success_criteria)
        result["format_retry_count"] = attempt
        last_result = result
        if result.get("format_valid"):
            return result
        retry_note = _format_retry_instruction(result, success_criteria)

    last_result["format_retry_exhausted"] = True
    return last_result


def judge_one_trial(
    vlm: VLMBackend,
    trial: EpisodeTrial,
    n_fallback_keysteps: int = 1,
    max_image_long_side: int = 768,
    eval_view: str = "top_head",
    format_retries: int = 2,
    api_retries: int = 5,
    api_retry_backoff: float = 2.0,
    use_reference_frame: bool = True,
) -> Dict[str, Any]:
    """Run physical-status judging for a single trial."""
    spec = load_task_spec(trial.task_id, trial.instruction, trial.task_spec)
    prompt_set = get_prompt_set(trial.task_id)

    eval_video_dir = _resolve_view_dir(trial.video_dir, eval_view)
    frame_files = list_frame_files(eval_video_dir)
    if not frame_files:
        return {
            "score": 0.0,
            "final": {
                "satisfied": False,
                "score": 0.0,
                "explanation": f"no frames found in {eval_video_dir}",
            },
            "key_steps": [],
            "instruction": spec.instruction,
            "eval_view": eval_view,
        }

    # Reference image: the GT episode's FIRST frame (initial scene). Given to the
    # VLM alongside grounding_objects so it can recall each object's / basket's
    # appearance before judging. Prefer the GT video dir; fall back to the eval dir.
    reference_image: Optional[Image.Image] = None
    if use_reference_frame:
        ref_dir = (
            _resolve_view_dir(trial.gt_video_dir, eval_view)
            if trial.gt_video_dir
            else eval_video_dir
        )
        try:
            ref_imgs = load_frames_as_pil(
                ref_dir, indices=[0], max_long_side=max_image_long_side
            )
            reference_image = ref_imgs[0] if ref_imgs else None
        except Exception as e:  # noqa: BLE001 - reference is best-effort
            logger.warning("ep=%s: could not load reference first frame: %s", trial.episode_id, e)

    if trial.prompt_segments:
        steps = _select_prompt_segments(trial.prompt_segments, len(frame_files))
    else:
        steps = _select_indices(
            n_frames=len(frame_files),
            keysteps=trial.keysteps,
            n_fallback=n_fallback_keysteps,
        )

    step_results: List[Dict[str, Any]] = []
    final_result: Dict[str, Any] = {}
    scores: List[float] = []

    for step in steps:
        idx = step["frame"]
        images = load_frames_as_pil(
            eval_video_dir, indices=[idx], max_long_side=max_image_long_side
        )
        success_criteria = step.get("success_criteria")
        if success_criteria:
            desc = (
                f"Segment {step.get('segment_id', len(step_results))} ending at "
                f"generated frame {idx}"
            )
            if step.get("source_frame") != idx:
                desc += f" (mapped from GT frame {step.get('source_frame')})"
            desc += "."
        elif step.get("is_final"):
            desc = "Final frame: judge the overall task completion."
        else:
            desc = (
                f"Intermediate key step at frame {idx}: judge whether the "
                "expected progress has been made so far."
            )
        result = _judge_step(
            vlm,
            images,
            spec,
            desc,
            prompt_set,
            success_criteria=success_criteria,
            format_retries=format_retries,
            api_retries=api_retries,
            api_retry_backoff=api_retry_backoff,
            grounding_objects=trial.grounding_objects,
            reference_image=reference_image,
        )
        result["frame"] = idx
        if "source_frame" in step:
            result["source_frame"] = step["source_frame"]
        if "segment_id" in step:
            result["segment_id"] = step["segment_id"]
        if success_criteria:
            result["success_criteria"] = success_criteria
        result["is_final"] = bool(step.get("is_final"))
        step_results.append(result)
        if step.get("is_final"):
            final_result = result
        try:
            scores.append(float(result.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "score": mean_score,
        "final": final_result,
        "key_steps": step_results,
        "instruction": spec.instruction,
        "eval_view": eval_view,
        "video_dir": eval_video_dir,
    }


def compute_physical_judge(
    vlm: VLMBackend,
    trials: List[EpisodeTrial],
    n_fallback_keysteps: int = 1,
    max_image_long_side: int = 768,
    eval_view: str = "top_head",
    format_retries: int = 2,
    api_retries: int = 5,
    api_retry_backoff: float = 2.0,
    progress: bool = True,
    use_reference_frame: bool = True,
    save_root: "str | Path | None" = None,
    worker_vlms: Optional[Sequence[VLMBackend]] = None,
) -> Dict[str, Any]:
    """Run physical-status judging across all trials.

    Returns a nested dict keyed by `task_id -> episode_id -> trial_id`.

    When ``save_root`` (the generated-dataset root / val_base) is set, each
    episode's trials are written to
    ``<save_root>/<task>/<ep>/physical_judge.json`` as they finish, so results
    are crash-safe and resumable.
    """
    evaluation_settings = {
        "n_fallback_keysteps": int(n_fallback_keysteps),
        "max_image_long_side": int(max_image_long_side),
        "eval_view": str(eval_view),
        "format_retries": int(format_retries),
        "use_reference_frame": bool(use_reference_frame),
    }
    def evaluate_trial(
        backend: VLMBackend,
        trial: EpisodeTrial,
    ) -> tuple[Dict[str, Any], bool]:
        succeeded = True
        try:
            per_trial = judge_one_trial(
                backend,
                trial,
                n_fallback_keysteps=n_fallback_keysteps,
                max_image_long_side=max_image_long_side,
                eval_view=eval_view,
                format_retries=format_retries,
                api_retries=api_retries,
                api_retry_backoff=api_retry_backoff,
                use_reference_frame=use_reference_frame,
            )
        except Exception as e:  # pragma: no cover - keep eval robust
            succeeded = False
            logger.exception("physical_judge failed for %s: %s", trial.uid, e)
            per_trial = {
                "metric_version": 1,
                "status": "error",
                "score": 0.0,
                "final": {"satisfied": False, "score": 0.0, "explanation": f"runtime error: {e}"},
                "key_steps": [],
                "instruction": trial.instruction,
                "evaluation_settings": evaluation_settings,
            }
        if succeeded:
            per_trial.update(
                {
                    "metric_version": 1,
                    "status": "complete",
                    "uid": trial.uid,
                    "task_id": trial.task_id,
                    "episode_id": trial.episode_id,
                    "trial_id": trial.trial_id,
                    "evaluation_settings": evaluation_settings,
                }
            )
        return per_trial, succeeded

    backends = list(worker_vlms or [vlm])
    if not backends:
        raise ValueError("At least one VLM backend is required")
    backends = backends[: max(1, min(len(backends), len(trials) or 1))]
    completed: List[Optional[tuple[Dict[str, Any], bool]]] = [
        None
    ] * len(trials)
    persisted_episodes: Dict[tuple[str, str], Dict[str, Any]] = {}

    def persist_success(
        trial: EpisodeTrial,
        per_trial: Dict[str, Any],
        succeeded: bool,
    ) -> None:
        if save_root is None or not succeeded:
            return
        key = (trial.task_id, trial.episode_id)
        if key not in persisted_episodes:
            existing = episode_store.load_metric(
                save_root,
                trial.task_id,
                trial.episode_id,
                "physical_judge",
            )
            persisted_episodes[key] = (
                dict(existing) if isinstance(existing, dict) else {}
            )
        persisted_episodes[key][trial.trial_id] = per_trial
        # Parallel workers finish out of order. Canonicalize the mapping before
        # every crash-safe write so both the persisted JSON and the returned
        # result remain deterministic across runs.
        persisted_episodes[key] = dict(
            sorted(persisted_episodes[key].items())
        )
        path = episode_store.save_metric(
            save_root,
            trial.task_id,
            trial.episode_id,
            "physical_judge",
            persisted_episodes[key],
        )
        logger.info("Saved %s physical_judge result to %s", trial.uid, path)

    if len(backends) == 1:
        iterator = (
            tqdm(
                enumerate(trials),
                total=len(trials),
                desc="physical_judge",
            )
            if progress
            else enumerate(trials)
        )
        for index, trial in iterator:
            completed[index] = evaluate_trial(backends[0], trial)
            persist_success(trial, *completed[index])
    else:
        work_queue: queue.Queue[tuple[int, EpisodeTrial]] = queue.Queue()
        result_queue: queue.Queue[
            tuple[int, Dict[str, Any], bool]
        ] = queue.Queue()
        for index, trial in enumerate(trials):
            work_queue.put((index, trial))

        def worker(backend: VLMBackend) -> None:
            while True:
                try:
                    index, trial = work_queue.get_nowait()
                except queue.Empty:
                    return
                per_trial, succeeded = evaluate_trial(backend, trial)
                result_queue.put((index, per_trial, succeeded))
                work_queue.task_done()

        threads = [
            threading.Thread(
                target=worker,
                args=(backend,),
                name=f"physical-judge-{index}",
            )
            for index, backend in enumerate(backends)
        ]
        for thread in threads:
            thread.start()
        progress_bar = (
            tqdm(total=len(trials), desc="physical_judge")
            if progress
            else None
        )
        for _ in range(len(trials)):
            index, per_trial, succeeded = result_queue.get()
            completed[index] = (per_trial, succeeded)
            persist_success(trials[index], per_trial, succeeded)
            if progress_bar is not None:
                progress_bar.update(1)
        for thread in threads:
            thread.join()
        if progress_bar is not None:
            progress_bar.close()

    # Assemble deterministically and persist from this single writer. This
    # avoids read-modify-write races for legacy episodes with multiple trials.
    results: Dict[str, Any] = {}
    for trial, outcome in zip(trials, completed):
        if outcome is None:
            raise RuntimeError(f"No physical judge result for {trial.uid}")
        per_trial, succeeded = outcome
        task_results = results.setdefault(trial.task_id, {})
        if trial.episode_id not in task_results:
            existing = (
                episode_store.load_metric(
                    save_root,
                    trial.task_id,
                    trial.episode_id,
                    "physical_judge",
                )
                if save_root is not None
                else None
            )
            task_results[trial.episode_id] = (
                dict(existing) if isinstance(existing, dict) else {}
            )
        ep_trials = task_results[trial.episode_id]
        ep_trials[trial.trial_id] = per_trial
    return results
