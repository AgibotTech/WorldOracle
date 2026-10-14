#!/usr/bin/env python3
"""Review SAM3 first-frame masks and save point prompts without video tracking.

Run this tool twice:
  * ``--target prompt`` annotates task objects in ``top_head``.
  * ``--target multiview`` annotates robot arms in all three camera views.

The UI runs SAM3 only on one image, so point refinement is responsive. The
separate ``sam3_track_saved_points.py`` tool later tracks all saved prompts.
"""

# pylint: disable=import-error,no-name-in-module

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import gradio as gr  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]


DEFAULT_DATASET = Path("/path/to/yam-ground-truth")
DEFAULT_POINTS = Path("/path/to/sam3-points")
DEFAULT_VIEWS = ("top_head", "hand_left", "hand_right")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FRAME_NUMBER = re.compile(r"(\d+)(?!.*\d)")


def add_sam3_to_path(path: Path | None) -> None:
    candidates = [
        path,
        Path(os.environ["SAM3_ROOT"]) if os.environ.get("SAM3_ROOT") else None,
        Path("/path/to/sam3"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "sam3").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return
    raise SystemExit("SAM3 source not found; pass --sam3-root or set SAM3_ROOT")


@dataclass
class ObjectPrompt:
    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class WorkItem:
    task: str
    episode: str
    episode_dir: Path
    target: str
    view: str
    frame_dir: Path
    first_frame_path: Path
    frame_count: int
    grounding_objects: tuple[str, ...]
    points_path: Path

    @property
    def key(self) -> str:
        return f"{self.task}/{self.episode}/{self.target}/{self.view}"


@dataclass
class AppState:
    index: int = -1
    item: WorkItem | None = None
    first_frame: np.ndarray | None = None
    objects: dict[int, ObjectPrompt] = field(
        default_factory=lambda: {1: ObjectPrompt()}
    )
    active_object: int = 1
    session_id: str | None = None
    outputs: dict[str, Any] | None = None


def natural_key(path: Path) -> tuple[int, str]:
    match = FRAME_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def frame_paths(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_key,
    )


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_grounding_objects(episode_dir: Path) -> tuple[str, ...]:
    path = episode_dir / "prompt" / "prompts.json"
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    objects = data.get("grounding_objects") or []
    if not isinstance(objects, list):
        objects = [objects]
    return tuple(str(value) for value in objects if str(value).strip())


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


class FirstFrameReviewApp:
    def __init__(
        self,
        dataset_root: Path,
        points_root: Path,
        tasks: list[str] | None,
        target: str,
        checkpoint_path: str,
        version: str,
        use_fa3: bool,
        compile_model: bool,
        seed: int,
        deterministic: bool,
    ) -> None:
        from sam3.model_builder import build_sam3_predictor  # type: ignore[import-not-found]
        from sam3.visualization_utils import (  # type: ignore[import-not-found]
            COLORS,
            render_masklet_frame,
        )

        self.dataset_root = dataset_root
        self.points_root = points_root
        self.points_root.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.seed = seed
        self.deterministic = deterministic
        self.colors = COLORS
        self.render_masklet_frame = render_masklet_frame
        self.build_sam3_predictor = build_sam3_predictor
        self.predictor = None
        self.predictor_options = {
            "checkpoint_path": checkpoint_path,
            "version": version,
            "compile": compile_model,
            "use_fa3": use_fa3,
        }
        self.items = self._scan(tasks)
        self.state = AppState()

    def _ensure_predictor(self) -> None:
        if self.predictor is None:
            seed_everything(self.seed, self.deterministic)
            self.predictor = self.build_sam3_predictor(
                **self.predictor_options
            )

    def _scan(self, requested_tasks: list[str] | None) -> list[WorkItem]:
        task_dirs = sorted(path for path in self.dataset_root.iterdir() if path.is_dir())
        if requested_tasks:
            wanted = set(requested_tasks)
            task_dirs = [path for path in task_dirs if path.name in wanted]
        views = ("top_head",) if self.target == "prompt" else DEFAULT_VIEWS
        items: list[WorkItem] = []
        for task_dir in task_dirs:
            for episode_dir in sorted(task_dir.glob("ep*")):
                if not episode_dir.is_dir():
                    continue
                grounding = read_grounding_objects(episode_dir)
                for view in views:
                    directory = episode_dir / "video" / view
                    frames = frame_paths(directory)
                    if not frames:
                        continue
                    points_path = (
                        self.points_root
                        / task_dir.name
                        / episode_dir.name
                        / f"{self.target}__{view}.json"
                    )
                    items.append(
                        WorkItem(
                            task=task_dir.name,
                            episode=episode_dir.name,
                            episode_dir=episode_dir,
                            target=self.target,
                            view=view,
                            frame_dir=directory,
                            first_frame_path=frames[0],
                            frame_count=len(frames),
                            grounding_objects=grounding,
                            points_path=points_path,
                        )
                    )
        return items

    def choices(self) -> list[str]:
        return [
            f"{index:04d} | {'SAVED' if item.points_path.is_file() else 'pending'} "
            f"| {item.key}"
            for index, item in enumerate(self.items)
        ]

    @staticmethod
    def choice_index(choice: str) -> int:
        try:
            return int(choice.split("|", 1)[0].strip())
        except (ValueError, AttributeError) as exc:
            raise gr.Error(f"invalid work-item choice: {choice}") from exc

    def _close_session(self, state: AppState) -> None:
        if state.session_id is not None and self.predictor is not None:
            with contextlib.suppress(Exception):
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": state.session_id}
                )
        state.session_id = None
        state.outputs = None

    def _ensure_session(self, state: AppState) -> None:
        if state.item is None:
            raise gr.Error("Load an episode/view first.")
        if state.session_id is not None:
            return
        self._ensure_predictor()
        assert self.predictor is not None
        seed_everything(self.seed, self.deterministic)
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                # Image-only session: fast first-frame validation, no video tracking.
                "resource_path": str(state.item.first_frame_path),
            }
        )
        state.session_id = response["session_id"]

    def _load_saved(self, item: WorkItem) -> dict[int, ObjectPrompt]:
        if not item.points_path.is_file():
            return {1: ObjectPrompt()}
        try:
            data = json.loads(item.points_path.read_text(encoding="utf-8"))
            objects = {
                int(object_id): ObjectPrompt(
                    points=[
                        [float(point[0]), float(point[1])]
                        for point in record.get("points", [])
                    ],
                    labels=[int(label) for label in record.get("labels", [])],
                )
                for object_id, record in (data.get("objects") or {}).items()
            }
            return objects or {1: ObjectPrompt()}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {1: ObjectPrompt()}

    def _point_count(self, state: AppState) -> int:
        return sum(len(prompt.points) for prompt in state.objects.values())

    def _reset_and_segment(self, state: AppState) -> None:
        nonempty = {
            object_id: prompt
            for object_id, prompt in state.objects.items()
            if prompt.points
        }
        if not nonempty:
            state.outputs = None
            return
        self._ensure_session(state)
        with contextlib.suppress(Exception):
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": state.session_id}
            )
        state.outputs = None
        assert state.first_frame is not None
        height, width = state.first_frame.shape[:2]
        for object_id, prompt in sorted(nonempty.items()):
            points = torch.tensor(
                [[x / width, y / height] for x, y in prompt.points],
                dtype=torch.float32,
            )
            labels = torch.tensor(prompt.labels, dtype=torch.int32)
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": state.session_id,
                    "frame_index": 0,
                    "points": points,
                    "point_labels": labels,
                    "obj_id": object_id,
                }
            )
            state.outputs = response["outputs"]

    def _render(self, state: AppState) -> np.ndarray:
        if state.first_frame is None:
            raise gr.Error("Load an episode/view first.")
        image = state.first_frame.copy()
        if state.outputs and len(state.outputs.get("out_probs", [])) > 0:
            image = self.render_masklet_frame(
                image, state.outputs, frame_idx=0, alpha=0.5
            )
        for object_id, prompt in state.objects.items():
            color = self.colors[object_id % len(self.colors)]
            ring = tuple(int(channel * 255) for channel in color)
            for (x, y), label in zip(prompt.points, prompt.labels):
                center = (int(round(x)), int(round(y)))
                cv2.circle(image, center, 10, ring, -1)
                cv2.circle(
                    image,
                    center,
                    5,
                    (0, 255, 0) if label == 1 else (255, 0, 0),
                    -1,
                )
                if object_id == state.active_object:
                    cv2.circle(image, center, 12, (255, 255, 255), 2)
        return image

    def _labels(self, state: AppState) -> tuple[str, str]:
        if state.item is None:
            return "No item loaded", ""
        item = state.item
        status = (
            f"[{state.index + 1}/{len(self.items)}] {item.key} | "
            f"{item.frame_count} frames | points={self._point_count(state)} | "
            f"{'saved' if item.points_path.is_file() else 'not saved'}"
        )
        inventory = (
            ", ".join(item.grounding_objects)
            if item.grounding_objects
            else "(no grounding_objects)"
        )
        guidance = (
            f"Target=task objects; inventory: {inventory}"
            if item.target == "prompt"
            else "Target=robot arms; normally create one object row per arm."
        )
        return status, guidance

    def load(self, choice: str, state: AppState):
        index = self.choice_index(choice)
        if not 0 <= index < len(self.items):
            raise gr.Error("work-item index out of range")
        self._close_session(state)
        item = self.items[index]
        state.index = index
        state.item = item
        state.first_frame = read_rgb(item.first_frame_path)
        state.objects = self._load_saved(item)
        state.active_object = min(state.objects)
        status, guidance = self._labels(state)
        return (
            self._render(state),
            status,
            guidance,
            gr.update(choices=sorted(state.objects), value=state.active_object),
            state,
        )

    def add_point(
        self,
        event: gr.SelectData,
        point_type: str,
        live_preview: bool,
        state: AppState,
    ):
        if state.first_frame is None:
            raise gr.Error("Load an episode/view first.")
        x, y = event.index
        prompt = state.objects.setdefault(state.active_object, ObjectPrompt())
        prompt.points.append([float(x), float(y)])
        prompt.labels.append(1 if point_type == "positive" else 0)
        started = time.monotonic()
        if live_preview:
            self._reset_and_segment(state)
        status, _ = self._labels(state)
        status += f" | preview={time.monotonic() - started:.2f}s"
        return self._render(state), status, state

    def segment(self, state: AppState):
        if self._point_count(state) == 0:
            raise gr.Error("Add at least one point.")
        started = time.monotonic()
        self._reset_and_segment(state)
        status, _ = self._labels(state)
        return (
            self._render(state),
            status + f" | first-frame SAM3 response={time.monotonic() - started:.2f}s",
            state,
        )

    def add_object(self, state: AppState):
        object_id = max(state.objects, default=0) + 1
        state.objects[object_id] = ObjectPrompt()
        state.active_object = object_id
        return (
            self._render(state),
            gr.update(choices=sorted(state.objects), value=object_id),
            f"Created object {object_id}.",
            state,
        )

    def set_active(self, object_id: int, state: AppState):
        state.active_object = int(object_id)
        return self._render(state), state

    def undo(self, state: AppState):
        prompt = state.objects.get(state.active_object)
        if prompt and prompt.points:
            prompt.points.pop()
            prompt.labels.pop()
            self._reset_and_segment(state)
        return self._render(state), self._labels(state)[0], state

    def clear(self, state: AppState):
        self._close_session(state)
        state.objects = {1: ObjectPrompt()}
        state.active_object = 1
        return (
            self._render(state),
            self._labels(state)[0],
            gr.update(choices=[1], value=1),
            state,
        )

    def save(self, state: AppState):
        if state.item is None or state.first_frame is None:
            raise gr.Error("Load an episode/view first.")
        nonempty = {
            object_id: prompt
            for object_id, prompt in state.objects.items()
            if prompt.points
        }
        if not nonempty:
            raise gr.Error("Add at least one point.")
        # Saving always recomputes the visible response, so points are never
        # accepted without SAM3 first-frame validation.
        self._reset_and_segment(state)
        height, width = state.first_frame.shape[:2]
        item = state.item
        data = {
            "schema_version": 1,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "task": item.task,
            "episode": item.episode,
            "episode_dir": str(item.episode_dir),
            "target": item.target,
            "view": item.view,
            "resource_path": str(item.frame_dir),
            "first_frame_path": str(item.first_frame_path),
            "frame_width": width,
            "frame_height": height,
            "num_frames": item.frame_count,
            "confirmed_first_frame_response": True,
            "objects": {
                str(object_id): {
                    "points": prompt.points,
                    "labels": prompt.labels,
                }
                for object_id, prompt in sorted(nonempty.items())
            },
        }
        item.points_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = item.points_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(item.points_path)
        status, _ = self._labels(state)
        return self._render(state), status + f" | wrote {item.points_path}", str(
            item.points_path
        ), state

    def save_next(self, state: AppState):
        _, _, saved_path, state = self.save(state)
        next_index = state.index + 1
        if next_index >= len(self.items):
            status, guidance = self._labels(state)
            return (
                self._render(state),
                status + " | all work items completed",
                guidance,
                gr.update(),
                gr.update(),
                saved_path,
                state,
            )
        next_choice = self.choices()[next_index]
        image, status, guidance, active, state = self.load(next_choice, state)
        return (
            image,
            status,
            guidance,
            gr.update(value=next_choice, choices=self.choices()),
            active,
            saved_path,
            state,
        )


def build_ui(application: FirstFrameReviewApp) -> gr.Blocks:
    title = f"SAM3 first-frame review — {application.target}"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"""
            # {title}
            Click positive/negative points and inspect the live SAM3 response.
            **Save points** stores prompts only; full-video tracking is a separate
            background step. Target: **{application.target}**.
            """
        )
        state = gr.State(application.state)
        choices = application.choices()
        with gr.Row():
            with gr.Column(scale=1):
                item = gr.Dropdown(
                    choices=choices,
                    value=choices[0] if choices else None,
                    label="Episode / target / view",
                )
                load_button = gr.Button("Load selected")
                status = gr.Textbox(label="Status", interactive=False)
                guidance = gr.Textbox(label="What to segment", interactive=False)
                active = gr.Dropdown(choices=[1], value=1, label="Active object row")
                add_object = gr.Button("New object row")
                point_type = gr.Radio(
                    ["positive", "negative"], value="positive", label="Point type"
                )
                live = gr.Checkbox(value=True, label="Live SAM3 response")
                with gr.Row():
                    undo = gr.Button("Undo point")
                    clear = gr.Button("Clear all")
                segment = gr.Button("Refresh first-frame response")
                save = gr.Button("Save points", variant="secondary")
                save_next = gr.Button("Save + next target", variant="primary")
                saved_file = gr.File(label="Saved annotation JSON")
            with gr.Column(scale=2):
                image = gr.Image(
                    type="numpy",
                    interactive=True,
                    label="First frame: mask response + point prompts",
                )

        load_button.click(
            application.load,
            [item, state],
            [image, status, guidance, active, state],
        )
        image.select(
            application.add_point,
            [point_type, live, state],
            [image, status, state],
        )
        add_object.click(
            application.add_object,
            [state],
            [image, active, status, state],
        )
        active.change(application.set_active, [active, state], [image, state])
        undo.click(application.undo, [state], [image, status, state])
        clear.click(
            application.clear,
            [state],
            [image, status, active, state],
        )
        segment.click(application.segment, [state], [image, status, state])
        save.click(
            application.save,
            [state],
            [image, status, saved_file, state],
        )
        save_next.click(
            application.save_next,
            [state],
            [image, status, guidance, item, active, saved_file, state],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--points-root", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--target", choices=("prompt", "multiview"), required=True
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--sam3-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-fa3", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Validate dataset discovery without starting Gradio or loading SAM3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_sam3_to_path(args.sam3_root)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    application = FirstFrameReviewApp(
        dataset_root=dataset_root,
        points_root=args.points_root.expanduser().resolve(),
        tasks=args.tasks,
        target=args.target,
        checkpoint_path=args.checkpoint_path,
        version=args.version,
        use_fa3=not args.no_fa3,
        compile_model=args.compile,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    if not application.items:
        raise SystemExit("no matching dataset frames found")
    print(
        f"[sam3-review] target={args.target} items={len(application.items)} "
        f"points_root={application.points_root} seed={args.seed}",
        flush=True,
    )
    if args.scan_only:
        return
    demo = build_ui(application)
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[
            str(dataset_root),
            str(application.points_root),
        ],
    )


if __name__ == "__main__":
    main()
