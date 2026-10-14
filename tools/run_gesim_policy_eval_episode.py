#!/usr/bin/env python3
"""Replay one compact policy-eval episode through GE-Sim closed-loop inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
import pyarrow.parquet as pq
from PIL import Image  # type: ignore[import-not-found]


CAMERAS = ("top_head", "hand_left", "hand_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--geometry-profile", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--model-type", choices=("dmd", "teacher"), default="dmd")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--flow-sigmas", default="1.0,0.9375,0.8333,0.625")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        path, quality=95
    )


def main() -> None:
    args = parse_args()
    ge_sim_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ge_sim_root / "scripts"))

    from infer_inference_dataset import (  # type: ignore[import-not-found]  # pylint: disable=import-error,import-outside-toplevel
        DEFAULT_CONFIG,
        EpisodeJob,
        GESimClosedLoopService,
        MODEL_CHUNK_SIZE,
        build_inference_runtime,
    )
    from serve_inference_dataset import (  # type: ignore[import-not-found]  # pylint: disable=import-error,import-outside-toplevel
        profile_initial_loader,
    )

    asset_root = args.asset_root.expanduser().resolve()
    profile = args.geometry_profile.expanduser().resolve()
    episode_name = f"ep{args.episode:06d}"
    asset_dir = asset_root / args.task / episode_name
    output_dir = args.out_root.expanduser().resolve() / args.task / episode_name
    if not asset_dir.is_dir():
        raise FileNotFoundError(asset_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)

    metadata = json.loads((asset_dir / "metadata.json").read_text())
    schema = json.loads((profile / "schema.json").read_text())
    image_sizes = json.loads((profile / "image_sizes.json").read_text())
    table = pq.read_table(
        asset_dir / "trajectory.parquet",
        columns=["action", "observation.state"],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float32
    )
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"expected action shape (T, 16), got {actions.shape}")
    if states.shape != actions.shape:
        raise ValueError(
            f"expected state shape {actions.shape}, got {states.shape}"
        )

    views = {
        camera: load_image(asset_dir / "observations" / f"{camera}.png")
        for camera in CAMERAS
    }
    job = EpisodeJob(
        asset_dir=profile,
        task_name=args.task,
        episode_index=args.episode,
        metadata={
            "fps": float(metadata["fps"]),
            "calibration_source_sizes": image_sizes,
        },
        schema=schema,
    )
    trainer, pipe, model_path = build_inference_runtime(
        config=args.config or DEFAULT_CONFIG,
        model_path=args.model_path,
        model_type=args.model_type,
        flow_sigmas=args.flow_sigmas,
    )
    sample_size = tuple(
        int(value) for value in trainer.args.data["val"]["sample_size"]
    )
    service = GESimClosedLoopService(
        trainer=trainer,
        pipe=pipe,
        jobs=[job],
        setup_root=args.out_root.expanduser().resolve() / "_setup",
        urdf_path=profile / "kinematics.urdf",
        sample_size=sample_size,
        show_progress=args.show_progress,
        num_inference_steps=args.num_inference_steps,
        initial_loader=profile_initial_loader(profile),
    )
    service.reset(
        task_name=args.task,
        episode_index=args.episode,
        seed=args.seed,
        views=views,
        state=states[0],
        prompt=args.task,
    )
    for camera in CAMERAS:
        save_image(views[camera], output_dir / "video" / camera / "frame_00000.jpg")

    output_index = 1
    for start in range(1, len(actions), MODEL_CHUNK_SIZE):
        stop = min(len(actions), start + MODEL_CHUNK_SIZE)
        result = service.step(actions[start:stop])
        for camera in CAMERAS:
            frames = np.asarray(result["frames"][camera], dtype=np.uint8)
            for local_index, frame in enumerate(frames):
                save_image(
                    frame,
                    output_dir
                    / "video"
                    / camera
                    / f"frame_{start + local_index:05d}.jpg",
                )
        output_index = stop
        print(
            f"[window] input {start}:{stop}, output {output_index}/{len(actions)}",
            flush=True,
        )

    result_metadata = {
        "status": "complete",
        "task": args.task,
        "episode": episode_name,
        "input_frames": len(actions),
        "output_frames": output_index,
        "model_type": args.model_type,
        "model_path": str(model_path),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result_metadata, indent=2) + "\n"
    )
    print(f"[done] {output_dir}: {output_index} frames", flush=True)


if __name__ == "__main__":
    main()
