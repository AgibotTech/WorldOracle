"""Build the gt_dataset/ tree from LeRobot source datasets.

Layout produced (per the benchmark spec)::

    data_dir/
      gt_dataset/
        {task_name}/
          {episode_id}/                          e.g. ep000012
            prompt/
              init_frame_hand_left.png
              init_frame_hand_right.png
              init_frame_top_head.png
              prompt.json                        per-episode manifest
              data.npz                           action + K/dist/T_end_cam x 3 + meta
              enerverse_top_head/                only when calibration_present:
                action.npy                       (T, 16) absolute pose for EnerVerse-AC
                intrinsics.npy                   (3, 3) top_head K
                extrinsics.npy                   (4, 4) cam-to-world (currently T_end_cam)
                frame.png                        copy of init_frame_top_head.png
            raw_video/
              hand_left.mp4                      original HEVC (sliced for v3.0)
              hand_right.mp4
              top_head.mp4
            video/
              hand_left/   frame_00000.jpg ... frame_NNNNN.jpg
              hand_right/  ...
              top_head/    ...

Source dataset layouts supported:

* LeRobot v2   (each episode is its own mp4 + parquet)::

    <src>/meta/{info.json,episodes.jsonl,tasks.jsonl}
    <src>/data/chunk-XXX/episode_NNNNNN.parquet
    <src>/videos/chunk-XXX/observation.images.<cam>/episode_NNNNNN.mp4

* LeRobot v3.0 (many episodes packed into one big parquet + mp4)::

    <src>/meta/info.json
    <src>/meta/episodes/chunk-XXX/file-YYY.parquet       # rows index every episode
    <src>/data/chunk-XXX/file-YYY.parquet                # rows for the whole chunk
    <src>/videos/observation.images.<cam>/chunk-XXX/file-YYY.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMERAS = ("hand_left", "hand_right", "top_head")

# Camera-keys used in the source `camera_parameters` dict (LeRobot convention).
_CAM_SOURCE_KEYS: dict[str, tuple[str, str]] = {
    "hand_left":  ("hand_left_rgb",  "end_T_hand_left_rgbd"),
    "hand_right": ("hand_right_rgb", "end_T_hand_right_rgbd"),
    "top_head":   ("head_front_rgb", "end_T_head_front_rgbd"),
}


# --------------------------------------------------------------------------- #
# Source abstraction
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeSlice:
    """Everything needed to extract one episode from a (v2 or v3) source."""

    episode_index: int
    task_text: str
    length: int  # number of frames
    fps: int

    # action / state parquet locator
    parquet_path: Path
    row_from: int  # inclusive
    row_to: int    # exclusive

    # mp4 locator (one big mp4 may hold many episodes in v3.0)
    mp4_paths: dict[str, Path]  # cam -> path
    mp4_from_ts: float  # 0.0 for v2, real start for v3.0
    mp4_to_ts: float    # length/fps for v2, real end for v3.0


@dataclass
class SourceTask:
    name: str
    src_dir: Path
    episodes: list[int]
    info: dict = field(default_factory=dict)
    version: str = ""   # "v2" or "v3"
    _ep_meta: dict | None = None   # cached: ep_idx -> dict

    def __post_init__(self) -> None:
        self.info = json.loads((self.src_dir / "meta/info.json").read_text())
        cv = self.info.get("codebase_version", "")
        self.version = "v3" if cv.startswith("v3") else "v2"

    @property
    def chunk_size(self) -> int:
        return int(self.info.get("chunks_size", 1000))

    def _ensure_ep_meta_v3(self) -> dict:
        if self._ep_meta is not None:
            return self._ep_meta
        out: dict[int, dict] = {}
        files = sorted((self.src_dir / "meta/episodes").rglob("*.parquet"))
        for f in files:
            t = pq.read_table(f)
            ep_idx_col = t.column("episode_index").to_pylist()
            len_col = t.column("length").to_pylist()
            tasks_col = t.column("tasks").to_pylist()
            df_idx = t.column("dataset_from_index").to_pylist()
            dt_idx = t.column("dataset_to_index").to_pylist()
            d_chunk = t.column("data/chunk_index").to_pylist()
            d_file = t.column("data/file_index").to_pylist()
            cam_data: dict[str, dict[str, list]] = {}
            for cam in CAMERAS:
                p = f"videos/observation.images.{cam}"
                cam_data[cam] = {
                    "chunk": t.column(f"{p}/chunk_index").to_pylist(),
                    "file":  t.column(f"{p}/file_index").to_pylist(),
                    "from":  t.column(f"{p}/from_timestamp").to_pylist(),
                    "to":    t.column(f"{p}/to_timestamp").to_pylist(),
                }
            for i, ep in enumerate(ep_idx_col):
                out[int(ep)] = {
                    "length": int(len_col[i] or 0),
                    "tasks": tasks_col[i] or [],
                    "row_from": int(df_idx[i]),
                    "row_to": int(dt_idx[i]),
                    "data_chunk": int(d_chunk[i]),
                    "data_file": int(d_file[i]),
                    "video": {
                        cam: {
                            "chunk": int(cam_data[cam]["chunk"][i] or 0),
                            "file": int(cam_data[cam]["file"][i] or 0),
                            "from": float(cam_data[cam]["from"][i] or 0.0),
                            "to":   float(cam_data[cam]["to"][i] or 0.0),
                        }
                        for cam in CAMERAS
                    },
                }
        self._ep_meta = out
        return out

    def _slice_v2(self, ep: int) -> EpisodeSlice:
        c = ep // self.chunk_size
        mp4s = {
            cam: self.src_dir
            / f"videos/chunk-{c:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"
            for cam in CAMERAS
        }
        pq_path = self.src_dir / f"data/chunk-{c:03d}/episode_{ep:06d}.parquet"
        # task text + length come from episodes.jsonl
        task_text = ""
        length = 0
        ep_file = self.src_dir / "meta/episodes.jsonl"
        if ep_file.exists():
            with ep_file.open() as f:
                for line in f:
                    row = json.loads(line)
                    if int(row.get("episode_index", -1)) == ep:
                        tasks = row.get("tasks") or []
                        task_text = " ".join(str(t) for t in tasks)
                        length = int(row.get("length", 0))
                        break
        fps = int(self.info.get("fps", 30))
        # row_from/to: full parquet
        if length == 0:
            # cheaper than reading: peek 1 column row count
            length = pq.read_metadata(pq_path).num_rows
        return EpisodeSlice(
            episode_index=ep,
            task_text=task_text,
            length=length,
            fps=fps,
            parquet_path=pq_path,
            row_from=0,
            row_to=length,
            mp4_paths=mp4s,
            mp4_from_ts=0.0,
            mp4_to_ts=length / fps,
        )

    def _slice_v3(self, ep: int) -> EpisodeSlice:
        meta = self._ensure_ep_meta_v3()
        m = meta.get(ep)
        if m is None:
            raise KeyError(f"episode {ep} not found in v3 episodes meta")
        c, f = m["data_chunk"], m["data_file"]
        pq_path = self.src_dir / f"data/chunk-{c:03d}/file-{f:03d}.parquet"
        mp4s = {
            cam: self.src_dir
            / f"videos/observation.images.{cam}/chunk-{m['video'][cam]['chunk']:03d}/file-{m['video'][cam]['file']:03d}.mp4"
            for cam in CAMERAS
        }
        tasks = m["tasks"]
        task_text = " ".join(str(t) for t in tasks) if tasks else ""
        # Slice uses hand_left's timestamps (all three cams share them).
        from_ts = m["video"]["hand_left"]["from"]
        to_ts = m["video"]["hand_left"]["to"]
        return EpisodeSlice(
            episode_index=ep,
            task_text=task_text,
            length=m["length"],
            fps=int(self.info.get("fps", 30)),
            parquet_path=pq_path,
            row_from=m["row_from"],
            row_to=m["row_to"],
            mp4_paths=mp4s,
            mp4_from_ts=from_ts,
            mp4_to_ts=to_ts,
        )

    def slice(self, ep: int) -> EpisodeSlice:
        return self._slice_v3(ep) if self.version == "v3" else self._slice_v2(ep)


# --------------------------------------------------------------------------- #
# ffmpeg + parquet helpers
# --------------------------------------------------------------------------- #


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstderr:\n{res.stderr}"
        )


def _extract_init_frame(src_mp4: Path, from_ts: float, dst_png: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{from_ts:.3f}",
        "-i", str(src_mp4),
        "-frames:v", "1",
        "-vsync", "0",
        str(dst_png),
    ]
    _run(cmd)


def _extract_frames(
    src_mp4: Path, from_ts: float, duration: float, dst_dir: Path, jpg_quality: int
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    qscale = {95: "2", 90: "3", 85: "4"}.get(jpg_quality, "2")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{from_ts:.3f}",
        "-i", str(src_mp4),
        "-t", f"{duration:.3f}",
        "-q:v", qscale,
        "-vsync", "0",
        "-start_number", "0",
        str(dst_dir / "frame_%05d.jpg"),
    ]
    _run(cmd)
    return sum(1 for _ in dst_dir.glob("frame_*.jpg"))


def _copy_or_slice_mp4(
    src_mp4: Path,
    from_ts: float,
    duration: float,
    dst_mp4: Path,
    is_full_episode: bool,
) -> None:
    """v2: just copy. v3: stream-copy slice from a multi-episode mp4."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if is_full_episode:
        shutil.copy2(src_mp4, dst_mp4)
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{from_ts:.3f}",
        "-i", str(src_mp4),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst_mp4),
    ]
    _run(cmd)


def _read_actions(slc: EpisodeSlice) -> np.ndarray:
    """Read the action rows for this episode, sorted by frame_index."""
    t = pq.read_table(slc.parquet_path, columns=["action", "frame_index"])
    actions = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    fidx = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
    actions = actions[slc.row_from : slc.row_to]
    fidx = fidx[slc.row_from : slc.row_to]
    order = np.argsort(fidx)
    return actions[order]


# --------------------------------------------------------------------------- #
# Camera parameter parsing
# --------------------------------------------------------------------------- #


def _intrinsic_to_K_dist(intr: dict | None) -> tuple[np.ndarray, np.ndarray, dict]:
    if not intr:
        return (
            np.full((3, 3), np.nan, dtype=np.float64),
            np.full((5,), np.nan, dtype=np.float64),
            {},
        )
    fx = float(intr["Fx"]); fy = float(intr["Fy"])
    cx = float(intr["Cx"]); cy = float(intr["Cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(
        [
            float(intr.get("k1", 0.0)),
            float(intr.get("k2", 0.0)),
            float(intr.get("p1", 0.0)),
            float(intr.get("p2", 0.0)),
            float(intr.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    extra = {k: intr[k] for k in ("SN",) if k in intr}
    return K, dist, extra


def _quat_xyz_to_T(extr: dict | None) -> np.ndarray:
    if not extr:
        return np.full((4, 4), np.nan, dtype=np.float64)
    r = extr["rotation"]; t = extr["translation"]
    qw, qx, qy, qz = float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"])
    n = qw*qw + qx*qx + qy*qy + qz*qz
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s*qx*qw, s*qy*qw, s*qz*qw
    xx, xy, xz = s*qx*qx, s*qy*qx, s*qz*qx
    yy, yz = s*qy*qy, s*qz*qy
    zz = s*qz*qz
    R = np.array([
        [1.0 - (yy+zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx+zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx+yy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(t["x"]), float(t["y"]), float(t["z"])]
    return T


def _resolve_camera_params(
    src_info: dict, episode_index: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict, bool]:
    cp_all = src_info.get("camera_parameters") or {}
    cp = cp_all.get(str(episode_index)) or None
    if not cp:
        # Fallback: any non-empty episode in the dataset.
        for ep_str, ep_cp in cp_all.items():
            if ep_cp:
                cp = ep_cp
                break
    calibration_present = bool(cp) and "intrinsic_head_front_rgb" in cp

    per_cam_arrays: dict[str, dict[str, np.ndarray]] = {}
    per_cam_meta: dict[str, dict] = {}
    for cam, (intr_key, extr_key) in _CAM_SOURCE_KEYS.items():
        intr = (cp or {}).get(f"intrinsic_{intr_key}")
        extr = (cp or {}).get(f"extrinsic_{extr_key}")
        K, dist, intr_extra = _intrinsic_to_K_dist(intr)
        T = _quat_xyz_to_T(extr)
        per_cam_arrays[cam] = {"K": K, "dist": dist, "T_end_cam": T}
        res = (cp or {}).get("camera_resolution", {}).get(intr_key, {})
        per_cam_meta[cam] = {
            "source_intrinsic_key": f"intrinsic_{intr_key}",
            "source_extrinsic_key": f"extrinsic_{extr_key}",
            "intrinsic_present": intr is not None,
            "extrinsic_present": extr is not None,
            "capture_resolution": [
                int(res["width"]) if "width" in res else None,
                int(res["height"]) if "height" in res else None,
            ] if res else None,
            **intr_extra,
        }
    return per_cam_arrays, per_cam_meta, calibration_present


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _write_data_npz(
    actions: np.ndarray,
    per_cam_arrays: dict[str, dict[str, np.ndarray]],
    per_cam_meta: dict[str, dict],
    calib_present: bool,
    slc: EpisodeSlice,
    task_name: str,
    src_info: dict,
    src_dir: Path,
    dst_npz: Path,
) -> None:
    fd = (src_info.get("features", {}).get("action") or {}).get("field_descriptions")
    action_layout: dict[str, list[int]] = (
        {k: list(v.get("indices", [])) for k, v in fd.items()} if fd else {}
    )
    meta = {
        "schema_version": 1,
        "task": slc.task_text,
        "task_name": task_name,
        "episode_index": int(slc.episode_index),
        "fps": int(slc.fps),
        "length": int(slc.length),
        "action": {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "field_layout": action_layout,
        },
        "cameras": list(CAMERAS),
        "calibration_present": bool(calib_present),
        "frame_convention": "T_end_cam: end-frame -> camera-frame (column-major homogeneous)",
        "distortion_model": "opencv_radial_tangential_[k1,k2,p1,p2,k3]",
        "camera_meta": per_cam_meta,
        "source_dataset": str(src_dir),
        "source_codebase_version": src_info.get("codebase_version", ""),
    }
    arrays: dict[str, np.ndarray] = {"action": actions}
    for cam in CAMERAS:
        a = per_cam_arrays[cam]
        arrays[f"K_{cam}"] = a["K"]
        arrays[f"dist_{cam}"] = a["dist"]
        arrays[f"T_end_{cam}"] = a["T_end_cam"]
    arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False), dtype="U")
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **arrays)


def _build_enerverse_action_16d(
    actions: np.ndarray,
    src_info: dict,
    gripper_scale: float,
) -> np.ndarray:
    """Pack our raw action into the (T, 16) layout EnerVerse-AC consumes."""
    fd = (src_info.get("features", {}).get("action") or {}).get("field_descriptions") or {}
    le_idx = (fd.get("action/left_effector/position") or {}).get("indices") or []
    re_idx = (fd.get("action/right_effector/position") or {}).get("indices") or []
    ep_idx = (fd.get("action/end/position") or {}).get("indices") or []
    eo_idx = (fd.get("action/end/orientation") or {}).get("indices") or []
    if len(le_idx) != 1 or len(re_idx) != 1 or len(ep_idx) != 6 or len(eo_idx) != 8:
        raise RuntimeError(
            f"unexpected action field dims: le={le_idx} re={re_idx} ep={ep_idx} eo={eo_idx}"
        )
    T = actions.shape[0]
    out = np.zeros((T, 16), dtype=np.float64)
    out[:, 0:3]   = actions[:, ep_idx[0:3]]      # left xyz
    out[:, 8:11]  = actions[:, ep_idx[3:6]]      # right xyz
    # LeRobot stores quaternions as w,x,y,z; EnerVerse-AC expects x,y,z,w (scipy).
    le_quat_wxyz = actions[:, eo_idx[0:4]]
    re_quat_wxyz = actions[:, eo_idx[4:8]]
    out[:, 3:7]   = le_quat_wxyz[:, [1, 2, 3, 0]]
    out[:, 11:15] = re_quat_wxyz[:, [1, 2, 3, 0]]
    out[:, 7]  = actions[:, le_idx[0]] * gripper_scale   # left gripper 0..1 -> 0..120
    out[:, 15] = actions[:, re_idx[0]] * gripper_scale   # right gripper
    return out


def _write_enerverse_top_head(
    actions: np.ndarray,
    per_cam_arrays: dict[str, dict[str, np.ndarray]],
    src_info: dict,
    dst_dir: Path,
    init_frame_png: Path,
    gripper_scale: float,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    act16 = _build_enerverse_action_16d(actions, src_info, gripper_scale)
    np.save(dst_dir / "action.npy", act16)
    np.save(dst_dir / "intrinsics.npy", per_cam_arrays["top_head"]["K"])
    # EnerVerse-AC expects a camera-to-world transform.  We have T_end_cam from the
    # static factory calibration (end-frame -> camera-frame).  In task_4932 the
    # robot base is stationary throughout each episode, so using this static
    # extrinsic as a stand-in c2w is a reasonable first cut.  Down-stream code
    # can swap in the per-frame aligned extrinsic from observation.state if a
    # more accurate world frame is needed.
    np.save(dst_dir / "extrinsics.npy", per_cam_arrays["top_head"]["T_end_cam"])
    if init_frame_png.exists():
        shutil.copy2(init_frame_png, dst_dir / "frame.png")


# --------------------------------------------------------------------------- #
# Per-episode driver
# --------------------------------------------------------------------------- #


def _build_episode(
    task: SourceTask,
    ep: int,
    out_root: Path,
    jpg_quality: int,
    overwrite: bool,
    gripper_scale: float,
) -> dict:
    ep_id = f"ep{ep:06d}"
    ep_out = out_root / task.name / ep_id
    prompt_dir = ep_out / "prompt"
    video_dir = ep_out / "video"
    raw_video_dir = ep_out / "raw_video"

    if ep_out.exists() and not overwrite:
        return {"task": task.name, "episode": ep, "status": "skip-exists"}
    if ep_out.exists() and overwrite:
        shutil.rmtree(ep_out)

    slc = task.slice(ep)
    for cam in CAMERAS:
        if not slc.mp4_paths[cam].exists():
            raise FileNotFoundError(f"missing source mp4: {slc.mp4_paths[cam]}")
    if not slc.parquet_path.exists():
        raise FileNotFoundError(f"missing source parquet: {slc.parquet_path}")

    prompt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, slc.mp4_to_ts - slc.mp4_from_ts)
    is_full = task.version == "v2"
    try:
        for cam in CAMERAS:
            _extract_init_frame(
                slc.mp4_paths[cam], slc.mp4_from_ts, prompt_dir / f"init_frame_{cam}.png"
            )

        actions = _read_actions(slc)
        per_cam_arrays, per_cam_meta, calib_present = _resolve_camera_params(
            task.info, ep
        )
        _write_data_npz(
            actions,
            per_cam_arrays,
            per_cam_meta,
            calib_present,
            slc,
            task.name,
            task.info,
            task.src_dir,
            prompt_dir / "data.npz",
        )

        frame_counts: dict[str, int] = {}
        for cam in CAMERAS:
            n = _extract_frames(
                slc.mp4_paths[cam],
                slc.mp4_from_ts,
                duration,
                video_dir / cam,
                jpg_quality,
            )
            frame_counts[cam] = n

        raw_video_info: dict[str, dict] = {}
        for cam in CAMERAS:
            dst = raw_video_dir / f"{cam}.mp4"
            _copy_or_slice_mp4(
                slc.mp4_paths[cam], slc.mp4_from_ts, duration, dst, is_full
            )
            raw_video_info[cam] = {
                "path": f"raw_video/{cam}.mp4",
                "bytes": dst.stat().st_size if dst.exists() else None,
                "source": str(slc.mp4_paths[cam]),
                "from_ts": slc.mp4_from_ts,
                "to_ts": slc.mp4_to_ts,
            }

        # EnerVerse-AC export (top_head only). Only when we have a calibrated
        # camera; otherwise the .npy files would just be NaN.
        if calib_present:
            _write_enerverse_top_head(
                actions=actions,
                per_cam_arrays=per_cam_arrays,
                src_info=task.info,
                dst_dir=prompt_dir / "enerverse_top_head",
                init_frame_png=prompt_dir / "init_frame_top_head.png",
                gripper_scale=gripper_scale,
            )

        prompt = {
            "task": slc.task_text,
            "task_name": task.name,
            "episode_index": ep,
            "length": slc.length,
            "fps": slc.fps,
            "data_npz": "prompt/data.npz",
            "action": {
                "length": int(actions.shape[0]),
                "dim": int(actions.shape[1]),
                "dtype": "float32",
            },
            "calibration_present": bool(calib_present),
            "cameras": list(CAMERAS),
            "frames_per_camera": frame_counts,
            "raw_video": raw_video_info,
            "enerverse_action_map": (
                {
                    "dir": "prompt/enerverse_top_head",
                    "files": ["action.npy", "intrinsics.npy", "extrinsics.npy", "frame.png"],
                    "action_shape": [int(actions.shape[0]), 16],
                    "gripper_scale": gripper_scale,
                    "quat_order": "xyzw",
                    "extrinsic_kind": "static T_end_cam (treated as c2w)",
                    "camera": "top_head",
                }
                if calib_present
                else None
            ),
            "source_dataset": str(task.src_dir),
            "source_codebase_version": task.info.get("codebase_version", ""),
            "jpg_quality": jpg_quality,
        }
        (prompt_dir / "prompt.json").write_text(
            json.dumps(prompt, indent=2, ensure_ascii=False)
        )
    except Exception:
        shutil.rmtree(ep_out, ignore_errors=True)
        raise
    return {
        "task": task.name,
        "episode": ep,
        "status": "ok",
        "calibration_present": calib_present,
        "actions": int(actions.shape[0]),
        "frames": frame_counts,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=str, default="/path/to/output-data")
    p.add_argument(
        "--lerobot-root",
        type=str,
        default="/path/to/lerobot-source",
        help="Root containing the LeRobot source datasets",
    )
    p.add_argument(
        "--task",
        action="append",
        default=None,
        help=(
            "Task spec: <out_name>=<src_rel>:<ep1,ep2,...>. "
            "May be passed multiple times. If absent the default sample set is used."
        ),
    )
    p.add_argument("--jpg-quality", type=int, default=95)
    p.add_argument(
        "--gripper-scale",
        type=float,
        default=120.0,
        help="Source gripper is 0..1, EnerVerse-AC expects 0..120. Scale factor.",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


_DEFAULT_TASKS = [
    (
        "throw_trash",
        "5620/2025121901/gripper/task_4932",
        [0, 12, 45],
    ),
]


def _resolve_tasks(args) -> list[SourceTask]:
    lerobot_root = Path(args.lerobot_root)
    raw = args.task or [
        f"{n}={rel}:{','.join(str(e) for e in eps)}" for n, rel, eps in _DEFAULT_TASKS
    ]
    out: list[SourceTask] = []
    for spec in raw:
        try:
            name, rest = spec.split("=", 1)
            rel, eps = rest.split(":", 1)
            ep_list = [int(x) for x in eps.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"bad --task spec: {spec!r}") from exc
        out.append(SourceTask(name=name, src_dir=lerobot_root / rel, episodes=ep_list))
    return out


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH")

    out_root = Path(args.data_dir) / "gt_dataset"
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = _resolve_tasks(args)

    print(f"[build_gt] data_dir   = {args.data_dir}")
    print(f"[build_gt] output     = {out_root}")
    print(f"[build_gt] jpg_quality= {args.jpg_quality}")
    print(f"[build_gt] gripper x  = {args.gripper_scale}")
    print(f"[build_gt] tasks      = {len(tasks)}")
    for t in tasks:
        print(f"  - {t.name} ({t.version}) <- {t.src_dir}  episodes={t.episodes}")

    jobs: list[tuple[SourceTask, int]] = []
    for t in tasks:
        if not t.src_dir.is_dir():
            print(f"!! skip missing src: {t.src_dir}", file=sys.stderr)
            continue
        for ep in t.episodes:
            jobs.append((t, ep))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(
                _build_episode, t, ep, out_root, args.jpg_quality, args.overwrite,
                args.gripper_scale,
            ): (t, ep)
            for t, ep in jobs
        }
        for fut in as_completed(futs):
            t, ep = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                tag = f"calib={'Y' if r.get('calibration_present') else 'N'}"
                print(f"[ok] {t.name}/ep{ep:06d} {r['status']} {tag}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR] {t.name}/ep{ep:06d}: {exc}", file=sys.stderr, flush=True)
                results.append(
                    {"task": t.name, "episode": ep, "status": "error", "error": str(exc)}
                )

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[build_gt] done: {ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
