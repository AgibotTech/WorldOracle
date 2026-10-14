"""
Render EnerVerse-AC style action maps (trajectory maps) for every episode
under ``data/gt_dataset/<task>/<ep>``, for all three cameras:
``top_head``, ``hand_left``, ``hand_right``.

Per episode this script reads:
- ``prompt/data.npz``      action[T, 40], K_<cam>, T_end_<cam> for each camera
- ``prompt/prompt.json``   gripper_scale, fps, camera capture resolution
- ``prompt/init_frame_<cam>.png``  optional, used only as a sanity check

It writes:
- ``action_map/<cam>.mp4`` one mp4 per camera (default 640x480 to match the
  resized rgb frames in ``video/<cam>/``)

The renderer is a stand-alone reimplementation of
``LatentVisualDiffusion.get_traj`` from EnerVerse-AC -- no diffusion model
is loaded.  Action layout, gripper colormap and key-point conventions all
match the original code so the outputs are byte-comparable with the
``examples/`` shipped in EnerVerse-AC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import av  # PyAV (h264) - preferred, matches EnerVerse-AC training code
    _HAS_AV = True
except ImportError:
    _HAS_AV = False

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


# --------------------------------------------------------------------------- #
# Constants (mirrored from EnerVerse-AC/lvdm/data/traj_vis_statistics.py)
# --------------------------------------------------------------------------- #
ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0, 1.0],
        [0.0, 0.1, 0.0, 1.0],
        [0.0, 0.0, 0.1, 1.0],
    ],
    dtype=np.float64,
)
Gripper2EEFCvt = np.array(
    [
        [1.0, 0.0, 0.0, 0.00],
        [0.0, 1.0, 0.0, 0.00],
        [0.0, 0.0, 1.0, 0.23],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)

CAMERAS = ("top_head", "hand_left", "hand_right")


# --------------------------------------------------------------------------- #
# Action packing: data.npz raw action (T, 40) -> EnerVerse (T, 16)
# --------------------------------------------------------------------------- #
def _meta_field_layout(meta: dict) -> dict:
    """Resolve the {field_name: [indices]} action layout from data.npz meta."""
    action_meta = meta.get("action", {})
    layout = action_meta.get("field_layout")
    if layout:
        return layout
    fd = action_meta.get("field_descriptions", {}) or {}
    return {k: (v or {}).get("indices", []) for k, v in fd.items()}


def build_enerverse_action_16d(
    actions: np.ndarray,
    meta: dict,
    gripper_scale: float,
) -> np.ndarray:
    """
    Convert the raw (T, 40) action from ``data.npz`` to the (T, 16) layout
    EnerVerse-AC expects:  [left_xyz(3), left_quat_xyzw(4), left_grip(1),
                            right_xyz(3), right_quat_xyzw(4), right_grip(1)].
    """
    layout = _meta_field_layout(meta)
    le_idx = layout.get("action/left_effector/position", [])
    re_idx = layout.get("action/right_effector/position", [])
    ep_idx = layout.get("action/end/position", [])
    eo_idx = layout.get("action/end/orientation", [])
    if len(le_idx) != 1 or len(re_idx) != 1 or len(ep_idx) != 6 or len(eo_idx) != 8:
        raise RuntimeError(
            f"unexpected action field dims: le={le_idx} re={re_idx} "
            f"ep={ep_idx} eo={eo_idx}"
        )
    T = actions.shape[0]
    out = np.zeros((T, 16), dtype=np.float64)
    out[:, 0:3] = actions[:, ep_idx[0:3]]            # left xyz
    out[:, 8:11] = actions[:, ep_idx[3:6]]           # right xyz
    le_quat_wxyz = actions[:, eo_idx[0:4]]
    re_quat_wxyz = actions[:, eo_idx[4:8]]
    # LeRobot stores wxyz; EnerVerse-AC / scipy expect xyzw.
    out[:, 3:7] = le_quat_wxyz[:, [1, 2, 3, 0]]
    out[:, 11:15] = re_quat_wxyz[:, [1, 2, 3, 0]]
    out[:, 7] = actions[:, le_idx[0]] * gripper_scale
    out[:, 15] = actions[:, re_idx[0]] * gripper_scale
    return out


def load_action_16d(episode_dir: Path, gripper_scale: float) -> np.ndarray:
    """Prefer the pre-built enerverse_top_head/action.npy (camera-independent);
    fall back to rebuilding from data.npz when absent."""
    pre = episode_dir / "prompt" / "enerverse_top_head" / "action.npy"
    if pre.exists():
        return np.load(pre).astype(np.float64)
    npz = np.load(episode_dir / "prompt" / "data.npz", allow_pickle=False)
    meta = json.loads(str(npz["meta"]))
    return build_enerverse_action_16d(npz["action"], meta, gripper_scale)


# --------------------------------------------------------------------------- #
# Intrinsic scaling (same rule as EnerVerse-AC LatentVisualDiffusion.inference)
# --------------------------------------------------------------------------- #
def scale_intrinsic(
    intrinsic: np.ndarray, orig_hw: tuple, target_hw: tuple
) -> np.ndarray:
    intr = intrinsic.copy().astype(np.float64)
    h_scale = float(target_hw[0]) / float(orig_hw[0])
    w_scale = float(target_hw[1]) / float(orig_hw[1])
    intr[0, 0] *= w_scale
    intr[0, 2] *= w_scale
    intr[1, 1] *= h_scale
    intr[1, 2] *= h_scale
    return intr


# --------------------------------------------------------------------------- #
# Frame-convention bridge: benchmark action xyz uses (x_fwd, y_vertical,
# z_lateral) while the static T_end_<cam> matrices were calibrated under an
# AgiBot-style convention (x_fwd, y_lateral, z_vertical).  The two are related
# by a 90-deg rotation about the +x axis -- equivalently the axis permutation
# (x, y, z) -> (x, z, -y).  Apply it to ALL action SE(3) poses before composing
# with the camera extrinsics.
# --------------------------------------------------------------------------- #
_PRESET_AXIS_PERM = {
    # name -> (M  matrix)
    "identity": np.eye(3),
    "xzy":     np.array([[1, 0, 0], [0, 0, 1], [0, 1,  0]], dtype=np.float64),
    "xz-y":    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "x-zy":    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "x-z-y":   np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64),
    "yxz":     np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
    "y-x-z":   np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64),
}


def axis_perm_M4(name: str) -> np.ndarray:
    M = _PRESET_AXIS_PERM.get(name)
    if M is None:
        raise ValueError(
            f"unknown --axis_perm {name!r}; choices={list(_PRESET_AXIS_PERM)}"
        )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = M
    return out


# --------------------------------------------------------------------------- #
# Core renderer (mirror of ddpm3d.py:LatentVisualDiffusion.get_traj, numpy)
# --------------------------------------------------------------------------- #
def _xyz_quat_to_se3(xyz_quat_xyzw: np.ndarray) -> np.ndarray:
    """Pack (T, 7) [xyz + xyzw quat] into (T, 4, 4) homogeneous transforms."""
    T = xyz_quat_xyzw.shape[0]
    out = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    out[:, :3, :3] = Rotation.from_quat(xyz_quat_xyzw[:, 3:7]).as_matrix()
    out[:, :3, 3] = xyz_quat_xyzw[:, :3]
    return out


def render_action_map(
    pose16: np.ndarray,
    intrinsic: np.ndarray,
    out_hw: tuple,
    cam_kind: str,                 # 'static' or 'wrist_left' or 'wrist_right'
    T_end_cam: np.ndarray,         # static factory calibration matrix
    axis_perm: np.ndarray,         # (4, 4) bridging action <-> calibration frames
    radius: int = 50,
    line_thickness: int = 8,
    bg_value: int = 50,
) -> np.ndarray:
    """
    Render (T, H, W, 3) uint8 RGB action map frames.

    For ``cam_kind == 'static'`` the extrinsic is the (assumed-static) factory
    matrix `T_end_cam` (treated as c2w per the build_gt_dataset.py convention).

    For ``cam_kind == 'wrist_left' / 'wrist_right'`` the camera moves with that
    arm, so per-frame c2w is computed via forward kinematics:
        c2w(t) = T_world_<eef>(t) @ T_<eef>_cam,
    where ``T_<eef>_cam`` is the static factory matrix and ``T_world_<eef>(t)``
    comes from the action stream itself (in axis-permuted world frame).
    """
    h, w = out_hw
    T = pose16.shape[0]
    Mp3 = axis_perm[:3, :3]

    # Action stores rotation already self-consistent with the calibration frame
    # while the xyz uses a different axis convention.  Empirically the cleanest
    # alignment is obtained by permuting ONLY the translation column, leaving
    # the quaternion-derived rotation matrix untouched.  (Applying the full
    # SE(3) similarity Mp @ T @ Mp^{-1} would also rotate the static gripper
    # offset ``Gripper2EEFCvt`` into a wrong direction and push one arm well
    # outside the image.)
    pose_l = _xyz_quat_to_se3(pose16[:, 0:7])     # (T, 4, 4)
    pose_r = _xyz_quat_to_se3(pose16[:, 8:15])
    pose_l[:, :3, 3] = pose_l[:, :3, 3] @ Mp3.T   # row-vec @ M.T == M @ col-vec
    pose_r[:, :3, 3] = pose_r[:, :3, 3] @ Mp3.T

    if cam_kind == "static":
        w2c = np.linalg.inv(T_end_cam.astype(np.float64))
        w2c_l = w2c_r = np.broadcast_to(w2c, (T, 4, 4))
    elif cam_kind in ("wrist_left", "wrist_right"):
        host = pose_l if cam_kind == "wrist_left" else pose_r
        c2w_t = host @ T_end_cam.astype(np.float64)[None]
        w2c_t = np.linalg.inv(c2w_t)
        w2c_l = w2c_r = w2c_t
    else:
        raise ValueError(f"unknown cam_kind {cam_kind!r}")

    cvt = Gripper2EEFCvt[None]
    ee2cam_l = w2c_l @ pose_l @ cvt
    ee2cam_r = w2c_r @ pose_r @ cvt

    pts_homo = EndEffectorPts.T[None]
    pts_l = ee2cam_l @ pts_homo
    pts_r = ee2cam_r @ pts_homo

    eps = 1e-6
    z_l = np.where(np.abs(pts_l[:, 2:3, :]) < eps, np.nan, pts_l[:, 2:3, :])
    z_r = np.where(np.abs(pts_r[:, 2:3, :]) < eps, np.nan, pts_r[:, 2:3, :])

    uvs_l = (intrinsic[None] @ pts_l[:, :3, :]) / z_l
    uvs_r = (intrinsic[None] @ pts_r[:, :3, :]) / z_r
    uvs_l = np.nan_to_num(uvs_l[:, :2, :], nan=-1e9).transpose(0, 2, 1)
    uvs_r = np.nan_to_num(uvs_r[:, :2, :], nan=-1e9).transpose(0, 2, 1)
    z_pos_l = pts_l[:, 2, 0] > 0
    z_pos_r = pts_r[:, 2, 0] > 0

    frames = np.zeros((T, h, w, 3), dtype=np.uint8)
    frames[:] = bg_value
    for i in range(T):
        nl = float(pose16[i, 7]) / 120.0
        nr = float(pose16[i, 15]) / 120.0
        col_l = tuple(int(c * 255) for c in ColorMapLeft(nl)[:3])
        col_r = tuple(int(c * 255) for c in ColorMapRight(nr)[:3])

        for points, color, ok in (
            (uvs_l[i], col_l, z_pos_l[i]), (uvs_r[i], col_r, z_pos_r[i])
        ):
            if not ok:
                continue
            base = points[0]
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            cv2.circle(frames[i], (int(base[0]), int(base[1])), radius, color, -1)

        for points, axes_colors, ok in (
            (uvs_l[i], ColorListLeft, z_pos_l[i]),
            (uvs_r[i], ColorListRight, z_pos_r[i]),
        ):
            if not ok:
                continue
            base = points[0]
            if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                continue
            base_pt = (int(base[0]), int(base[1]))
            for k in range(1, points.shape[0]):
                tip = (int(points[k, 0]), int(points[k, 1]))
                cv2.line(frames[i], base_pt, tip, axes_colors[k - 1], line_thickness)
    return frames


# --------------------------------------------------------------------------- #
# Mp4 writers (av -> imageio -> cv2 fallback chain)
# --------------------------------------------------------------------------- #
def _write_mp4_av(frames: np.ndarray, out_path: str, fps: int):
    container = av.open(out_path, "w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    for i in range(frames.shape[0]):
        frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_mp4_imageio(frames: np.ndarray, out_path: str, fps: int):
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as w:
        for i in range(frames.shape[0]):
            w.append_data(frames[i])


def _write_mp4_cv2(frames: np.ndarray, out_path: str, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, ww = frames.shape[1], frames.shape[2]
    wr = cv2.VideoWriter(out_path, fourcc, fps, (ww, h))
    if not wr.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
    for i in range(frames.shape[0]):
        wr.write(frames[i][:, :, ::-1])  # RGB -> BGR
    wr.release()


def write_mp4(frames: np.ndarray, out_path: str, fps: int):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    last = None
    if _HAS_AV:
        try:
            _write_mp4_av(frames, out_path, fps)
            return
        except Exception as e:
            last = e
            print(f"  [warn] PyAV failed ({e}); falling back.")
    if _HAS_IMAGEIO:
        try:
            _write_mp4_imageio(frames, out_path, fps)
            return
        except Exception as e:
            last = e
            print(f"  [warn] imageio failed ({e}); falling back.")
    try:
        _write_mp4_cv2(frames, out_path, fps)
    except Exception as e:
        raise RuntimeError(f"all mp4 backends failed; last error: {last or e}") from e


# --------------------------------------------------------------------------- #
# Per-episode driver
# --------------------------------------------------------------------------- #
def render_episode(
    episode_dir: Path,
    out_root: Path | None,
    out_hw: tuple,
    cameras: tuple,
    fps: int | None,
    overwrite: bool,
    radius: int,
    line_thickness: int,
    axis_perm_name: str = "xz-y",
) -> dict:
    """Render the per-camera action map mp4s for a single episode."""
    prompt_dir = episode_dir / "prompt"
    npz_path = prompt_dir / "data.npz"
    prompt_json_path = prompt_dir / "prompt.json"
    if not npz_path.exists():
        return {"episode": str(episode_dir), "status": "skip-no-data.npz"}

    npz = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(npz["meta"]))

    with open(prompt_json_path) as f:
        prompt_meta = json.load(f)
    gripper_scale = float(
        prompt_meta.get("enerverse_action_map", {}).get("gripper_scale", 120.0)
    )
    src_fps = int(prompt_meta.get("fps", 30))
    out_fps = fps if fps is not None else src_fps

    pose16 = load_action_16d(episode_dir, gripper_scale)

    cam_meta = meta.get("camera_meta", {})
    target_dir = (out_root / episode_dir.parent.name / episode_dir.name
                  if out_root is not None else episode_dir / "action_map")
    target_dir.mkdir(parents=True, exist_ok=True)

    Mp = axis_perm_M4(axis_perm_name)
    cam_kind_map = {
        "top_head": "static",
        "hand_left": "wrist_left",
        "hand_right": "wrist_right",
    }

    written = []
    for cam in cameras:
        if cam not in CAMERAS:
            raise ValueError(f"unknown camera: {cam}")
        K = npz.get(f"K_{cam}")
        T_end = npz.get(f"T_end_{cam}")
        if K is None or T_end is None:
            print(f"  [skip] {cam}: missing K or T_end in data.npz")
            continue

        out_path = target_dir / f"{cam}.mp4"
        if out_path.exists() and not overwrite:
            written.append(str(out_path))
            print(f"  [skip] {cam}: {out_path} already exists")
            continue

        cap_w, cap_h = cam_meta.get(cam, {}).get("capture_resolution", [None, None])
        if cap_w is None or cap_h is None:
            init_png = prompt_dir / f"init_frame_{cam}.png"
            if init_png.exists():
                size = Image.open(init_png).size
                cap_w, cap_h = size[0], size[1]
            else:
                raise RuntimeError(
                    f"capture_resolution missing for {cam} and "
                    f"{init_png} not found"
                )
        scaled_K = scale_intrinsic(np.asarray(K), (cap_h, cap_w), out_hw)

        frames = render_action_map(
            pose16=pose16,
            intrinsic=scaled_K,
            out_hw=out_hw,
            cam_kind=cam_kind_map[cam],
            T_end_cam=np.asarray(T_end),
            axis_perm=Mp,
            radius=radius,
            line_thickness=line_thickness,
        )
        write_mp4(frames, str(out_path), fps=out_fps)
        written.append(str(out_path))
        print(
            f"  [ok]   {cam} ({cam_kind_map[cam]}): {out_path}  "
            f"({frames.shape[0]} frames, {out_hw[1]}x{out_hw[0]}, fps={out_fps})"
        )
    return {"episode": str(episode_dir), "status": "ok", "written": written}


# --------------------------------------------------------------------------- #
# Dataset driver
# --------------------------------------------------------------------------- #
def iter_episodes(dataset_root: Path, task_filter, episode_filter):
    for task_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if task_filter and task_dir.name not in task_filter:
            continue
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if episode_filter and ep_dir.name not in episode_filter:
                continue
            if not (ep_dir / "prompt" / "data.npz").exists():
                continue
            yield ep_dir


def main():
    parser = argparse.ArgumentParser(
        description="Render EnerVerse-AC action maps for top_head / hand_left / "
                    "hand_right across every episode in a gt_dataset tree."
    )
    parser.add_argument(
        "--dataset_root", "-r", type=str,
        default="/path/to/main-ground-truth",
        help="Root directory containing <task>/<ep> subfolders.",
    )
    parser.add_argument(
        "--out_root", "-o", type=str, default=None,
        help="If set, write mp4s under <out_root>/<task>/<ep>/<cam>.mp4. "
             "Default writes into each episode's action_map/ folder.",
    )
    parser.add_argument(
        "--cameras", "-c", type=str, nargs="+",
        default=list(CAMERAS), choices=list(CAMERAS),
        help="Subset of cameras to render. Default: all three.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Only process the given task names (e.g. throw_trash). "
             "Default: every task under dataset_root.",
    )
    parser.add_argument(
        "--episodes", type=str, nargs="*", default=None,
        help="Only process the given episode ids (e.g. ep000000). "
             "Default: every episode under each task.",
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"),
        help="Output frame size. Default 480 640 to overlay on stored rgb frames.",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Output fps. Default uses fps from each episode's prompt.json.",
    )
    parser.add_argument(
        "--radius", type=int, default=50, help="EEF circle radius in px.",
    )
    parser.add_argument(
        "--line_thickness", type=int, default=8,
        help="Axis-line thickness in px.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite any existing per-camera mp4.",
    )
    parser.add_argument(
        "--axis_perm", type=str, default="xz-y",
        choices=sorted(_PRESET_AXIS_PERM.keys()),
        help="Axis permutation bridging action xyz frame to the calibration "
             "frame.  Default 'xz-y' i.e. (x, y, z) -> (x, z, -y), which lines "
             "up the benchmark dataset action with the static T_end_cam calib.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        sys.exit(f"dataset_root not found: {dataset_root}")
    out_root = Path(args.out_root).resolve() if args.out_root else None
    out_hw = (int(args.size[0]), int(args.size[1]))

    task_filter = set(args.tasks) if args.tasks else None
    episode_filter = set(args.episodes) if args.episodes else None

    summary = []
    for ep_dir in iter_episodes(dataset_root, task_filter, episode_filter):
        print(f"[ep] {ep_dir.relative_to(dataset_root)}")
        info = render_episode(
            episode_dir=ep_dir,
            out_root=out_root,
            out_hw=out_hw,
            cameras=tuple(args.cameras),
            fps=args.fps,
            overwrite=args.overwrite,
            radius=args.radius,
            line_thickness=args.line_thickness,
            axis_perm_name=args.axis_perm,
        )
        summary.append(info)

    n_ok = sum(1 for s in summary if s.get("status") == "ok")
    print(f"\nDone: {n_ok}/{len(summary)} episodes rendered.")


if __name__ == "__main__":
    main()
