#!/usr/bin/env python3
"""Smooth web viewer for GT and generated baseline videos.

Frames are matched by frame number and transcoded once into cached H.264 clips.
The browser then plays two synchronized videos instead of requesting every
frame separately.

Example:
    python tools/visualize_baseline_results.py \
        /path/to/model-predictions \
        --baseline-name cosmos3_g02 --host 0.0.0.0
"""

# pylint: disable=global-statement,import-error

from __future__ import annotations

import argparse
import hashlib
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")
EPISODE_RE = re.compile(r"ep(\d+)$")


@dataclass
class Episode:
    task: str
    name: str
    index: int
    gt_frames: dict[int, Path]
    generated_frames: dict[int, Path]

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.gt_frames.keys() & self.generated_frames.keys())

    def summary(self) -> dict[str, object]:
        indices = self.frame_indices
        return {
            "task": self.task,
            "name": self.name,
            "index": self.index,
            "frame_count": len(indices),
            "first_frame": indices[0] if indices else None,
            "last_frame": indices[-1] if indices else None,
            "gt_count": len(self.gt_frames),
            "generated_count": len(self.generated_frames),
        }


app = Flask(__name__)
GENERATED_ROOT = Path()
GT_ROOT = Path()
CACHE_ROOT = Path()
VIEW = "top_head"
FPS = 30.0
BASELINE_NAME = ""
CRF = 25
PRESET = "veryfast"
CACHE_VERSION = secrets.token_hex(8)
REQUESTED_TASKS: list[str] | None = None
EXPECTED_TASKS = 4
EXPECTED_EPISODES = 100
_EPISODES: dict[tuple[str, str], Episode] = {}
_SCAN_LOCK = threading.Lock()
_CLIP_LOCKS_LOCK = threading.Lock()
_CLIP_LOCKS: dict[Path, threading.Lock] = {}
_PREWARM_LOCK = threading.Lock()
_PREWARM_STATE: dict[str, int | bool] = {
    "running": False,
    "total": 0,
    "done": 0,
    "failed": 0,
}


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ baseline_name }} · Baseline Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d11; --panel: #151820; --panel2: #1d222c;
      --line: #303746; --text: #eef1f7; --muted: #96a0b3;
      --accent: #75a9ff; --ratio: 4 / 3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; background: var(--bg); color: var(--text);
      font: 14px/1.4 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    header {
      position: sticky; top: 0; z-index: 10; display: flex; align-items: center;
      gap: 16px; padding: 14px 22px; border-bottom: 1px solid var(--line);
      background: rgba(11,13,17,.96);
    }
    h1 { margin: 0; font-size: 18px; }
    .baseline {
      padding: 4px 9px; border-radius: 6px; color: #b9d0ff;
      background: #1e3153; font-weight: 700;
    }
    .root { margin-left: auto; color: var(--muted); font-size: 12px; }
    main { max-width: 1800px; margin: auto; padding: 20px; }
    .selectors {
      display: grid; grid-template-columns: minmax(170px,1fr) minmax(220px,1fr) auto;
      gap: 12px; margin-bottom: 16px;
    }
    select, button, input {
      min-height: 38px; padding: 7px 11px; border: 1px solid var(--line);
      border-radius: 8px; background: var(--panel2); color: var(--text);
    }
    button { cursor: pointer; font-weight: 650; }
    button:hover { border-color: var(--accent); }
    .comparison {
      display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px;
    }
    .view {
      position: relative; overflow: hidden; border: 1px solid var(--line);
      border-radius: 10px; background: #000; aspect-ratio: var(--ratio);
    }
    video { display: block; width: 100%; height: 100%; object-fit: fill; }
    .badge {
      position: absolute; left: 10px; top: 10px; padding: 5px 9px;
      border-radius: 6px; background: rgba(0,0,0,.72); font-weight: 750;
      pointer-events: none;
    }
    .generated .badge { color: #a7c5ff; }
    .loading {
      display: none; position: absolute; inset: 0; place-items: center;
      background: rgba(0,0,0,.65); z-index: 2; font-weight: 700;
    }
    .controls {
      margin-top: 14px; padding: 14px; border: 1px solid var(--line);
      border-radius: 10px; background: var(--panel);
    }
    .transport {
      display: grid; grid-template-columns: auto auto auto minmax(180px,1fr) auto;
      align-items: center; gap: 9px;
    }
    #slider { width: 100%; padding: 0; accent-color: var(--accent); }
    .status {
      display: flex; justify-content: space-between; gap: 12px;
      margin-top: 11px; color: var(--muted); font-variant-numeric: tabular-nums;
    }
    .error {
      display: none; margin-top: 12px; padding: 10px 12px; border-radius: 8px;
      color: #ffb5b5; background: #421d24;
    }
    kbd { padding: 1px 5px; border: 1px solid var(--line); border-radius: 4px; }
    @media (max-width: 850px) {
      .comparison,.selectors { grid-template-columns: 1fr; }
      .root { display: none; }
      .transport { grid-template-columns: repeat(3,auto); }
      #slider { grid-column: 1/-1; grid-row: 2; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Baseline Result Viewer</h1>
    <span class="baseline">{{ baseline_name }}</span>
    <span id="datasetCount"></span>
    <span class="root">{{ generated_root }}</span>
  </header>
  <main>
    <section class="selectors">
      <select id="task"></select>
      <select id="episode"></select>
      <button id="reload">Rescan dataset</button>
    </section>
    <section class="comparison" id="comparison">
      <div class="view">
        <video id="gt" muted playsinline preload="metadata"></video>
        <span class="badge">GT</span>
        <span class="loading">Preparing video…</span>
      </div>
      <div class="view generated">
        <video id="generated" muted playsinline preload="metadata"></video>
        <span class="badge">{{ baseline_name }}</span>
        <span class="loading">Preparing video…</span>
      </div>
    </section>
    <section class="controls">
      <div class="transport">
        <button id="previousEpisode">Prev EP</button>
        <button id="play">Play</button>
        <button id="nextEpisode">Next EP</button>
        <input id="slider" type="range" min="0" max="1" value="0" step="0.001">
        <span id="frameLabel">frame —</span>
      </div>
      <div class="status">
        <span id="episodeInfo">Loading…</span>
        <span id="cacheInfo">H.264 clip cache</span>
        <span><kbd>←</kbd>/<kbd>→</kbd> frame · <kbd>Space</kbd> play</span>
      </div>
      <div class="error" id="error"></div>
    </section>
  </main>
  <script>
    const fps = {{ fps | tojson }};
    const cacheVersion = {{ cache_version | tojson }};
    const taskSelect = document.querySelector("#task");
    const episodeSelect = document.querySelector("#episode");
    const slider = document.querySelector("#slider");
    const gt = document.querySelector("#gt");
    const generated = document.querySelector("#generated");
    const playButton = document.querySelector("#play");
    const errorBox = document.querySelector("#error");
    let episodes = [];
    let indices = [];
    let syncing = false;

    async function getJSON(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }
    function showError(message="") {
      errorBox.textContent = message;
      errorBox.style.display = message ? "block" : "none";
    }
    function clipURL(source) {
      const task = encodeURIComponent(taskSelect.value);
      const ep = encodeURIComponent(episodeSelect.value);
      return `/clip.mp4?source=${source}&task=${task}&episode=${ep}&v=${cacheVersion}`;
    }
    function showLoading(show) {
      document.querySelectorAll(".loading").forEach(
        node => node.style.display = show ? "grid" : "none");
    }
    function updateTimeline() {
      if (!Number.isFinite(gt.duration)) return;
      slider.max = gt.duration;
      slider.value = gt.currentTime;
      const position = Math.min(
        indices.length - 1, Math.max(0, Math.floor(gt.currentTime * fps)));
      const original = indices[position];
      document.querySelector("#frameLabel").textContent =
        original === undefined ? "frame —" :
        `frame ${String(original).padStart(5,"0")} · ${position + 1}/${indices.length}`;
      if (!generated.paused && Math.abs(generated.currentTime - gt.currentTime) > 0.08) {
        generated.currentTime = gt.currentTime;
      }
    }
    async function loadTasks(rescan=false) {
      showError();
      const data = await getJSON(`/api/tasks${rescan ? "?reload=1" : ""}`);
      taskSelect.replaceChildren(...data.tasks.map(task => new Option(task, task)));
      document.querySelector("#datasetCount").textContent =
        `${data.task_count} tasks · ${data.episode_count} episodes`;
      if (!data.tasks.length) throw new Error("No common tasks found");
      await loadEpisodes();
    }
    async function loadEpisodes() {
      pauseVideos();
      const data = await getJSON(`/api/episodes/${encodeURIComponent(taskSelect.value)}`);
      episodes = data.episodes;
      episodeSelect.replaceChildren(...episodes.map(ep =>
        new Option(`${ep.name}  (${ep.frame_count} frames)`, ep.name)));
      if (!episodes.length) throw new Error("No common episodes");
      await loadEpisode();
    }
    async function loadEpisode() {
      pauseVideos();
      showError();
      showLoading(true);
      const task = encodeURIComponent(taskSelect.value);
      const ep = encodeURIComponent(episodeSelect.value);
      const data = await getJSON(`/api/episode/${task}/${ep}`);
      indices = data.frame_indices;
      document.querySelector("#episodeInfo").textContent =
        `${data.task}/${data.name} · matched ${data.frame_count} · ` +
        `GT ${data.gt_count} · generated ${data.generated_count}`;
      gt.src = clipURL("gt");
      generated.src = clipURL("generated");
      gt.load();
      generated.load();
    }
    function pauseVideos() {
      gt.pause(); generated.pause(); playButton.textContent = "Play";
    }
    async function togglePlay() {
      if (gt.paused) {
        generated.currentTime = gt.currentTime;
        try {
          await Promise.all([gt.play(), generated.play()]);
          playButton.textContent = "Pause";
        } catch (error) { showError(error.message); }
      } else {
        pauseVideos();
      }
    }
    function seek(time) {
      const duration = Number.isFinite(gt.duration) ? gt.duration : 0;
      const target = Math.max(0, Math.min(time, duration));
      syncing = true;
      gt.currentTime = target;
      generated.currentTime = target;
      syncing = false;
      updateTimeline();
    }
    function switchEpisode(offset) {
      const next = episodeSelect.selectedIndex + offset;
      if (next < 0 || next >= episodeSelect.options.length) return;
      episodeSelect.selectedIndex = next;
      loadEpisode().catch(error => showError(error.message));
    }

    gt.addEventListener("loadedmetadata", () => {
      slider.max = gt.duration;
      if (gt.videoWidth && gt.videoHeight) {
        document.querySelector("#comparison").style.setProperty(
          "--ratio", `${gt.videoWidth}/${gt.videoHeight}`);
      }
    });
    for (const video of [gt, generated]) {
      video.addEventListener("canplay", () => {
        if (gt.readyState >= 3 && generated.readyState >= 3) showLoading(false);
      });
      video.addEventListener("error", () => {
        showLoading(false);
        showError("Video preparation failed. Check the server log and ffmpeg.");
      });
    }
    gt.addEventListener("timeupdate", updateTimeline);
    gt.addEventListener("pause", () => {
      if (!syncing) { generated.pause(); playButton.textContent = "Play"; }
    });
    gt.addEventListener("ended", pauseVideos);
    slider.addEventListener("input", () => seek(Number(slider.value)));
    playButton.addEventListener("click", togglePlay);
    taskSelect.addEventListener("change", () =>
      loadEpisodes().catch(error => showError(error.message)));
    episodeSelect.addEventListener("change", () =>
      loadEpisode().catch(error => showError(error.message)));
    document.querySelector("#reload").addEventListener("click", () =>
      loadTasks(true).catch(error => showError(error.message)));
    document.querySelector("#previousEpisode").addEventListener("click", () => switchEpisode(-1));
    document.querySelector("#nextEpisode").addEventListener("click", () => switchEpisode(1));
    document.addEventListener("keydown", event => {
      if (event.target.matches("select,input")) return;
      if (event.code === "Space") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft") seek(gt.currentTime - 1 / fps);
      if (event.key === "ArrowRight") seek(gt.currentTime + 1 / fps);
    });
    loadTasks().catch(error => showError(error.message));
  </script>
</body>
</html>
"""


def natural_key(value: str | Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(value).name)
    ]


def indexed_frames(directory: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    if not directory.is_dir():
        return frames
    for path in sorted(directory.iterdir(), key=natural_key):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = FRAME_NUMBER_RE.search(path.stem)
        if match is not None:
            frames.setdefault(int(match.group(1)), path)
    return frames


def directory_names(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir()}


def scan_dataset() -> dict[tuple[str, str], Episode]:
    if not GENERATED_ROOT.is_dir():
        raise RuntimeError(f"Generated root does not exist: {GENERATED_ROOT}")
    if not GT_ROOT.is_dir():
        raise RuntimeError(f"GT root does not exist: {GT_ROOT}")
    common_tasks = directory_names(GENERATED_ROOT) & directory_names(GT_ROOT)
    tasks = REQUESTED_TASKS or sorted(common_tasks, key=natural_key)
    episodes: dict[tuple[str, str], Episode] = {}
    for task in tasks:
        generated_task = GENERATED_ROOT / task
        gt_task = GT_ROOT / task
        if not generated_task.is_dir() or not gt_task.is_dir():
            continue
        names = directory_names(generated_task) & directory_names(gt_task)
        for name in sorted(names, key=natural_key):
            match = EPISODE_RE.fullmatch(name)
            if match is None:
                continue
            gt_frames = indexed_frames(gt_task / name / "video" / VIEW)
            generated_frames = indexed_frames(
                generated_task / name / "video" / VIEW
            )
            if not (gt_frames.keys() & generated_frames.keys()):
                continue
            episodes[(task, name)] = Episode(
                task, name, int(match.group(1)), gt_frames, generated_frames
            )
    return episodes


def refresh_dataset() -> None:
    global _EPISODES
    with _SCAN_LOCK:
        _EPISODES = scan_dataset()


def get_episode(task: str, name: str) -> Episode:
    episode = _EPISODES.get((task, name))
    if episode is None:
        abort(404)
    return episode


def dataset_cache_tag() -> str:
    identity = f"{GENERATED_ROOT}|{GT_ROOT}|{VIEW}|{FPS:.8g}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def clip_path(episode: Episode, source: str) -> Path:
    return (
        CACHE_ROOT
        / dataset_cache_tag()
        / episode.task
        / f"{episode.name}__{source}.mp4"
    )


def clip_lock(path: Path) -> threading.Lock:
    with _CLIP_LOCKS_LOCK:
        return _CLIP_LOCKS.setdefault(path, threading.Lock())


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def target_resolution(episode: Episode) -> tuple[int, int]:
    first_index = episode.frame_indices[0]
    with Image.open(episode.gt_frames[first_index]) as image:
        width, height = image.size
    return width - width % 2, height - height % 2


def transcode_clip(episode: Episode, source: str, destination: Path) -> None:
    if source not in {"gt", "generated"}:
        raise ValueError(f"Unknown source: {source}")
    frame_map = episode.gt_frames if source == "gt" else episode.generated_frames
    files = [frame_map[index] for index in episode.frame_indices]
    width, height = target_resolution(episode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    frame_duration = 1.0 / FPS
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ffconcat",
        prefix=f"{episode.task}_{episode.name}_{source}_",
        dir=destination.parent,
        encoding="utf-8",
        delete=False,
    ) as manifest:
        manifest_path = Path(manifest.name)
        manifest.write("ffconcat version 1.0\n")
        for path in files:
            manifest.write(f"file '{ffconcat_quote(path)}'\n")
            manifest.write(f"duration {frame_duration:.12f}\n")
        # concat demuxer ignores the final duration unless the last frame repeats.
        manifest.write(f"file '{ffconcat_quote(files[-1])}'\n")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-r",
        f"{FPS:.8g}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(destination)
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed for {episode.task}/{episode.name}/{source}"
        ) from error
    finally:
        manifest_path.unlink(missing_ok=True)


def ensure_clip(episode: Episode, source: str) -> Path:
    destination = clip_path(episode, source)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with clip_lock(destination):
        if not destination.is_file() or destination.stat().st_size == 0:
            transcode_clip(episode, source, destination)
    return destination


def prewarm_clips(workers: int) -> None:
    with _PREWARM_LOCK:
        if bool(_PREWARM_STATE["running"]):
            return
        jobs = [
            (episode, source)
            for episode in _EPISODES.values()
            for source in ("gt", "generated")
            if not clip_path(episode, source).is_file()
        ]
        _PREWARM_STATE.update(
            running=True, total=len(jobs), done=0, failed=0
        )

    def run() -> None:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(ensure_clip, episode, source)
                for episode, source in jobs
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                    _PREWARM_STATE["done"] = int(_PREWARM_STATE["done"]) + 1
                except (OSError, RuntimeError, ValueError):
                    _PREWARM_STATE["failed"] = (
                        int(_PREWARM_STATE["failed"]) + 1
                    )
        _PREWARM_STATE["running"] = False

    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        generated_root=str(GENERATED_ROOT),
        baseline_name=BASELINE_NAME,
        fps=FPS,
        cache_version=CACHE_VERSION,
    )


@app.get("/api/tasks")
def api_tasks():
    if request.args.get("reload"):
        refresh_dataset()
    tasks = sorted({task for task, _ in _EPISODES}, key=natural_key)
    return jsonify(
        tasks=tasks,
        task_count=len(tasks),
        episode_count=len(_EPISODES),
        baseline_name=BASELINE_NAME,
    )


@app.get("/api/episodes/<task>")
def api_episodes(task: str):
    episodes = sorted(
        (
            episode.summary()
            for (episode_task, _), episode in _EPISODES.items()
            if episode_task == task
        ),
        key=lambda item: int(item["index"]),
    )
    return jsonify(task=task, episodes=episodes)


@app.get("/api/episode/<task>/<name>")
def api_episode(task: str, name: str):
    episode = get_episode(task, name)
    return jsonify(**episode.summary(), frame_indices=episode.frame_indices)


@app.get("/clip.mp4")
def clip():
    source = request.args.get("source", "")
    if source not in {"gt", "generated"}:
        abort(400)
    episode = get_episode(
        request.args.get("task", ""), request.args.get("episode", "")
    )
    path = ensure_clip(episode, source)
    return send_file(
        path,
        mimetype="video/mp4",
        conditional=True,
        max_age=86400,
    )


@app.route("/api/prewarm", methods=["GET", "POST"])
def api_prewarm():
    workers = max(1, min(16, int(request.args.get("workers", "4"))))
    prewarm_clips(workers)
    return jsonify(_PREWARM_STATE)


@app.get("/api/prewarm/status")
def api_prewarm_status():
    return jsonify(_PREWARM_STATE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve cached, synchronized GT/generated H.264 videos."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help="Generated dataset root, e.g. .../cosmos3_g02_dataset.",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        help="Same as positional dataset_root; retained for explicit usage.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path(
            "/path/to/main-ground-truth"
        ),
    )
    parser.add_argument(
        "--baseline-name",
        help="Name shown in the UI; inferred from generated root by default.",
    )
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--expected-tasks", type=int, default=4)
    parser.add_argument("--expected-total-episodes", type=int, default=100)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/acwmbench_baseline_viewer"),
    )
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the entire viewer cache before regenerating clips.",
    )
    cache_group.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing clips; by default this dataset's clips are regenerated.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def infer_baseline_name(root: Path) -> str:
    name = root.name
    return name[: -len("_dataset")] if name.endswith("_dataset") else name


def main() -> None:
    global GENERATED_ROOT, GT_ROOT, CACHE_ROOT, VIEW, FPS, BASELINE_NAME
    global CRF, PRESET, REQUESTED_TASKS, EXPECTED_TASKS, EXPECTED_EPISODES

    args = parse_args()
    root_arg = args.generated_root or args.dataset_root
    if root_arg is None:
        root_arg = Path("/path/to/model-predictions")
    GENERATED_ROOT = root_arg.expanduser().resolve()
    GT_ROOT = args.gt_root.expanduser().resolve()
    CACHE_ROOT = args.cache_dir.expanduser().resolve()
    VIEW = args.view
    FPS = args.fps
    BASELINE_NAME = args.baseline_name or infer_baseline_name(GENERATED_ROOT)
    CRF = args.crf
    PRESET = args.preset
    REQUESTED_TASKS = args.tasks
    EXPECTED_TASKS = args.expected_tasks
    EXPECTED_EPISODES = args.expected_total_episodes
    if FPS <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    if args.clear_cache and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    elif not args.reuse_cache:
        shutil.rmtree(CACHE_ROOT / dataset_cache_tag(), ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[baseline-viewer] baseline  = {BASELINE_NAME}", flush=True)
    print(f"[baseline-viewer] generated = {GENERATED_ROOT}", flush=True)
    print(f"[baseline-viewer] GT        = {GT_ROOT}", flush=True)
    print(f"[baseline-viewer] cache     = {CACHE_ROOT}", flush=True)
    print("[baseline-viewer] scanning datasets ...", flush=True)
    refresh_dataset()
    task_count = len({task for task, _ in _EPISODES})
    print(
        f"[baseline-viewer] found {task_count} tasks and "
        f"{len(_EPISODES)} matched episodes",
        flush=True,
    )
    if EXPECTED_TASKS and task_count != EXPECTED_TASKS:
        print(
            f"[baseline-viewer] WARNING: expected {EXPECTED_TASKS} tasks",
            flush=True,
        )
    if EXPECTED_EPISODES and len(_EPISODES) != EXPECTED_EPISODES:
        print(
            f"[baseline-viewer] WARNING: expected {EXPECTED_EPISODES} episodes",
            flush=True,
        )
    if not _EPISODES:
        raise SystemExit("No episodes with matching GT/generated frame indices")
    if args.prewarm:
        prewarm_clips(args.prewarm_workers)
        print(
            f"[baseline-viewer] prewarming clips with "
            f"{args.prewarm_workers} workers",
            flush=True,
        )

    url = f"http://{args.host}:{args.port}"
    print(f"[baseline-viewer] serving on {url}", flush=True)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
