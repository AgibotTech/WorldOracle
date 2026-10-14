#!/usr/bin/env python3
"""Single-dataset web viewer for policy_eval_dataset (no GT matching).

Example:
    python tools/visualize_policy_eval_dataset.py \
        /path/to/policy-inputs \
        --prewarm --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import hashlib
import re
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
    frames: dict[int, Path]

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.frames.keys())

    def summary(self) -> dict[str, object]:
        indices = self.frame_indices
        return {
            "task": self.task,
            "name": self.name,
            "index": self.index,
            "frame_count": len(indices),
            "first_frame": indices[0] if indices else None,
            "last_frame": indices[-1] if indices else None,
        }


app = Flask(__name__)
DATASET_ROOT = Path()
CACHE_ROOT = Path()
VIEW = "top_head"
FPS = 30.0
DATASET_NAME = ""
CRF = 25
PRESET = "veryfast"
REQUESTED_TASKS: list[str] | None = None
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
  <title>{{ dataset_name }} · Dataset Viewer</title>
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
    main { max-width: 1200px; margin: auto; padding: 20px; }
    .selectors {
      display: grid; grid-template-columns: minmax(170px,1fr) minmax(220px,1fr) auto auto;
      gap: 12px; margin-bottom: 16px;
    }
    select, button, input {
      min-height: 38px; padding: 7px 11px; border: 1px solid var(--line);
      border-radius: 8px; background: var(--panel2); color: var(--text);
    }
    button { cursor: pointer; font-weight: 650; }
    button:hover { border-color: var(--accent); }
    .view {
      position: relative; overflow: hidden; border: 1px solid var(--line);
      border-radius: 10px; background: #000; aspect-ratio: var(--ratio);
    }
    video { display: block; width: 100%; height: 100%; object-fit: fill; }
    .badge {
      position: absolute; left: 10px; top: 10px; padding: 5px 9px;
      border-radius: 6px; background: rgba(0,0,0,.72); font-weight: 750;
      pointer-events: none; color: #a7c5ff;
    }
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
      .selectors { grid-template-columns: 1fr; }
      .root { display: none; }
      .transport { grid-template-columns: repeat(3,auto); }
      #slider { grid-column: 1/-1; grid-row: 2; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Dataset Viewer</h1>
    <span class="baseline">{{ dataset_name }}</span>
    <span id="datasetCount"></span>
    <span class="root">{{ dataset_root }}</span>
  </header>
  <main>
    <section class="selectors">
      <select id="task"></select>
      <select id="episode"></select>
      <select id="view">
        <option value="top_head">top_head</option>
        <option value="hand_left">hand_left</option>
        <option value="hand_right">hand_right</option>
      </select>
      <button id="reload">Rescan dataset</button>
    </section>
    <section class="view" id="player">
      <video id="video" muted playsinline preload="metadata"></video>
      <span class="badge" id="viewBadge">{{ view }}</span>
      <span class="loading">Preparing video…</span>
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
    const defaultView = {{ view | tojson }};
    const taskSelect = document.querySelector("#task");
    const episodeSelect = document.querySelector("#episode");
    const viewSelect = document.querySelector("#view");
    const slider = document.querySelector("#slider");
    const video = document.querySelector("#video");
    const playButton = document.querySelector("#play");
    const errorBox = document.querySelector("#error");
    let episodes = [];
    let indices = [];

    viewSelect.value = defaultView;

    async function getJSON(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    }
    function showError(message="") {
      errorBox.textContent = message;
      errorBox.style.display = message ? "block" : "none";
    }
    function clipURL() {
      const task = encodeURIComponent(taskSelect.value);
      const ep = encodeURIComponent(episodeSelect.value);
      const view = encodeURIComponent(viewSelect.value);
      return `/clip.mp4?task=${task}&episode=${ep}&view=${view}`;
    }
    function showLoading(show) {
      document.querySelector(".loading").style.display = show ? "grid" : "none";
    }
    function updateTimeline() {
      if (!Number.isFinite(video.duration)) return;
      slider.max = video.duration;
      slider.value = video.currentTime;
      const position = Math.min(
        indices.length - 1, Math.max(0, Math.floor(video.currentTime * fps)));
      const original = indices[position];
      document.querySelector("#frameLabel").textContent =
        original === undefined ? "frame —" :
        `frame ${String(original).padStart(5,"0")} · ${position + 1}/${indices.length}`;
    }
    async function loadTasks(rescan=false) {
      showError();
      const view = encodeURIComponent(viewSelect.value);
      const data = await getJSON(
        `/api/tasks?view=${view}${rescan ? "&reload=1" : ""}`);
      taskSelect.replaceChildren(...data.tasks.map(task => new Option(task, task)));
      document.querySelector("#datasetCount").textContent =
        `${data.task_count} tasks · ${data.episode_count} episodes`;
      if (!data.tasks.length) throw new Error("No tasks found");
      await loadEpisodes();
    }
    async function loadEpisodes() {
      pauseVideo();
      const view = encodeURIComponent(viewSelect.value);
      const data = await getJSON(
        `/api/episodes/${encodeURIComponent(taskSelect.value)}?view=${view}`);
      episodes = data.episodes;
      episodeSelect.replaceChildren(...episodes.map(ep =>
        new Option(`${ep.name}  (${ep.frame_count} frames)`, ep.name)));
      if (!episodes.length) throw new Error("No episodes found");
      await loadEpisode();
    }
    async function loadEpisode() {
      pauseVideo();
      showError();
      showLoading(true);
      const task = encodeURIComponent(taskSelect.value);
      const ep = encodeURIComponent(episodeSelect.value);
      const view = encodeURIComponent(viewSelect.value);
      document.querySelector("#viewBadge").textContent = viewSelect.value;
      const data = await getJSON(`/api/episode/${task}/${ep}?view=${view}`);
      indices = data.frame_indices;
      document.querySelector("#episodeInfo").textContent =
        `${data.task}/${data.name}/${viewSelect.value} · ${data.frame_count} frames`;
      video.src = clipURL();
      video.load();
    }
    function pauseVideo() {
      video.pause(); playButton.textContent = "Play";
    }
    async function togglePlay() {
      if (video.paused) {
        try {
          await video.play();
          playButton.textContent = "Pause";
        } catch (error) { showError(error.message); }
      } else {
        pauseVideo();
      }
    }
    function seek(time) {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      video.currentTime = Math.max(0, Math.min(time, duration));
      updateTimeline();
    }
    function switchEpisode(offset) {
      const next = episodeSelect.selectedIndex + offset;
      if (next < 0 || next >= episodeSelect.options.length) return;
      episodeSelect.selectedIndex = next;
      loadEpisode().catch(error => showError(error.message));
    }

    video.addEventListener("loadedmetadata", () => {
      slider.max = video.duration;
      if (video.videoWidth && video.videoHeight) {
        document.querySelector("#player").style.setProperty(
          "--ratio", `${video.videoWidth}/${video.videoHeight}`);
      }
    });
    video.addEventListener("canplay", () => showLoading(false));
    video.addEventListener("error", () => {
      showLoading(false);
      showError("Video preparation failed. Check the server log and ffmpeg.");
    });
    video.addEventListener("timeupdate", updateTimeline);
    video.addEventListener("pause", () => { playButton.textContent = "Play"; });
    video.addEventListener("ended", pauseVideo);
    slider.addEventListener("input", () => seek(Number(slider.value)));
    playButton.addEventListener("click", togglePlay);
    taskSelect.addEventListener("change", () =>
      loadEpisodes().catch(error => showError(error.message)));
    episodeSelect.addEventListener("change", () =>
      loadEpisode().catch(error => showError(error.message)));
    viewSelect.addEventListener("change", () =>
      loadTasks(true).catch(error => showError(error.message)));
    document.querySelector("#reload").addEventListener("click", () =>
      loadTasks(true).catch(error => showError(error.message)));
    document.querySelector("#previousEpisode").addEventListener("click", () => switchEpisode(-1));
    document.querySelector("#nextEpisode").addEventListener("click", () => switchEpisode(1));
    document.addEventListener("keydown", event => {
      if (event.target.matches("select,input")) return;
      if (event.code === "Space") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft") seek(video.currentTime - 1 / fps);
      if (event.key === "ArrowRight") seek(video.currentTime + 1 / fps);
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


def scan_dataset(view: str) -> dict[tuple[str, str], Episode]:
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"Dataset root does not exist: {DATASET_ROOT}")
    tasks = REQUESTED_TASKS or sorted(directory_names(DATASET_ROOT), key=natural_key)
    episodes: dict[tuple[str, str], Episode] = {}
    for task in tasks:
        task_dir = DATASET_ROOT / task
        if not task_dir.is_dir():
            continue
        for name in sorted(directory_names(task_dir), key=natural_key):
            match = EPISODE_RE.fullmatch(name)
            if match is None:
                continue
            frames = indexed_frames(task_dir / name / "video" / view)
            if not frames:
                continue
            episodes[(task, name)] = Episode(
                task, name, int(match.group(1)), frames
            )
    return episodes


def refresh_dataset(view: str | None = None) -> None:
    global _EPISODES, VIEW
    if view is not None:
        VIEW = view
    with _SCAN_LOCK:
        _EPISODES = scan_dataset(VIEW)


def get_episode(task: str, name: str) -> Episode:
    episode = _EPISODES.get((task, name))
    if episode is None:
        abort(404)
    return episode


def dataset_cache_tag() -> str:
    identity = f"{DATASET_ROOT}|{VIEW}|{FPS:.8g}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def clip_path(episode: Episode) -> Path:
    return (
        CACHE_ROOT
        / dataset_cache_tag()
        / episode.task
        / f"{episode.name}__{VIEW}.mp4"
    )


def clip_lock(path: Path) -> threading.Lock:
    with _CLIP_LOCKS_LOCK:
        return _CLIP_LOCKS.setdefault(path, threading.Lock())


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def target_resolution(episode: Episode) -> tuple[int, int]:
    first_index = episode.frame_indices[0]
    with Image.open(episode.frames[first_index]) as image:
        width, height = image.size
    return width - width % 2, height - height % 2


def transcode_clip(episode: Episode, destination: Path) -> None:
    files = [episode.frames[index] for index in episode.frame_indices]
    width, height = target_resolution(episode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    frame_duration = 1.0 / FPS
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ffconcat",
        prefix=f"{episode.task}_{episode.name}_{VIEW}_",
        dir=destination.parent,
        encoding="utf-8",
        delete=False,
    ) as manifest:
        manifest_path = Path(manifest.name)
        manifest.write("ffconcat version 1.0\n")
        for path in files:
            manifest.write(f"file '{ffconcat_quote(path)}'\n")
            manifest.write(f"duration {frame_duration:.12f}\n")
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
            f"ffmpeg failed for {episode.task}/{episode.name}/{VIEW}"
        ) from error
    finally:
        manifest_path.unlink(missing_ok=True)


def ensure_clip(episode: Episode) -> Path:
    destination = clip_path(episode)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with clip_lock(destination):
        if not destination.is_file() or destination.stat().st_size == 0:
            transcode_clip(episode, destination)
    return destination


def prewarm_clips(workers: int) -> None:
    with _PREWARM_LOCK:
        if bool(_PREWARM_STATE["running"]):
            return
        jobs = [
            episode
            for episode in _EPISODES.values()
            if not clip_path(episode).is_file()
        ]
        _PREWARM_STATE.update(running=True, total=len(jobs), done=0, failed=0)

    def run() -> None:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(ensure_clip, episode) for episode in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                    _PREWARM_STATE["done"] = int(_PREWARM_STATE["done"]) + 1
                except (OSError, RuntimeError, ValueError):
                    _PREWARM_STATE["failed"] = int(_PREWARM_STATE["failed"]) + 1
        _PREWARM_STATE["running"] = False

    threading.Thread(target=run, daemon=True).start()


def maybe_switch_view(view: str | None) -> None:
    if view and view != VIEW:
        refresh_dataset(view)


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        dataset_root=str(DATASET_ROOT),
        dataset_name=DATASET_NAME,
        fps=FPS,
        view=VIEW,
    )


@app.get("/api/tasks")
def api_tasks():
    maybe_switch_view(request.args.get("view"))
    if request.args.get("reload"):
        refresh_dataset()
    tasks = sorted({task for task, _ in _EPISODES}, key=natural_key)
    return jsonify(
        tasks=tasks,
        task_count=len(tasks),
        episode_count=len(_EPISODES),
        dataset_name=DATASET_NAME,
        view=VIEW,
    )


@app.get("/api/episodes/<task>")
def api_episodes(task: str):
    maybe_switch_view(request.args.get("view"))
    episodes = sorted(
        (
            episode.summary()
            for (episode_task, _), episode in _EPISODES.items()
            if episode_task == task
        ),
        key=lambda item: int(item["index"]),
    )
    return jsonify(task=task, episodes=episodes, view=VIEW)


@app.get("/api/episode/<task>/<name>")
def api_episode(task: str, name: str):
    maybe_switch_view(request.args.get("view"))
    episode = get_episode(task, name)
    return jsonify(**episode.summary(), frame_indices=episode.frame_indices, view=VIEW)


@app.get("/clip.mp4")
def clip():
    maybe_switch_view(request.args.get("view"))
    episode = get_episode(
        request.args.get("task", ""), request.args.get("episode", "")
    )
    path = ensure_clip(episode)
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
        description="Serve a single dataset as cached H.264 clips (no GT matching)."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path(
            "/path/to/policy-inputs"
        ),
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/acwmbench_dataset_viewer"),
    )
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT, CACHE_ROOT, VIEW, FPS, DATASET_NAME
    global CRF, PRESET, REQUESTED_TASKS

    args = parse_args()
    DATASET_ROOT = args.dataset_root.expanduser().resolve()
    CACHE_ROOT = args.cache_dir.expanduser().resolve()
    VIEW = args.view
    FPS = args.fps
    DATASET_NAME = args.dataset_name or DATASET_ROOT.name
    CRF = args.crf
    PRESET = args.preset
    REQUESTED_TASKS = args.tasks
    if FPS <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH")
    if args.clear_cache and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[dataset-viewer] name = {DATASET_NAME}", flush=True)
    print(f"[dataset-viewer] root = {DATASET_ROOT}", flush=True)
    print(f"[dataset-viewer] view = {VIEW}", flush=True)
    print(f"[dataset-viewer] cache = {CACHE_ROOT}", flush=True)
    print("[dataset-viewer] scanning dataset ...", flush=True)
    refresh_dataset()
    task_count = len({task for task, _ in _EPISODES})
    print(
        f"[dataset-viewer] found {task_count} tasks and "
        f"{len(_EPISODES)} episodes",
        flush=True,
    )
    if not _EPISODES:
        raise SystemExit("No episodes with frames found")
    if args.prewarm:
        prewarm_clips(args.prewarm_workers)
        print(
            f"[dataset-viewer] prewarming clips with "
            f"{args.prewarm_workers} workers",
            flush=True,
        )

    url = f"http://{args.host}:{args.port}"
    print(f"[dataset-viewer] serving on {url}", flush=True)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
