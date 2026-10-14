#!/usr/bin/env python3
"""Synchronously compare YAM GT with multiple inference results.

The viewer discovers common task/episode directories quickly, then lazily
transcodes the selected episode's aligned image sequence into cached H.264
clips. GT frame names define the timeline for every baseline.

Example:
    python tools/visualize_yam_comparison.py \
      --gt-root /path/to/yam-ground-truth \
      --baseline model-a=/path/to/model-a-predictions \
      --baseline model-b=/path/to/model-b-predictions
"""

# pylint: disable=global-statement,import-error

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    frame_names: list[str] | None = field(default=None, repr=False)
    frame_indices: list[int] | None = field(default=None, repr=False)


app = Flask(__name__)
GT_ROOT = Path()
SOURCES: dict[str, Path] = {}
CACHE_ROOT = Path()
VIEW = "top_head"
FPS = 30.0
CRF = 25
PRESET = "veryfast"
CLIP_WORKERS = 2
REQUESTED_TASKS: list[str] | None = None
EPISODES: dict[tuple[str, str], Episode] = {}
CLIP_LOCKS: dict[Path, threading.Lock] = {}
CLIP_LOCKS_GUARD = threading.Lock()
TRANSCODE_SEMAPHORE = threading.Semaphore(CLIP_WORKERS)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YAM · Aligned Comparison</title>
  <style>
    :root{color-scheme:dark;--bg:#090c12;--panel:#151a24;--line:#30394a;
      --text:#eef2f8;--muted:#98a4b7;--accent:#72a9ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
      font:14px/1.45 system-ui,sans-serif}
    header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;
      padding:12px 18px;background:#0d121bcc;border-bottom:1px solid var(--line)}
    h1{font-size:17px;margin:0}.muted{color:var(--muted)}.grow{flex:1}
    main{max-width:1900px;margin:auto;padding:16px}
    .selectors{display:grid;grid-template-columns:minmax(150px,1fr)
      minmax(180px,1fr) auto auto auto;gap:9px;margin-bottom:12px}
    select,button,input{min-height:36px;padding:6px 10px;color:var(--text);
      background:var(--panel);border:1px solid var(--line);border-radius:7px}
    button{cursor:pointer;font-weight:650}button:hover{border-color:var(--accent)}
    .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
    .card{position:relative;background:#000;border:1px solid var(--line);
      border-radius:9px;overflow:hidden;aspect-ratio:4/3}
    video{display:block;width:100%;height:100%;object-fit:contain}
    .badge{position:absolute;left:8px;top:8px;padding:4px 8px;border-radius:5px;
      background:#000c;font-weight:750;color:#bdd5ff;pointer-events:none}
    .card:first-child .badge{color:#9df0c3}
    .loading{position:absolute;inset:0;display:grid;place-items:center;
      background:#000b;color:#dbe7ff;font-weight:700}
    .controls{margin-top:12px;padding:12px;background:var(--panel);
      border:1px solid var(--line);border-radius:9px}
    .transport{display:grid;grid-template-columns:auto auto auto minmax(220px,1fr)
      auto auto;gap:8px;align-items:center}
    #timeline{width:100%;padding:0}.status{display:flex;justify-content:space-between;
      gap:10px;margin-top:8px;color:var(--muted);font-variant-numeric:tabular-nums}
    .error{display:none;margin-top:10px;padding:9px;color:#ffb9b9;background:#421d24;
      border-radius:6px}
    kbd{padding:1px 5px;border:1px solid var(--line);border-radius:4px}
    @media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:700px){.grid,.selectors{grid-template-columns:1fr}
      .transport{grid-template-columns:repeat(3,auto)}#timeline{grid-column:1/-1}}
  </style>
</head>
<body>
<header><h1>YAM Aligned Episode Comparison</h1>
  <span id="count" class="muted"></span><span class="grow"></span>
  <span class="muted">{{ view }} · {{ fps }} FPS</span>
</header>
<main>
  <section class="selectors">
    <select id="task"></select><select id="episode"></select>
    <button id="prev">Previous EP</button><button id="next">Next EP</button>
    <button id="reload">Rescan</button>
  </section>
  <section class="grid" id="grid"></section>
  <section class="controls">
    <div class="transport">
      <button id="back">−1 frame</button><button id="play">Play</button>
      <button id="forward">+1 frame</button>
      <input id="timeline" type="range" min="0" max="1" value="0" step="0.001">
      <span id="position">frame —</span>
      <select id="speed">
        <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
        <option value="1" selected>1×</option><option value="2">2×</option>
      </select>
    </div>
    <div class="status"><span id="info">Loading…</span>
      <span><kbd>←</kbd>/<kbd>→</kbd> frame · <kbd>Space</kbd> play</span></div>
    <div class="error" id="error"></div>
  </section>
</main>
<script>
const fps={{ fps|tojson }};
const sourceNames={{ source_names|tojson }};
const taskSelect=document.querySelector("#task");
const episodeSelect=document.querySelector("#episode");
const timeline=document.querySelector("#timeline");
const playButton=document.querySelector("#play");
const errorBox=document.querySelector("#error");
let videos=[],master=null,episodes=[],frameIndices=[];

function showError(message=""){
  errorBox.textContent=message;errorBox.style.display=message?"block":"none";
}
async function getJSON(url){
  const response=await fetch(url);
  if(!response.ok)throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}
function buildGrid(){
  const grid=document.querySelector("#grid");grid.replaceChildren();
  videos=sourceNames.map((name,index)=>{
    const card=document.createElement("div");card.className="card";
    const video=document.createElement("video");
    video.muted=true;video.playsInline=true;video.preload="metadata";
    const badge=document.createElement("span");badge.className="badge";
    badge.textContent=name;
    const loading=document.createElement("span");loading.className="loading";
    loading.textContent="Preparing aligned clip…";
    card.append(video,badge,loading);grid.append(card);
    video.addEventListener("canplay",()=>{
      loading.style.display="none";
      if(index===0&&video.videoWidth)card.style.aspectRatio=
        `${video.videoWidth}/${video.videoHeight}`;
    });
    video.addEventListener("error",()=>{
      loading.style.display="none";
      showError(`Failed to prepare ${name}. Check ffmpeg/server output.`);
    });
    return video;
  });
  master=videos[0];
  master.addEventListener("timeupdate",updateTimeline);
  master.addEventListener("ended",pauseAll);
  master.addEventListener("pause",()=>{
    if(videos.some(video=>!video.paused))pauseAll();
  });
}
function pauseAll(){videos.forEach(video=>video.pause());playButton.textContent="Play"}
function syncFollowers(force=false){
  if(!master)return;
  videos.slice(1).forEach(video=>{
    if(force||Math.abs(video.currentTime-master.currentTime)>0.06)
      video.currentTime=master.currentTime;
  });
}
function updateTimeline(){
  if(!master||!Number.isFinite(master.duration))return;
  timeline.max=master.duration;timeline.value=master.currentTime;
  const position=Math.min(frameIndices.length-1,
    Math.max(0,Math.floor(master.currentTime*fps)));
  const original=frameIndices[position];
  document.querySelector("#position").textContent=original===undefined?"frame —":
    `frame ${String(original).padStart(5,"0")} · ${master.currentTime.toFixed(2)}s`;
  syncFollowers(false);
}
function seek(time){
  const duration=master&&Number.isFinite(master.duration)?master.duration:0;
  const target=Math.max(0,Math.min(time,duration));
  videos.forEach(video=>video.currentTime=target);updateTimeline();
}
async function togglePlay(){
  if(!master)return;
  if(master.paused){
    syncFollowers(true);
    try{await Promise.all(videos.map(video=>video.play()));
      playButton.textContent="Pause";
    }catch(error){showError(error.message)}
  }else pauseAll();
}
async function loadTasks(reload=false){
  showError();const data=await getJSON(`/api/tasks${reload?"?reload=1":""}`);
  taskSelect.replaceChildren(...data.tasks.map(task=>new Option(task,task)));
  document.querySelector("#count").textContent=
    `${data.task_count} tasks · ${data.episode_count} common episodes`;
  await loadEpisodes();
}
async function loadEpisodes(){
  pauseAll();
  const data=await getJSON(`/api/episodes/${encodeURIComponent(taskSelect.value)}`);
  episodes=data.episodes;
  episodeSelect.replaceChildren(...episodes.map(item=>new Option(item.name,item.name)));
  await loadEpisode();
}
async function loadEpisode(){
  pauseAll();showError();
  document.querySelectorAll(".loading").forEach(node=>node.style.display="grid");
  const task=encodeURIComponent(taskSelect.value);
  const episode=encodeURIComponent(episodeSelect.value);
  const data=await getJSON(`/api/episode/${task}/${episode}`);
  frameIndices=data.frame_indices;
  document.querySelector("#info").textContent=
    `${data.task}/${data.name} · ${data.frame_count} aligned frames · `+
    `${(data.frame_count/fps).toFixed(2)}s`;
  videos.forEach((video,index)=>{
    video.src=`/clip.mp4?source=${encodeURIComponent(sourceNames[index])}`+
      `&task=${task}&episode=${episode}`;video.load();
  });
}
function switchEpisode(offset){
  const next=episodeSelect.selectedIndex+offset;
  if(next<0||next>=episodeSelect.options.length)return;
  episodeSelect.selectedIndex=next;loadEpisode().catch(error=>showError(error.message));
}
buildGrid();
taskSelect.addEventListener("change",()=>loadEpisodes().catch(error=>showError(error.message)));
episodeSelect.addEventListener("change",()=>loadEpisode().catch(error=>showError(error.message)));
document.querySelector("#reload").addEventListener("click",()=>loadTasks(true).catch(error=>showError(error.message)));
document.querySelector("#prev").addEventListener("click",()=>switchEpisode(-1));
document.querySelector("#next").addEventListener("click",()=>switchEpisode(1));
document.querySelector("#back").addEventListener("click",()=>seek(master.currentTime-1/fps));
document.querySelector("#forward").addEventListener("click",()=>seek(master.currentTime+1/fps));
playButton.addEventListener("click",togglePlay);
timeline.addEventListener("input",()=>seek(Number(timeline.value)));
document.querySelector("#speed").addEventListener("change",event=>{
  videos.forEach(video=>video.playbackRate=Number(event.target.value));
});
document.addEventListener("keydown",event=>{
  if(event.target.matches("select,input"))return;
  if(event.code==="Space"){event.preventDefault();togglePlay()}
  if(event.key==="ArrowLeft")seek(master.currentTime-1/fps);
  if(event.key==="ArrowRight")seek(master.currentTime+1/fps);
});
setInterval(()=>{if(master&&!master.paused)syncFollowers(false)},250);
loadTasks().catch(error=>showError(error.message));
</script>
</body>
</html>"""


def natural_key(value: str | Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(value).name)
    ]


def directory_names(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir()}


def video_dir(root: Path, episode: Episode) -> Path:
    return root / episode.task / episode.name / "video" / VIEW


def scan_dataset() -> dict[tuple[str, str], Episode]:
    roots = [GT_ROOT, *SOURCES.values()]
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Dataset root does not exist: {root}")
    common_tasks = set.intersection(*(directory_names(root) for root in roots))
    tasks = REQUESTED_TASKS or sorted(common_tasks, key=natural_key)
    episodes: dict[tuple[str, str], Episode] = {}
    for task in tasks:
        if task not in common_tasks:
            continue
        common_names = set.intersection(
            *(directory_names(root / task) for root in roots)
        )
        for name in sorted(common_names, key=natural_key):
            match = EPISODE_RE.fullmatch(name)
            if match is None:
                continue
            episode = Episode(task, name, int(match.group(1)))
            if all(video_dir(root, episode).is_dir() for root in roots):
                episodes[(task, name)] = episode
    return episodes


def get_episode(task: str, name: str) -> Episode:
    episode = EPISODES.get((task, name))
    if episode is None:
        abort(404)
    if episode.frame_names is None:
        paths = sorted(
            (
                path
                for path in video_dir(GT_ROOT, episode).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
        names: list[str] = []
        indices: list[int] = []
        for path in paths:
            match = FRAME_NUMBER_RE.search(path.stem)
            if match is not None:
                names.append(path.name)
                indices.append(int(match.group(1)))
        if not names:
            abort(404)
        episode.frame_names = names
        episode.frame_indices = indices
    return episode


def cache_tag() -> str:
    identity = "|".join(
        [str(GT_ROOT), *(f"{name}={root}" for name, root in SOURCES.items())]
        + [VIEW, f"{FPS:.8g}", str(CRF), PRESET]
    )
    return hashlib.sha1(identity.encode()).hexdigest()[:14]


def clip_path(episode: Episode, source: str) -> Path:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
    return CACHE_ROOT / cache_tag() / episode.task / f"{episode.name}__{safe_source}.mp4"


def clip_lock(path: Path) -> threading.Lock:
    with CLIP_LOCKS_GUARD:
        return CLIP_LOCKS.setdefault(path, threading.Lock())


def ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def target_resolution(episode: Episode) -> tuple[int, int]:
    assert episode.frame_names
    with Image.open(video_dir(GT_ROOT, episode) / episode.frame_names[0]) as image:
        width, height = image.size
    return width - width % 2, height - height % 2


def transcode_clip(episode: Episode, source: str, destination: Path) -> None:
    root = GT_ROOT if source == "GT" else SOURCES[source]
    assert episode.frame_names
    files = [video_dir(root, episode) / name for name in episode.frame_names]
    width, height = target_resolution(episode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ffconcat",
        dir=destination.parent,
        encoding="utf-8",
        delete=False,
    ) as manifest:
        manifest_path = Path(manifest.name)
        manifest.write("ffconcat version 1.0\n")
        for path in files:
            manifest.write(f"file '{ffconcat_quote(path)}'\n")
            manifest.write(f"duration {1.0 / FPS:.12f}\n")
        manifest.write(f"file '{ffconcat_quote(files[-1])}'\n")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(manifest_path), "-vf", f"scale={width}:{height}:flags=lanczos",
        "-r", f"{FPS:.8g}", "-fps_mode", "cfr", "-c:v", "libx264",
        "-preset", PRESET, "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", str(temporary),
    ]
    try:
        with TRANSCODE_SEMAPHORE:
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
    if destination.is_file() and destination.stat().st_size:
        return destination
    with clip_lock(destination):
        if not destination.is_file() or not destination.stat().st_size:
            transcode_clip(episode, source, destination)
    return destination


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        source_names=["GT", *SOURCES],
        fps=FPS,
        view=VIEW,
    )


@app.get("/api/tasks")
def api_tasks():
    global EPISODES
    if request.args.get("reload"):
        EPISODES = scan_dataset()
    tasks = sorted({task for task, _ in EPISODES}, key=natural_key)
    return jsonify(
        tasks=tasks,
        task_count=len(tasks),
        episode_count=len(EPISODES),
    )


@app.get("/api/episodes/<task>")
def api_episodes(task: str):
    episodes = sorted(
        (
            {"name": episode.name, "index": episode.index}
            for (episode_task, _), episode in EPISODES.items()
            if episode_task == task
        ),
        key=lambda item: int(item["index"]),
    )
    return jsonify(task=task, episodes=episodes)


@app.get("/api/episode/<task>/<name>")
def api_episode(task: str, name: str):
    episode = get_episode(task, name)
    return jsonify(
        task=task,
        name=name,
        frame_count=len(episode.frame_names or []),
        frame_indices=episode.frame_indices,
    )


@app.get("/clip.mp4")
def clip():
    source = request.args.get("source", "")
    if source != "GT" and source not in SOURCES:
        abort(400)
    episode = get_episode(
        request.args.get("task", ""),
        request.args.get("episode", ""),
    )
    return send_file(
        ensure_clip(episode, source),
        mimetype="video/mp4",
        conditional=True,
        max_age=86400,
    )


def parse_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--baseline must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--baseline must be NAME=PATH")
    return name.strip(), Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/path/to/yam-ground-truth"),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=parse_baseline,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/acwmbench_yam_comparison"),
    )
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--clip-workers", type=int, default=2)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    global GT_ROOT, SOURCES, CACHE_ROOT, VIEW, FPS, CRF, PRESET
    global CLIP_WORKERS, TRANSCODE_SEMAPHORE, REQUESTED_TASKS, EPISODES
    args = parse_args()
    GT_ROOT = args.gt_root.expanduser().resolve()
    SOURCES = {name: path.resolve() for name, path in args.baseline}
    if len(SOURCES) != len(args.baseline):
        raise SystemExit("Baseline names must be unique")
    CACHE_ROOT = args.cache_dir.expanduser().resolve()
    VIEW = args.view
    FPS = float(args.fps)
    CRF = int(args.crf)
    PRESET = args.preset
    CLIP_WORKERS = max(1, int(args.clip_workers))
    TRANSCODE_SEMAPHORE = threading.Semaphore(CLIP_WORKERS)
    REQUESTED_TASKS = args.tasks
    if FPS <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found")
    if args.clear_cache:
        shutil.rmtree(CACHE_ROOT / cache_tag(), ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    EPISODES = scan_dataset()
    if not EPISODES:
        raise SystemExit("No common task/episode/view directories found")
    task_count = len({task for task, _ in EPISODES})
    print(f"[yam-viewer] GT: {GT_ROOT}", flush=True)
    for name, root in SOURCES.items():
        print(f"[yam-viewer] {name}: {root}", flush=True)
    print(
        f"[yam-viewer] found {task_count} tasks and {len(EPISODES)} common episodes",
        flush=True,
    )
    if args.prewarm:
        def prewarm() -> None:
            jobs = [
                (episode, source)
                for episode in EPISODES.values()
                for source in ["GT", *SOURCES]
            ]
            with ThreadPoolExecutor(max_workers=CLIP_WORKERS) as executor:
                list(executor.map(lambda job: ensure_clip(*job), jobs))
        threading.Thread(target=prewarm, daemon=True).start()
        print("[yam-viewer] prewarming all clips in background", flush=True)
    print(f"[yam-viewer] http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
