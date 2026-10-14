#!/usr/bin/env python3
"""Browse top-head MP4 episodes in a LeRobot v3 policy-eval dataset.

Example:
    python tools/visualize_policy_eval_lerobot_videos.py \
      /path/to/policy-inputs \
      --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, send_file


TOP_HEAD_KEY = "observation.images.top_head"
FILE_RE = re.compile(r"file-(\d+)\.mp4$", re.IGNORECASE)
CHUNK_RE = re.compile(r"chunk-(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EpisodeVideo:
    task: str
    episode_id: str
    chunk: int
    file_index: int
    path: Path

    def payload(self) -> dict[str, object]:
        return {
            "task": self.task,
            "episode_id": self.episode_id,
            "chunk": self.chunk,
            "file_index": self.file_index,
            "size_bytes": self.path.stat().st_size,
            "url": f"/video/{self.task}/{self.episode_id}.mp4",
        }


app = Flask(__name__)
DATASET_ROOT = Path()
FPS = 30.0
EPISODES: dict[tuple[str, str], EpisodeVideo] = {}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Policy Eval · top_head</title>
  <style>
    :root{color-scheme:dark;--bg:#0b0e14;--panel:#151a24;--line:#30394a;
      --text:#eef2f8;--muted:#98a4b7;--accent:#6da9ff}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
      font:14px/1.45 system-ui,sans-serif}
    header{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:14px;
      padding:13px 20px;background:#10151ecc;border-bottom:1px solid var(--line)}
    h1{margin:0;font-size:17px}.muted{color:var(--muted)}.grow{flex:1}
    main{max-width:1250px;margin:auto;padding:20px}
    .selectors{display:grid;grid-template-columns:1fr 1fr auto auto;gap:10px;
      margin-bottom:14px}
    select,button{min-height:38px;padding:7px 11px;color:var(--text);
      background:var(--panel);border:1px solid var(--line);border-radius:7px}
    button{cursor:pointer;font-weight:650}button:hover{border-color:var(--accent)}
    .player{position:relative;background:#000;border:1px solid var(--line);
      border-radius:10px;overflow:hidden;aspect-ratio:16/10}
    video{display:block;width:100%;height:100%;object-fit:contain}
    .badge{position:absolute;left:10px;top:10px;padding:5px 8px;border-radius:5px;
      background:#000b;color:#bcd5ff;font-weight:700;pointer-events:none}
    .meta{display:flex;justify-content:space-between;gap:15px;margin-top:12px;
      padding:11px 13px;background:var(--panel);border:1px solid var(--line);
      border-radius:8px;color:var(--muted);font-variant-numeric:tabular-nums}
    @media(max-width:700px){.selectors{grid-template-columns:1fr 1fr}
      header .root{display:none}.meta{flex-direction:column}}
  </style>
</head>
<body>
<header>
  <h1>Policy Eval Dataset</h1>
  <span>top_head MP4</span><span class="grow"></span>
  <span class="muted root">{{ root }}</span>
</header>
<main>
  <section class="selectors">
    <select id="task"></select>
    <select id="episode"></select>
    <button id="prev">Previous</button>
    <button id="next">Next</button>
  </section>
  <section class="player">
    <video id="video" controls muted playsinline preload="metadata"></video>
    <span class="badge" id="badge">top_head</span>
  </section>
  <section class="meta">
    <span id="identity">Loading…</span>
    <span id="position">—</span>
    <span id="count">—</span>
  </section>
</main>
<script>
const fps={{ fps|tojson }};
const taskSelect=document.querySelector("#task");
const episodeSelect=document.querySelector("#episode");
const video=document.querySelector("#video");
let dataset={},episodes=[];

function selected(){
  return episodes[episodeSelect.selectedIndex] || null;
}
function renderPosition(){
  const frame=Math.max(0,Math.round(video.currentTime*fps));
  const duration=Number.isFinite(video.duration)?video.duration:0;
  document.querySelector("#position").textContent=
    `frame ${frame} · ${video.currentTime.toFixed(2)}s / ${duration.toFixed(2)}s`;
}
function loadVideo(){
  const item=selected();
  if(!item)return;
  video.src=item.url;
  video.load();
  document.querySelector("#identity").textContent=
    `${item.task}/${item.episode_id} · chunk-${String(item.chunk).padStart(3,"0")} / file-${String(item.file_index).padStart(3,"0")}`;
  document.querySelector("#count").textContent=
    `${episodeSelect.selectedIndex+1} / ${episodes.length}`;
  document.querySelector("#badge").textContent=`${item.task} · ${item.episode_id}`;
}
function loadEpisodes(){
  episodes=dataset[taskSelect.value] || [];
  episodeSelect.replaceChildren(...episodes.map(
    item=>new Option(item.episode_id,item.episode_id)));
  loadVideo();
}
function move(offset){
  const next=episodeSelect.selectedIndex+offset;
  if(next<0||next>=episodes.length)return;
  episodeSelect.selectedIndex=next;
  loadVideo();
}
fetch("/api/dataset").then(response=>response.json()).then(data=>{
  dataset=data.tasks;
  taskSelect.replaceChildren(...Object.keys(dataset).map(
    task=>new Option(`${task} (${dataset[task].length})`,task)));
  loadEpisodes();
});
taskSelect.addEventListener("change",loadEpisodes);
episodeSelect.addEventListener("change",loadVideo);
document.querySelector("#prev").addEventListener("click",()=>move(-1));
document.querySelector("#next").addEventListener("click",()=>move(1));
video.addEventListener("timeupdate",renderPosition);
video.addEventListener("loadedmetadata",renderPosition);
document.addEventListener("keydown",event=>{
  if(event.target.matches("select,button"))return;
  if(event.key==="ArrowUp"){event.preventDefault();move(-1)}
  if(event.key==="ArrowDown"){event.preventDefault();move(1)}
  if(event.key==="ArrowLeft")video.currentTime=Math.max(0,video.currentTime-1/fps);
  if(event.key==="ArrowRight")video.currentTime=Math.min(video.duration||0,video.currentTime+1/fps);
  if(event.code==="Space"){event.preventDefault();video.paused?video.play():video.pause()}
});
</script>
</body>
</html>"""


def _resolve_dataset_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    nested = resolved / "robot_local"
    if nested.is_dir():
        return nested
    return resolved


def _video_sort_key(path: Path) -> tuple[int, int]:
    chunk_match = CHUNK_RE.fullmatch(path.parent.name)
    file_match = FILE_RE.fullmatch(path.name)
    if chunk_match is None or file_match is None:
        raise ValueError(f"Unexpected LeRobot video path: {path}")
    return int(chunk_match.group(1)), int(file_match.group(1))


def discover_videos(root: Path) -> dict[tuple[str, str], EpisodeVideo]:
    videos: dict[tuple[str, str], EpisodeVideo] = {}
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        video_root = task_dir / "videos" / TOP_HEAD_KEY
        if not video_root.is_dir():
            continue
        paths = sorted(video_root.glob("chunk-*/file-*.mp4"), key=_video_sort_key)
        for episode_index, path in enumerate(paths):
            chunk, file_index = _video_sort_key(path)
            episode_id = f"ep{episode_index:06d}"
            videos[(task_dir.name, episode_id)] = EpisodeVideo(
                task=task_dir.name,
                episode_id=episode_id,
                chunk=chunk,
                file_index=file_index,
                path=path.resolve(),
            )
    return videos


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        root=str(DATASET_ROOT),
        fps=FPS,
    )


@app.get("/api/dataset")
def api_dataset():
    tasks: dict[str, list[dict[str, object]]] = {}
    for episode in EPISODES.values():
        tasks.setdefault(episode.task, []).append(episode.payload())
    for items in tasks.values():
        items.sort(key=lambda item: str(item["episode_id"]))
    return jsonify(
        root=str(DATASET_ROOT),
        view="top_head",
        episode_count=len(EPISODES),
        tasks=dict(sorted(tasks.items())),
    )


@app.get("/video/<task>/<episode_id>.mp4")
def video(task: str, episode_id: str):
    episode = EPISODES.get((task, episode_id))
    if episode is None:
        abort(404)
    return send_file(
        episode.path,
        mimetype="video/mp4",
        conditional=True,
        max_age=86400,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path(
            "/path/to/policy-inputs"
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    global DATASET_ROOT, FPS, EPISODES
    args = parse_args()
    DATASET_ROOT = _resolve_dataset_root(args.dataset_root)
    FPS = float(args.fps)
    if FPS <= 0:
        raise SystemExit("--fps must be positive")
    if not DATASET_ROOT.is_dir():
        raise SystemExit(f"Dataset root does not exist: {DATASET_ROOT}")
    EPISODES = discover_videos(DATASET_ROOT)
    if not EPISODES:
        raise SystemExit(
            f"No videos/{TOP_HEAD_KEY}/chunk-*/file-*.mp4 found in "
            f"{DATASET_ROOT}"
        )
    task_count = len({task for task, _ in EPISODES})
    print(f"[viewer] root: {DATASET_ROOT}", flush=True)
    print(
        f"[viewer] found {len(EPISODES)} top_head videos across "
        f"{task_count} tasks",
        flush=True,
    )
    print(f"[viewer] http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
