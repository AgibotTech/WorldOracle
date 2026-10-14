#!/usr/bin/env python3
"""Compare policy_eval GT with three policy/world-model rollouts.

Layout:

    <gt_root>/<task>/epXXXXXX/prompt/prompts.json
    <gt_root>/<task>/epXXXXXX/video/<view>/frame_*.jpg
    <rollout_root>/<task>/epXXXXXX/{0,1,2}/video/<view>/frame_*.jpg

Example:

    python tools/visualize_policy_rollouts.py \
      --gt-root /path/to/policy-ground-truth \
      --rollout-root /path/to/policy-rollouts \
      --host 127.0.0.1 --port 8766 --prewarm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, abort, make_response, request, send_file


CAMERAS = ("top_head", "hand_left", "hand_right")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
EPISODE_RE = re.compile(r"ep(\d+)$")


@dataclass(frozen=True)
class Segment:
    key_frame: int
    success_criteria: tuple[str, ...]


@dataclass
class Episode:
    task: str
    episode_id: str
    n_frames: int
    grounding_objects: list[str] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    available: dict[str, bool] = field(default_factory=dict)

    @property
    def index(self) -> int:
        match = EPISODE_RE.fullmatch(self.episode_id)
        return int(match.group(1)) if match else -1


GT_ROOT = Path()
ROLLOUT_ROOT = Path()
CACHE_ROOT = Path()
FPS = 30.0
PREVIEW_HEIGHT = 480
CRF = 28
PRESET = "veryfast"
ROLLOUT_IDS: tuple[str, ...] = ("0", "1", "2")
TASK_FILTER: set[str] | None = None
_EPISODES: list[Episode] | None = None
_CACHE_LOCK = threading.Lock()
_CLIP_LOCKS: dict[str, threading.Lock] = {}
_CLIP_LOCKS_LOCK = threading.Lock()
_PREWARM = {"running": False, "done": 0, "total": 0, "failed": 0}


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Policy rollout comparison</title>
<style>
:root{--bg:#0d1016;--card:#171c27;--soft:#202737;--line:#30394c;--fg:#eef2f8;--dim:#9aa7ba;--blue:#64a8ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;padding:10px 16px;background:#131824;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center}
header b{font-size:16px} header span{color:var(--dim);font-size:12px}.grow{flex:1}
select,button,input{background:var(--soft);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:6px 9px}
button{cursor:pointer}.primary{background:var(--blue);color:#07111e;font-weight:700}
main{max-width:2400px;margin:auto;padding:15px}.episode{margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid var(--line)}.meta{display:flex;gap:14px;align-items:center;margin-bottom:10px;color:var(--dim)}
.videos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:9px;overflow:hidden;position:relative}
.label{position:absolute;left:8px;top:8px;z-index:2;background:#000b;padding:4px 8px;border-radius:4px;font-weight:700}
video{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:fill}.missing{aspect-ratio:4/3;display:grid;place-items:center;color:#ff9c9c}
.transport,.keyframes{display:flex;gap:8px;align-items:center;margin-top:11px}.transport input{flex:1}.frame{min-width:135px;text-align:right;color:var(--dim)}
.kf{padding:4px 8px}.kf.active{border-color:var(--blue);color:#beddff}
.annotation{margin-top:12px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px}
.annotation h3{margin:0 0 7px}.objects{color:#b9d2ff}.criteria li{margin:5px 0}
@media(max-width:900px){header{flex-wrap:wrap}.videos{min-width:1100px}main{overflow-x:auto}}
</style>
</head>
<body data-fps="{{ fps }}">
<header>
  <b>GT × {{ rollout_ids|length }} policy rollouts</b>
  <span>{{ episodes|length }} episodes · {{ rollout_root }}</span><span class="grow"></span>
  <form method="get">
    <select name="task" onchange="this.form.submit()">
      <option value="" {% if not selected_task %}selected{% endif %}>All tasks</option>
      {% for value in tasks %}<option value="{{ value }}" {% if value==selected_task %}selected{% endif %}>{{ value }}</option>{% endfor %}
    </select>
    <select name="view" onchange="this.form.submit()">
      {% for value in cameras %}<option value="{{ value }}" {% if value==view %}selected{% endif %}>{{ value }}</option>{% endfor %}
    </select>
  </form>
  <span id="warm"></span>
</header>
<main>
{% for episode in episodes %}
<section class="episode" data-criteria='{{ criteria_by_episode[episode.task ~ "/" ~ episode.episode_id]|tojson }}'>
  <div class="meta"><b>{{ episode.task }}/{{ episode.episode_id }}</b><span>{{ episode.n_frames }} frames @ {{ fps }} fps</span></div>
  <div class="videos">
    {% for source,label in sources %}
    <div class="panel">
      <div class="label">{{ label }}</div>
      {% if episode.available[source] %}
      <video class="sync-video" muted playsinline preload="metadata"
        src="/clip.mp4?task={{ episode.task }}&ep={{ episode.episode_id }}&source={{ source }}&view={{ view }}"></video>
      {% else %}<div class="missing">missing {{ label }}</div>{% endif %}
    </div>
    {% endfor %}
  </div>
  <div class="transport">
    <button class="play primary">Play</button>
    <button class="continue">Continue</button>
    <input class="seek" type="range" min="0" max="{{ episode.n_frames - 1 }}" value="0" step="1">
    <span class="frame">frame 0 / {{ episode.n_frames - 1 }}</span>
  </div>
  <div class="keyframes"><span>Keyframes:</span>
    {% for segment in episode.segments %}<button class="kf" data-frame="{{ segment.key_frame }}">{{ segment.key_frame }}</button>{% endfor %}
  </div>
  <div class="annotation">
    <div class="objects"><b>Grounding objects:</b> {{ episode.grounding_objects|join(', ') or '(none)' }}</div>
    <h3 class="criteria-title">Success criteria</h3>
    <ul class="criteria"><li>Select or play to a keyframe.</li></ul>
  </div>
</section>
{% endfor %}
</main>
<script>
const fps=Number(document.body.dataset.fps)||30;
document.querySelectorAll('.episode').forEach(card=>{
  const videos=[...card.querySelectorAll('.sync-video')],master=videos[0],seek=card.querySelector('.seek'),frameLabel=card.querySelector('.frame'),play=card.querySelector('.play');
  const criteria=JSON.parse(card.dataset.criteria||'{}'),keyframes=Object.keys(criteria).map(Number).sort((a,b)=>a-b);
  let lastFrame=0,pausedAt=-1,syncing=false;
  const currentFrame=()=>master?Math.max(0,Math.round(master.currentTime*fps)):0;
  const sync=(force=false)=>{if(!master||syncing)return;syncing=true;for(const v of videos.slice(1)){if(force||Math.abs(v.currentTime-master.currentTime)>.12){try{v.currentTime=master.currentTime}catch(_){}}}syncing=false};
  const render=f=>{frameLabel.textContent=`frame ${f} / ${seek.max}`;seek.value=f;card.querySelectorAll('.kf').forEach(b=>b.classList.toggle('active',Number(b.dataset.frame)===f));if(criteria[String(f)]){card.querySelector('.criteria-title').textContent=`Success criteria @ frame ${f}`;card.querySelector('.criteria').innerHTML=criteria[String(f)].map(x=>`<li>${x}</li>`).join('')}};
  const pauseAll=()=>videos.forEach(v=>v.pause());
  const playAll=async()=>{if(!master)return;sync(true);await Promise.allSettled(videos.map(v=>v.play()));play.textContent='Pause'};
  const seekAll=f=>{const t=Number(f)/fps;videos.forEach(v=>{try{v.currentTime=t}catch(_){}});lastFrame=Number(f);pausedAt=-1;render(Number(f))};
  if(master){master.addEventListener('timeupdate',()=>{const f=currentFrame();sync();seek.value=f;frameLabel.textContent=`frame ${f} / ${seek.max}`;for(const k of keyframes){if(k>lastFrame&&k<=f&&k!==pausedAt){pauseAll();pausedAt=k;seekAll(k);render(k);play.textContent='Play';break}}lastFrame=f});master.addEventListener('ended',()=>{pauseAll();play.textContent='Play'})}
  play.onclick=()=>{if(!master)return;if(master.paused)playAll();else{pauseAll();play.textContent='Play'}};
  card.querySelector('.continue').onclick=()=>{pausedAt=-1;lastFrame=currentFrame();card.querySelectorAll('.kf').forEach(b=>b.classList.remove('active'));playAll()};
  seek.oninput=()=>seekAll(seek.value);
  card.querySelectorAll('.kf').forEach(b=>b.onclick=()=>{pauseAll();seekAll(b.dataset.frame);pausedAt=Number(b.dataset.frame);render(pausedAt);play.textContent='Play'});
  videos.forEach(v=>{v.onclick=()=>master&&master.paused?playAll():pauseAll()});
});
async function warm(){try{const r=await fetch('/api/prewarm/status'),s=await r.json();document.querySelector('#warm').textContent=s.total?`prewarm ${s.done}/${s.total}${s.failed?' · failed '+s.failed:''}`:'';if(s.running)setTimeout(warm,1500)}catch(_){}}warm();
</script>
</body></html>"""


app = Flask(__name__)


def _frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    indexed = []
    for path in directory.iterdir():
        match = FRAME_RE.fullmatch(path.name)
        if path.is_file() and match:
            indexed.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed)]


def _source_dir(task: str, episode: str, source: str, view: str) -> Path:
    if source == "gt":
        return GT_ROOT / task / episode / "video" / view
    if source not in ROLLOUT_IDS:
        raise ValueError(source)
    return ROLLOUT_ROOT / task / episode / source / "video" / view


def _load_prompt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _discover() -> list[Episode]:
    episodes = []
    for task_dir in sorted(path for path in GT_ROOT.iterdir() if path.is_dir()):
        if TASK_FILTER and task_dir.name not in TASK_FILTER:
            continue
        for episode_dir in sorted(
            path for path in task_dir.iterdir()
            if path.is_dir() and EPISODE_RE.fullmatch(path.name)
        ):
            prompt = _load_prompt(episode_dir / "prompt" / "prompts.json")
            gt_frames = _frames(episode_dir / "video" / "top_head")
            if not gt_frames:
                continue
            available = {"gt": True}
            for rollout_id in ROLLOUT_IDS:
                available[rollout_id] = bool(
                    _frames(
                        ROLLOUT_ROOT
                        / task_dir.name
                        / episode_dir.name
                        / rollout_id
                        / "video"
                        / "top_head"
                    )
                )
            if not any(available[value] for value in ROLLOUT_IDS):
                continue
            segments = []
            for item in prompt.get("segments", []):
                if not isinstance(item, dict):
                    continue
                segments.append(
                    Segment(
                        int(item.get("key_frame", 0)),
                        tuple(str(value) for value in item.get("success_criteria", [])),
                    )
                )
            episodes.append(
                Episode(
                    task=task_dir.name,
                    episode_id=episode_dir.name,
                    n_frames=int(prompt.get("n_frames", len(gt_frames))),
                    grounding_objects=[
                        str(value) for value in prompt.get("grounding_objects", [])
                    ],
                    segments=sorted(segments, key=lambda value: value.key_frame),
                    available=available,
                )
            )
    return episodes


def _episodes(reload: bool = False) -> list[Episode]:
    global _EPISODES
    with _CACHE_LOCK:
        if reload or _EPISODES is None:
            _EPISODES = _discover()
        return list(_EPISODES)


def _clip_key(task: str, episode: str, source: str, view: str) -> str:
    directory = _source_dir(task, episode, source, view)
    frames = _frames(directory)
    if not frames:
        raise FileNotFoundError(directory)
    state = "|".join(
        f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in frames
    )
    digest = hashlib.sha1(
        f"{directory}|{FPS}|{PREVIEW_HEIGHT}|{CRF}|{state}".encode()
    ).hexdigest()[:20]
    return f"{task}_{episode}_{source}_{view}_{digest}"


def _ensure_clip(task: str, episode: str, source: str, view: str) -> Path:
    source_dir = _source_dir(task, episode, source, view)
    frames = _frames(source_dir)
    if not frames:
        raise FileNotFoundError(source_dir)
    key = _clip_key(task, episode, source, view)
    destination = CACHE_ROOT / f"{key}.mp4"
    if destination.is_file():
        return destination
    with _CLIP_LOCKS_LOCK:
        lock = _CLIP_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if destination.is_file():
            return destination
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as temporary:
            temp = Path(temporary)
            manifest = temp / "frames.txt"
            duration = 1.0 / FPS
            lines = []
            for frame in frames:
                escaped = str(frame.resolve()).replace("'", "'\\''")
                lines.extend((f"file '{escaped}'", f"duration {duration:.10f}"))
            escaped = str(frames[-1].resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            output = temp / "clip.mp4"
            filters = []
            if PREVIEW_HEIGHT > 0:
                filters.append(f"scale=-2:{PREVIEW_HEIGHT}")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(manifest),
                "-an", "-c:v", "libx264", "-preset", PRESET,
                "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            ]
            if filters:
                command.extend(("-vf", ",".join(filters)))
            command.append(str(output))
            subprocess.run(command, check=True)
            shutil.move(str(output), destination)
    return destination


def _episode(task: str, episode: str) -> Episode:
    for item in _episodes():
        if item.task == task and item.episode_id == episode:
            return item
    raise KeyError((task, episode))


@app.route("/")
def index():
    all_episodes = _episodes(reload=bool(request.args.get("reload")))
    if not all_episodes:
        return "No matched GT/generated episode pairs.", 404
    tasks = sorted({item.task for item in all_episodes})
    selected_task = request.args.get("task", "").strip()
    visible = (
        [item for item in all_episodes if item.task == selected_task]
        if selected_task in tasks
        else all_episodes
    )
    view = request.args.get("view", "top_head")
    if view not in CAMERAS:
        view = "top_head"
    criteria_by_episode = {
        f"{episode.task}/{episode.episode_id}": {
            str(segment.key_frame): list(segment.success_criteria)
            for segment in episode.segments
        }
        for episode in visible
    }
    sources = [("gt", "Ground Truth")] + [
        (rollout_id, f"Generation {rollout_id}") for rollout_id in ROLLOUT_IDS
    ]
    from flask import render_template_string

    response = make_response(
        render_template_string(
            PAGE,
            episodes=visible,
            tasks=tasks,
            selected_task=selected_task,
            cameras=CAMERAS,
            view=view,
            fps=FPS,
            criteria_by_episode=criteria_by_episode,
            sources=sources,
            rollout_ids=ROLLOUT_IDS,
            rollout_root=str(ROLLOUT_ROOT),
        )
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/clip.mp4")
def clip():
    task = request.args.get("task", "")
    episode = request.args.get("ep", "")
    source = request.args.get("source", "")
    view = request.args.get("view", "top_head")
    try:
        _episode(task, episode)
        path = _ensure_clip(task, episode, source, view)
    except (KeyError, ValueError, FileNotFoundError):
        abort(404)
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/api/prewarm/status")
def prewarm_status():
    return dict(_PREWARM)


def _start_prewarm(workers: int) -> None:
    if _PREWARM["running"]:
        return
    jobs = []
    for episode in _episodes():
        for source in ("gt", *ROLLOUT_IDS):
            if episode.available.get(source):
                for view in CAMERAS:
                    if _frames(_source_dir(episode.task, episode.episode_id, source, view)):
                        jobs.append((episode.task, episode.episode_id, source, view))
    _PREWARM.update(running=True, done=0, total=len(jobs), failed=0)

    def one(job: tuple[str, str, str, str]) -> None:
        try:
            _ensure_clip(*job)
        except Exception:
            _PREWARM["failed"] += 1
        finally:
            _PREWARM["done"] += 1

    def run() -> None:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(one, jobs))
        _PREWARM["running"] = False

    threading.Thread(target=run, daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        default="/path/to/policy-ground-truth",
    )
    parser.add_argument(
        "--rollout-root",
        default="/path/to/policy-rollouts",
    )
    parser.add_argument("--rollouts", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cache-dir", default="~/.cache/policy_rollout_viewer")
    parser.add_argument("--preview-height", type=int, default=480)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-workers", type=int, default=4)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    global GT_ROOT, ROLLOUT_ROOT, CACHE_ROOT, FPS
    global PREVIEW_HEIGHT, CRF, PRESET, ROLLOUT_IDS, TASK_FILTER
    args = parse_args()
    GT_ROOT = Path(args.gt_root).expanduser().resolve()
    ROLLOUT_ROOT = Path(args.rollout_root).expanduser().resolve()
    CACHE_ROOT = Path(args.cache_dir).expanduser().resolve()
    if not GT_ROOT.is_dir():
        raise SystemExit(f"GT root does not exist: {GT_ROOT}")
    if not ROLLOUT_ROOT.is_dir():
        raise SystemExit(f"rollout root does not exist: {ROLLOUT_ROOT}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    FPS = float(args.fps)
    PREVIEW_HEIGHT = max(0, int(args.preview_height))
    CRF = int(args.crf)
    PRESET = str(args.preset)
    ROLLOUT_IDS = tuple(str(value) for value in args.rollouts)
    TASK_FILTER = set(args.tasks) if args.tasks else None
    if args.clear_cache and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    episodes = _episodes()
    print(
        f"[policy_rollout_viewer] {len(episodes)} episodes; "
        f"GT={GT_ROOT}; rollout={ROLLOUT_ROOT}",
        flush=True,
    )
    if args.prewarm:
        _start_prewarm(max(1, int(args.prewarm_workers)))
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
