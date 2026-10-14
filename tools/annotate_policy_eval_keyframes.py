#!/usr/bin/env python3
"""Annotate policy-eval keyframes from the top-head video in a browser.

This reuses the episode discovery and lazy MP4 cache from
``visualize_policy_eval_prompts.py``. Saving preserves every prompt field and
only replaces ``segments``; criteria attached to unchanged keyframes survive.

Example:
    python tools/annotate_policy_eval_keyframes.py \
      --root /path/to/policy-ground-truth \
      --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template_string, request

import visualize_policy_eval_prompts as viewer


app = Flask(__name__)
_SAVE_LOCK = threading.Lock()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Policy-eval keyframe annotation</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d;
      --text:#e6edf3; --muted:#8b949e; --blue:#2f81f7; --red:#f85149; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text);
      font:14px/1.45 system-ui,-apple-system,sans-serif; }
    header { position:sticky; top:0; z-index:2; padding:12px 18px;
      background:#0d1117ee; border-bottom:1px solid var(--line); }
    .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    select,button { color:var(--text); background:#21262d; border:1px solid var(--line);
      border-radius:6px; padding:7px 10px; }
    button { cursor:pointer; }
    button:hover { border-color:#8b949e; }
    button.primary { background:var(--blue); border-color:var(--blue); }
    button.danger { color:#ffb3ad; }
    main { max-width:1280px; margin:auto; padding:18px; }
    video { display:block; width:100%; max-height:72vh; background:#000;
      border:1px solid var(--line); border-radius:8px; }
    .toolbar { margin-top:12px; padding:12px; background:var(--panel);
      border:1px solid var(--line); border-radius:8px; }
    .frame { font:600 16px ui-monospace,monospace; min-width:180px; }
    .keyframes { display:flex; gap:7px; flex-wrap:wrap; margin-top:12px; min-height:34px; }
    .keyframes button { font-family:ui-monospace,monospace; }
    .keyframes button.active { background:#9e6a03; border-color:#d29922; }
    .keyframes button.filled::after { content:" ✓"; color:#56d364; }
    .muted { color:var(--muted); }
    #status.ok { color:#56d364; } #status.error { color:#ff7b72; }
    .help { margin-top:10px; color:var(--muted); }
  </style>
</head>
<body>
<header>
  <div class="row">
    <button id="prevEp">← Previous episode</button>
    <select id="episode"></select>
    <button id="nextEp">Next episode →</button>
    <span id="progress" class="muted"></span>
    <span id="status"></span>
  </div>
</header>
<main>
  <video id="video" controls muted preload="metadata"></video>
  <section class="toolbar">
    <div class="row">
      <span id="frame" class="frame">frame 0</span>
      <button id="back">−1 frame</button>
      <button id="forward">+1 frame</button>
      <button id="add" class="primary">Add current frame (K)</button>
      <button id="remove" class="danger">Remove selected (D)</button>
      <button id="save">Save (S)</button>
    </div>
    <div id="keyframes" class="keyframes"></div>
    <div class="help">
      Space: play/pause · ←/→: one frame · K: add · D: remove selected ·
      S: save. Click a keyframe to seek to it. Existing success criteria are
      preserved unless that keyframe is removed.
    </div>
  </section>
</main>
<script>
const FPS = {{ fps|tojson }};
const episodes = {{ episodes|tojson }};
const video = document.querySelector('#video');
const select = document.querySelector('#episode');
const statusNode = document.querySelector('#status');
let episodeIndex = 0;
let keyframes = [];
let selected = null;
let dirty = false;

const current = () => episodes[episodeIndex];
const frameNow = () => {
  const ep = current();
  return Math.max(0, Math.min(ep.n_frames - 1, Math.round(video.currentTime * FPS)));
};
const setStatus = (text, kind='') => {
  statusNode.textContent = text;
  statusNode.className = kind;
};
const updateFrame = () => {
  const ep = current();
  document.querySelector('#frame').textContent =
    `frame ${frameNow()} / ${Math.max(0, ep.n_frames - 1)}  (${video.currentTime.toFixed(2)}s)`;
};
const seek = (frame) => {
  video.currentTime = Math.max(0, Math.min(current().n_frames - 1, frame)) / FPS;
  updateFrame();
};
function renderKeys() {
  const box = document.querySelector('#keyframes');
  box.replaceChildren();
  if (!keyframes.length) {
    box.innerHTML = '<span class="muted">No keyframes yet.</span>';
    return;
  }
  for (const item of keyframes) {
    const b = document.createElement('button');
    b.textContent = `frame ${item.key_frame}`;
    if (item.key_frame === selected) b.classList.add('active');
    if (item.success_criteria?.length) b.classList.add('filled');
    b.onclick = () => { selected = item.key_frame; seek(item.key_frame); renderKeys(); };
    box.appendChild(b);
  }
}
function loadEpisode(index) {
  if (dirty && !confirm('Discard unsaved keyframe changes?')) {
    select.value = episodeIndex;
    return;
  }
  episodeIndex = Math.max(0, Math.min(episodes.length - 1, index));
  const ep = current();
  select.value = episodeIndex;
  keyframes = ep.segments.map(x => ({
    key_frame:x.key_frame, success_criteria:[...(x.success_criteria || [])]
  }));
  selected = keyframes.length ? keyframes[0].key_frame : null;
  dirty = false;
  setStatus('');
  document.querySelector('#progress').textContent =
    `${episodeIndex + 1}/${episodes.length} · ${ep.task}/${ep.name}`;
  video.src = `/clip.mp4?task=${encodeURIComponent(ep.task)}&ep=${ep.episode_index}`;
  video.load();
  renderKeys();
  updateFrame();
}
function addCurrent() {
  const frame = frameNow();
  if (!keyframes.some(x => x.key_frame === frame)) {
    keyframes.push({key_frame:frame, success_criteria:[]});
    keyframes.sort((a,b) => a.key_frame - b.key_frame);
    dirty = true;
  }
  selected = frame;
  renderKeys();
}
function removeSelected() {
  if (selected === null) return;
  keyframes = keyframes.filter(x => x.key_frame !== selected);
  selected = null;
  dirty = true;
  renderKeys();
}
async function save() {
  const ep = current();
  setStatus('Saving…');
  try {
    const response = await fetch('/api/keyframes', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task:ep.task, episode_index:ep.episode_index,
                           keyframes:keyframes.map(x => x.key_frame)})
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    ep.segments = body.segments;
    keyframes = body.segments.map(x => ({
      key_frame:x.key_frame, success_criteria:[...(x.success_criteria || [])]
    }));
    dirty = false;
    renderKeys();
    setStatus(`Saved ${keyframes.length} keyframe(s)`, 'ok');
  } catch (error) {
    setStatus(`Save failed: ${error.message}`, 'error');
  }
}
episodes.forEach((ep, i) => {
  const option = document.createElement('option');
  option.value = i;
  option.textContent = `${ep.task}/${ep.name} (${ep.segments.length} keyframes)`;
  select.appendChild(option);
});
select.onchange = () => loadEpisode(Number(select.value));
document.querySelector('#prevEp').onclick = () => loadEpisode(episodeIndex - 1);
document.querySelector('#nextEp').onclick = () => loadEpisode(episodeIndex + 1);
document.querySelector('#back').onclick = () => seek(frameNow() - 1);
document.querySelector('#forward').onclick = () => seek(frameNow() + 1);
document.querySelector('#add').onclick = addCurrent;
document.querySelector('#remove').onclick = removeSelected;
document.querySelector('#save').onclick = save;
video.addEventListener('timeupdate', updateFrame);
video.addEventListener('loadedmetadata', updateFrame);
window.addEventListener('beforeunload', e => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});
document.addEventListener('keydown', e => {
  if (['SELECT','INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); seek(frameNow() - 1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); seek(frameNow() + 1); }
  else if (e.key.toLowerCase() === 'k') addCurrent();
  else if (e.key.toLowerCase() === 'd') removeSelected();
  else if (e.key.toLowerCase() === 's') save();
});
if (episodes.length) loadEpisode(0);
</script>
</body>
</html>
"""


def _episode_payload(ep: viewer.Episode) -> dict[str, Any]:
    return {
        "task": ep.task,
        "name": Path(ep.dir_name).name,
        "episode_index": ep.episode_index,
        "n_frames": ep.n_frames,
        "segments": [
            {
                "key_frame": segment.key_frame,
                "success_criteria": segment.success_criteria,
            }
            for segment in ep.segments
        ],
    }


def _prompt_path(ep: viewer.Episode) -> Path:
    path = viewer.ROOT / ep.dir_name / "prompt" / "prompts.json"
    if not path.is_file():
        abort(404, description=f"missing {path}")
    return path


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@app.get("/")
def index():
    episodes = [_episode_payload(ep) for ep in viewer._get_episodes()]
    return render_template_string(PAGE, episodes=episodes, fps=viewer.FPS)


@app.get("/clip.mp4")
def clip():
    task = (request.args.get("task") or "").strip()
    try:
        ep_idx = int(request.args["ep"])
    except (KeyError, ValueError):
        abort(400)
    return viewer._stream_file_with_range(viewer._ensure_clip(task, ep_idx))


@app.post("/api/keyframes")
def save_keyframes():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="expected a JSON object"), 400
    try:
        task = str(body["task"])
        ep_idx = int(body["episode_index"])
        raw_frames = body["keyframes"]
        if not isinstance(raw_frames, list):
            raise TypeError("keyframes must be a list")
        frames = sorted({int(value) for value in raw_frames})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    ep = viewer._get_episode(task, ep_idx)
    if any(frame < 0 or frame >= ep.n_frames for frame in frames):
        return jsonify(error=f"keyframes must be in [0, {ep.n_frames - 1}]"), 400

    with _SAVE_LOCK:
        path = _prompt_path(ep)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("prompt root must be a JSON object")
            existing: dict[int, list[str]] = {}
            for segment in data.get("segments", []) or []:
                if isinstance(segment, dict) and "key_frame" in segment:
                    criteria = segment.get("success_criteria") or []
                    if not isinstance(criteria, list):
                        criteria = [str(criteria)]
                    existing[int(segment["key_frame"])] = [
                        str(value) for value in criteria
                    ]
            segments = [
                {
                    "key_frame": frame,
                    "success_criteria": existing.get(frame, []),
                }
                for frame in frames
            ]
            data["segments"] = segments
            _atomic_write_json(path, data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return jsonify(error=str(exc)), 500

        global_cache = viewer._EPISODES_CACHE
        if global_cache is not None:
            for cached in global_cache:
                if cached.task == task and cached.episode_index == ep_idx:
                    cached.segments = [
                        viewer.Segment(
                            key_frame=item["key_frame"],
                            success_criteria=item["success_criteria"],
                        )
                        for item in segments
                    ]
                    break
    return jsonify(segments=segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/path/to/policy-ground-truth"
        ),
    )
    parser.add_argument("--view", default="top_head")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "POLICY_EVAL_VIEWER_CACHE",
                "~/.cache/acwmbench_policy_eval_viewer",
            )
        ).expanduser(),
    )
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    viewer.ROOT = root
    viewer.FRAME_VIEW = args.view
    viewer.FPS = float(args.fps)
    viewer.TASK_FILTER = args.tasks
    viewer.CLIP_CACHE_DIR = args.cache_dir.expanduser().resolve()
    viewer._EPISODES_CACHE = None
    if args.clear_cache and viewer.CLIP_CACHE_DIR.exists():
        shutil.rmtree(viewer.CLIP_CACHE_DIR)
    viewer.CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    episodes = viewer._get_episodes()
    if not episodes:
        raise SystemExit(f"no episodes with prompt/prompts.json found under {root}")
    print(
        f"[keyframe_annotator] root={root} view={args.view} "
        f"episodes={len(episodes)} fps={viewer.FPS:g}",
        flush=True,
    )
    if args.prewarm:
        viewer._prewarm()
    print(
        f"[keyframe_annotator] serving http://{args.host}:{args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
