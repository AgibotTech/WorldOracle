#!/usr/bin/env python3
"""Web viewer for policy_eval_dataset prompts (keyframe + success_criteria).

Mirrors ``lerobot_viewer/vlm_app.py`` playback: video plays and auto-pauses at
each keyframe; **Play** resumes until the next keyframe. While paused, that
keyframe's ``success_criteria`` (and episode ``grounding_objects``) are shown.

``policy_eval_dataset`` has extracted frames (no ``raw_video``), so clips are
transcoded on demand from ``video/<view>/frame_*.jpg``.

Layout::

    <root>/<task>/epXXXXXX/prompt/prompts.json
    <root>/<task>/epXXXXXX/video/<view>/frame_00000.jpg ...

Or a single-task root::

    <root>/epXXXXXX/prompt/prompts.json
    ...

Run::

    python tools/visualize_policy_eval_prompts.py \
        --root /path/to/policy-inputs \
        --host 127.0.0.1 --port 8765 --prewarm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (  # type: ignore[import-not-found]
    Flask,
    Response,
    abort,
    jsonify,
    make_response,
    request,
    send_file,
)


SUPPORTED_TASKS = ("pickfruit", "hangtowel", "pourwater")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_EP_DIR_RE = re.compile(r"ep(\d+)$")
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


@dataclass
class Segment:
    key_frame: int
    success_criteria: list[str]


@dataclass
class Episode:
    task: str
    episode_index: int
    dir_name: str  # relative to ROOT, e.g. pickfruit/ep000002
    n_frames: int
    grounding_objects: list[str] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    has_frames: bool = False

    @property
    def keyframes(self) -> list[int]:
        return [s.key_frame for s in self.segments]

    @property
    def criteria_map(self) -> dict[str, list[str]]:
        return {str(s.key_frame): s.success_criteria for s in self.segments}

    @property
    def ep_id(self) -> str:
        return f"{self.task}/{self.dir_name.split('/')[-1]}"


ROOT = Path()
FPS = 30.0
FRAME_VIEW = "top_head"
TASK_FILTER: list[str] | None = None

_CACHE_LOCK = threading.Lock()
_EPISODES_CACHE: list[Episode] | None = None

CLIP_CACHE_DIR = Path()
CLIP_CRF = 28
CLIP_PRESET = "veryfast"
CLIP_SCALE_H = 480
_CLIP_LOCK_REGISTRY_LOCK = threading.Lock()
_CLIP_LOCKS: dict[str, threading.Lock] = {}
_CLIP_EXECUTOR: ThreadPoolExecutor | None = None
_PREWARM_STATE: dict[str, Any] = {
    "total": 0,
    "done": 0,
    "failed": 0,
    "running": False,
}


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>policy_eval prompts viewer</title>
  <style>
    :root {
      --bg: #0f1115; --bg-soft: #161a22; --bg-card: #1b2030;
      --fg: #e6e6e6; --fg-dim: #9aa3b2; --accent: #4ea1ff; --border: #262c3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--fg);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: var(--bg-soft); padding: 1px 5px; border-radius: 4px; }
    .topbar {
      display: flex; align-items: baseline; gap: 16px; padding: 10px 18px;
      background: var(--bg-soft); border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 10;
    }
    .topbar .brand { font-weight: 700; }
    .topbar .meta { color: var(--fg-dim); font-size: 12px; margin-left: auto; }
    .controls {
      display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
      padding: 10px 18px; background: var(--bg-soft); border-bottom: 1px solid var(--border);
    }
    .controls label { display: inline-flex; align-items: center; gap: 6px; color: var(--fg-dim); font-size: 12px; }
    .controls select, .controls input[type=search] {
      background: var(--bg-card); border: 1px solid var(--border); color: var(--fg);
      border-radius: 4px; padding: 4px 8px; min-height: 30px;
    }
    .controls button, .btn {
      background: var(--accent); color: #08111c; border: 0; border-radius: 4px;
      padding: 5px 10px; font-weight: 700; cursor: pointer;
    }
    .controls button:hover, .btn:hover { filter: brightness(1.08); }
    .pager { color: var(--fg-dim); font-size: 12px; }
    .prewarm-status { color: var(--fg-dim); font-size: 12px; }
    main.grid { display: grid; grid-template-columns: 1fr; max-width: 1180px; margin: 0 auto; padding: 16px; gap: 18px; }
    .card {
      display: flex; gap: 16px; align-items: flex-start;
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
    }
    .vlm-left { flex: 1 1 60%; min-width: 0; }
    .vlm-right {
      flex: 1 1 40%; min-width: 260px; max-height: 560px; overflow-y: auto;
      background: #11161d; border: 1px solid #2a3340; border-radius: 8px; padding: 12px;
    }
    .card-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
    .ep-idx { font-weight: 700; }
    .ep-len { color: var(--fg-dim); font-size: 12px; }
    .grounding { color: #b9d0ff; font-size: 12px; margin-bottom: 8px; line-height: 1.45; }
    .vbox { position: relative; background: #000; border-radius: 8px; overflow: hidden; }
    video { display: block; width: 100%; max-height: 520px; background: #000; }
    .cam-label {
      position: absolute; left: 8px; top: 8px; padding: 3px 8px; border-radius: 4px;
      background: rgba(0,0,0,.7); color: #cfe3ff; font-size: 12px; font-weight: 700;
    }
    .scrub { display: flex; align-items: center; gap: 8px; padding: 8px 4px 0; }
    .seek { flex: 1; }
    .tlabel { color: var(--fg-dim); font-variant-numeric: tabular-nums; font-size: 12px; min-width: 90px; text-align: right; }
    .continue-btn { margin-top: 8px; }
    .no-video { color: #ff8f8f; padding: 24px; text-align: center; }
    .keyframes { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
    .kf-label { color: var(--fg-dim); font-size: 12px; margin-right: 4px; }
    .kf-btn {
      background: var(--bg-soft); color: var(--fg); border: 1px solid var(--border);
      border-radius: 4px; padding: 3px 8px; cursor: pointer; font-variant-numeric: tabular-nums;
    }
    .kf-btn.active { border-color: var(--accent); color: #cfe3ff; }
    .kf-btn.done { border-color: #3a7; }
    .kf-btn.empty { border-color: #a44; }
    .crit-head { font-weight: 700; margin-bottom: 8px; color: #cfe3ff; }
    .crit-head .kf-num { color: #ffd479; }
    .crit-list { margin: 0; padding-left: 18px; }
    .crit-list li { margin: 4px 0; line-height: 1.4; }
    .crit-empty { color: #ff8f8f; font-style: italic; }
    .crit-hint { color: #8aa0b6; font-style: italic; }
    @media (max-width: 900px) { .card { flex-direction: column; } }
  </style>
</head>
<body data-fps="{{ fps }}">
  <header class="topbar">
    <div class="brand">policy_eval prompts</div>
    <div class="title"><code>{{ root }}</code></div>
    <div class="meta">{{ total }} episodes · {{ fps }} fps · view={{ view }}</div>
  </header>

  <form class="controls" method="get">
    <label>task
      <select name="task" onchange="this.form.submit()">
        <option value="" {% if not task %}selected{% endif %}>all</option>
        {% for t in tasks %}
          <option value="{{ t }}" {% if t == task %}selected{% endif %}>{{ t }}</option>
        {% endfor %}
      </select>
    </label>
    <label>per page
      <select name="per_page" onchange="this.form.submit()">
        {% for n in [1, 2, 3, 4, 6, 8] %}
          <option value="{{ n }}" {% if n == per_page %}selected{% endif %}>{{ n }}</option>
        {% endfor %}
      </select>
    </label>
    <label>filter
      <input type="search" name="q" value="{{ q }}" placeholder="ep / task" />
    </label>
    <button type="submit">Apply</button>
    <span class="pager">
      {% if page > 1 %}
        <a href="?task={{ task }}&per_page={{ per_page }}&q={{ q }}&page={{ page - 1 }}">&laquo; prev</a>
      {% endif %}
      page {{ page }} / {{ n_pages }} ({{ total }} matched)
      {% if page < n_pages %}
        <a href="?task={{ task }}&per_page={{ per_page }}&q={{ q }}&page={{ page + 1 }}">next &raquo;</a>
      {% endif %}
    </span>
    <span class="globals">
      <label><input type="checkbox" id="g-autoplay" checked /> autoplay</label>
      <label><input type="checkbox" id="g-mute" checked /> mute</label>
      <a class="btn" href="?task={{ task }}&per_page={{ per_page }}&q={{ q }}&page={{ page }}&reload=1">↻ Reload prompts</a>
      <button type="button" id="btn-prewarm">Prewarm clips</button>
      <span id="prewarm-status" class="prewarm-status"></span>
    </span>
  </form>

  <main class="grid">
    {% for ep in episodes %}
    <div class="card"
         data-task="{{ ep.task }}"
         data-episode="{{ ep.episode_index }}"
         data-total-frames="{{ ep.n_frames }}">
      <script type="application/json" class="criteria-data">{{ ep.criteria_map | tojson }}</script>
      <div class="vlm-left">
        <div class="card-head">
          <span class="ep-idx">{{ ep.dir_name }}</span>
          <span class="ep-len">{{ ep.n_frames }} frames · {{ ep.keyframes | length }} keyframes</span>
        </div>
        {% if ep.grounding_objects %}
        <div class="grounding"><b>grounding_objects:</b> {{ ep.grounding_objects | join(', ') }}</div>
        {% else %}
        <div class="grounding"><b>grounding_objects:</b> <i>(empty)</i></div>
        {% endif %}
        <div class="videos">
          <div class="vbox" title="{{ view }}">
            {% if ep.has_frames %}
            <video
              preload="none"
              muted
              playsinline
              data-cam="{{ view }}"
              data-src="/clip.mp4?task={{ ep.task }}&ep={{ ep.episode_index }}"></video>
            <div class="cam-label">{{ view }}</div>
            <div class="scrub">
              <input type="range" class="seek" min="0" max="0" step="0.05" value="0" aria-label="seek" />
              <span class="tlabel">0 / {{ ep.n_frames }}</span>
            </div>
            <button type="button" class="continue-btn">Play ▶ (pause at next keyframe)</button>
            <button type="button" class="pause-btn">Pause ❚❚</button>
            {% else %}
            <div class="no-video">no frames under video/{{ view }}/</div>
            {% endif %}
          </div>
        </div>
        {% if ep.keyframes %}
        <div class="keyframes">
          <span class="kf-label">Keyframes:</span>
          {% for s in ep.segments %}
            <button type="button"
                    class="kf-btn {{ 'done' if s.success_criteria else 'empty' }}"
                    data-frame="{{ s.key_frame }}">{{ s.key_frame }}</button>
          {% endfor %}
        </div>
        {% endif %}
      </div>
      <div class="vlm-right">
        <div class="crit-head">success_criteria <span class="kf-num"></span></div>
        <div class="crit-body">
          <div class="crit-hint">Play — auto-pauses at each keyframe and shows criteria.
          Click Play to resume, Pause to stop anywhere, or click a keyframe button.</div>
        </div>
      </div>
    </div>
    {% endfor %}
  </main>

  <script>
  (() => {
    const videos = Array.from(document.querySelectorAll('.card video'));
    const gAutoplay = document.getElementById('g-autoplay');
    const gMute = document.getElementById('g-mute');
    const btnPrewarm = document.getElementById('btn-prewarm');
    const prewarmStatus = document.getElementById('prewarm-status');
    const fps = parseFloat(document.body.getAttribute('data-fps') || '30') || 30;

    const loadFlag = (k, fb) => {
      const v = localStorage.getItem(k);
      return v === null ? fb : v === '1';
    };
    const saveFlag = (k, on) => localStorage.setItem(k, on ? '1' : '0');
    if (gAutoplay) gAutoplay.checked = loadFlag('policy_eval.autoplay', true);
    if (gMute) gMute.checked = loadFlag('policy_eval.mute', true);

    const attach = (v) => { if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.load(); } };
    const detach = (v) => { if (v.src) { v.pause(); v.removeAttribute('src'); v.load(); } };
    const applyMute = () => { const m = gMute ? gMute.checked : true; videos.forEach(v => { v.muted = m; }); };
    const applyAutoplay = () => {
      const a = gAutoplay ? gAutoplay.checked : true;
      videos.forEach(v => {
        if (a && v.dataset._inView === '1') { attach(v); v.play().catch(() => {}); }
        else if (!a) v.pause();
      });
    };
    if (gAutoplay) gAutoplay.addEventListener('change', () => { saveFlag('policy_eval.autoplay', gAutoplay.checked); applyAutoplay(); });
    if (gMute) gMute.addEventListener('change', () => { saveFlag('policy_eval.mute', gMute.checked); applyMute(); });

    const readCriteria = (card) => {
      const node = card.querySelector('.criteria-data');
      if (!node) return {};
      try { return JSON.parse(node.textContent || '{}'); } catch (_) { return {}; }
    };
    const renderCriteria = (card, kfFrame) => {
      const crit = readCriteria(card);
      const head = card.querySelector('.crit-head .kf-num');
      const body = card.querySelector('.crit-body');
      if (head) head.textContent = `@ keyframe ${kfFrame}`;
      if (!body) return;
      const list = crit[String(kfFrame)] || [];
      if (!list.length) {
        body.innerHTML = '<div class="crit-empty">(no success_criteria for this keyframe yet)</div>';
        return;
      }
      const ul = document.createElement('ul');
      ul.className = 'crit-list';
      for (const c of list) {
        const li = document.createElement('li');
        li.textContent = c;
        ul.appendChild(li);
      }
      body.innerHTML = '';
      body.appendChild(ul);
    };

    videos.forEach(v => {
      v.muted = gMute ? gMute.checked : true;
      v.addEventListener('click', () => {
        if (!v.paused && !v.ended) v.pause();
        else { attach(v); v.play().catch(() => {}); }
      });

      const box = v.closest('.vbox');
      const card = v.closest('.card');
      const seek = box ? box.querySelector('.seek') : null;
      const label = box ? box.querySelector('.tlabel') : null;
      const continueBtn = box ? box.querySelector('.continue-btn') : null;
      const pauseBtn = box ? box.querySelector('.pause-btn') : null;
      const kfBtns = card ? Array.from(card.querySelectorAll('.kf-btn')) : [];
      const totalFrames = card ? (parseInt(card.getAttribute('data-total-frames'), 10) || 0) : 0;
      const keyframes = kfBtns
        .map(b => parseInt(b.getAttribute('data-frame'), 10))
        .filter(n => Number.isFinite(n))
        .sort((a, b) => a - b);

      const frameToTime = (f) => f / fps;
      const timeToFrame = (t) => Math.max(0, Math.min(totalFrames, Math.floor(t * fps + 0.5)));
      const syncLabel = () => { if (label) label.textContent = `${timeToFrame(v.currentTime)} / ${totalFrames}`; };
      if (v._lastFrame === undefined) v._lastFrame = 0;

      const highlightKf = (kfFrame) => {
        kfBtns.forEach(b => {
          const f = parseInt(b.getAttribute('data-frame'), 10);
          b.classList.toggle('active', f === kfFrame);
        });
      };

      const pauseAtKeyframe = (kfFrame) => {
        v.pause();
        v._lastFrame = kfFrame;
        highlightKf(kfFrame);
        renderCriteria(card, kfFrame);
      };

      const playToNextKeyframe = () => {
        attach(v);
        const currentFrame = timeToFrame(v.currentTime);
        if (v.ended || currentFrame >= Math.max(0, totalFrames - 1)) {
          try { v.currentTime = 0; } catch (_) {}
          v._lastFrame = 0;
        } else {
          v._lastFrame = currentFrame;
        }
        kfBtns.forEach(b => b.classList.remove('active'));
        v.play().catch(() => {});
      };
      if (continueBtn) continueBtn.addEventListener('click', playToNextKeyframe);
      if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
          v.pause();
          v._lastFrame = timeToFrame(v.currentTime);
        });
      }

      const checkKeyframePause = () => {
        if (!keyframes.length || v.paused) return;
        const curFrame = timeToFrame(v.currentTime);
        const last = v._lastFrame || 0;
        let crossed = null;
        for (const kf of keyframes) {
          if (kf > last && kf <= curFrame) { crossed = kf; break; }
        }
        if (crossed !== null) pauseAtKeyframe(crossed);
        v._lastFrame = curFrame;
      };

      if (seek && label) {
        let seeking = false;
        const onMeta = () => {
          const d = Number.isFinite(v.duration) ? v.duration : 0;
          seek.max = d > 0 ? d.toFixed(2) : 0;
          syncLabel();
        };
        v.addEventListener('loadedmetadata', onMeta);
        v.addEventListener('durationchange', onMeta);
        v.addEventListener('timeupdate', () => {
          if (!seeking) seek.value = v.currentTime;
          syncLabel();
          checkKeyframePause();
        });
        seek.addEventListener('pointerdown', () => { seeking = true; attach(v); });
        const endSeek = () => { seeking = false; v._lastFrame = timeToFrame(v.currentTime); };
        seek.addEventListener('pointerup', endSeek);
        seek.addEventListener('pointercancel', endSeek);
        seek.addEventListener('change', endSeek);
        seek.addEventListener('input', () => {
          const t = parseFloat(seek.value);
          if (Number.isFinite(t)) { try { v.currentTime = t; } catch (_) {} }
          syncLabel();
          v._lastFrame = timeToFrame(t);
        });
      }

      v.addEventListener('play', () => {
        kfBtns.forEach(b => b.classList.remove('active'));
      });

      kfBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          const frame = parseInt(btn.getAttribute('data-frame'), 10);
          if (!Number.isFinite(frame)) return;
          attach(v);
          const doSeek = () => { try { v.currentTime = frameToTime(frame); } catch (_) {} };
          if (Number.isFinite(v.duration) && v.duration > 0) doSeek();
          else v.addEventListener('loadedmetadata', doSeek, { once: true });
          pauseAtKeyframe(frame);
        });
      });
    });

    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const v = e.target;
        v.dataset._inView = e.isIntersecting ? '1' : '0';
        if (!e.isIntersecting) detach(v);
        else {
          attach(v);
          if (gAutoplay && gAutoplay.checked) v.play().catch(() => {});
        }
      }
    }, { rootMargin: '150px 0px', threshold: 0.01 });
    videos.forEach(v => io.observe(v));

    document.addEventListener('keydown', (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
      if (e.code === 'Space') {
        e.preventDefault();
        const anyPlaying = videos.some(v => !v.paused && !v.ended);
        videos.forEach(v => {
          if (anyPlaying) v.pause();
          else if (v.dataset._inView === '1') { attach(v); v.play().catch(() => {}); }
        });
      }
    });

    let prewarmTimer = null;
    function renderPrewarm(s) {
      if (!s || !s.total) { prewarmStatus.textContent = ''; return; }
      const pct = Math.round((s.done / s.total) * 100);
      const tag = s.running ? 'running' : (s.done >= s.total ? 'done' : 'idle');
      prewarmStatus.textContent = `prewarm: ${s.done}/${s.total} (${pct}%) ${tag}${s.failed ? ' · failed ' + s.failed : ''}`;
    }
    async function pollPrewarm() {
      try {
        const r = await fetch('/api/prewarm/status');
        const s = await r.json();
        renderPrewarm(s);
        if (s.running) prewarmTimer = setTimeout(pollPrewarm, 1500);
        else prewarmTimer = null;
      } catch (e) { prewarmTimer = null; }
    }
    if (btnPrewarm) {
      btnPrewarm.addEventListener('click', async () => {
        btnPrewarm.disabled = true;
        try { await fetch('/api/prewarm?workers=4', { method: 'POST' }); }
        finally { btnPrewarm.disabled = false; }
        if (!prewarmTimer) pollPrewarm();
      });
      pollPrewarm();
    }
  })();
  </script>
</body>
</html>
"""


app = Flask(__name__)


def _clip_lock(key: str) -> threading.Lock:
    with _CLIP_LOCK_REGISTRY_LOCK:
        return _CLIP_LOCKS.setdefault(key, threading.Lock())


def _parse_ep_index(name: str) -> int | None:
    m = _EP_DIR_RE.search(name)
    return int(m.group(1)) if m else None


def _load_prompt(ep_dir: Path) -> dict[str, Any] | None:
    for p in (ep_dir / "prompt" / "prompts.json", ep_dir / "prompt.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _has_view_frames(ep_dir: Path, view: str) -> bool:
    d = ep_dir / "video" / view
    if not d.is_dir():
        return False
    for path in d.iterdir():
        if path.is_file() and _FRAME_RE.match(path.name):
            return True
    return False


def _list_frame_paths(ep_dir: Path, view: str) -> list[Path]:
    d = ep_dir / "video" / view
    if not d.is_dir():
        return []
    frames: list[tuple[int, Path]] = []
    for path in d.iterdir():
        if not path.is_file():
            continue
        m = _FRAME_RE.match(path.name)
        if m:
            frames.append((int(m.group(1)), path))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]


def _iter_episode_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return (task, ep_dir) pairs under root."""
    out: list[tuple[str, Path]] = []
    # Single-task root: root/ep*
    direct = [p for p in root.iterdir() if p.is_dir() and _parse_ep_index(p.name) is not None]
    if direct:
        task = root.name if root.name in SUPPORTED_TASKS else root.name
        for ep_dir in sorted(direct, key=lambda p: _parse_ep_index(p.name) or 0):
            out.append((task, ep_dir))
        return out

    # Multi-task root: root/<task>/ep*
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        task = task_dir.name
        if TASK_FILTER and task not in TASK_FILTER:
            continue
        if task not in SUPPORTED_TASKS and TASK_FILTER is None:
            # Still accept unknown task folder names that contain ep*
            pass
        for ep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if _parse_ep_index(ep_dir.name) is None:
                continue
            out.append((task, ep_dir))
    return out


def _scan_episodes(root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for task, ep_dir in _iter_episode_dirs(root):
        idx = _parse_ep_index(ep_dir.name)
        if idx is None:
            continue
        data = _load_prompt(ep_dir)
        if not isinstance(data, dict):
            continue
        n_frames = int(data.get("n_frames", 0) or 0)
        grounding = data.get("grounding_objects") or []
        if not isinstance(grounding, list):
            grounding = [str(grounding)]
        segments: list[Segment] = []
        for seg in data.get("segments", []) or []:
            if not isinstance(seg, dict) or "key_frame" not in seg:
                continue
            crit = seg.get("success_criteria") or []
            if not isinstance(crit, list):
                crit = [str(crit)]
            segments.append(
                Segment(
                    key_frame=int(seg["key_frame"]),
                    success_criteria=[str(c) for c in crit],
                )
            )
        segments.sort(key=lambda s: s.key_frame)
        try:
            dir_name = str(ep_dir.relative_to(root))
        except ValueError:
            dir_name = f"{task}/{ep_dir.name}"
        episodes.append(
            Episode(
                task=task,
                episode_index=idx,
                dir_name=dir_name,
                n_frames=n_frames,
                grounding_objects=[str(x) for x in grounding if str(x).strip()],
                segments=segments,
                has_frames=_has_view_frames(ep_dir, FRAME_VIEW),
            )
        )
    episodes.sort(key=lambda e: (e.task, e.episode_index))
    return episodes


def _get_episodes() -> list[Episode]:
    global _EPISODES_CACHE
    with _CACHE_LOCK:
        if _EPISODES_CACHE is None:
            _EPISODES_CACHE = _scan_episodes(ROOT)
        return _EPISODES_CACHE


def _get_episode(task: str, ep_idx: int) -> Episode:
    for e in _get_episodes():
        if e.task == task and e.episode_index == ep_idx:
            return e
    abort(404)


def _root_tag() -> str:
    identity = f"{ROOT}|{FRAME_VIEW}|{FPS:.8g}|h{CLIP_SCALE_H}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]


def _clip_path(task: str, ep_idx: int) -> Path:
    return CLIP_CACHE_DIR / _root_tag() / task / f"ep{ep_idx:06d}.mp4"


def _ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", r"'\''")


def _transcode_from_frames(frames: list[Path], dst_mp4: Path) -> None:
    if not frames:
        raise RuntimeError("no frames to transcode")
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_mp4.with_suffix(".tmp.mp4")
    frame_duration = 1.0 / FPS
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ffconcat",
        prefix="policy_eval_",
        dir=dst_mp4.parent,
        encoding="utf-8",
        delete=False,
    ) as manifest:
        manifest_path = Path(manifest.name)
        manifest.write("ffconcat version 1.0\n")
        for path in frames:
            manifest.write(f"file '{_ffconcat_quote(path)}'\n")
            manifest.write(f"duration {frame_duration:.12f}\n")
        manifest.write(f"file '{_ffconcat_quote(frames[-1])}'\n")

    vf = []
    if CLIP_SCALE_H > 0:
        vf.append(f"scale=-2:{CLIP_SCALE_H}")
    # even dims for yuv420p
    vf.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    cmd = [
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
        ",".join(vf),
        "-r",
        f"{FPS:.8g}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        CLIP_PRESET,
        "-crf",
        str(CLIP_CRF),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(dst_mp4)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {dst_mp4}") from exc
    finally:
        manifest_path.unlink(missing_ok=True)


def _ensure_clip(task: str, ep_idx: int) -> Path:
    ep = _get_episode(task, ep_idx)
    ep_dir = ROOT / ep.dir_name
    frames = _list_frame_paths(ep_dir, FRAME_VIEW)
    if not frames:
        abort(404)
    dst = _clip_path(task, ep_idx)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    lock = _clip_lock(str(dst))
    with lock:
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        _transcode_from_frames(frames, dst)
    return dst


def _prewarm(workers: int = 4) -> None:
    global _CLIP_EXECUTOR
    eps = [e for e in _get_episodes() if e.has_frames]
    if _CLIP_EXECUTOR is None:
        _CLIP_EXECUTOR = ThreadPoolExecutor(max_workers=workers)
    _PREWARM_STATE.update(
        total=len(eps),
        done=sum(1 for e in eps if _clip_path(e.task, e.episode_index).exists()),
        failed=0,
        running=True,
    )

    def _one(task: str, ep_idx: int) -> None:
        try:
            _ensure_clip(task, ep_idx)
            _PREWARM_STATE["done"] += 1
        except Exception:
            _PREWARM_STATE["failed"] += 1

    def _runner() -> None:
        futures = [
            _CLIP_EXECUTOR.submit(_one, e.task, e.episode_index) for e in eps
        ]
        for f in futures:
            f.result()
        _PREWARM_STATE["running"] = False

    threading.Thread(target=_runner, daemon=True).start()


def _stream_file_with_range(file_path: Path) -> Response:
    if not file_path.is_file():
        abort(404)
    size = file_path.stat().st_size
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "application/octet-stream"
    range_header = request.headers.get("Range")
    if not range_header:

        def gen_all():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        resp = Response(gen_all(), mimetype=mime)
        resp.headers["Content-Length"] = str(size)
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    m = _RANGE_RE.fullmatch(range_header.strip())
    if not m:
        abort(416)
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        abort(416)
    if start_s == "":
        suffix = int(end_s)
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start >= size or end >= size or start > end:
        resp = Response(status=416)
        resp.headers["Content-Range"] = f"bytes */{size}"
        return resp
    length = end - start + 1

    def gen_range():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    resp = Response(gen_range(), status=206, mimetype=mime)
    resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(length)
    return resp


@app.route("/")
def index():
    global _EPISODES_CACHE
    if request.args.get("reload"):
        with _CACHE_LOCK:
            _EPISODES_CACHE = None
    episodes = _get_episodes()
    task = (request.args.get("task") or "").strip()
    if task:
        episodes = [e for e in episodes if e.task == task]
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(200, int(request.args.get("per_page", 2))))
    q = (request.args.get("q") or "").strip().lower()
    if q:
        episodes = [
            e
            for e in episodes
            if q in str(e.episode_index)
            or q in e.dir_name.lower()
            or q in e.task.lower()
        ]
    total = len(episodes)
    n_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, n_pages)
    start = (page - 1) * per_page
    visible = episodes[start : start + per_page]
    tasks = sorted({e.task for e in _get_episodes()})
    from flask import render_template_string  # type: ignore[import-not-found]

    resp = make_response(
        render_template_string(
            PAGE,
            root=str(ROOT),
            episodes=visible,
            page=page,
            n_pages=n_pages,
            per_page=per_page,
            q=q,
            total=total,
            fps=FPS,
            view=FRAME_VIEW,
            task=task,
            tasks=tasks,
        )
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/clip.mp4")
def clip_file():
    task = (request.args.get("task") or "").strip()
    try:
        ep_idx = int(request.args.get("ep", ""))
    except ValueError:
        abort(400)
    if not task:
        abort(400)
    dst = _ensure_clip(task, ep_idx)
    return _stream_file_with_range(dst)


@app.route("/frame.jpg")
def frame_file():
    task = (request.args.get("task") or "").strip()
    try:
        ep_idx = int(request.args.get("ep", ""))
        frame = int(request.args.get("f", ""))
    except ValueError:
        abort(400)
    ep = _get_episode(task, ep_idx)
    ep_dir = ROOT / ep.dir_name
    for ext in IMAGE_EXTS:
        cand = ep_dir / "video" / FRAME_VIEW / f"frame_{frame:05d}{ext}"
        if cand.is_file():
            return send_file(cand, mimetype="image/jpeg" if ext != ".png" else "image/png")
    abort(404)


@app.route("/api/episodes")
def api_episodes():
    return jsonify(
        {
            "root": str(ROOT),
            "fps": FPS,
            "view": FRAME_VIEW,
            "episodes": [
                {
                    "task": e.task,
                    "episode_index": e.episode_index,
                    "dir_name": e.dir_name,
                    "n_frames": e.n_frames,
                    "has_frames": e.has_frames,
                    "grounding_objects": e.grounding_objects,
                    "segments": [
                        {
                            "key_frame": s.key_frame,
                            "success_criteria": s.success_criteria,
                        }
                        for s in e.segments
                    ],
                }
                for e in _get_episodes()
            ],
        }
    )


@app.route("/api/prewarm", methods=["POST", "GET"])
def api_prewarm():
    workers = max(1, min(16, int(request.args.get("workers", 4))))
    _prewarm(workers=workers)
    return jsonify(_PREWARM_STATE)


@app.route("/api/prewarm/status")
def api_prewarm_status():
    return jsonify(_PREWARM_STATE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="policy_eval_dataset keyframe / success_criteria viewer"
    )
    p.add_argument(
        "--root",
        type=str,
        default="/path/to/policy-inputs",
        help="policy_eval_dataset root, or a single task subdir",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task whitelist, e.g. pickfruit hangtowel",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--view", type=str, default="top_head")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get(
            "POLICY_EVAL_PROMPTS_CACHE",
            os.path.expanduser("~/.cache/policy_eval_prompts_viewer"),
        ),
    )
    p.add_argument("--crf", type=int, default=28)
    p.add_argument("--preset", type=str, default="veryfast")
    p.add_argument("--preview-height", type=int, default=480)
    p.add_argument("--prewarm", action="store_true")
    p.add_argument("--prewarm-workers", type=int, default=4)
    p.add_argument("--clear-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    global ROOT, FPS, FRAME_VIEW, TASK_FILTER
    global CLIP_CACHE_DIR, CLIP_CRF, CLIP_PRESET, CLIP_SCALE_H, _CLIP_EXECUTOR

    args = parse_args()
    ROOT = Path(args.root).resolve()
    if not ROOT.is_dir():
        raise SystemExit(f"--root {ROOT} is not a directory")
    FPS = float(args.fps) or 30.0
    FRAME_VIEW = args.view
    TASK_FILTER = list(args.tasks) if args.tasks else None
    CLIP_CACHE_DIR = Path(args.cache_dir).resolve()
    if args.clear_cache and CLIP_CACHE_DIR.exists():
        shutil.rmtree(CLIP_CACHE_DIR)
    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_CRF = args.crf
    CLIP_PRESET = args.preset
    CLIP_SCALE_H = max(0, args.preview_height)
    _CLIP_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, args.prewarm_workers))

    if shutil.which("ffmpeg") is None:
        print("[policy_eval_prompts] WARNING: ffmpeg not on PATH", flush=True)

    print(f"[policy_eval_prompts] root  = {ROOT}", flush=True)
    print(f"[policy_eval_prompts] view  = {FRAME_VIEW}", flush=True)
    print(f"[policy_eval_prompts] cache = {CLIP_CACHE_DIR}", flush=True)
    print(f"[policy_eval_prompts] fps   = {FPS}", flush=True)
    print("[policy_eval_prompts] scanning ...", flush=True)
    eps = _get_episodes()
    with_frames = sum(1 for e in eps if e.has_frames)
    filled = sum(
        1
        for e in eps
        for s in e.segments
        if s.success_criteria
    )
    total_seg = sum(len(e.segments) for e in eps)
    print(
        f"[policy_eval_prompts] found = {len(eps)} episode(s), "
        f"{with_frames} with frames, {filled}/{total_seg} keyframes annotated",
        flush=True,
    )
    if args.prewarm:
        _prewarm(workers=args.prewarm_workers)
        print(
            f"[policy_eval_prompts] prewarm with {args.prewarm_workers} workers",
            flush=True,
        )
    print(
        f"[policy_eval_prompts] serving on http://{args.host}:{args.port}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
