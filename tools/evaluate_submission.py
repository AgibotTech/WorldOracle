#!/usr/bin/env python3
"""Restartable dynamic submission executor; delegates scoring to the original runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import fcntl
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL = ['keyarea_visual_quality', 'interaction_spatial_alignment', 'multiview_consistency']


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_trials(trials, manifest=None, limit=None):
    if manifest:
        pairs = [tuple(x) for x in json.loads(Path(manifest).read_text())]
        if len(set(pairs)) != len(pairs):
            raise ValueError('Duplicate manifest entries')
        lookup = {(t.task_id, t.episode_id): t for t in trials}
        if len(lookup) != len(trials):
            raise ValueError('Only one trial per episode is supported')
        return [lookup[p] for p in pairs]
    groups = {}
    for trial in sorted(trials, key=lambda t: (t.task_id, t.episode_id)):
        groups.setdefault(trial.task_id, []).append(trial)
    ordered = [v[i] for i in range(max(map(len, groups.values()), default=0))
               for v in groups.values() if i < len(v)]
    return ordered[:limit] if limit else ordered


def window_trials(trials, frame_limit):
    if frame_limit is None:
        return trials
    # The evaluated video is a prefix with exactly ``frame_limit`` source frames.
    # Keep segment frame indices in that same coordinate system so the physical
    # judge does not rescale them as though the full long-horizon clip were used.
    # Copy each segment to avoid mutating dataset metadata shared by other runs.
    return [replace(t, prompt_segments=[
        {**segment, 'source_n_frames': frame_limit}
        for segment in t.prompt_segments or []
        if 0 <= segment['end_frame'] < frame_limit
    ]) for t in trials]


def input_identity(trial, views, stat_pool=None, frame_limit=None):
    """Metadata identity: inputs must remain immutable; not a content-hash guarantee."""
    from acwmbench.execution_acceleration import discover_frames
    identity = []
    for view in views:
        sets = []
        for root in (trial.gt_video_dir, trial.video_dir):
            frames = discover_frames(root, view)
            if frame_limit is not None:
                frames = {i:p for i,p in frames.items() if 0 <= i < frame_limit}
            sets.append(list(frames))
            def metadata(p):
                s=p.stat()
                return s.st_size,s.st_mtime_ns
            stats=list(stat_pool.map(metadata,frames.values())) if stat_pool else [metadata(p) for p in frames.values()]
            identity.append([(i,str(p),s) for (i,p),s in zip(frames.items(),stats)])
        if sets[0] != sets[1] or not sets[0]:
            raise ValueError(f'{trial.uid}/{view}: GT/prediction frame indices differ')
        expected=json.loads((Path(trial.gt_video_dir).parent/'prompt/prompts.json').read_text())['n_frames']
        if frame_limit is not None:
            expected = min(int(expected), frame_limit)
        if sets[0] != list(range(int(expected))):
            raise ValueError(f'{trial.uid}/{view}: missing frames relative to GT n_frames')
    gt = Path(trial.gt_video_dir).parent
    required = [gt/'prompt/masks.npz', gt/'prompt/prompts.json']
    required += [gt/'multiview_masks'/v/'masks.npz' for v in views]
    for p in required:
        s = p.stat()
        identity.append((str(p.resolve()), s.st_size, s.st_mtime_ns))
    return digest(identity)


def validate_result(path, dims, trial=None):
    hashes = {}
    for dim in dims:
        f = path/(dim+'.json')
        r = json.loads(f.read_text())
        if dim == 'physical_judge':
            values = list(r.values())
            if trial is not None and set(r) != {trial.trial_id}:
                raise ValueError('PO trial coverage mismatch')
            if not values or any(v.get('status') != 'complete' or not v.get('key_steps')
                or any(not s.get('format_valid') for s in v['key_steps']) for v in values):
                raise ValueError(f'Incomplete/invalid PO result: {f}')
        else:
            frames = [x['frame'] for x in r.get('per_frame', [])]
            if not frames or len(frames) != len(set(frames)):
                raise ValueError(f'Incomplete/duplicate frames: {f}')
            if trial is not None and r.get('uid') != trial.uid:
                raise ValueError(f'Unexpected UID: {f}')
        hashes[dim] = file_hash(f)
    return hashes


def valid_receipt(out, trial, identity, dims):
    p = out/'receipts'/trial.task_id/(trial.episode_id+'.json')
    try:
        receipt = json.loads(p.read_text())
        return receipt['identity'] == identity and receipt['hashes'] == validate_result(
            out/'predictions'/trial.task_id/trial.episode_id, dims, trial)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def valid_po_result(out, trial):
    try:
        validate_result(
            out/'predictions'/trial.task_id/trial.episode_id,
            ['physical_judge'],
            trial,
        )
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def worker(rank, work, events, cfg, out, threads, accelerated, local_dims, frame_limit, cpu_heavy=False):
    log = open(out/f'worker{rank}.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        import torch
        import acwmbench.runner as runner
        from acwmbench.execution_acceleration import install, install_window
        torch.set_num_threads(threads)
        if accelerated:
            clear = install(cfg['dimensions'], frame_limit=frame_limit, decode_workers=16 if cpu_heavy else 8)
        else:
            install_window(frame_limit)
            clear = lambda: None
        dims = cfg['dimensions']
        # Preserve one-episode inference groups and the resolved original batch sizes.
        dims['keyarea_visual_quality'].setdefault('execution', {}).update(
            episode_workers=1, max_in_flight_episodes=1, prefetch_depth=1)
        for d in LOCAL[1:]:
            dims[d].setdefault('execution', {}).update(loader_workers=32 if cpu_heavy else 16,
                    prefetch_batches=16, matcher_workers=16 if cpu_heavy else 8)
        engine = runner.ACWMBenchmark('cuda', out/f'worker{rank}'/'aggregate')
        while True:
            job = work.get()
            if job is None:
                return
            trial, identity = job
            started = time.perf_counter()
            events.put({'event':'started', 'rank':rank, 'uid':trial.uid})
            try:
                # Discovery was already done once; no rescanning the full submission.
                runner.build_trial_list = lambda *a, **k: [trial]
                result = engine.evaluate(data_base=str(out/'predictions'), data_name='submission',
                    dimension_list=local_dims, gt_path=cfg['data']['gt_path'],
                    dim_configs=dims, hardware_config=cfg.get('hardware', {}), overwrite=True)
                for d in local_dims:
                    if len(result[d].get('per_episode', [])) != 1:
                        raise ValueError(f'{d}: episode skipped')
                path = out/'predictions'/trial.task_id/trial.episode_id
                hashes = validate_result(path, local_dims, trial)
                write(out/'receipts'/trial.task_id/(trial.episode_id+'.json'),
                      {'identity':identity, 'hashes':hashes, 'seconds':time.perf_counter()-started})
                events.put({'event':'complete', 'rank':rank, 'uid':trial.uid,
                            'seconds':time.perf_counter()-started})
            except Exception:
                events.put({'event':'error', 'rank':rank, 'uid':trial.uid, 'error':traceback.format_exc()})
                raise
            finally:
                clear()
    except BaseException:
        traceback.print_exc()
        raise


def po_worker(cfg, trials, out, workers, events, schedule='episode'):
    """Use original episode-level PO concurrency; no new prompt or aggregation logic."""
    log = open(out/'po.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    if schedule == 'checkpoint':
        from acwmbench.execution_po import run_checkpoints
        run_checkpoints(trials, cfg['dimensions']['physical_judge'], workers, out)
        for trial in trials:
            validate_result(out/'predictions'/trial.task_id/trial.episode_id, ['physical_judge'], trial)
        events.put({'event':'po_complete'})
        return
    import acwmbench.runner as runner
    runner.build_trial_list = lambda *a, **k: trials
    cfg['dimensions']['physical_judge'].setdefault('execution', {})['api_workers'] = workers
    result = runner.ACWMBenchmark('cpu', out/'po_aggregate').evaluate(
        data_base=str(out/'predictions'), data_name='submission', dimension_list=['physical_judge'],
        gt_path=cfg['data']['gt_path'], dim_configs=cfg['dimensions'], overwrite=False)
    for trial in trials:
        validate_result(out/'predictions'/trial.task_id/trial.episode_id, ['physical_judge'], trial)
    write(out/'po_usage.json', result.get('physical_judge_usage', {}))
    events.put({'event':'po_complete'})


def run(args):
    import subprocess
    import psutil
    from evaluate import load_merged_config
    from acwmbench.dataset import build_trial_list
    if args.with_po and not args.confirm_paid_api:
        raise ValueError('--with-po requires explicit --confirm-paid-api')
    out = Path(args.output).resolve()
    if out.exists() and not args.resume:
        raise ValueError('Output exists; use --resume or a new output')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        # Keep the fresh-run measurement; never replace it with a fast no-op resume.
        archive=out/'attempts'/str(time.time_ns())
        for name in ['summary.json','resources.json','events.json','live.json','failure.json']:
            source=out/name
            if source.exists():
                archive.mkdir(parents=True,exist_ok=True)
                source.replace(archive/name)
    started = time.perf_counter()
    frame_limit = args.seconds * args.fps if args.seconds else None
    local_dims = LOCAL if args.views == 'three' else LOCAL[:2]
    views = ['top_head','hand_left','hand_right'] if args.views == 'three' else ['top_head']
    cfg = load_merged_config(str(Path(args.base_config).resolve()), args.config_path)
    if args.po_config_path:
        po_cfg = load_merged_config(str(Path(args.base_config).resolve()), args.po_config_path)
        cfg['dimensions']['physical_judge'] = po_cfg['dimensions']['physical_judge']
    if args.predictions: cfg['data']['val_base'] = str(Path(args.predictions).resolve())
    if args.gt: cfg['data']['gt_path'] = str(Path(args.gt).resolve())
    cfg['data']['val_base'] = str(Path(cfg['data']['val_base']).resolve())
    cfg['data']['gt_path'] = str(Path(cfg['data']['gt_path']).resolve())
    if args.with_po and cfg['dimensions']['physical_judge'].get('vlm', {}).get('backend') not in {'openai','vllm','anthropic','gemini'}:
        raise ValueError('Concurrent PO requires a remote backend')
    trials = selected_trials(build_trial_list(cfg['data']['val_base'], cfg['data']['gt_path']), args.manifest, args.limit)
    trials = window_trials(trials, frame_limit)
    if args.with_po and any(not t.prompt_segments for t in trials):
        raise ValueError('PO window contains an episode with no eligible key steps')
    if not trials or len({(t.task_id,t.episode_id) for t in trials}) != len(trials):
        raise ValueError('Need nonempty submission with one trial per episode')
    if any(t.trial_id != 'default' for t in trials):
        raise ValueError('This executor currently supports the canonical default-trial layout only')
    if cfg['dimensions']['interaction_spatial_alignment'].get('eval_view','top_head') != 'top_head' or cfg['dimensions']['keyarea_visual_quality'].get('eval_view','top_head') != 'top_head' or cfg['dimensions']['keyarea_visual_quality'].get('secondary_mask_view','top_head')!='top_head':
        raise ValueError('This executor currently validates the top_head + hand_left + hand_right platform')
    sources=sorted((ROOT/'acwmbench').rglob('*.py'))+[Path(__file__),ROOT/'evaluate.py']
    source_hash = digest({str(p.relative_to(ROOT)):file_hash(p) for p in sources})
    weight_identity=[]
    for weight in [cfg['dimensions']['keyarea_visual_quality']['vjepa']['model_path'],
                   cfg['dimensions']['interaction_spatial_alignment']['checkpoint']]:
        p=Path(weight); s=p.stat(); weight_identity.append((str(p.resolve()),s.st_size,s.st_mtime_ns))
    config_hash = digest(cfg)  # Never persist configuration credentials.
    fingerprint = digest({'config':config_hash,'source':source_hash,'threads':args.threads,
                          'cpu_heavy':args.cpu_heavy,'po_schedule':args.po_schedule,
                          'views':args.views,'frame_limit':frame_limit,'fps':args.fps,
                          'accelerated':not args.reference,'weights':weight_identity,
                          'manifest':[(t.task_id,t.episode_id) for t in trials]})
    spec_path = out/'run_spec.json'
    persisted_source_hash = source_hash
    source_compatible_resume = False
    if args.resume:
        if not spec_path.exists():
            raise ValueError('Resume requested but run_spec.json is missing')
        previous_spec = json.loads(spec_path.read_text())
        if previous_spec.get('fingerprint') != fingerprint:
            previously_approved_source = (
                previous_spec.get('source_compatible_resume')
                and previous_spec.get('current_source_hash') == source_hash
            )
            if not (args.allow_source_change_on_resume or previously_approved_source):
                raise ValueError(
                    'Resume source differs; review the code change and retry with '
                    '--allow-source-change-on-resume'
                )
            selected_manifest = [(t.task_id,t.episode_id) for t in trials]
            existing_manifest_path = out/'manifest.json'
            existing_manifest = (
                [tuple(item) for item in json.loads(existing_manifest_path.read_text())]
                if existing_manifest_path.is_file()
                else None
            )
            compatible = (
                previous_spec.get('config_hash') == config_hash
                and previous_spec.get('threads') == args.threads
                and previous_spec.get('cpu_heavy') == args.cpu_heavy
                and previous_spec.get('po_schedule') == args.po_schedule
                and previous_spec.get('views') == args.views
                and previous_spec.get('seconds') == args.seconds
                and previous_spec.get('fps') == args.fps
                and previous_spec.get('frame_limit') == frame_limit
                and previous_spec.get('dimensions') == local_dims
                and previous_spec.get('po_model') == cfg['dimensions']['physical_judge'].get('vlm',{}).get('served_model_name')
                and existing_manifest == selected_manifest
            )
            if not compatible:
                raise ValueError('Resume configuration/input manifest differs')
            # A resume-support patch changes this script's source hash. Preserve
            # the original fingerprint so completed local-metric receipts remain
            # valid; input_identity below still verifies every frame and mask.
            fingerprint = previous_spec['fingerprint']
            persisted_source_hash = previous_spec.get('source_hash', source_hash)
            source_compatible_resume = True
            print(json.dumps({
                'event':'resume_source_compatibility',
                'previous_source_hash':persisted_source_hash,
                'current_source_hash':source_hash,
            }), flush=True)
    write(spec_path, {'fingerprint':fingerprint,'source_hash':persisted_source_hash,
                     'current_source_hash':source_hash,'source_compatible_resume':source_compatible_resume,
                     'config_hash':config_hash,
                     'input_identity':'paths, frame indices, sizes, mtimes; immutable input required',
                     'workers':args.workers,'threads':args.threads,'with_po':args.with_po,
                     'cpu_heavy':args.cpu_heavy,'po_schedule':args.po_schedule,'api_workers':args.api_workers,
                     'views':args.views,'seconds':args.seconds,'fps':args.fps,
                     'frame_limit':frame_limit,'dimensions':local_dims,
                     'po_model':cfg['dimensions']['physical_judge'].get('vlm',{}).get('served_model_name'),
                     'po_keypoints':sum(len(t.prompt_segments or []) for t in trials)})
    write(out/'manifest.json', [(t.task_id,t.episode_id) for t in trials])
    identities, copies, pending = {}, [], []
    with ThreadPoolExecutor(max_workers=args.preflight_workers) as stat_pool, ThreadPoolExecutor(max_workers=4) as episode_pool:
        def identify(trial):
            return digest((fingerprint,input_identity(trial,views,stat_pool,frame_limit)))
        prepared_identities=[]
        for i,identity in enumerate(episode_pool.map(identify,trials)):
            prepared_identities.append(identity)
            print(json.dumps({'event':'input_checked','episodes':i+1,'total':len(trials),'elapsed_sec':time.perf_counter()-started}),flush=True)
    for trial,identity in zip(trials,prepared_identities):
        identities[trial.uid] = identity
        dst = out/'predictions'/trial.task_id/trial.episode_id
        dst.mkdir(parents=True, exist_ok=True)
        video = dst/'video'
        if video.is_symlink():
            if video.resolve() != Path(trial.video_dir).resolve(): raise ValueError('Input symlink differs')
        elif video.exists(): raise ValueError('Prediction output contains unexpected video directory')
        else: video.symlink_to(Path(trial.video_dir).resolve(), target_is_directory=True)
        copied = replace(trial, video_dir=str(video))
        copies.append(copied)
        if not (args.resume and valid_receipt(out,copied,identity,local_dims)):
            pending.append((copied,identity))
    po_pending = (
        [trial for trial in copies if not valid_po_result(out, trial)]
        if args.with_po
        else []
    )
    po_reused = len(copies) - len(po_pending) if args.with_po else 0
    write(out/'input_identities.json', identities)
    preflight = time.perf_counter()-started
    print(json.dumps({'event':'preflight_complete','episodes':len(trials),'pending':len(pending),
                      'seconds':preflight,'po_enabled':args.with_po,
                      'po_pending':len(po_pending),'po_reused':po_reused}), flush=True)
    # Largest videos first; workers claim the next episode dynamically.
    pending.sort(key=lambda x: max((s.get('source_n_frames') or 0 for s in x[0].prompt_segments or []),default=0), reverse=True)
    env = {'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'OPENBLAS_NUM_THREADS':'1'}
    os.environ.update(env)
    ctx = mp.get_context('spawn'); work = ctx.Queue(); events = ctx.Queue()
    for job in pending: work.put(job)
    n = min(args.workers,len(pending))
    for _ in range(n): work.put(None)
    processes = [ctx.Process(target=worker,args=(i,work,events,cfg,out,args.threads,not args.reference,local_dims,frame_limit,args.cpu_heavy)) for i in range(n)]
    if args.with_po and po_pending:
        processes.append(ctx.Process(target=po_worker,args=(cfg,po_pending,out,args.api_workers,events,args.po_schedule)))
    samples, updates, cpu_last = [], [], {}
    execution_started = time.perf_counter()
    try:
        for proc in processes: proc.start()
        while any(p.is_alive() for p in processes):
            while True:
                try:
                    event = events.get_nowait(); event['elapsed_sec'] = time.perf_counter()-started
                    updates.append(event); print(json.dumps(event),flush=True)
                except queue.Empty: break
            if any(p.exitcode not in (None,0) for p in processes):
                raise RuntimeError('Worker failed; inspect worker logs')
            rss = 0
            for p in processes:
                try:
                    ps = psutil.Process(p.pid); rss += ps.memory_info().rss
                    cpu_last[p.pid] = sum(ps.cpu_times()[:2])
                except psutil.NoSuchProcess: pass
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                       '--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
            if len(gpu)!=1: raise ValueError('Expose one GPU with CUDA_VISIBLE_DEVICES for this executor')
            util,mem = map(float,gpu[0].split(','))
            row = {'elapsed_sec':time.perf_counter()-started,'execution_sec':time.perf_counter()-execution_started,
                   'cpu_seconds_sampled':sum(cpu_last.values()),'rss_gib':rss/2**30,'gpu_percent':util,'gpu_gib':mem/1024}
            samples.append(row); write(out/'live.json',row)
            if row['rss_gib']>args.ram_limit_gib or row['gpu_gib']>args.vram_limit_gib:
                raise RuntimeError('Safety memory limit exceeded; results remain resumable')
            time.sleep(2)
        for p in processes:
            p.join()
            if p.exitcode: raise RuntimeError('Worker failed; inspect logs')
        while True:
            try: updates.append(events.get_nowait())
            except queue.Empty: break
        for trial in copies:
            if not valid_receipt(out,trial,identities[trial.uid],local_dims):
                raise ValueError(f'Incomplete final coverage: {trial.uid}')
        if args.with_po:
            for trial in copies: validate_result(out/'predictions'/trial.task_id/trial.episode_id,['physical_judge'],trial)
        elapsed = time.perf_counter()-started
        # Reuse the official normalization and per-episode scoring function.
        from tools.summarize_acwmbench_scores import episode_score, load_normalization
        norm = load_normalization(ROOT/'configs/analysis/keyarea_results.yaml')
        scores = []
        for trial in copies:
            path = out/'predictions'/trial.task_id/trial.episode_id
            score_dims=local_dims+(['physical_judge'] if args.with_po else [])
            scores.append({'uid':trial.uid, **{d:episode_score(d,json.loads((path/(d+'.json')).read_text()),norm) for d in score_dims}})
        if any(row[d] is None for row in scores for d in score_dims):
            raise ValueError('One or more official episode scores are invalid')
        write(out/'scores.json',scores)
        summary = {'status':'complete','episodes':len(copies),'computed_episodes':len(pending),
            'resumed_episodes':len(copies)-len(pending),'dimensions':score_dims,
            'po_computed_episodes':len(po_pending),'po_resumed_episodes':po_reused,
            'views':args.views,'seconds':args.seconds,'fps':args.fps,
            'elapsed_sec':time.perf_counter()-started,'preflight_sec':preflight,
            'evaluation_and_validation_sec':elapsed-preflight,'deadline_sec':args.deadline_seconds,
            'deadline_met':time.perf_counter()-started<=args.deadline_seconds,
            'full_four_metric_run':args.with_po and args.views=='three','fresh_run':len(pending)==len(copies),
            'resource_sampling_scope':'worker execution; excludes preflight and final scoring',
            'gpu_mean_percent':sum(s['gpu_percent'] for s in samples)/max(1,len(samples)),
            'peak_gpu_gib':max((s['gpu_gib'] for s in samples),default=0),
            'peak_rss_gib':max((s['rss_gib'] for s in samples),default=0),
            'sampled_cpu_average_cores':sum(cpu_last.values())/max(.001,elapsed-preflight)}
        write(out/'summary.json',summary); print(json.dumps(summary),flush=True)
    except BaseException:
        write(out/'failure.json',{'elapsed_sec':time.perf_counter()-started,'error':traceback.format_exc()})
        raise
    finally:
        for p in processes:
            if p.pid and p.is_alive(): p.terminate()
        for p in processes:
            if p.pid:
                p.join(timeout=15)
                if p.is_alive(): p.kill();p.join()
        write(out/'resources.json',samples);write(out/'events.json',updates)
        # Aborted consumers leave queued jobs unread; do not hang at interpreter
        # shutdown waiting for the multiprocessing feeder to flush that pipe.
        for channel in (work, events):
            channel.cancel_join_thread()
            channel.close()
        lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-config',default=str(ROOT/'configs/base.yaml'))
    ap.add_argument('--config-path')
    ap.add_argument('--po-config-path',help='Import ONLY physical_judge from this merged config; leaves all local metrics unchanged')
    ap.add_argument('--predictions');ap.add_argument('--gt')
    ap.add_argument('--output',required=True);ap.add_argument('--manifest');ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--api-workers',type=int,default=32)
    ap.add_argument('--po-schedule',choices=['episode','checkpoint'],default='episode')
    ap.add_argument('--cpu-heavy',action='store_true',help='Increase CPU decode/load/matching concurrency only')
    ap.add_argument('--preflight-workers',type=int,default=32)
    ap.add_argument('--views',choices=['single','three'],default='three')
    ap.add_argument('--seconds',type=int,choices=[0,30],default=0,help='0: full video; 30: first 30 seconds')
    ap.add_argument('--fps',type=int,default=30,help='Declared source frame rate; must match input videos')
    ap.add_argument('--with-po',action='store_true');ap.add_argument('--confirm-paid-api',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--reference',action='store_true')
    ap.add_argument('--allow-source-change-on-resume',action='store_true',
                    help='Reuse validated receipts after an explicitly reviewed code-only change')
    ap.add_argument('--deadline-seconds',type=float,default=3600)
    ap.add_argument('--ram-limit-gib',type=float,default=88);ap.add_argument('--vram-limit-gib',type=float,default=22.5)
    args=ap.parse_args()
    if min(args.workers,args.threads,args.api_workers,args.preflight_workers,args.fps,args.deadline_seconds,args.ram_limit_gib,args.vram_limit_gib)<=0 or (args.limit is not None and args.limit<=0):
        ap.error('Counts, limits and deadline must be positive')
    if args.limit and args.manifest: ap.error('Use --manifest or --limit, not both')
    run(args)


if __name__=='__main__': main()
