"""Process-local checkpoint scheduling; original judge and aggregation unchanged."""
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import threading
import time


def run_checkpoints(trials, config, workers, out):
    from acwmbench.models.vlm import build_vlm_backend
    from scripts.benchmark_physical_judge_concurrency import _judge_kwargs, _normalized_vlm_config, _merge_usage
    from tools.evaluate_submission import write
    judge = importlib.import_module('acwmbench.metrics.physical_judge')
    original = judge._judge_step
    kwargs = _judge_kwargs(config)
    vc = _normalized_vlm_config(config, model=None, base_url=None)
    jobs, counts = [], []
    started = time.perf_counter()

    def capture(_backend, *args, **kw):
        jobs.append((args, kw))
        return {'score': 0.0, 'format_valid': True}

    # Prepare exactly the calls that the original serial judge would issue.
    # No external requests occur during preparation or replay.
    try:
        judge._judge_step = capture
        for trial in trials:
            before = len(jobs)
            judge.judge_one_trial(None, trial, **kwargs)
            counts.append(len(jobs)-before)
    finally:
        judge._judge_step = original
    if not jobs or any(n == 0 for n in counts):
        raise ValueError('Every episode must have at least one PO checkpoint')
    prepared = time.perf_counter()-started
    clients, lock, local = [], threading.Lock(), threading.local()
    values = [None] * len(jobs)

    def initialize():
        local.backend = build_vlm_backend(dict(vc))
        with lock:
            clients.append(local.backend)

    def evaluate(index):
        args, kw = jobs[index]
        t = time.perf_counter()
        result = original(local.backend, *args, **kw)
        return index, result, time.perf_counter()-t

    try:
        with ThreadPoolExecutor(max_workers=min(workers,len(jobs)), initializer=initialize) as pool:
            futures = [pool.submit(evaluate, i) for i in range(len(jobs))]
            for completed, future in enumerate(as_completed(futures), 1):
                i, value, duration = future.result()
                values[i] = value
                write(out/'po_checkpoints'/f'{i:06d}.json', {'index':i,'seconds':duration,'result':value})
                write(out/'po_progress.json', {'completed':completed,'total':len(jobs),
                      'elapsed_sec':time.perf_counter()-started,'prepare_sec':prepared,
                      'concurrency':min(workers,len(jobs))})
        if any(not v or not v.get('format_valid') for v in values):
            raise ValueError('Incomplete or invalid PO checkpoints')
        cursor = 0
        for trial, count in zip(trials, counts):
            selected = values[cursor:cursor+count]
            responses = iter(selected)

            def replay(_backend, *args, **kw):
                return copy.deepcopy(next(responses))

            try:
                judge._judge_step = replay
                result = judge.judge_one_trial(None, trial, **kwargs)
                if len(result['key_steps']) != count or next(responses, None) is not None:
                    raise ValueError('PO replay coverage mismatch')
            finally:
                judge._judge_step = original
            result.update(metric_version=1, status='complete', uid=trial.uid,
                          task_id=trial.task_id, episode_id=trial.episode_id, trial_id=trial.trial_id,
                          evaluation_settings={k:kwargs[k] for k in ['n_fallback_keysteps',
                              'max_image_long_side','eval_view','format_retries','use_reference_frame']})
            write(out/'predictions'/trial.task_id/trial.episode_id/'physical_judge.json', {trial.trial_id:result})
            cursor += count
        write(out/'po_timing.json', {'elapsed_sec':time.perf_counter()-started,
              'prepare_sec':prepared,'checkpoints':len(jobs),'concurrency':min(workers,len(jobs)),
              'aggregation':'original judge_one_trial replay; original _judge_step outputs'})
    finally:
        write(out/'po_usage.json', _merge_usage([c.get_usage_summary() for c in clients]))
        for c in clients:
            c.close()
