#!/usr/bin/env python3
"""Fresh open-loop measurements; GPU scenarios run serially, PO overlaps each.

No closed-loop substitution: supply the real GT, masks and predictions separately.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_submission import write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--config-path', default='configs/tasks/ctrlworld.yaml')
    ap.add_argument('--po-config-path')
    ap.add_argument('--predictions')
    ap.add_argument('--short-reference', required=True, help='Completed same-four-episode 30s reference run')
    ap.add_argument('--full-reference', help='Completed full100 local run for full single-view regression')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--verified-gate', help='Reuse an already completed four-episode accuracy check, not its performance results')
    ap.add_argument('--with-po', action='store_true')
    ap.add_argument('--confirm-paid-api', action='store_true')
    args = ap.parse_args()
    if args.episodes < 1 or args.workers < 1:
        ap.error('episodes and workers must be positive')
    if args.with_po and not args.confirm_paid_api:
        ap.error('Explicit paid API confirmation required')
    ref = Path(args.short_reference).resolve()
    summary = json.loads((ref/'summary.json').read_text())
    spec = json.loads((ref/'run_spec.json').read_text())
    if summary.get('status') != 'complete' or summary.get('episodes') != 4 or spec.get('seconds') != 30 or spec.get('views') != 'three':
        ap.error('Need a completed four-episode, three-view, 30s reference')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-u', str(ROOT/'tools/evaluate_submission.py'),
              '--config-path', args.config_path, '--threads', '4']
    if args.predictions:
        common += ['--predictions', args.predictions]
    if args.po_config_path:
        common += ['--po-config-path', args.po_config_path]
    state = {'status': 'running', 'with_po': args.with_po, 'completed': [],
             'episodes_per_scenario':args.episodes, 'workers':args.workers,
             'closed_loop': 'NOT RUN: known dataset has 36 episodes and no required masks',
             'started_unix': time.time()}

    def invoke(name, command):
        state['current'] = name
        write(out/'matrix_status.json', state)
        print(json.dumps({'starting': name}), flush=True)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def compare(name, candidate, reference):
        invoke(name, [sys.executable, str(ROOT/'scripts/compare_hour_budget.py'),
                      '--run', str(candidate), '--baseline-run', str(reference)])
        report = json.loads((candidate/'regression.json').read_text())
        if not report['episode_tolerance_0_01_pass']:
            raise RuntimeError('Score regression exceeds 0.01/100; refusing larger run')
        return report['max_episode_score_delta_100']

    try:
        gate = Path(args.verified_gate).resolve() if args.verified_gate else out/'short_accelerated4'
        if args.verified_gate:
            previous = json.loads((gate/'summary.json').read_text())
            if previous.get('status') != 'complete' or previous.get('episodes') != 4 or previous.get('seconds') != 30 or previous.get('views') != 'three':
                raise ValueError('Verified gate must be a complete four-episode 30s three-view run')
            state['reused_accuracy_gate'] = str(gate)
        else:
            invoke('short_accelerated4', common + ['--manifest', str(ref/'manifest.json'),
                   '--seconds', '30', '--views', 'three', '--workers', '4', '--output', str(gate)])
        delta = compare('short_regression', gate, ref)
        measured = json.loads((gate/'summary.json').read_text())
        state['four_episode_comparison'] = {
            'reference_seconds': summary['elapsed_sec'], 'accelerated_seconds': measured['elapsed_sec'],
            'wall_speedup': summary['elapsed_sec']/measured['elapsed_sec'],
            'max_score_delta_100': delta,
            'caveat': 'Reference precedes candidate; filesystem cache warmth may differ; not a 100-episode speedup'}
        cases = [('short_three',30,'three'), ('short_single',30,'single'), ('long_single',0,'single')]
        if args.with_po:
            cases.append(('long_three',0,'three'))
        else:
            state['long_three'] = {'status':'previous_measurement_only', 'path':args.full_reference,
                                  'po_included':False}
        for name, seconds, views in cases:
            destination = out/name
            command = common + ['--limit',str(args.episodes),'--seconds',str(seconds),'--views',views,
                                '--workers',str(args.workers),'--output',str(destination)]
            if args.with_po:
                command += ['--with-po','--confirm-paid-api','--api-workers','32']
            invoke(name, command)
            result = json.loads((destination/'summary.json').read_text())
            if result['episodes'] != args.episodes or result['computed_episodes'] != args.episodes:
                raise RuntimeError(f'Expected {args.episodes} freshly computed episodes')
            if name == 'long_single' and args.full_reference:
                result['regression_delta_100'] = compare('long_single_regression', destination, Path(args.full_reference).resolve())
            usage = destination/'po_usage.json'
            if usage.exists():
                result['po_usage'] = json.loads(usage.read_text())
            state['completed'].append({'scenario':name,'result':result})
            write(out/'matrix_status.json',state)
        state['status'] = 'complete_open_loop' if args.with_po else 'complete_local_only'
        state['current'] = None
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        state['updated_unix'] = time.time()
        write(out/'matrix_status.json',state)


if __name__ == '__main__':
    main()
