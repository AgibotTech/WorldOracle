"""Isolated throughput experiments: no production metric edits or result reuse."""
import argparse
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def select(n):
    gt = ROOT/'data/gt_dataset'
    pred = ROOT/'data/eval_workspace/ctrlworld'
    episodes = []
    for task in sorted(p for p in gt.iterdir() if p.is_dir()):
        eps = [p for p in sorted(task.glob('ep*')) if (pred/task.name/p.name/'video').exists()]
        episodes.extend((task.name,p.name) for p in eps[:n])
    return episodes


def install_lazy_io(decode_workers=1):
    """Preserve original frame coordinates, stride, arrays, and inference batches.

    Decode images only when production code slices the sequence. Cache exact
    stride-selected GT masks across metrics; no resizing/quantization here.
    Full NPZ decompression is timed, once per unique GT mask path per worker.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from acwmbench.io_utils import list_frame_files, load_frames_as_array
    kro = importlib.import_module('acwmbench.metrics.keyarea_visual_quality')
    ia = importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
    mvc = importlib.import_module('acwmbench.metrics.multiview_consistency')
    def decode(path):
        with Image.open(path) as img:
            return np.asarray(img.convert('RGB'))
    class LazyFrames:
        def __init__(self, path, preferred_view=None):
            self.files = list_frame_files(path, preferred_view=preferred_view)
        def __len__(self):
            return len(self.files)
        def __getitem__(self, index):
            if not isinstance(index,slice):
                raise TypeError('Only production time slicing is supported')
            paths=self.files[index]
            if decode_workers<=1:
                return np.stack([decode(p) for p in paths])
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                return np.stack(list(pool.map(decode,paths)))
    def lazy_frames(path, indices=None, preferred_view=None):
        if indices is not None:
            return load_frames_as_array(path, indices=indices, preferred_view=preferred_view)
        return LazyFrames(path, preferred_view)
    class SampledMasks:
        def __init__(self, array):
            self.shape = array.shape
            self.ndim = array.ndim
            assert self.ndim == 4
            self.data = array[:,::10].copy()
        def __getitem__(self, index):
            if not isinstance(index,tuple) or len(index)!=2:
                raise TypeError(f'Unexpected production mask index {index!r}')
            mask, frame = index
            if isinstance(frame,slice):
                start,stop,step = frame.indices(self.shape[1])
                assert start%10==0 and step%10==0
                return self.data[mask, start//10:(stop+9)//10:step//10]
            assert frame%10==0
            return self.data[mask,frame//10]
    cache = {}
    lock = threading.Lock()
    original_masks = ia._load_masks
    def cached_masks(path):
        key = str(Path(path).resolve())
        with lock:
            if key not in cache:
                cache[key] = SampledMasks(original_masks(Path(path)))
            return cache[key]
    kro.load_frames_as_array = lazy_frames
    kro._load_npz_masks = cached_masks
    ia._load_masks = cached_masks
    mvc._load_masks = cached_masks
    return cache


def child(args):
    import torch
    from evaluate import load_merged_config
    from acwmbench.runner import ACWMBenchmark
    from scripts.measure_submission_cost import Monitor
    import acwmbench.runner as runner
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(args.threads)
    cfg = load_merged_config(str(ROOT/'configs/base.yaml'),None)
    if args.fast_index:
        ia=importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
        mvc=importlib.import_module('acwmbench.metrics.multiview_consistency')
        def discover(root,view):
            root=Path(root)
            for directory in (root/view,root,root/'video'/view,root/'video'):
                if not directory.is_dir():continue
                with os.scandir(directory) as entries:
                    files=sorted(Path(e.path) for e in entries if Path(e.name).suffix.lower() in ia.IMAGE_EXTENSIONS and e.is_file())
                if not files:continue
                parsed={}
                for ordinal,path in enumerate(files):
                    index=ia._frame_number(path)
                    index=ordinal if index is None else index
                    if index in parsed:raise ValueError(f'Duplicate frame index {index}: {path}')
                    parsed[index]=path
                return parsed
            raise FileNotFoundError(f'No image frames found under {root}')
        ia._discover_frames=mvc._discover_frames=discover
    if args.reuse_dino:
        ia=importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
        mvc=importlib.import_module('acwmbench.metrics.multiview_consistency')
        original_dino=ia.DinoExtractor
        resident={}
        def shared_dino(**kwargs):
            key=tuple(sorted((k,str(v)) for k,v in kwargs.items()))
            if key not in resident:
                resident[key]=original_dino(**kwargs)
            return resident[key]
        ia.DinoExtractor=mvc.DinoExtractor=shared_dino
    cache = install_lazy_io(args.decode_workers) if args.lazy else None
    cfg['dimensions']['keyarea_visual_quality']['execution'].update(
        episode_workers=args.prepare_workers, max_in_flight_episodes=1, prefetch_depth=1)
    for dim in ['interaction_spatial_alignment','multiview_consistency']:
        for field in ['loader_workers','prefetch_batches','matcher_workers']:
            value=getattr(args,field)
            if value is not None:
                cfg['dimensions'][dim]['execution'][field]=value
    episodes = json.loads(Path(args.manifest).read_text())[args.rank::args.workers]
    out = Path(args.output)/f'worker{args.rank}'
    out.mkdir(parents=True,exist_ok=False)
    val = out/'predictions'
    for task,ep in episodes:
        dst = val/task/ep
        dst.mkdir(parents=True)
        (dst/'video').symlink_to((ROOT/'data/eval_workspace/ctrlworld'/task/ep/'video').resolve(),target_is_directory=True)
    dims = ['keyarea_visual_quality','interaction_spatial_alignment','multiview_consistency']
    report = {}
    for dim in dims:
        original = getattr(runner,'compute_'+dim)
        def measured(*a,_fn=original,_dim=dim,**kw):
            torch.cuda.synchronize()
            with Monitor() as mon:
                result = _fn(*a,**kw)
                torch.cuda.synchronize()
            r=mon.report()
            r['num_episodes']=len(result.get('per_episode',[]))
            write(out/(_dim+'_resources.json'),r)
            report[_dim]={k:v for k,v in r.items() if k!='samples'}
            write(out/'summary.json',report)
            if r['num_episodes']!=len(episodes):
                raise RuntimeError(f'{_dim}: incomplete coverage')
            return result
        setattr(runner,'compute_'+dim,measured)
    started=time.perf_counter()
    ACWMBenchmark('cuda',out/'aggregate').evaluate(
        data_base=str(val),data_name='hour_budget',dimension_list=dims,
        gt_path=str(ROOT/'data/gt_dataset'),dim_configs=cfg['dimensions'],
        hardware_config=cfg.get('hardware',{}))
    write(out/'complete.json',{'seconds':time.perf_counter()-started,'episodes':episodes,
                             'mask_cache_paths':len(cache) if cache is not None else 0})


def parent(args):
    import psutil
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    episodes=select(args.per_task)
    if args.limit is not None:
        # Round-robin across tasks: ten episodes are 3/3/2/2, selected
        # before timing and without inspecting per-episode runtimes.
        tasks=sorted({t for t,e in episodes})
        groups={t:[(tt,e) for tt,e in episodes if tt==t] for t in tasks}
        episodes=[groups[t][i] for i in range(args.per_task) for t in tasks if i<len(groups[t])][:args.limit]
        assert len(episodes)==args.limit
    assert len(episodes)>=args.workers
    write(out/'manifest.json',episodes)
    write(out/'config.json',vars(args))
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
             OPENBLAS_NUM_THREADS='1',PYTHONWARNINGS='ignore')
    children=[];logs=[];samples=[]
    started=time.perf_counter()
    try:
        for rank in range(args.workers):
            log=(out/f'worker{rank}.log').open('w');logs.append(log)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--output',str(out),
                 '--manifest',str(out/'manifest.json'),'--rank',str(rank),'--workers',str(args.workers),
                 '--threads',str(args.threads)]
            if args.lazy:cmd.append('--lazy')
            if args.reuse_dino:cmd.append('--reuse-dino')
            if args.fast_index:cmd.append('--fast-index')
            for field in ['decode_workers','prepare_workers','loader_workers','prefetch_batches','matcher_workers']:
                value=getattr(args,field)
                if value is not None:cmd.extend(['--'+field.replace('_','-'),str(value)])
            children.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            row={'t':time.perf_counter()-started,'rss_gib':0.0,'cpu_seconds':0.0}
            for p in children:
                if p.poll() is None:
                    try:
                        proc=psutil.Process(p.pid)
                        row['rss_gib']+=proc.memory_info().rss/2**30
                        row['cpu_seconds']+=sum(proc.cpu_times()[:2])
                    except psutil.NoSuchProcess:pass
            values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                          '--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
            row.update(gpu_percent=float(values[0]),gpu_memory_gib=float(values[1])/1024)
            samples.append(row)
            write(out/'live.json',row)
            if row['rss_gib']>88 or row['gpu_memory_gib']>22.5:
                raise RuntimeError('Safety memory budget exceeded')
            if any(p.poll() not in (None,0) for p in children):
                raise RuntimeError('A worker failed; see worker logs')
            time.sleep(2)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Worker failed')
        elapsed=time.perf_counter()-started
        result={'elapsed_sec':elapsed,'episode_count':len(episodes),'workers':args.workers,
                'lazy':args.lazy,'estimated_100_sec':elapsed*100/len(episodes),
                'gpu_mean_percent':sum(r['gpu_percent'] for r in samples)/len(samples),
                'peak_gpu_gib':max(r['gpu_memory_gib'] for r in samples),
                'peak_rss_gib':max(r['rss_gib'] for r in samples)}
        cpu=sum(sum(v['cpu_seconds'] for v in json.loads((out/f'worker{i}/summary.json').read_text()).values()) for i in range(args.workers))
        result['metric_cpu_seconds']=cpu
        result['metric_cpu_average_cores']=cpu/elapsed
        write(out/'resources.json',samples);write(out/'summary.json',result)
        print(json.dumps(result),flush=True)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--per-task',type=int,default=1)
    ap.add_argument('--limit',type=int,help='Task-balanced exact episode count')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--lazy',action='store_true')
    ap.add_argument('--reuse-dino',action='store_true')
    ap.add_argument('--fast-index',action='store_true')
    ap.add_argument('--decode-workers',type=int,default=1)
    ap.add_argument('--prepare-workers',type=int,default=1)
    ap.add_argument('--loader-workers',type=int)
    ap.add_argument('--prefetch-batches',type=int)
    ap.add_argument('--matcher-workers',type=int)
    ap.add_argument('--child',action='store_true')
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--manifest')
    args=ap.parse_args()
    (child if args.child else parent)(args)


if __name__=='__main__':main()
