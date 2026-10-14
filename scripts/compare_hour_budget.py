"""Numerical/coverage regression against unmodified scoring outputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_acwmbench_scores import episode_score,load_normalization
from scripts.experiment_hour_budget import write


def check_nested(a,b,path=''):
    """Verify nested frame/mask/view coverage, not just episode averages."""
    maxima={}
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),f'Keys differ: {path}'
        for key in a:
            for field,value in check_nested(a[key],b[key],path+'/'+key).items():
                maxima[field]=max(maxima.get(field,0),value)
    elif isinstance(a,list):
        assert isinstance(b,list),f'Type differs: {path}'
        if path.rsplit('/',1)[-1].startswith('match_'):
            # Matching results are derived outputs; threshold-level float changes
            # can change their length, unlike the evaluated frame/mask coverage.
            maxima[path+'/length_delta']=abs(len(a)-len(b))
            if len(a)!=len(b):
                return maxima
        else:
            assert len(a)==len(b),f'Coverage differs: {path}'
        for x,y in zip(a,b):
            for field,value in check_nested(x,y,path+'/*').items():
                maxima[field]=max(maxima.get(field,0),value)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)),f'Numeric value missing: {path}'
        if path.rsplit('/',1)[-1] in {'frame','mask_index','num_masks','num_sources','num_frames_evaluated'}:
            assert a==b,f'Index/count differs: {path}'
        maxima[path]=abs(a-b)
    elif a is None:
        assert b is None,f'Validity differs: {path}'
    return maxima


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--baseline',default='output/fresh_cost_20260918_long_default/predictions')
    ap.add_argument('--baseline-run',help='Compare all episodes against another sharded experiment')
    ap.add_argument('--report-name',default='regression.json')
    args=ap.parse_args();run=Path(args.run);baseline=Path(args.baseline)
    references={}
    roots=sorted(Path(args.baseline_run).glob('worker*/predictions')) if args.baseline_run else [baseline]
    if args.baseline_run and (Path(args.baseline_run)/'predictions').is_dir():
        roots=[Path(args.baseline_run)/'predictions']
    for root in roots:
        for path in root.glob('*/*/*.json'):
            key=(path.parent.parent.name,path.parent.name,path.name)
            assert key not in references,f'Duplicate baseline {key}'
            references[key]=path
    norm=load_normalization(ROOT/'configs/analysis/keyarea_results.yaml')
    result=[]
    candidates=[run] if (run/'predictions').is_dir() else sorted(run.glob('worker*'))
    for worker in candidates:
        if not worker.is_dir():continue
        for path in sorted((worker/'predictions').glob('*/*/*.json')):
            dim=path.stem
            if dim not in ['keyarea_visual_quality','interaction_spatial_alignment','multiview_consistency']:continue
            ref=references.get((path.parent.parent.name,path.parent.name,path.name))
            if ref is None:continue
            a=json.loads(ref.read_text());b=json.loads(path.read_text())
            x=episode_score(dim,a,norm);y=episode_score(dim,b,norm)
            assert x is not None and y is not None
            aa={f['frame']:f for f in a['per_frame']};bb={f['frame']:f for f in b['per_frame']}
            assert aa.keys()==bb.keys(),f'Frame coverage mismatch {path}'
            nested={}
            for section in ['per_frame','mask_sources','per_view']:
                if section in a:
                    nested.update(check_nested(a[section],b[section],section))
            numeric={}
            for k in ['psnr','lpips','vjepa_similarity','alignment_score','multiview_consistency_score']:
                deltas=[abs(aa[f][k]-bb[f][k]) for f in aa if isinstance(aa[f].get(k),(float,int)) and isinstance(bb[f].get(k),(float,int))]
                if deltas:numeric[k]=max(deltas)
            result.append({'episode':str(path.parent.relative_to(worker/'predictions')),
                           'metric':dim,'frames':len(aa),'baseline_score_100':x,
                           'candidate_score_100':y,'score_delta_100':y-x,
                           'max_per_frame_raw_delta':numeric,
                           'nested_coverage_pass':True,'nested_max_raw_deltas':nested})
    assert result,'No comparable results'
    spec_path=run/'run_spec.json'
    dims=json.loads(spec_path.read_text()).get('dimensions') if spec_path.exists() else None
    dims=dims or ['keyarea_visual_quality','interaction_spatial_alignment','multiview_consistency']
    expected=sum((t,e,d+'.json') in references for t,e in json.loads((run/'manifest.json').read_text()) for d in dims)
    if args.baseline_run:
        assert expected==len(dims)*len(json.loads((run/'manifest.json').read_text())), 'Baseline must cover every episode'
    assert len(result)==expected,f'Incomplete metric coverage: {len(result)}/{expected}'
    summary={'baseline_reference':str(args.baseline_run or baseline),'records':result,'max_episode_score_delta_100':max(abs(r['score_delta_100']) for r in result),
             'episode_tolerance_0_01_pass':all(abs(r['score_delta_100'])<=.01 for r in result)}
    write(run/args.report_name,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
