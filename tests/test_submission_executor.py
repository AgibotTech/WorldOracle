import json
from types import SimpleNamespace
import numpy as np
import pytest
from acwmbench.execution_acceleration import SampledMasks, discover_frames
from tools.evaluate_submission import selected_trials, validate_result, valid_receipt, write


def test_po_window_uses_evaluated_prefix_coordinates_without_mutation():
    from dataclasses import dataclass
    from tools.evaluate_submission import window_trials
    @dataclass
    class Trial:
        prompt_segments: list
        video_dir: str = '/unchanged/video'
    segments = [{'end_frame': i, 'source_n_frames': 3600} for i in [0, 899, 900, 3599]]
    trial = Trial(segments)
    clipped = window_trials([trial], 900)[0]
    assert clipped.video_dir == trial.video_dir
    assert [s['end_frame'] for s in clipped.prompt_segments] == [0, 899]
    assert all(s['source_n_frames'] == 900 for s in clipped.prompt_segments)
    assert len(trial.prompt_segments) == 4
    assert all(s['source_n_frames'] == 3600 for s in trial.prompt_segments)


def test_window_adapters_same_pixels_and_indices(tmp_path, monkeypatch):
    import importlib
    from PIL import Image
    from acwmbench.execution_acceleration import install, install_window
    kro = importlib.import_module('acwmbench.metrics.keyarea_visual_quality')
    ia = importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
    mvc = importlib.import_module('acwmbench.metrics.multiview_consistency')
    # Register all process-local hooks for restoration after each variant.
    hooks = [(kro,'load_frames_as_array'),(kro,'_load_npz_masks'),(kro,'ensure_lpips'),
             (ia,'_load_masks'),(mvc,'_load_masks'),(ia,'_discover_frames'),
             (mvc,'_discover_frames'),(ia,'DinoExtractor'),(mvc,'DinoExtractor')]
    view = tmp_path/'top_head'; view.mkdir()
    for i in range(7):
        Image.fromarray(np.full((4,4,3),i,dtype=np.uint8)).save(view/f'{i:06d}.png')
    original = kro.load_frames_as_array(tmp_path, preferred_view='top_head')[:5:2]
    for accelerated in [False, True]:
        with monkeypatch.context() as patch:
            for module, name in hooks:
                patch.setattr(module, name, getattr(module, name))
            if accelerated:
                install({'keyarea_visual_quality':{'downsample_rate':2},
                         'interaction_spatial_alignment':{'downsample_rate':2}}, frame_limit=5)
            else:
                install_window(5)
            frames = kro.load_frames_as_array(tmp_path, preferred_view='top_head')
            assert len(frames) == 5
            np.testing.assert_array_equal(frames[::2], original)
            assert list(ia._discover_frames(tmp_path,'top_head')) == list(range(5))
            assert list(mvc._discover_frames(tmp_path,'top_head')) == list(range(5))


def test_mask_slice_exact():
    x=np.arange(2*37*3*4).reshape(2,37,3,4)
    a=SampledMasks(x,2)
    np.testing.assert_array_equal(a[:, :35:10],x[:, :35:10])
    np.testing.assert_array_equal(a[1, 12],x[1,12])
    assert a.shape==x.shape
    with pytest.raises(ValueError):a[:,::3]
    with pytest.raises(ValueError):a[0,3]


def test_frame_discovery_same(tmp_path):
    from acwmbench.metrics.interaction_spatial_alignment import _discover_frames
    p=tmp_path/'top_head';p.mkdir()
    for n in ['frame_000.png','frame_010.png','frame_020.png','junk.txt']:(p/n).touch()
    assert discover_frames(tmp_path,'top_head')==_discover_frames(tmp_path,'top_head')
    (p/'other_010.jpg').touch()
    with pytest.raises(ValueError):discover_frames(tmp_path,'top_head')


def test_selection(tmp_path):
    trials=[SimpleNamespace(task_id=t,episode_id=str(i)) for t in ['a','b'] for i in range(3)]
    assert [(t.task_id,t.episode_id) for t in selected_trials(trials,limit=3)]==[('a','0'),('b','0'),('a','1')]
    p=tmp_path/'manifest.json';p.write_text(json.dumps([['b','2'],['a','0']]))
    assert selected_trials(trials,str(p))[0].episode_id=='2'
    p.write_text(json.dumps([['b','2'],['b','2']]))
    with pytest.raises(ValueError):selected_trials(trials,str(p))


def test_resume_checks_result_content(tmp_path):
    trial=SimpleNamespace(task_id='a',episode_id='e',uid='a__e__default')
    p=tmp_path/'predictions/a/e';p.mkdir(parents=True)
    d='interaction_spatial_alignment'
    write(p/(d+'.json'),{'uid':trial.uid,'per_frame':[{'frame':0,'alignment_score':5}]})
    hashes=validate_result(p,[d],trial)
    write(tmp_path/'receipts/a/e.json',{'identity':'id','hashes':hashes})
    assert valid_receipt(tmp_path,trial,'id',[d])
    assert not valid_receipt(tmp_path,trial,'different',[d])
    write(p/(d+'.json'),{'uid':trial.uid,'per_frame':[{'frame':0,'alignment_score':9}]})
    assert not valid_receipt(tmp_path,trial,'id',[d])


def test_po_invalid_is_not_complete(tmp_path):
    write(tmp_path/'physical_judge.json',{'default':{'status':'complete','key_steps':[{'format_valid':False}]}})
    with pytest.raises(ValueError):validate_result(tmp_path,['physical_judge'])
