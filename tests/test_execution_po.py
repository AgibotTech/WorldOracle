import hashlib
import json
from pathlib import Path
import threading

from PIL import Image
from acwmbench.dataset import EpisodeTrial
from acwmbench.execution_po import run_checkpoints
from acwmbench.metrics.physical_judge import judge_one_trial


def test_checkpoint_messages_and_original_aggregation_identical(tmp_path, monkeypatch):
    from acwmbench.models import vlm
    messages = []
    lock = threading.Lock()

    class Backend:
        def __init__(self):
            self.calls = 0

        def chat(self, value):
            h = hashlib.sha256()
            def walk(x):
                if isinstance(x, dict):
                    for k in sorted(x):h.update(k.encode());walk(x[k])
                elif isinstance(x, list):
                    for v in x:walk(v)
                elif isinstance(x, Image.Image):
                    h.update(str((x.mode,x.size)).encode());h.update(x.tobytes())
                else:h.update(json.dumps(x).encode())
            walk(value)
            digest = h.hexdigest()
            with lock:messages.append(digest)
            self.calls += 1
            return json.dumps({'aspect_scores':[int(digest[-1],16)%2]})

        def get_usage_summary(self):return {'n_requests':self.calls,'total_tokens':self.calls*10}
        def close(self):pass

    video = tmp_path/'video/top_head';video.mkdir(parents=True)
    for i in range(5):Image.new('RGB',(16,16),(i*40,0,0)).save(video/f'{i:06d}.png')
    trials = [EpisodeTrial(task_id='pickfruit',episode_id=f'ep{i:06d}',trial_id='default',
              video_dir=str(video.parent),gt_video_dir=str(video.parent),instruction='Pick fruit',
              prompt_segments=[{'segment_id':j,'end_frame':j+1,'source_n_frames':5,
                                'success_criteria':['Fruit is on table']} for j in range(4)]) for i in range(2)]
    expected = [judge_one_trial(Backend(),t) for t in trials]
    expected_messages = sorted(messages);messages.clear()
    monkeypatch.setattr(vlm,'build_vlm_backend',lambda cfg:Backend())
    run_checkpoints(trials,{'vlm':{'backend':'fake'}},4,tmp_path/'out')
    assert sorted(messages)==expected_messages
    for t, ref in zip(trials,expected):
        value=json.loads((tmp_path/'out/predictions'/t.task_id/t.episode_id/'physical_judge.json').read_text())[t.trial_id]
        from tools.summarize_acwmbench_scores import physical_episode_score
        assert physical_episode_score({t.trial_id:value}) is not None
        assert value.pop('status')=='complete'
        assert value.pop('metric_version') == 1
        assert value.pop('uid') == t.uid
        assert value.pop('task_id') == t.task_id
        assert value.pop('episode_id') == t.episode_id
        assert value.pop('trial_id') == t.trial_id
        value.pop('evaluation_settings')
        assert value==ref
    usage=json.loads((tmp_path/'out/po_usage.json').read_text())
    assert usage['n_requests']==8
