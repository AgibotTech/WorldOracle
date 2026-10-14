from __future__ import annotations

import time

from acwmbench import episode_store
from acwmbench.dataset import EpisodeTrial
from acwmbench.metrics.physical_judge import compute_physical_judge
from scripts import benchmark_physical_judge_concurrency as benchmark


class _FakeBackend:
    def __init__(self):
        self.requests = 0
        self.closed = False

    def get_usage_summary(self):
        return {
            "backend": "fake",
            "model": "fake",
            "n_requests": self.requests,
            "total_tokens": self.requests * 10,
        }

    def close(self):
        self.closed = True


def _trial(index: int) -> EpisodeTrial:
    return EpisodeTrial(
        task_id="pickfruit",
        episode_id=f"ep{index:06d}",
        trial_id="default",
        video_dir=f"/tmp/video/{index}",
        gt_video_dir=f"/tmp/gt/{index}",
    )


def test_concurrent_runner_preserves_order_and_merges_usage(
    monkeypatch,
) -> None:
    created: list[_FakeBackend] = []

    def build(_):
        backend = _FakeBackend()
        created.append(backend)
        return backend

    def judge(backend, trial, **_):
        backend.requests += 1
        time.sleep(0.002 * (4 - int(trial.episode_id[-1])))
        score = int(trial.episode_id[-1]) % 2
        return {
            "score": float(score),
            "key_steps": [
                {
                    "frame": 10,
                    "segment_id": 0,
                    "aspect_scores": [score, 1],
                }
            ],
        }

    monkeypatch.setattr(benchmark, "build_vlm_backend", build)
    monkeypatch.setattr(benchmark, "judge_one_trial", judge)
    trials = [_trial(index) for index in range(4)]
    result = benchmark._run_concurrent(
        trials,
        workers=2,
        vlm_config={"backend": "fake"},
        judge_kwargs={},
    )

    assert [record["uid"] for record in result["records"]] == [
        trial.uid for trial in trials
    ]
    assert result["successful_trials"] == 4
    assert result["failed_trials"] == 0
    assert result["usage"]["n_requests"] == 4
    assert result["usage"]["total_tokens"] == 40
    assert len(created) == 2
    assert all(backend.closed for backend in created)


def test_run_comparison_reports_judge_agreement() -> None:
    baseline = {
        "records": [
            {
                "uid": "a",
                "status": "complete",
                "result": {
                    "score": 1.0,
                    "key_steps": [
                        {
                            "frame": 10,
                            "segment_id": 0,
                            "aspect_scores": [1, 1],
                        }
                    ],
                },
            }
        ]
    }
    candidate = {
        "records": [
            {
                "uid": "a",
                "status": "complete",
                "result": {
                    "score": 0.5,
                    "key_steps": [
                        {
                            "frame": 10,
                            "segment_id": 0,
                            "aspect_scores": [1, 0],
                        }
                    ],
                },
            }
        ]
    }
    comparison = benchmark._compare_runs(baseline, candidate)
    assert comparison["score_mae"] == 0.5
    assert comparison["max_score_delta"] == 0.5
    assert comparison["criterion_agreement"] == 0.5
    assert comparison["step_exact_agreement"] == 0.0


def test_workload_counts_bundled_criteria() -> None:
    trial = _trial(1)
    trial.prompt_segments = [
        {"success_criteria": ["a", "b"]},
        {"success_criteria": ["c"]},
    ]
    assert benchmark._trial_workload(trial) == (2, 3)


def test_step_limit_is_even_and_includes_final_segment() -> None:
    trial = _trial(1)
    trial.prompt_segments = [
        {"segment_id": index, "success_criteria": [str(index)]}
        for index in range(10)
    ]
    limited = benchmark._limit_trial_steps([trial], 4)
    assert [
        segment["segment_id"]
        for segment in limited[0].prompt_segments
    ] == [0, 3, 6, 9]


def test_production_parallel_judge_has_single_writer_and_stable_order(
    tmp_path, monkeypatch
) -> None:
    trials = [
        EpisodeTrial(
            task_id="pickfruit",
            episode_id="ep000001",
            trial_id=f"trial{index}",
            video_dir="/unused",
        )
        for index in range(3)
    ]
    backends = [_FakeBackend(), _FakeBackend()]

    def judge(backend, trial, **_):
        backend.requests += 1
        time.sleep(0.003 * (3 - int(trial.trial_id[-1])))
        return {
            "score": float(trial.trial_id[-1]),
            "final": {},
            "key_steps": [],
            "instruction": "",
            "eval_view": "top_head",
        }

    monkeypatch.setattr(
        "acwmbench.metrics.physical_judge.judge_one_trial", judge
    )
    result = compute_physical_judge(
        vlm=backends[0],
        worker_vlms=backends,
        trials=trials,
        save_root=tmp_path,
        progress=False,
    )
    episode = result["pickfruit"]["ep000001"]
    assert list(episode) == ["trial0", "trial1", "trial2"]
    persisted = episode_store.load_metric(
        tmp_path,
        "pickfruit",
        "ep000001",
        "physical_judge",
    )
    assert set(persisted) == {"trial0", "trial1", "trial2"}
    assert sum(backend.requests for backend in backends) == 3
