"""ACWMBench CLI entrypoint.

Examples:

    # use whatever model_path is written in configs/base.yaml
    python evaluate.py --dimension physical_judge --config_path configs/base.yaml

    # KeyArea Visual Quality (no VLM needed)
    # With V-JEPA 2.1 from local checkpoint:
    python evaluate.py --dimension keyarea_visual_quality --config_path configs/base.yaml

    # DINOv3 object/gripper interaction alignment:
    python evaluate.py --dimension interaction_spatial_alignment \
        --config_path configs/base.yaml

    # Example config snippet:
    # dimensions:
    #   keyarea_visual_quality:
    #     eval_view: top_head
    #     vjepa:
    #       model_path: /path/to/vjepa2.1_vit_base.pth

    # one-line model swap (overrides config; alias OR path OR repo id)
    python evaluate.py --dimension physical_judge --config_path configs/base.yaml \\
        --vlm-model qwen3_vl_8b
    python evaluate.py --dimension physical_judge --config_path configs/base.yaml \\
        --vlm-model qwen3_5_9b
    python evaluate.py --dimension physical_judge --config_path configs/base.yaml \\
        --vlm-model /abs/path/to/some/ckpt

    # env var also works (CLI wins if both are set)
    ACWMBENCH_VLM_MODEL=qwen3_5_9b python evaluate.py ...

    # Generate PI0.5 -> GE-Sim AR rollouts only (no VLM calls).
    python evaluate.py --dimension policy_eval --config_path configs/base.yaml \
        --task pickfruit --episode ep000000 --overwrite

"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml

from acwmbench import ACWMBenchmark
from acwmbench.configuration import (
    configure_distribution_paths,
    configure_policy_eval_paths,
)
from acwmbench.distributed import dist_init, print0
from acwmbench.models.model_zoo import MODEL_ALIASES, resolve_model_path

# Base config is always loaded first; --config_path (e.g. a per-model file that
# only overrides data.val_base) is deep-merged on top. Resolve relative to this
# file so it works regardless of the current working directory.
DEFAULT_BASE_CONFIG = str(Path(__file__).resolve().parent / "configs" / "base.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACWMBench",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dimension",
        nargs="+",
        default=None,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>. "
        "Not required when using --check_data.",
    )
    parser.add_argument(
        "--check_data",
        action="store_true",
        help="Only check that generated data is complete (each episode's per-view "
        "frame count matches n_frames in the GT prompt/prompts.json), then exit. "
        "Does not require --dimension or any model.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the automatic pre-evaluation data-completeness check. By "
        "default, evaluation first checks completeness and evaluates ONLY the "
        "complete episodes, skipping (not failing on) incomplete ones.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional override config deep-merged on top of the base config. "
        "Typically a small per-model file that only sets data.val_base, e.g. "
        "configs/tasks/cosmos3.yaml, so different model outputs can be evaluated "
        "without touching the base config.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base YAML config, always loaded first (default: {DEFAULT_BASE_CONFIG}).",
    )
    parser.add_argument(
        "--distribution",
        default=None,
        help=(
            "Select a configured GT distribution, for example "
            "'human_operated' or 'policy_rollout'."
        ),
    )
    parser.add_argument(
        "--val-base",
        default=None,
        help="Prediction root override: <task>/<episode>/video/<view>/frame_*.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Aggregate report directory override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute selected metrics and overwrite their per-episode JSON files. "
        "By default, existing <val_base>/<task>/<episode>/<metric>.json files "
        "are reused.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Dump debug visualizations. For keyarea_visual_quality this saves, "
        "per sampled frame/mask, a panel of gt frame, predicted frame, the mask, "
        "and the cropped-masked gt/pred pair actually fed to the metrics. "
        "Written under <val_base>/<task>/<episode>/visualize/. Off by default.",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("ACWMBENCH_VLM_MODEL"),
        help=(
            "Override physical_judge.vlm.model_path. Accepts a short alias, a "
            "HuggingFace repo id, or an on-disk path. "
            f"Known aliases: {sorted(set(MODEL_ALIASES))}."
        ),
    )
    parser.add_argument(
        "--ckpt-root",
        default="ckpt",
        help="Root directory where downloaded checkpoints live (default: ckpt/).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only evaluate the first N episodes (sorted by task/episode id). "
        "Useful for quick prompt-engineering / debugging passes.",
    )
    parser.add_argument(
        "--top-n-per-task",
        type=int,
        default=None,
        help="Evaluate the first N episodes independently within every task, "
        "sorted by episode id. Applied before the global --top-n limit.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        help="Only evaluate these subtask(s) (dataset subfolder name under data_base), "
        "e.g. --task pickfruit. Applied before --top-n.",
    )
    parser.add_argument(
        "--episode",
        nargs="+",
        default=None,
        help="Only evaluate these episode(s) (episode dir name under a subtask), "
        "e.g. --episode ep000000. The leading 'ep' is optional (000000 also works). "
        "Applied together with --task, before --top-n.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins)."""
    import copy

    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_merged_config(base_config: str, override_config: str | None) -> dict:
    """Load the base config and deep-merge an optional override config on top."""
    config = load_config(base_config)
    if override_config:
        config = _deep_merge(config, load_config(override_config))
    return config


def main() -> None:
    args = parse_args()

    if not args.check_data and not args.dimension:
        raise SystemExit(
            "Nothing to do: pass --dimension <dim ...> to evaluate, "
            "or --check_data to only verify generated-data completeness."
        )

    dist_init()
    print0(f"[acwmbench] args: {args}")

    config = load_merged_config(args.base_config, args.config_path)
    configure_distribution_paths(
        config,
        distribution=args.distribution,
        val_base=args.val_base,
        output_dir=args.output_dir,
    )
    configure_policy_eval_paths(config, args.dimension)
    print0(f"[acwmbench] base config   = {args.base_config}")
    if args.config_path:
        print0(f"[acwmbench] override config = {args.config_path}")

    save_path = config.get("save_path", "./output")
    data_cfg = config.get("data", {})
    data_base = data_cfg.get("val_base")
    gt_path = data_cfg.get("gt_path")

    # --- Data completeness check (no model / GPU needed) -------------------
    if args.check_data:
        _run_data_check(
            gt_path=gt_path,
            val_base=data_base,
            save_path=save_path,
            config=config,
            task_filter=args.task,
            episode_filter=args.episode,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_standard_data = any(dim != "policy_eval" for dim in args.dimension)
    if needs_standard_data and (not gt_path or not data_base):
        raise ValueError(
            "config must define `data.gt_path` and, for video dimensions, "
            "`data.val_base` "
            f"(got {data_cfg!r})"
        )
    if not data_base:
        data_base = gt_path

    data_name = config.get(
        "data_name", os.path.basename(str(data_base).rstrip("/")).replace("_dataset", "")
    )

    dim_configs = config.get("dimensions", {})

    if args.vlm_model:
        resolved = resolve_model_path(args.vlm_model, ckpt_root=args.ckpt_root)
        pj = dim_configs.setdefault("physical_judge", {}).setdefault("vlm", {})
        previous = pj.get("model_path")
        pj["model_path"] = resolved
        print0(
            f"[acwmbench] --vlm-model {args.vlm_model!r} -> "
            f"model_path={resolved} (was {previous!r})"
        )

    print0(f"[acwmbench] save_path = {save_path}")
    print0(
        "[acwmbench] data.distribution = "
        f"{data_cfg.get('distribution', 'custom')}"
    )
    print0(f"[acwmbench] data.gt_path = {gt_path}")
    print0(f"[acwmbench] data.val_base = {data_base}  (name={data_name})")
    if "policy_eval" in args.dimension:
        policy_cfg = (dim_configs.get("policy_eval") or {})
        print0(
            "[acwmbench] policy_eval: "
            f"mode={policy_cfg.get('mode')}, "
            f"rollout_root={policy_cfg.get('rollout_root')}, "
            f"gt_root={policy_cfg.get('gt_root')}"
        )
    print0(f"[acwmbench] dimensions = {args.dimension}")

    # --- Auto data-completeness check (video dimensions only) --------------
    # By default we check every generated episode first and evaluate ONLY the
    # complete ones, skipping (not failing on) the rest. --skip-check disables.
    include_episodes = None
    check_report = None
    autocheck = (
        not args.skip_check
        and bool(data_base)
        and any(dim != "policy_eval" for dim in args.dimension)
    )
    if autocheck:
        from acwmbench.check_data import check_generated_data, print_check_report

        check_report = check_generated_data(
            gt_path=gt_path,
            val_base=data_base,
            task_filter=args.task,
            episode_filter=args.episode,
        )
        print_check_report(check_report, printer=print0)
        include_episodes = {
            (r["task_id"], r["episode_id"])
            for r in check_report["episodes"]
            if r["status"] == "ok"
        }
        if not include_episodes:
            raise SystemExit(
                "[acwmbench] data check: no complete episodes to evaluate. "
                "Fix the data, or pass --skip-check to force evaluation anyway."
            )

    bench = ACWMBenchmark(device=device, output_path=save_path)
    results = bench.evaluate(
        data_base=data_base,
        data_name=data_name,
        dimension_list=args.dimension,
        gt_path=gt_path,
        dim_configs=dim_configs,
        hardware_config=config.get("hardware", {}),
        overwrite=args.overwrite,
        limit_episodes=args.top_n,
        limit_episodes_per_task=args.top_n_per_task,
        task_filter=args.task,
        episode_filter=args.episode,
        include_episodes=include_episodes,
        visualize=args.visualize,
    )

    if check_report is not None:
        _report_eval_coverage(check_report, save_path, data_name)

    if "physical_judge" in args.dimension:
        _report_physical_judge_accuracy(results, save_path, data_name)
        _report_physical_judge_token_usage(results)

    print0("[acwmbench] done")


def _report_eval_coverage(check_report: dict, save_path: str, data_name: str) -> None:
    """Print which episodes were evaluated vs skipped (by the data check)."""
    import json

    episodes = check_report.get("episodes", [])
    evaluated = [r for r in episodes if r["status"] == "ok"]
    skipped = [r for r in episodes if r["status"] != "ok"]

    print0("")
    print0("================ evaluation coverage ================")
    print0(f"  evaluated episodes : {len(evaluated)}")
    for r in evaluated:
        print0(f"    [OK] {r['task_id']}/{r['episode_id']}")
    print0(f"  skipped episodes   : {len(skipped)}  (incomplete data, not evaluated)")
    for r in skipped:
        reason = r["issues"][0] if r.get("issues") else r["status"]
        print0(f"    [{r['status']}] {r['task_id']}/{r['episode_id']} — {reason}")
    print0("=====================================================")

    # Persist the coverage split alongside the results.
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{data_name}_eval_coverage.json")
    coverage = {
        "evaluated": [
            {"task_id": r["task_id"], "episode_id": r["episode_id"]} for r in evaluated
        ],
        "skipped": [
            {
                "task_id": r["task_id"],
                "episode_id": r["episode_id"],
                "status": r["status"],
                "issues": r.get("issues", []),
            }
            for r in skipped
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print0(f"[acwmbench] eval coverage saved to {out_path}")


def _run_data_check(
    gt_path,
    val_base,
    save_path: str,
    config: dict,
    task_filter=None,
    episode_filter=None,
) -> None:
    """Check generated-data completeness, print + save a report, then exit."""
    import json

    from acwmbench.check_data import check_generated_data, print_check_report

    if not gt_path or not val_base:
        raise ValueError(
            "--check_data needs both `data.gt_path` (for n_frames) and "
            f"`data.val_base` (generated frames) in the config (got {config.get('data', {})!r})"
        )

    data_name = config.get(
        "data_name", os.path.basename(str(val_base).rstrip("/")).replace("_dataset", "")
    )
    print0(f"[acwmbench] check_data: gt_path  = {gt_path}")
    print0(f"[acwmbench] check_data: val_base = {val_base}  (name={data_name})")

    report = check_generated_data(
        gt_path=gt_path,
        val_base=val_base,
        task_filter=task_filter,
        episode_filter=episode_filter,
    )
    print_check_report(report, printer=print0)

    complete = {
        (r["task_id"], r["episode_id"])
        for r in report["episodes"]
        if r["status"] == "ok"
    }
    metric_names = (
        "physical_judge",
        "keyarea_visual_quality",
        "interaction_spatial_alignment",
        "multiview_consistency",
    )
    status = {}
    for task_id, ep_id in sorted(complete):
        ep_dir = Path(val_base) / task_id / ep_id
        completed = []
        for metric in metric_names:
            metric_file = ep_dir / f"{metric}.json"
            if not metric_file.is_file():
                continue
            if metric == "physical_judge":
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if not any(
                        isinstance(result, dict)
                        and result.get("metric_version") == 1
                        and result.get("status") == "complete"
                        for result in cached.values()
                    ):
                        continue
                except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    continue
            elif metric == "keyarea_visual_quality":
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                        if (
                            cached.get("metrics")
                            != ["psnr", "lpips", "vjepa_similarity"]
                            or cached.get("metric_version") != 12
                        ):
                            continue
                except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    continue
            elif metric == "multiview_consistency":
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("metric_version") != 5:
                        continue
                    interaction_file = (
                        metric_file.parent
                        / "interaction_spatial_alignment.json"
                    )
                    with open(
                        interaction_file,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        interaction = json.load(f)
                    from acwmbench.metrics.multiview_consistency import (
                        interaction_robot_arm_digest,
                    )

                    if cached.get(
                        "top_head_source_digest"
                    ) != interaction_robot_arm_digest(interaction):
                        continue
                except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    continue
            elif metric == "interaction_spatial_alignment":
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("metric_version") != 2:
                        continue
                except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    continue
            completed.append(metric)
        status[(task_id, ep_id)] = completed
    no_results = [key for key, metrics in status.items() if not metrics]

    print0("")
    print0("================ evaluation status ================")
    for metric in metric_names:
        done = sum(metric in metrics for metrics in status.values())
        print0(f"  {metric + '.json':38s}: {done}/{len(complete)} episode(s)")
    print0(f"  no metric result yet                  : {len(no_results)}")
    for (task_id, ep_id), metrics in status.items():
        label = ", ".join(metrics) if metrics else "pending"
        print0(f"    [{label}] {task_id}/{ep_id}")
    print0(f"  incomplete (will be skipped)          : {len(report['problems'])}")
    print0("===================================================")

    report["evaluation_status"] = [
        {
            "task_id": task_id,
            "episode_id": ep_id,
            "completed_metrics": metrics,
            "pending_metrics": [m for m in metric_names if m not in metrics],
        }
        for (task_id, ep_id), metrics in status.items()
    ]

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{data_name}_data_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print0(f"[acwmbench] check_data: report saved to {out_path}")

    if not report["summary"]["all_ok"]:
        # Non-zero exit so this can gate a pipeline before evaluation.
        raise SystemExit(1)


def _report_physical_judge_accuracy(results: dict, save_path: str, data_name: str) -> None:
    """Print a GT-accuracy report and save the summary JSON + failures CSV.

    When val_base == gt_path, every success criterion was authored from that
    exact key frame, so a perfect judge should reach criterion_accuracy == 1.0.
    The printed worst episodes + the failures CSV are the hooks for prompt
    engineering / data filtering.
    """
    from acwmbench.aggregation import write_physical_judge_accuracy_report

    summary = write_physical_judge_accuracy_report(results, save_path, data_name)
    if not summary:
        print0("[acwmbench] physical_judge: no criteria found to summarize.")
        return

    overall = summary["overall"]

    def _pct(x):
        return f"{100.0 * x:.2f}%" if isinstance(x, (int, float)) else "n/a"

    print0("")
    print0("================ physical_judge GT accuracy ================")
    print0(f"  episodes                : {overall['n_episodes']}")
    print0(f"  segments (key frames)   : {overall['n_segments']}")
    print0(f"  criteria total          : {overall['n_criteria']}")
    print0(
        f"  criterion accuracy      : {_pct(overall['criterion_accuracy'])} "
        f"({overall['n_satisfied']}/{overall['n_criteria']} satisfied)"
    )
    print0(
        f"  segment accuracy (all-1): {_pct(overall['segment_accuracy'])} "
        f"({overall['n_segments_fully_satisfied']}/{overall['n_segments']})"
    )
    print0(f"  format-invalid segments : {overall['n_format_invalid_segments']}")
    print0(f"  mean trial score        : {_pct(overall['mean_trial_score'])}")

    # Worst episodes (lowest criterion accuracy) to guide data filtering.
    per_ep = summary.get("per_episode") or {}
    ranked = sorted(
        (
            (k, v)
            for k, v in per_ep.items()
            if isinstance(v.get("criterion_accuracy"), (int, float))
        ),
        key=lambda kv: kv[1]["criterion_accuracy"],
    )
    worst = [kv for kv in ranked if kv[1]["criterion_accuracy"] < 1.0][:10]
    if worst:
        print0("  worst episodes:")
        for ep_key, v in worst:
            print0(
                f"    {ep_key}: {_pct(v['criterion_accuracy'])} "
                f"({v['n_satisfied']}/{v['n_criteria']})"
            )
    paths = summary.get("_paths", {})
    if paths:
        print0(f"  summary  : {paths.get('summary_json')}")
        print0(f"  failures : {paths.get('failures_csv')}")
    print0("===========================================================")


def _report_physical_judge_token_usage(results: dict) -> None:
    usage = results.get("physical_judge_usage")
    if not isinstance(usage, dict):
        print0("[acwmbench] physical_judge token usage: not available in results.")
        return

    n_requests = int(usage.get("n_requests", 0) or 0)
    n_with_usage = int(usage.get("n_requests_with_usage", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)

    avg_total = (total_tokens / n_requests) if n_requests else 0.0
    avg_prompt = (prompt_tokens / n_requests) if n_requests else 0.0
    avg_completion = (completion_tokens / n_requests) if n_requests else 0.0

    print0("")
    print0("================ physical_judge token usage ===============")
    print0(f"  backend/model            : {usage.get('backend')} / {usage.get('model')}")
    print0(f"  requests                 : {n_requests} (usage reported on {n_with_usage})")
    print0(f"  prompt tokens            : {prompt_tokens}")
    print0(f"  completion tokens        : {completion_tokens}")
    print0(f"  total tokens             : {total_tokens}")
    print0(f"  reasoning tokens         : {reasoning_tokens}")
    print0(f"  avg prompt / request     : {avg_prompt:.2f}")
    print0(f"  avg completion / request : {avg_completion:.2f}")
    print0(f"  avg total / request      : {avg_total:.2f}")
    print0("===========================================================")


if __name__ == "__main__":
    main()
