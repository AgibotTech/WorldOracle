"""Lightweight configuration post-processing helpers."""

from __future__ import annotations


def configure_distribution_paths(
    config: dict,
    *,
    distribution: str | None = None,
    val_base: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Select a GT distribution and apply runtime path overrides.

    Distribution profiles only describe GT-side data. Prediction roots remain
    model-specific and are supplied with ``--val-base``.
    """
    data = config.setdefault("data", {})
    if distribution:
        profiles = data.get("distributions") or {}
        profile = profiles.get(distribution)
        if not isinstance(profile, dict):
            available = ", ".join(sorted(str(key) for key in profiles)) or "(none)"
            raise ValueError(
                f"unknown distribution {distribution!r}; available: {available}"
            )
        gt_path = profile.get("gt_path")
        if not gt_path:
            raise ValueError(
                f"data.distributions.{distribution}.gt_path is required"
            )
        data["distribution"] = distribution
        data["gt_path"] = str(gt_path)
    if val_base:
        data["val_base"] = str(val_base)
    if output_dir:
        config["save_path"] = str(output_dir)


def configure_policy_eval_paths(
    config: dict,
    dimensions: list[str] | None,
) -> None:
    """Map task-level policy data roots onto the policy-eval dimension.

    ``data.policy_val_base`` means generated closed-loop rollouts that are
    ready for offline scoring, so its presence selects ``wm_score`` rather
    than inheriting base.yaml's rollout-generation mode.
    """
    if not dimensions or "policy_eval" not in dimensions:
        return
    data = config.get("data") or {}
    policy_val_base = data.get("policy_val_base")
    policy_gt_path = data.get("policy_gt_path")
    if not policy_val_base:
        return
    policy_cfg = config.setdefault("dimensions", {}).setdefault(
        "policy_eval",
        {},
    )
    policy_cfg["rollout_root"] = str(policy_val_base)
    policy_cfg["mode"] = str(
        data.get("policy_eval_mode", "wm_score")
    )
    if policy_gt_path:
        policy_cfg["gt_root"] = str(policy_gt_path)
