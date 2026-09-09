"""Metric calculations used by the W1 training controller."""

from __future__ import annotations

from typing import Any

from .config import FastDecisionConfig


def _values(summary: dict[str, Any], *, sigma: float) -> list[float]:
    selected = {
        float(row["rate"]): float(row["unobserved_free_space_nmse"])
        for row in summary["rows"]
        if row["variant"] == "w1"
        and row["stage"] == "fast_w1"
        and float(row["noise_std"]) == float(sigma)
        and row["method"] == "no_da"
    }
    if set(selected) != {1.0, 3.0}:
        raise ValueError(f"fast_w1 requires p1/p3 at sigma={sigma:g}")
    return [selected[1.0], selected[3.0]]


def fast_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    settings: FastDecisionConfig,
) -> dict[str, Any]:
    candidate_clean = sum(_values(candidate, sigma=0.0)) / 2
    baseline_clean = sum(_values(baseline, sigma=0.0)) / 2
    candidate_noisy = sum(_values(candidate, sigma=0.05)) / 2
    baseline_noisy = sum(_values(baseline, sigma=0.05)) / 2
    clean_improvement = 1.0 - candidate_clean / baseline_clean
    noisy_regression = candidate_noisy / baseline_noisy - 1.0
    return {
        "clean_relative_improvement": clean_improvement,
        "noisy_relative_regression": noisy_regression,
        "eligible": (
            clean_improvement >= settings.minimum_clean_relative_improvement - 1e-12
            and noisy_regression <= settings.maximum_noisy_relative_regression + 1e-12
        ),
        "robust_enough_for_plateau": (
            noisy_regression <= settings.maximum_noisy_relative_regression + 1e-12
        ),
    }


def stopping_reason(
    *,
    step: int,
    max_steps: int,
    history: list[dict[str, Any]],
    plateau_count: int,
    settings: FastDecisionConfig,
) -> str:
    if step >= max_steps:
        return "max_steps_without_full_pass"
    if step >= settings.no_eligible_stop_step and not any(item["eligible"] for item in history):
        return "no_fast_eligible_checkpoint"
    recent = history[-settings.excessive_noisy_evaluations :]
    if len(recent) == settings.excessive_noisy_evaluations and all(
        item["noisy_relative_regression"] > settings.excessive_noisy_regression + 1e-12
        for item in recent
    ):
        return "repeated_excessive_noisy_regression"
    if step >= settings.plateau_minimum_step and plateau_count >= settings.plateau_evaluations:
        return "robust_fast_metric_plateau"
    return ""


__all__ = ["fast_comparison", "stopping_reason"]
