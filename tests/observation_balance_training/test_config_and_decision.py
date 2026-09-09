from __future__ import annotations

from pathlib import Path

from rmdm_hvdit_v4_x0_observation_balance.config import (
    FastDecisionConfig,
    load_observation_balance_config,
)
from rmdm_hvdit_v4_x0_observation_balance.decision import fast_comparison, stopping_reason


ROOT = Path(__file__).resolve().parents[2]


def _summary(clean_p1: float, clean_p3: float, noisy_p1: float, noisy_p3: float) -> dict:
    rows = []
    for sigma, values in ((0.0, (clean_p1, clean_p3)), (0.05, (noisy_p1, noisy_p3))):
        for rate, value in zip((1.0, 3.0), values):
            rows.append(
                {
                    "variant": "w1",
                    "stage": "fast_w1",
                    "rate": rate,
                    "noise_std": sigma,
                    "method": "no_da",
                    "unobserved_free_space_nmse": value,
                }
            )
    return {"rows": rows}


def test_fixed_training_config_loads() -> None:
    config = load_observation_balance_config(
        ROOT / "configs/hvdit_v4_x0_observation_balance/w1_v1.yaml"
    )
    assert config.max_steps == 12000
    assert config.segment_steps == 1000
    assert config.noise.probabilities == [0.50, 0.20, 0.15, 0.15]
    assert config.alignment.peak_weight == 0.25


def test_fast_comparison_uses_clean_and_noisy_constraints() -> None:
    baseline = _summary(1.0, 1.0, 2.0, 2.0)
    candidate = _summary(0.9, 0.9, 2.04, 2.04)
    result = fast_comparison(candidate, baseline, FastDecisionConfig())
    assert abs(result["clean_relative_improvement"] - 0.1) < 1e-12
    assert abs(result["noisy_relative_regression"] - 0.02) < 1e-12
    assert result["eligible"] is True


def test_stopping_rules_are_explicit() -> None:
    settings = FastDecisionConfig()
    history = [
        {"eligible": False, "noisy_relative_regression": 0.0},
        {"eligible": False, "noisy_relative_regression": 0.0},
    ]
    assert stopping_reason(
        step=6000,
        max_steps=12000,
        history=history,
        plateau_count=0,
        settings=settings,
    ) == "no_fast_eligible_checkpoint"
    assert stopping_reason(
        step=4000,
        max_steps=12000,
        history=history,
        plateau_count=3,
        settings=settings,
    ) == "robust_fast_metric_plateau"


def test_fast_thresholds_include_exact_boundary() -> None:
    baseline = _summary(1.0, 1.0, 1.0, 1.0)
    settings = FastDecisionConfig()
    assert fast_comparison(_summary(0.95, 0.95, 1.03, 1.03), baseline, settings)["eligible"]
    assert not fast_comparison(_summary(0.950001, 0.950001, 1.03, 1.03), baseline, settings)["eligible"]
    assert not fast_comparison(_summary(0.95, 0.95, 1.030001, 1.030001), baseline, settings)["eligible"]


def test_exact_eight_percent_regression_does_not_stop_training() -> None:
    history = [{"eligible": True, "noisy_relative_regression": 1.08 - 1.0}] * 2
    assert stopping_reason(
        step=4000, max_steps=12000, history=history, plateau_count=0,
        settings=FastDecisionConfig(),
    ) == ""
