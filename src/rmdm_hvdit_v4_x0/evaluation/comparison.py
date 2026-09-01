"""Compare the x0 pilot with the aligned V4-epsilon step-10k result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RATES = ("1", "2", "3")
DOMAINS = ("full_image", "unobserved_free_space")


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        rate: {
            domain: {
                "nmse": float(payload["rates"][rate]["metrics"][domain]["nmse"]),
                "psnr": float(payload["rates"][rate]["metrics"][domain]["psnr"]),
            }
            for domain in DOMAINS
        }
        for rate in RATES
    }
    full = {rate: metrics[rate]["full_image"]["nmse"] for rate in RATES}
    return {
        "rates": metrics,
        "macro_full_image_nmse": sum(full.values()) / len(full),
        "rate_response": {
            "p1_to_p2_absolute_nmse_drop": full["1"] - full["2"],
            "p2_to_p3_absolute_nmse_drop": full["2"] - full["3"],
            "p1_to_p3_relative_nmse_drop": (full["1"] - full["3"]) / full["1"],
        },
    }


def build_step10k_comparison(
    x0_result: dict[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    baseline_path = root / (
        "runs/rmdm_hvdit_v4_joint/t1_pretrain/validation/"
        "stage_a_step_010000.json"
    )
    baseline = _load(baseline_path)
    x0 = _summary(x0_result)
    epsilon = _summary(baseline)
    return {
        "schema": "rmdm_hvdit_v4_x0_step10k_comparison_v1",
        "protocol": "same Stage-A 30-video/all100/DDIM20 p1-p2-p3",
        "x0_prediction": x0,
        "epsilon_prediction": epsilon,
        "x0_minus_epsilon": {
            rate: {
                domain: {
                    metric: x0["rates"][rate][domain][metric]
                    - epsilon["rates"][rate][domain][metric]
                    for metric in ("nmse", "psnr")
                }
                for domain in DOMAINS
            }
            for rate in RATES
        },
        "baseline_path": str(baseline_path),
    }
