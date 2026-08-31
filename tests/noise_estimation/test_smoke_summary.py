from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_w16_noise_smoke.py"
SPEC = importlib.util.spec_from_file_location("w16_noise_smoke_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _unit(scene: str, true_sigma: float, estimate: float) -> dict:
    return {
        "scene_id": scene,
        "true_sigma": true_sigma,
        "methods": {"calibrated_ensemble_mle": {"sigma": estimate}},
    }


def test_smoke_summary_reports_scene_detection_and_passes() -> None:
    units = [
        _unit("scene_a", 0.0, 0.001),
        _unit("scene_a", 0.01, 0.009),
        _unit("scene_b", 0.0, 0.002),
        _unit("scene_b", 0.01, 0.011),
    ]
    summary = MODULE.build_summary(
        units,
        method="calibrated_ensemble_mle",
        expected_sigmas=[0.0, 0.01],
        expected_scenes=2,
    )

    assert summary["smoke_passed"] is True
    assert summary["overall"]["0.01"]["windows"] == 2
    assert all(item["detected"] for item in summary["scene_detection"])
