"""Protocol-checked comparison against the retained factorized unified baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rmdm_hvdit_v4_joint.provenance import sha256_file


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _validate_stage_a(result: dict[str, Any], *, label: str) -> None:
    if result.get("subset_stage") != "stage_a" or int(result.get("ddim_steps", -1)) != 20:
        raise ValueError(f"{label} is not Stage-A DDIM20")
    if int(result.get("video_count", -1)) != 30 or int(result.get("window_count", -1)) != 180:
        raise ValueError(f"{label} does not cover 30 videos / 180 first96 windows")
    if set(result.get("rates", {})) != {"1", "2", "3"}:
        raise ValueError(f"{label} rate set is not exactly p1/p2/p3")


def _rate_nmse(result: dict[str, Any]) -> dict[str, float]:
    return {
        rate: float(result["rates"][rate]["metrics"]["full_image"]["nmse"])
        for rate in ("1", "2", "3")
    }


def compare_with_factorized_baseline(
    *,
    factorized_run_dir: str | Path,
    hvdit_selection_path: str | Path,
) -> dict[str, Any]:
    baseline_root = Path(factorized_run_dir).expanduser().resolve()
    baseline_paths = sorted((baseline_root / "validation").glob("stage_a_epoch_*.json"))
    if not baseline_paths:
        raise FileNotFoundError(f"No committed factorized Stage-A results under {baseline_root}")
    baseline_candidates = []
    for path in baseline_paths:
        result = _load(path)
        _validate_stage_a(result, label=str(path))
        baseline_candidates.append(
            {
                "path": path,
                "result": result,
                "score": float(result["macro_full_image_nmse_p1_p2_p3"]),
            }
        )
    baseline = min(baseline_candidates, key=lambda item: (item["score"], str(item["path"])))

    selection_path = Path(hvdit_selection_path).expanduser().resolve()
    selection = _load(selection_path)
    selected = selection["selected"]
    hvdit_stage_a_path = Path(selected["stage_a"]).expanduser().resolve()
    hvdit = _load(hvdit_stage_a_path)
    _validate_stage_a(hvdit, label=str(hvdit_stage_a_path))
    baseline_rates = _rate_nmse(baseline["result"])
    hvdit_rates = _rate_nmse(hvdit)
    baseline_macro = float(baseline["score"])
    hvdit_macro = float(hvdit["macro_full_image_nmse_p1_p2_p3"])
    report: dict[str, Any] = {
        "schema": "rmdm_hvdit_v4_joint_factorized_comparison_v1",
        "protocol": {
            "videos": 30,
            "frames_per_video": 96,
            "starts": [0, 16, 32, 48, 64, 80],
            "rates": [1, 2, 3],
            "ddim_steps": 20,
            "metric": "full_image_nmse",
        },
        "factorized": {
            "run_dir": str(baseline_root),
            "available_stage_a_count": len(baseline_candidates),
            "best_stage_a": str(baseline["path"]),
            "best_stage_a_sha256": sha256_file(baseline["path"]),
            "rates": baseline_rates,
            "macro": baseline_macro,
        },
        "hvdit_v4_joint": {
            "selection": str(selection_path),
            "stage_a": str(hvdit_stage_a_path),
            "stage_a_sha256": sha256_file(hvdit_stage_a_path),
            "rates": hvdit_rates,
            "macro": hvdit_macro,
        },
        "delta_hvdit_minus_factorized": {
            "rates": {rate: hvdit_rates[rate] - baseline_rates[rate] for rate in ("1", "2", "3")},
            "macro": hvdit_macro - baseline_macro,
            "relative_macro": hvdit_macro / baseline_macro - 1.0,
        },
        "winner": "hvdit_v4_joint" if hvdit_macro < baseline_macro else "factorized",
    }
    baseline_final_path = baseline_root / "final_selection.json"
    if baseline_final_path.is_file():
        baseline_final = _load(baseline_final_path)
        report["factorized"]["final_selection"] = str(baseline_final_path)
        report["factorized"]["final_combined_score"] = float(
            baseline_final["selected"]["combined_score"]
        )
        report["hvdit_v4_joint"]["final_combined_score"] = float(selected["combined_score"])
        report["delta_hvdit_minus_factorized"]["combined_macro"] = (
            float(selected["combined_score"]) - float(baseline_final["selected"]["combined_score"])
        )
    return report
