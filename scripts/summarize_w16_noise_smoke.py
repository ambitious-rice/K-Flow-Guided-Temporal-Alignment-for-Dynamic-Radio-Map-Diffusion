#!/usr/bin/env python3
"""Summarize a small W16 validation noise-estimation smoke run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def _sigmas(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item]
    if not values or any(item < 0.0 for item in values):
        raise ValueError("expected sigmas must be non-negative")
    return values


def _sigma_key(value: float) -> str:
    return f"{value:.6g}"


def _load_units(units_dir: Path, method: str) -> list[dict]:
    units = []
    for path in sorted(units_dir.glob("rank_*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            unit = json.load(handle)
        if method not in unit.get("methods", {}):
            raise ValueError(f"{path} does not contain method {method!r}")
        units.append(unit)
    if not units:
        raise ValueError(f"no JSON estimation units found below {units_dir}")
    return units


def _stats(units: list[dict], method: str) -> dict:
    true_sigma = float(units[0]["true_sigma"])
    estimates = [float(unit["methods"][method]["sigma"]) for unit in units]
    errors = [abs(value - true_sigma) for value in estimates]
    return {
        "true_sigma": true_sigma,
        "windows": len(units),
        "mean_estimate": mean(estimates),
        "median_estimate": median(estimates),
        "mae": mean(errors),
        "rmse": math.sqrt(mean(error**2 for error in errors)),
    }


def build_summary(
    units: list[dict], *, method: str, expected_sigmas: list[float], expected_scenes: int
) -> dict:
    by_sigma: dict[str, list[dict]] = defaultdict(list)
    by_scene_sigma: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for unit in units:
        key = _sigma_key(float(unit["true_sigma"]))
        by_sigma[key].append(unit)
        by_scene_sigma[str(unit["scene_id"])][key].append(unit)

    overall = {key: _stats(group, method) for key, group in sorted(by_sigma.items())}
    per_scene = {
        scene: {key: _stats(group, method) for key, group in sorted(groups.items())}
        for scene, groups in sorted(by_scene_sigma.items())
    }
    expected_keys = [_sigma_key(value) for value in expected_sigmas]
    nonzero_keys = [key for key in expected_keys if float(key) > 0.0]
    complete_sigmas = all(key in overall for key in expected_keys)
    covered_scenes = all(
        sum(key in groups for groups in per_scene.values()) >= expected_scenes
        for key in expected_keys
    )
    scene_detection = []
    if "0" in expected_keys:
        for scene, groups in per_scene.items():
            for key in nonzero_keys:
                if "0" in groups and key in groups:
                    scene_detection.append(
                        {
                            "scene_id": scene,
                            "true_sigma": float(key),
                            "clean_mean": groups["0"]["mean_estimate"],
                            "noisy_mean": groups[key]["mean_estimate"],
                            "detected": groups[key]["mean_estimate"] > groups["0"]["mean_estimate"],
                        }
                    )
    nonzero_better_than_zero = all(
        overall[key]["mae"] < overall[key]["true_sigma"]
        for key in nonzero_keys
        if key in overall
    )
    checks = {
        "all_expected_sigmas_present": complete_sigmas,
        "expected_scene_coverage": covered_scenes,
        "each_scene_detects_nonzero_noise": bool(scene_detection)
        and all(item["detected"] for item in scene_detection),
        "nonzero_mae_beats_zero_estimate": nonzero_better_than_zero,
    }
    return {
        "schema": "w16_noise_estimation_smoke_summary_v1",
        "primary_method": method,
        "unit_count": len(units),
        "expected_sigmas": expected_sigmas,
        "expected_scenes": expected_scenes,
        "overall": overall,
        "per_scene": per_scene,
        "scene_detection": scene_detection,
        "checks": checks,
        "smoke_passed": all(checks.values()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, summary: dict, run_config: dict | None) -> None:
    lines = [
        "# W16 noise-estimation smoke report",
        "",
        f"- Smoke passed: **{summary['smoke_passed']}**",
        f"- Method: `{summary['primary_method']}`",
        f"- Windows: {summary['unit_count']}",
        f"- Expected sigmas: {', '.join(str(value) for value in summary['expected_sigmas'])}",
        "",
        "## Overall estimates",
        "",
        "| Injected sigma | Windows | Mean estimate | Median estimate | MAE | RMSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["overall"].values():
        lines.append(
            "| {true_sigma:.6g} | {windows} | {mean_estimate:.6g} | "
            "{median_estimate:.6g} | {mae:.6g} | {rmse:.6g} |".format(**item)
        )
    lines.extend(["", "## Per-scene estimates", ""])
    for scene, groups in summary["per_scene"].items():
        lines.extend([f"### {scene}", "", "| Injected sigma | Windows | Mean estimate | MAE |", "| ---: | ---: | ---: | ---: |"])
        for item in groups.values():
            lines.append(
                "| {true_sigma:.6g} | {windows} | {mean_estimate:.6g} | {mae:.6g} |".format(**item)
            )
        lines.append("")
    lines.extend(["## Smoke checks", ""])
    lines.extend(f"- `{name}`: {value}" for name, value in summary["checks"].items())
    if run_config:
        lines.extend(["", "## Experiment configuration", "", "```json", json.dumps(run_config, indent=2, sort_keys=True), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-config")
    parser.add_argument("--method", default="calibrated_ensemble_mle")
    parser.add_argument("--expected-sigmas", required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    args = parser.parse_args()
    if args.expected_scenes < 1:
        raise ValueError("expected-scenes must be positive")
    units = _load_units(Path(args.units_dir), args.method)
    run_config = None
    if args.run_config:
        with Path(args.run_config).open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
    summary = build_summary(
        units,
        method=args.method,
        expected_sigmas=_sigmas(args.expected_sigmas),
        expected_scenes=args.expected_scenes,
    )
    _write_json(Path(args.output), summary)
    _write_report(Path(args.report), summary, run_config)


if __name__ == "__main__":
    main()
