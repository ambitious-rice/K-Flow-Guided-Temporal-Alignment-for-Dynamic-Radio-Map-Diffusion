#!/usr/bin/env python3
"""Prepare, run, summarize, and compare the fixed observation-balance tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Collection

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

# Avoid importing the torch-dependent evaluation package in this local CLI.
SUITE_MODULE_PATH = REPOSITORY_ROOT / "src" / "rmdm" / "evaluation" / "observation_balance_suite.py"
SUITE_SPEC = importlib.util.spec_from_file_location("observation_balance_suite", SUITE_MODULE_PATH)
if SUITE_SPEC is None or SUITE_SPEC.loader is None:
    raise RuntimeError(f"cannot load test-suite helpers from {SUITE_MODULE_PATH}")
SUITE_MODULE = importlib.util.module_from_spec(SUITE_SPEC)
SUITE_SPEC.loader.exec_module(SUITE_MODULE)
RESULT_SCHEMA = SUITE_MODULE.RESULT_SCHEMA
balanced_video_ids = SUITE_MODULE.balanced_video_ids
expand_cells = SUITE_MODULE.expand_cells
load_suite = SUITE_MODULE.load_suite
read_json = SUITE_MODULE.read_json
write_json_atomic = SUITE_MODULE.write_json_atomic


def _root(args: argparse.Namespace) -> Path:
    return Path(args.output_root).expanduser().resolve() / args.candidate_id


def prepare(args: argparse.Namespace) -> None:
    suite_path = Path(args.suite).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    suite = load_suite(suite_path)
    manifest = read_json(manifest_path)
    requested_stages = args.stages.split(",") if args.stages else None
    cells = [
        cell
        for cell in expand_cells(suite, stages=requested_stages)
        if cell["variant"] == args.variant
    ]
    if not cells:
        raise ValueError("the selected stages contain no cells for this variant")
    root = _root(args)
    cell_dir = root / "cells"
    result_dir = root / "results"
    for cell in cells:
        cell["video_ids"] = balanced_video_ids(
            manifest,
            cell["subset_stage"],
            cell["videos_per_scene"],
        )
        cell["config"] = str(config_path)
        cell["checkpoint"] = str(checkpoint_path)
        cell["manifest"] = str(manifest_path)
        cell["result"] = str(result_dir / f"{cell['cell_id']}.json")
        write_json_atomic(cell_dir / f"{cell['cell_id']}.json", cell)
    plan = {
        "schema": "rmdm_observation_balance_plan_v1",
        "suite": str(suite_path),
        "suite_id": suite["suite_id"],
        "candidate_id": args.candidate_id,
        "variant": args.variant,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "cell_files": [str(cell_dir / f"{cell['cell_id']}.json") for cell in cells],
    }
    write_json_atomic(root / "plan.json", plan)
    print(json.dumps({"plan": str(root / "plan.json"), "cells": len(cells)}))


def run(args: argparse.Namespace) -> None:
    root = _root(args)
    plan = read_json(root / "plan.json")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    environment["PYTHONPATH"] = f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = 0
    skipped = 0
    for index, cell_file in enumerate(plan["cell_files"]):
        cell = read_json(cell_file)
        output = Path(cell["result"])
        if output.exists() and not args.rerun:
            result = read_json(output)
            if result.get("schema") == RESULT_SCHEMA:
                skipped += 1
                continue
        if len(gpus) != int(cell["world_size"]):
            raise RuntimeError(
                f"cell {cell['cell_id']} requires {cell['world_size']} GPUs, got {len(gpus)}"
            )
        command = [
            args.python,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            str(cell["world_size"]),
            "--mixed_precision",
            "bf16",
            "--main_process_port",
            str(args.base_port + index),
            str(REPOSITORY_ROOT / "scripts" / "evaluate_observation_balance_cell.py"),
            "--cell",
            str(cell_file),
        ]
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
        completed += 1
    print(json.dumps({"completed": completed, "skipped": skipped, "root": str(root)}))


def _collect(root: Path) -> list[dict[str, Any]]:
    plan = read_json(root / "plan.json")
    results = []
    for cell_file in plan["cell_files"]:
        cell = read_json(cell_file)
        output = Path(cell["result"])
        if output.exists():
            result = read_json(output)
            if result.get("schema") == RESULT_SCHEMA:
                results.append(result)
    return results


def _mean(values: Collection[float]) -> float:
    return sum(values) / len(values)


def summarize(args: argparse.Namespace) -> None:
    root = _root(args)
    results = _collect(root)
    rows: list[dict[str, Any]] = []
    for result in results:
        cell = result["cell"]
        for method, value in result["methods"].items():
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "stage": cell["stage"],
                    "variant": cell["variant"],
                    "rate": cell["rate"],
                    "noise_std": cell["noise_std"],
                    "method": method,
                    "full_image_nmse": value["metrics"]["full_image"]["nmse"],
                    "unobserved_free_space_nmse": value["metrics"]["unobserved_free_space"]["nmse"],
                    "observed_point_mse": value["observed_points"]["metrics"]["mse"],
                }
            )
    by_stage: dict[str, dict[str, float]] = {}
    for stage in sorted({row["stage"] for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        by_stage[stage] = {}
        for method in sorted({row["method"] for row in stage_rows}):
            selected = [row for row in stage_rows if row["method"] == method]
            by_stage[stage][f"{method}_mean_unobserved_nmse"] = _mean(
                [row["unobserved_free_space_nmse"] for row in selected]
            )
    summary = {
        "schema": "rmdm_observation_balance_summary_v1",
        "candidate_id": args.candidate_id,
        "completed_result_count": len(results),
        "rows": rows,
        "stage_means": by_stage,
    }
    write_json_atomic(root / "summary.json", summary)
    print(json.dumps({"summary": str(root / "summary.json"), "rows": len(rows)}))


def compare(args: argparse.Namespace) -> None:
    suite = load_suite(args.suite)
    root = _root(args)
    candidate = read_json(root / "summary.json")
    baseline = read_json(Path(args.output_root).expanduser().resolve() / args.baseline_id / "summary.json")
    settings = suite["selection"][args.variant]

    def metrics(summary: dict[str, Any], stage: str, sigma: float, method: str) -> dict[float, float]:
        selected = {
            float(row["rate"]): row["unobserved_free_space_nmse"]
            for row in summary["rows"]
            if row["variant"] == args.variant and row["stage"] == stage
            and row["noise_std"] == sigma and row["method"] == method
        }
        if set(selected) != set(suite["stages"][stage]["rates"]):
            raise ValueError(
                f"{summary['candidate_id']}: incomplete rates for {stage}, sigma={sigma}, {method}"
            )
        return selected

    def relative_improvement(stage: str, sigma: float) -> float:
        candidate_mean = _mean(metrics(candidate, stage, sigma, "no_da").values())
        baseline_mean = _mean(metrics(baseline, stage, sigma, "no_da").values())
        return 1.0 - candidate_mean / baseline_mean

    clean_relative = relative_improvement(settings["clean_stage"], settings["clean_noise_std"])
    noisy_relative_regression = -relative_improvement(
        settings["noisy_stage"], settings["robustness_noise_std"]
    )
    da = metrics(candidate, settings["da_stage"], settings["da_noise_std"], "known_noise_da")
    no_da = metrics(candidate, settings["noisy_stage"], settings["da_noise_std"], "no_da")
    da_improvements = {rate: 1.0 - value / no_da[rate] for rate, value in da.items()}
    mean_da = _mean(da_improvements.values())
    # Inclusive thresholds should tolerate floating-point roundoff.
    tolerance = 1e-12
    gates = {
        "clean": clean_relative >= float(settings["minimum_clean_relative_improvement"]) - tolerance,
        "robustness": noisy_relative_regression <= float(settings["maximum_noisy_relative_regression"]) + tolerance,
        "da_mean": mean_da >= float(settings["minimum_da_relative_improvement"]) - tolerance,
        "da_each_rate": (
            all(value > 0 for value in da_improvements.values())
            if settings["require_da_improvement_each_rate"] else True
        ),
    }
    comparison = {
        "schema": "rmdm_observation_balance_comparison_v1",
        "candidate_id": args.candidate_id,
        "baseline_id": args.baseline_id,
        "variant": args.variant,
        "clean_relative_improvement": clean_relative,
        "noisy_relative_regression": noisy_relative_regression,
        "known_noise_da_relative_improvement": mean_da,
        "known_noise_da_by_rate": da_improvements,
        "gates": gates,
        "passed": all(gates.values()),
    }
    destination = root / f"comparison_to_{args.baseline_id}.json"
    write_json_atomic(destination, comparison)
    print(json.dumps(comparison, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", required=True)
    common.add_argument("--candidate-id", required=True)

    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--suite", required=True)
    prepare_parser.add_argument("--variant", choices=("w1", "w16"), required=True)
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--checkpoint", required=True)
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--stages", default="")
    prepare_parser.set_defaults(function=prepare)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--gpus", required=True)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--base-port", type=int, default=29700)
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.set_defaults(function=run)

    summary_parser = subparsers.add_parser("summarize", parents=[common])
    summary_parser.set_defaults(function=summarize)

    compare_parser = subparsers.add_parser("compare", parents=[common])
    compare_parser.add_argument("--suite", required=True)
    compare_parser.add_argument("--baseline-id", required=True)
    compare_parser.add_argument("--variant", choices=("w1", "w16"), required=True)
    compare_parser.set_defaults(function=compare)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
