#!/usr/bin/env python3
"""Alternate W1 training and fixed validation until the branch passes or stops."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from rmdm.evaluation.observation_balance_suite import read_json, write_json_atomic
from rmdm_hvdit_v4_x0_observation_balance.config import load_observation_balance_config
from rmdm_hvdit_v4_x0_observation_balance.decision import fast_comparison, stopping_reason


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def _validation(
    *,
    python: str,
    environment: dict[str, str],
    candidate_id: str,
    stages: str,
    checkpoint: Path,
    model_config: Path,
    suite: Path,
    manifest: Path,
    output_root: Path,
    gpus: list[int],
    base_port: int,
    baseline_id: str,
    compare: bool,
) -> dict[str, Any]:
    script = str(REPOSITORY_ROOT / "scripts" / "run_observation_balance_validation.py")
    common = ["--candidate-id", candidate_id, "--output-root", str(output_root)]
    _run(
        [
            python, script, "prepare", *common,
            "--suite", str(suite), "--variant", "w1",
            "--config", str(model_config), "--checkpoint", str(checkpoint),
            "--manifest", str(manifest), "--stages", stages,
        ],
        environment=environment,
    )
    _run(
        [
            python, script, "run", *common,
            "--gpus", ",".join(str(value) for value in gpus),
            "--python", python, "--base-port", str(base_port),
        ],
        environment=environment,
    )
    _run([python, script, "summarize", *common], environment=environment)
    if compare:
        _run(
            [
                python, script, "compare", *common,
                "--suite", str(suite), "--variant", "w1",
                "--baseline-id", baseline_id,
            ],
            environment=environment,
        )
        return read_json(output_root / candidate_id / f"comparison_to_{baseline_id}.json")
    return read_json(output_root / candidate_id / "summary.json")


def _initial_state() -> dict[str, Any]:
    return {
        "state": "training",
        "branch_step": 0,
        "latest_checkpoint": "",
        "fast_history": [],
        "best_robust_clean_improvement": None,
        "plateau_count": 0,
        "last_full_clean_improvement": None,
        "full_tested_candidates": [],
        "selected_checkpoint": "",
        "reason": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--python", default="")
    parser.add_argument("--gpus", default="")
    args = parser.parse_args()

    root = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_observation_balance_config(config_path)
    evaluation = config.evaluation
    python = args.python or evaluation.python
    gpus = (
        [int(value) for value in args.gpus.split(",") if value.strip()]
        if args.gpus
        else list(evaluation.gpus)
    )
    if len(gpus) != len(evaluation.gpus):
        raise ValueError("the controller GPU count must match evaluation.gpus")

    output = _resolve(root, config.output_dir)
    state_path = output / "controller_state.json"
    state = read_json(state_path) if state_path.exists() else _initial_state()
    if state["state"] in {"complete", "stopped"}:
        print(json.dumps(state, ensure_ascii=False, sort_keys=True))
        return
    model_config = _resolve(root, config.model_config)
    source_checkpoint = _resolve(root, config.source_checkpoint)
    suite = _resolve(root, evaluation.suite)
    manifest = _resolve(root, evaluation.manifest)
    validation_root = _resolve(root, evaluation.output_root)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpus)
    environment["PYTHONPATH"] = f"{root / 'src'}:{root}"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    validate = partial(
        _validation, python=python, environment=environment, model_config=model_config,
        suite=suite, manifest=manifest, output_root=validation_root, gpus=gpus,
        base_port=evaluation.base_port, baseline_id=evaluation.baseline_id,
    )
    baseline_summary = validation_root / evaluation.baseline_id / "summary.json"
    if not baseline_summary.exists():
        validate(
            candidate_id=evaluation.baseline_id,
            stages="fast_w1,full_w1,w1_da_gate",
            checkpoint=source_checkpoint,
            compare=False,
        )
    baseline = read_json(baseline_summary)

    while int(state["branch_step"]) < config.max_steps:
        train_command = [
            python,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            str(len(gpus)),
            "--mixed_precision",
            config.mixed_precision,
            "--main_process_port",
            str(evaluation.base_port - 1),
            str(REPOSITORY_ROOT / "scripts" / "train_w1_observation_balance_segment.py"),
            "--config",
            str(config_path),
            "--repository-root",
            str(root),
        ]
        if state["latest_checkpoint"]:
            train_command.extend(["--resume-from", state["latest_checkpoint"]])
        status_path = output / "status.json"
        training_status = read_json(status_path) if status_path.exists() else {}
        # A finished segment may still be waiting for validation after an interruption.
        if not (training_status.get("checkpoint") and
                int(training_status["branch_step"]) > int(state["branch_step"])):
            _run(train_command, environment=environment)
            training_status = read_json(status_path)
        step = int(training_status["branch_step"])
        checkpoint = Path(training_status["checkpoint"]).resolve()
        candidate_id = f"{config.run_id}_step{step:06d}"

        fast_summary = validate(
            candidate_id=candidate_id, stages="fast_w1", checkpoint=checkpoint, compare=False,
        )
        fast = fast_comparison(fast_summary, baseline, config.fast_decision)
        fast["branch_step"] = step
        fast["candidate_id"] = candidate_id
        state["fast_history"].append(fast)
        state["branch_step"] = step
        state["latest_checkpoint"] = str(checkpoint)

        if fast["robust_enough_for_plateau"]:
            best = state["best_robust_clean_improvement"]
            if best is None or fast["clean_relative_improvement"] >= (
                float(best) + config.fast_decision.plateau_improvement - 1e-12
            ):
                state["best_robust_clean_improvement"] = fast["clean_relative_improvement"]
                state["plateau_count"] = 0
            else:
                state["plateau_count"] = int(state["plateau_count"]) + 1

        last_full = state["last_full_clean_improvement"]
        run_full = bool(fast["eligible"]) and (
            last_full is None
            or fast["clean_relative_improvement"] >= (
                float(last_full) + config.fast_decision.full_test_improvement_increment - 1e-12
            )
            or step >= config.max_steps
        )
        if run_full:
            comparison = validate(
                candidate_id=candidate_id, stages="full_w1,w1_da_gate", checkpoint=checkpoint, compare=True,
            )
            state["last_full_clean_improvement"] = fast["clean_relative_improvement"]
            state["full_tested_candidates"].append(candidate_id)
            if bool(comparison["passed"]):
                state["state"] = "complete"
                state["selected_checkpoint"] = str(checkpoint)
                state["reason"] = "full_w1_passed"
                write_json_atomic(state_path, state)
                print(json.dumps(state, ensure_ascii=False, sort_keys=True))
                return

        reason = stopping_reason(
            step=step,
            max_steps=config.max_steps,
            history=state["fast_history"],
            plateau_count=int(state["plateau_count"]),
            settings=config.fast_decision,
        )
        if reason:
            state["state"] = "stopped"
            state["reason"] = reason
            write_json_atomic(state_path, state)
            print(json.dumps(state, ensure_ascii=False, sort_keys=True))
            return
        write_json_atomic(state_path, state)

    state["state"] = "stopped"
    state["reason"] = "max_steps_without_full_pass"
    write_json_atomic(state_path, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
