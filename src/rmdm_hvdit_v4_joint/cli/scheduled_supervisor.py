"""Run the two commands declared by the 8-GPU-to-4-GPU schedule YAML."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from rmdm_hvdit_v4_joint.config import load_config
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_joint.training.execution import (
    AUTHORIZED_PIPELINE_GPU_PROFILES,
    AUTHORIZED_T1_GPU_PROFILES,
    parse_wall_clock,
)

from .timed_supervisor import (
    _alive_owned_processes,
    _descendant_processes,
    _gpu_snapshot,
    _iso,
    _remember_processes,
    _stop_owned_processes,
)


def _load_schedule(path: Path, root: Path) -> tuple[dict[str, Any], Path, Any, datetime]:
    schedule = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "experiment_config",
        "switch_at",
        "pre_switch",
        "post_switch",
        "max_automatic_restarts",
        "restart_backoff_seconds",
    }
    if not isinstance(schedule, dict) or set(schedule) != expected:
        raise ValueError(f"Execution schedule keys must be exactly {sorted(expected)}")
    if schedule["schema"] != "rmdm_hvdit_v4_joint_execution_schedule_v1":
        raise ValueError("Unsupported execution schedule schema")
    config_path = (root / schedule["experiment_config"]).resolve()
    config = load_config(config_path)
    pre = schedule["pre_switch"]
    post = schedule["post_switch"]
    if set(pre) != {
        "physical_gpus",
        "gradient_accumulation_steps",
        "pause_before_validation_seconds",
        "graceful_pause_lead_seconds",
    }:
        raise ValueError("Invalid pre_switch execution profile")
    if set(post) != {"physical_gpus"}:
        raise ValueError("Invalid post_switch execution profile")
    if tuple(pre["physical_gpus"]) not in AUTHORIZED_T1_GPU_PROFILES:
        raise ValueError("The pre-switch T1 GPU profile is not authorized")
    if tuple(post["physical_gpus"]) not in AUTHORIZED_PIPELINE_GPU_PROFILES:
        raise ValueError("The post-switch pipeline GPU profile is not authorized")
    if list(post["physical_gpus"]) != config.pipeline.allowed_physical_gpus:
        raise ValueError("The post-switch profile must equal the canonical experiment config placement")
    represented = (
        len(pre["physical_gpus"])
        * config.t1_train.per_gpu_batch_size
        * int(pre["gradient_accumulation_steps"])
    )
    if represented != config.t1_train.effective_global_batch_size:
        raise ValueError("The pre-switch profile changes T1 effective global batch size")
    if int(pre["pause_before_validation_seconds"]) < 0:
        raise ValueError("pause_before_validation_seconds cannot be negative")
    if int(pre["graceful_pause_lead_seconds"]) < 60:
        raise ValueError("graceful_pause_lead_seconds must leave at least 60 seconds for checkpointing")
    if int(schedule["max_automatic_restarts"]) < 1 or int(schedule["restart_backoff_seconds"]) < 1:
        raise ValueError("The automatic restart policy must be positive")
    deadline = parse_wall_clock(str(schedule["switch_at"]))
    assert deadline is not None
    return schedule, config_path, config, deadline


def _environment(root: Path, visible: list[int] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if visible is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible))
    return environment


def _safe_gpu_snapshot(indices: list[int]) -> dict[str, Any]:
    try:
        return {"rows": _gpu_snapshot(indices), "error": None}
    except BaseException as error:
        # Telemetry is advisory and must not terminate training.
        return {"rows": [], "error": repr(error)}


def _terminal_t1_gate_failure(returncode: int | None, status: dict[str, Any]) -> bool:
    """A scientific gate rejection is final, not a retryable infrastructure fault."""

    return returncode not in (None, 0) and status.get("state") == "gate_failed"


def _pre_command(schedule: dict[str, Any], config_path: Path, config: Any, root: Path) -> tuple[list[str], dict[str, str]]:
    profile = schedule["pre_switch"]
    gpus = list(profile["physical_gpus"])
    switch_at = parse_wall_clock(str(schedule["switch_at"]))
    assert switch_at is not None
    stop_at = switch_at - timedelta(seconds=int(profile["graceful_pause_lead_seconds"]))
    environment_path = Path(config.pipeline.environment_path).expanduser().resolve()
    command = [
        str(environment_path / "bin" / "accelerate"),
        "launch", "--main_process_port", "0", "--num_processes", str(len(gpus)),
        "--multi_gpu", "--mixed_precision", "bf16",
        "-m", "rmdm_hvdit_v4_joint.cli.train_t1",
        "--config", str(config_path), "--repository-root", str(root),
        "--execution-gpus", ",".join(map(str, gpus)),
        "--gradient-accumulation-steps", str(profile["gradient_accumulation_steps"]),
        "--stop-at", stop_at.isoformat(timespec="seconds"),
        "--pause-before-validation-seconds", str(profile["pause_before_validation_seconds"]),
    ]
    last = Path(config.pipeline.output_root).expanduser().resolve() / "t1_pretrain" / "checkpoints" / "last.pth"
    if last.is_file():
        command.extend(("--resume-from", str(last)))
    return command, _environment(root, gpus)


def _post_command(config_path: Path, config: Any, root: Path) -> tuple[list[str], dict[str, str]]:
    python = Path(config.pipeline.environment_path).expanduser().resolve() / "bin" / "python"
    return (
        [
            str(python), "-m", "rmdm_hvdit_v4_joint.cli.run_pipeline",
            "--config", str(config_path), "--repository-root", str(root),
        ],
        _environment(root),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    schedule_path = Path(args.schedule).expanduser().resolve()
    schedule, config_path, config, deadline = _load_schedule(schedule_path, root)
    output = Path(config.pipeline.output_root).expanduser().resolve()
    state_path = output / "scheduled_supervisor.json"
    pipeline_state_path = output / "pipeline_state.json"
    t1_status_path = output / "t1_pretrain" / "status.json"
    log_path = output / "logs" / "scheduled_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "schema": "rmdm_hvdit_v4_joint_scheduled_supervisor_v1",
        "status": "starting",
        "supervisor_pid": os.getpid(),
        "schedule": str(schedule_path),
        "experiment_config": str(config_path),
        "switch_at": deadline.isoformat(timespec="seconds"),
        "started_at": _iso(datetime.now().astimezone()),
        "launches": [],
    }
    write_json_atomic(state_path, state)
    received_signal: list[int] = []

    def request_stop(signum: int, _frame: Any) -> None:
        received_signal.append(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    phase = "pre_switch" if datetime.now().astimezone() < deadline else "post_switch"
    consecutive_failures = 0

    while not received_signal:
        if phase == "pre_switch":
            command, environment = _pre_command(schedule, config_path, config, root)
            physical_gpus = list(schedule["pre_switch"]["physical_gpus"])
        else:
            command, environment = _post_command(config_path, config, root)
            physical_gpus = list(schedule["post_switch"]["physical_gpus"])
        with log_path.open("a", encoding="utf-8") as log:
            log.write("SUPERVISED_COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launch = {
            "phase": phase,
            "pid": process.pid,
            "process_group": process.pid,
            "physical_gpus": physical_gpus,
            "started_at": _iso(datetime.now().astimezone()),
            "command": command,
        }
        state["launches"].append(launch)
        state["active_launch"] = launch
        state["status"] = f"running_{phase}"
        write_json_atomic(state_path, state)
        known: dict[int, dict[str, int]] = {}
        started = time.monotonic()
        last_heartbeat = 0.0
        switch_due = False
        while process.poll() is None and not received_signal:
            _remember_processes(known, _descendant_processes(process.pid))
            now = datetime.now().astimezone()
            if phase == "pre_switch" and now >= deadline:
                switch_due = True
                break
            if time.monotonic() - last_heartbeat >= 30:
                state["heartbeat_at"] = _iso(now)
                state["owned_processes"] = _alive_owned_processes(known)
                state["gpu"] = _safe_gpu_snapshot(physical_gpus)
                write_json_atomic(state_path, state)
                last_heartbeat = time.monotonic()
            time.sleep(2)

        _remember_processes(known, _descendant_processes(process.pid))
        remaining = _alive_owned_processes(known)
        cleanup_attempts: list[dict[str, Any]] = []
        if process.poll() is None or remaining:
            cleanup_attempts, remaining = _stop_owned_processes(
                process,
                process.pid,
                known,
                interrupt_grace_seconds=10,
                terminate_grace_seconds=10,
                kill_grace_seconds=5,
            )
        returncode = process.poll()
        state["last_returncode"] = returncode
        state["last_cleanup_attempts"] = cleanup_attempts
        state["owned_processes_after_cleanup"] = remaining
        state["last_finished_at"] = _iso(datetime.now().astimezone())
        write_json_atomic(state_path, state)
        if remaining:
            state["status"] = "cleanup_failed"
            write_json_atomic(state_path, state)
            raise SystemExit(3)
        if received_signal:
            break

        t1_status = json.loads(t1_status_path.read_text(encoding="utf-8")) if t1_status_path.is_file() else {}
        if _terminal_t1_gate_failure(returncode, t1_status):
            state["status"] = "t1_gate_failed"
            state["finished_at"] = _iso(datetime.now().astimezone())
            state["t1_status"] = t1_status
            write_json_atomic(state_path, state)
            return

        if phase == "pre_switch":
            if switch_due or t1_status.get("state") == "paused":
                while datetime.now().astimezone() < deadline and not received_signal:
                    state["status"] = "waiting_for_switch"
                    state["heartbeat_at"] = _iso(datetime.now().astimezone())
                    write_json_atomic(state_path, state)
                    time.sleep(min(2.0, max(0.1, (deadline - datetime.now().astimezone()).total_seconds())))
                phase = "post_switch"
                consecutive_failures = 0
                state["switched_at"] = _iso(datetime.now().astimezone())
                write_json_atomic(state_path, state)
                continue
            if returncode == 0 and t1_status.get("state") == "passed":
                # T1 finished before 08:00; moving early to the already-restricted
                # 4-7 profile avoids leaving the GPUs idle.
                phase = "post_switch"
                consecutive_failures = 0
                state["switched_early_after_t1_completion_at"] = _iso(datetime.now().astimezone())
                write_json_atomic(state_path, state)
                continue
        else:
            pipeline_state = (
                json.loads(pipeline_state_path.read_text(encoding="utf-8"))
                if pipeline_state_path.is_file()
                else {}
            )
            if returncode == 0 and pipeline_state.get("state") == "completed":
                state["status"] = "completed"
                state["finished_at"] = _iso(datetime.now().astimezone())
                write_json_atomic(state_path, state)
                return

        ran_seconds = time.monotonic() - started
        consecutive_failures += 1
        state["consecutive_failures"] = consecutive_failures
        state["last_failed_run_seconds"] = ran_seconds
        if consecutive_failures > int(schedule["max_automatic_restarts"]):
            state["status"] = "retry_budget_exhausted"
            write_json_atomic(state_path, state)
            raise SystemExit(4)
        state["status"] = "retry_backoff"
        write_json_atomic(state_path, state)
        backoff_end = time.monotonic() + int(schedule["restart_backoff_seconds"])
        while time.monotonic() < backoff_end and not received_signal:
            time.sleep(min(2.0, backoff_end - time.monotonic()))

    state["status"] = "stopped_on_signal"
    state["received_signal"] = signal.Signals(received_signal[0]).name if received_signal else None
    state["finished_at"] = _iso(datetime.now().astimezone())
    write_json_atomic(state_path, state)


if __name__ == "__main__":
    main()
