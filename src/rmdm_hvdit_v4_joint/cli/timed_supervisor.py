"""Run V4 under a hard deadline and clean every owned descendant process."""

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

from rmdm_hvdit_v4_joint.training.engine import write_json_atomic

from .common import config_argument, load_arguments


def _iso(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


def _parse_deadline(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("wall-clock deadline must include an explicit UTC offset")
    return parsed


def _process_group_pids(process_group: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if len(fields) > 2 and int(fields[2]) == process_group:
                members.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return sorted(members)


def _process_table() -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if len(fields) <= 19:
                continue
            pid = int(entry.name)
            table[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "process_group": int(fields[2]),
                "session": int(fields[3]),
                # Linux /proc stat field 22, stable for the lifetime of a PID.
                "start_time_ticks": int(fields[19]),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return table


def _descendant_processes(root_pid: int) -> list[dict[str, int]]:
    table = _process_table()
    if root_pid not in table:
        return []
    children: dict[int, list[int]] = {}
    for pid, info in table.items():
        children.setdefault(info["ppid"], []).append(pid)
    result: list[dict[str, int]] = []
    stack = [(root_pid, 0)]
    while stack:
        pid, depth = stack.pop()
        info = table.get(pid)
        if info is None:
            continue
        result.append({**info, "depth": depth})
        stack.extend((child, depth + 1) for child in children.get(pid, ()))
    return sorted(result, key=lambda item: (item["depth"], item["pid"]))


def _remember_processes(known: dict[int, dict[str, int]], processes: list[dict[str, int]]) -> None:
    for info in processes:
        known[info["pid"]] = info


def _alive_owned_processes(known: dict[int, dict[str, int]]) -> list[dict[str, int]]:
    table = _process_table()
    alive = []
    for pid, expected in known.items():
        current = table.get(pid)
        if current is not None and current["start_time_ticks"] == expected["start_time_ticks"]:
            alive.append({**current, "depth": expected.get("depth", 0)})
    return sorted(alive, key=lambda item: (item["depth"], item["pid"]))


def _gpu_snapshot(indices: list[int]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(map(str, indices)),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, memory, utilization, temperature = [int(part.strip()) for part in line.split(",")]
        rows.append(
            {
                "index": index,
                "memory_used_mib": memory,
                "utilization_percent": utilization,
                "temperature_c": temperature,
            }
        )
    return rows


def _stop_owned_processes(
    process: subprocess.Popen[Any],
    process_group: int,
    known: dict[int, dict[str, int]],
    *,
    interrupt_grace_seconds: int = 60,
    terminate_grace_seconds: int = 30,
    kill_grace_seconds: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Gracefully stop the launcher, then hard-stop identity-checked descendants."""

    _remember_processes(known, _descendant_processes(process.pid))
    attempts: list[dict[str, Any]] = []
    alive = _alive_owned_processes(known)
    if alive:
        attempts.append({
            "signal": "SIGINT",
            "target": "outer_process_group",
            "sent_at": _iso(datetime.now().astimezone()),
            "owned_before": alive,
            "grace_seconds": interrupt_grace_seconds,
        })
        try:
            os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            pass
        end = time.monotonic() + interrupt_grace_seconds
        while time.monotonic() < end:
            process.poll()
            _remember_processes(known, _descendant_processes(process.pid))
            if not _alive_owned_processes(known):
                break
            time.sleep(1)

    for stop_signal, grace_seconds in (
        (signal.SIGTERM, terminate_grace_seconds),
        (signal.SIGKILL, kill_grace_seconds),
    ):
        alive = _alive_owned_processes(known)
        if not alive:
            break
        attempts.append({
            "signal": signal.Signals(stop_signal).name,
            "target": "identity_checked_owned_pids",
            "sent_at": _iso(datetime.now().astimezone()),
            "owned_before": alive,
            "grace_seconds": grace_seconds,
        })
        # Children first prevents a killed launcher from orphaning an untracked worker.
        for info in reversed(alive):
            try:
                os.kill(info["pid"], stop_signal)
            except ProcessLookupError:
                continue
        end = time.monotonic() + grace_seconds
        while time.monotonic() < end:
            process.poll()
            if not _alive_owned_processes(known):
                break
            time.sleep(1)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    return attempts, _alive_owned_processes(known)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--wall-clock-deadline", required=True)
    parser.add_argument("--through", choices=("w16_validation", "formal_test"), default="w16_validation")
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    if config.pipeline.allowed_physical_gpus != [4, 5, 6, 7]:
        raise RuntimeError("timed supervisor is authorized only for physical GPUs 4-7")
    if args.duration_hours <= 0:
        raise ValueError("duration-hours must be positive")

    start = datetime.now().astimezone()
    duration_deadline = start + timedelta(hours=args.duration_hours)
    wall_clock_deadline = _parse_deadline(args.wall_clock_deadline).astimezone()
    deadline = min(duration_deadline, wall_clock_deadline)
    if deadline <= start:
        raise ValueError("resolved deadline is not in the future")

    output = Path(config.pipeline.output_root).expanduser().resolve()
    state_path = output / "timed_supervisor.json"
    log_path = output / "logs" / "timed_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    python = Path(config.pipeline.environment_path).expanduser().resolve() / "bin" / "python"
    command = [
        str(python),
        "-m",
        "rmdm_hvdit_v4_joint.cli.run_pipeline",
        "--config",
        str(config_path),
        "--repository-root",
        str(root),
        "--through",
        args.through,
    ]
    environment = dict(os.environ)
    source = str(root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    with log_path.open("a", encoding="utf-8") as log:
        log.write("TIMED_COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
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
    process_group = process.pid
    state: dict[str, Any] = {
        "schema": "rmdm_hvdit_v4_joint_timed_supervisor_v1",
        "status": "running",
        "supervisor_pid": os.getpid(),
        "pipeline_pid": process.pid,
        "pipeline_process_group": process_group,
        "command": command,
        "authorized_physical_gpus": config.pipeline.allowed_physical_gpus,
        "started_at": _iso(start),
        "duration_hours": args.duration_hours,
        "duration_deadline": _iso(duration_deadline),
        "wall_clock_deadline": _iso(wall_clock_deadline),
        "resolved_deadline": _iso(deadline),
        "log": str(log_path),
        "gpu_at_start": _gpu_snapshot(config.pipeline.allowed_physical_gpus),
    }
    write_json_atomic(state_path, state)

    received_signal: list[int] = []

    def request_stop(signum: int, _frame: Any) -> None:
        received_signal.append(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    timed_out = False
    last_heartbeat = 0.0
    known_processes: dict[int, dict[str, int]] = {}
    while process.poll() is None:
        now = datetime.now().astimezone()
        if received_signal or now >= deadline:
            timed_out = not received_signal and now >= deadline
            break
        if time.monotonic() - last_heartbeat >= 30:
            descendants = _descendant_processes(process.pid)
            _remember_processes(known_processes, descendants)
            state["heartbeat_at"] = _iso(now)
            state["pipeline_process_group_members"] = _process_group_pids(process_group)
            state["owned_descendant_processes"] = descendants
            state["gpu"] = _gpu_snapshot(config.pipeline.allowed_physical_gpus)
            write_json_atomic(state_path, state)
            last_heartbeat = time.monotonic()
        time.sleep(2)

    _remember_processes(known_processes, _descendant_processes(process.pid))
    cleanup_attempts: list[dict[str, Any]] = []
    remaining = _alive_owned_processes(known_processes)
    if process.poll() is None or remaining:
        state["status"] = "stopping_at_deadline" if timed_out else "stopping_on_signal"
        state["stop_started_at"] = _iso(datetime.now().astimezone())
        write_json_atomic(state_path, state)
        cleanup_attempts, remaining = _stop_owned_processes(process, process_group, known_processes)
    returncode = process.poll()
    remaining = _alive_owned_processes(known_processes)
    state.update(
        {
            "status": (
                "stopped_at_deadline"
                if timed_out and not remaining
                else "stopped_on_signal"
                if received_signal and not remaining
                else "completed_early"
                if returncode == 0 and not remaining
                else "failed_early"
                if not remaining
                else "cleanup_failed"
            ),
            "finished_at": _iso(datetime.now().astimezone()),
            "pipeline_returncode": returncode,
            "received_signal": signal.Signals(received_signal[0]).name if received_signal else None,
            "cleanup_attempts": cleanup_attempts,
            "owned_processes_after_cleanup": remaining,
            "gpu_after_cleanup": _gpu_snapshot(config.pipeline.allowed_physical_gpus),
            "cleanup_complete": not remaining,
        }
    )
    write_json_atomic(state_path, state)
    if remaining:
        raise SystemExit(3)
    if not timed_out and not received_signal and returncode not in (0, None):
        raise SystemExit(int(returncode) if isinstance(returncode, int) and returncode > 0 else 1)


if __name__ == "__main__":
    main()
