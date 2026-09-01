"""Resumable, no-confirmation T1 -> gate -> inflation -> W16 state machine."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from rmdm_hvdit_v4_joint.provenance import build_dependency_manifest, sha256_file
from rmdm_hvdit_v4_joint.evaluation import validate_t1_reference
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_joint.training.execution import (
    AUTHORIZED_PIPELINE_GPU_PROFILES,
)

from .common import config_argument, load_arguments


class _LoggedCommandError(RuntimeError):
    def __init__(self, returncode: int, log_path: Path, current_output: str) -> None:
        super().__init__(f"Command failed with exit code {returncode}; see {log_path}")
        self.returncode = int(returncode)
        self.log_path = log_path
        self.current_output = current_output


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another HV-DiT v4 joint pipeline owns {path}") from error
        yield


def _gpu_snapshot(indices: list[int]) -> tuple[bool, dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    free = {}
    for line in result.stdout.splitlines():
        index, memory = [part.strip() for part in line.split(",", 1)]
        free[int(index)] = int(memory)
    pmon = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-i", ",".join(map(str, indices))],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active = []
    for line in pmon.stdout.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 2:
            continue
        if fields[1] != "-":
            active.append({"gpu": int(fields[0]), "pid": int(fields[1])})
    return not active, {"free_memory_mib": {str(index): free.get(index, -1) for index in indices}, "active": active}


def _wait_for_resources(
    config,
    state_path: Path,
    state: dict[str, Any],
    physical_gpus: list[int],
    *,
    allow_gpu_co_tenancy: bool,
) -> None:
    required = config.pipeline.free_memory_mib
    consecutive = 0
    while True:
        no_processes, snapshot = _gpu_snapshot(physical_gpus)
        enough_memory = all(value >= required for value in snapshot["free_memory_mib"].values())
        placement_available = enough_memory and (
            no_processes or allow_gpu_co_tenancy
        )
        consecutive = consecutive + 1 if placement_available else 0
        state["resource_wait"] = {
            "consecutive_free_polls": consecutive,
            "physical_gpus": physical_gpus,
            "co_tenancy_allowed": bool(allow_gpu_co_tenancy),
            "exclusive": bool(no_processes),
            **snapshot,
        }
        write_json_atomic(state_path, state)
        if consecutive >= config.pipeline.consecutive_free_polls:
            return
        if not config.pipeline.wait_for_gpus:
            raise RuntimeError(f"Authorized GPUs are not free: {snapshot}")
        time.sleep(config.pipeline.gpu_poll_seconds)


def _environment(config, root: Path, *, visible: str) -> dict[str, str]:
    environment = dict(os.environ)
    source = str(root / "src")
    environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["CUDA_VISIBLE_DEVICES"] = visible
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return environment


def _run_logged(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    current_start = log_path.stat().st_size if log_path.is_file() else 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        with log_path.open("rb") as log:
            log.seek(current_start)
            current_output = log.read().decode("utf-8", errors="replace")
        raise _LoggedCommandError(result.returncode, log_path, current_output)


def _is_cuda_oom(error: _LoggedCommandError) -> bool:
    output = error.current_output.lower()
    return any(
        marker in output
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "torch.outofmemoryerror",
            "cublas_status_alloc_failed",
        )
    )


def _mark(state_path: Path, state: dict[str, Any], phase: str, status: str, **details: Any) -> None:
    state.setdefault("phases", {})[phase] = {"status": status, **details}
    state["current_phase"] = phase
    write_json_atomic(state_path, state)


def _complete(state: dict[str, Any], phase: str) -> bool:
    return state.get("phases", {}).get(phase, {}).get("status") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    os.chdir(root)
    physical_gpus = list(config.pipeline.allowed_physical_gpus)
    if tuple(physical_gpus) not in AUTHORIZED_PIPELINE_GPU_PROFILES:
        raise ValueError(f"Pipeline placement {physical_gpus} is not an authorized execution profile")
    allow_gpu_co_tenancy = bool(config.pipeline.allow_gpu_co_tenancy)
    through = config.pipeline.default_through
    output = Path(config.pipeline.output_root).expanduser().resolve()
    state_path = output / "pipeline_state.json"
    lock_path = Path(config.pipeline.lock_file).expanduser().resolve()
    python = Path(config.pipeline.environment_path).expanduser().resolve() / "bin" / "python"
    accelerate = Path(config.pipeline.environment_path).expanduser().resolve() / "bin" / "accelerate"
    if not python.is_file() or not accelerate.is_file():
        raise FileNotFoundError("The independent HV-DiT v4 joint environment is not installed")
    config_hash = sha256_file(config_path)
    dependency_hash = build_dependency_manifest(
        config,
        config_path=config_path,
        repository_root=root,
    )["manifest_sha256"]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_hash:
            raise ValueError("Existing pipeline state belongs to a different config")
        if state.get("dependency_manifest_sha256") != dependency_hash:
            raise ValueError("Existing pipeline state was created from different code or dependencies")
        if state.get("through") != through:
            raise ValueError("Pipeline scope cannot be expanded from validation to formal test after launch")
    else:
        state = {
            "schema": "rmdm_hvdit_v4_joint_pipeline_state_v1",
            "config": str(config_path),
            "config_sha256": config_hash,
            "dependency_manifest_sha256": dependency_hash,
            "through": through,
            "authorized_physical_gpus": config.pipeline.allowed_physical_gpus,
            "phases": {},
        }
        write_json_atomic(state_path, state)

    if state.get("state") == "completed":
        print(json.dumps({"state": "completed", "pipeline_state": str(state_path)}), flush=True)
        return
    visible_all = ",".join(map(str, physical_gpus))
    env_all = _environment(config, root, visible=visible_all)
    audit_phase = "environment_audit"
    smoke_phase = "cuda_smoke"
    with _exclusive_lock(lock_path):
        try:
            state["current_phase"] = "resource_wait"
            write_json_atomic(state_path, state)
            state.setdefault("execution_placements", []).append(
                {
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "physical_gpus": physical_gpus,
                    "allow_gpu_co_tenancy": allow_gpu_co_tenancy,
                    "effective_t1_global_batch_size": config.t1_train.effective_global_batch_size,
                }
            )
            write_json_atomic(state_path, state)
            _wait_for_resources(
                config,
                state_path,
                state,
                physical_gpus,
                allow_gpu_co_tenancy=allow_gpu_co_tenancy,
            )

            if not _complete(state, audit_phase):
                _mark(state_path, state, audit_phase, "running")
                audit_path = output / "audit" / "preflight.json"
                _run_logged(
                    [
                        str(python),
                        "-m",
                        "rmdm_hvdit_v4_joint.cli.audit",
                        "--config",
                        str(config_path),
                        "--repository-root",
                        str(root),
                        "--output",
                        str(audit_path),
                        "--execution-gpus",
                        visible_all,
                    ],
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "environment_audit.log",
                )
                _mark(state_path, state, audit_phase, "completed", artifact=str(audit_path))

            reference_path = Path(config.evaluation.t1_reference_summary).expanduser().resolve()
            if not _complete(state, "sf_reference"):
                _mark(state_path, state, "sf_reference", "running")
                if not reference_path.is_file():
                    reference_path.parent.mkdir(parents=True, exist_ok=True)
                    reference_command = [
                        str(accelerate),
                        "launch",
                        "--main_process_port",
                        "0",
                        "--num_processes",
                        str(len(physical_gpus)),
                        "--multi_gpu",
                        "--mixed_precision",
                        "bf16",
                        str(root / "evaluate_sparse_dynamic_rmdm.py"),
                        "--checkpoint",
                        str(Path(config.evaluation.sf_reference_checkpoint).expanduser().resolve()),
                        "--manifest",
                        str(Path(config.evaluation.sf_mask_manifest).expanduser().resolve()),
                        "--split",
                        "val",
                        "--rates",
                        "1,2,3",
                        "--ddim_steps",
                        "20",
                        "--batch_size",
                        str(config.evaluation.sf_reference_per_gpu_batch_size),
                        "--workers",
                        "4",
                        "--seed",
                        str(config.sampling.seed),
                        "--subset_manifest",
                        str(Path(config.evaluation.subset_manifest).expanduser().resolve()),
                        "--subset_stage",
                        "stage_a",
                        "--output_dir",
                        str(reference_path.parent),
                        "--mixed_precision",
                        "bf16",
                    ]
                    if (reference_path.parent / "run_config.json").is_file():
                        reference_command.append("--resume")
                    _run_logged(
                        reference_command,
                        root=root,
                        environment=env_all,
                        log_path=output / "logs" / "sf_reference.log",
                    )
                if not reference_path.is_file():
                    raise RuntimeError(f"Fixed-SF reference was not produced: {reference_path}")
                reference_validation = validate_t1_reference(config, reference_path)
                _mark(
                    state_path,
                    state,
                    "sf_reference",
                    "completed",
                    artifact=str(reference_path),
                    **reference_validation,
                )

            if not _complete(state, smoke_phase):
                _mark(state_path, state, smoke_phase, "running")
                smoke_root = output / "smoke"
                env_one = _environment(config, root, visible=str(physical_gpus[0]))
                _run_logged(
                    [
                        str(python), "-m", "rmdm_hvdit_v4_joint.cli.smoke",
                        "--config", str(config_path), "--repository-root", str(root),
                        "--variant", "t1", "--batch-size", str(config.t1_train.per_gpu_batch_size),
                        "--physical-gpu", str(physical_gpus[0]),
                        "--output", str(smoke_root / "t1.json"),
                    ],
                    root=root,
                    environment=env_one,
                    log_path=output / "logs" / "smoke_t1.log",
                )
                successes_by_gpu: dict[str, list[int]] = {}
                failures: list[dict[str, Any]] = []
                for physical_gpu in physical_gpus:
                    gpu_successes: list[int] = []
                    env_gpu = _environment(config, root, visible=str(physical_gpu))
                    for candidate in sorted(config.w16_train.microbatch_candidates):
                        try:
                            _run_logged(
                                [
                                    str(python), "-m", "rmdm_hvdit_v4_joint.cli.smoke",
                                    "--config", str(config_path), "--repository-root", str(root),
                                    "--variant", "w16", "--batch-size", str(candidate),
                                    "--physical-gpu", str(physical_gpu),
                                    "--output", str(smoke_root / f"w16_gpu{physical_gpu}_b{candidate}.json"),
                                ],
                                root=root,
                                environment=env_gpu,
                                log_path=output / "logs" / f"smoke_w16_gpu{physical_gpu}_b{candidate}.log",
                            )
                            gpu_successes.append(candidate)
                        except _LoggedCommandError as error:
                            if not _is_cuda_oom(error):
                                raise
                            failures.append(
                                {
                                    "physical_gpu": physical_gpu,
                                    "batch_size": candidate,
                                    "error": str(error),
                                }
                            )
                            break
                    if not gpu_successes:
                        raise RuntimeError(f"W16 CUDA smoke failed at microbatch=1 on physical GPU {physical_gpu}")
                    successes_by_gpu[str(physical_gpu)] = gpu_successes
                selected_microbatch = min(max(values) for values in successes_by_gpu.values())
                resolution = {
                    "schema": "rmdm_hvdit_v4_joint_smoke_resolution_v1",
                    "physical_gpus": physical_gpus,
                    "t1_batch_size": config.t1_train.per_gpu_batch_size,
                    "successful_microbatches_by_gpu": successes_by_gpu,
                    "failures": failures,
                    "selected_microbatch": selected_microbatch,
                    "gradient_accumulation_steps": config.w16_train.effective_global_batch_size
                    // (len(physical_gpus) * selected_microbatch),
                }
                write_json_atomic(smoke_root / "resolution.json", resolution)
                _mark(state_path, state, smoke_phase, "completed", **resolution)

            if not _complete(state, "t1_training_and_gate"):
                prior_t1_phase = dict(state.get("phases", {}).get("t1_training_and_gate", {}))
                _mark(state_path, state, "t1_training_and_gate", "running")
                t1_command = [
                    str(accelerate), "launch", "--main_process_port", "0", "--num_processes", str(len(physical_gpus)), "--multi_gpu", "--mixed_precision", "bf16",
                    "-m", "rmdm_hvdit_v4_joint.cli.train_t1",
                    "--config", str(config_path), "--repository-root", str(root),
                    "--execution-gpus", visible_all,
                    "--gradient-accumulation-steps", str(
                        config.t1_train.effective_global_batch_size
                        // (len(physical_gpus) * config.t1_train.per_gpu_batch_size)
                    ),
                ]
                t1_last = output / "t1_pretrain" / "checkpoints" / "last.pth"
                declared_resume = prior_t1_phase.get("resume_checkpoint")
                resume_checkpoint = Path(declared_resume).expanduser().resolve() if declared_resume else t1_last
                if resume_checkpoint.is_file():
                    t1_command.extend(("--resume-from", str(resume_checkpoint)))
                _run_logged(
                    t1_command,
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "t1_training.log",
                )
                gate_path = output / "t1_pretrain" / "gate.json"
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                if not gate.get("passed"):
                    raise RuntimeError("T1 gate failed; automatic inflation is forbidden")
                best_t1 = output / "t1_pretrain" / "checkpoints" / "best.pth"
                _mark(
                    state_path,
                    state,
                    "t1_training_and_gate",
                    "completed",
                    best_checkpoint=str(best_t1),
                    gate=str(gate_path),
                    gate_sha256=sha256_file(gate_path),
                )

            if not _complete(state, "w16_inflation_audit"):
                _mark(state_path, state, "w16_inflation_audit", "running")
                best_t1 = Path(state["phases"]["t1_training_and_gate"]["best_checkpoint"])
                initialization = output / "w16_init" / "from_t1_best.pth"
                _run_logged(
                    [
                        str(python), "-m", "rmdm_hvdit_v4_joint.cli.inflate",
                        "--config", str(config_path), "--repository-root", str(root),
                        "--t1-checkpoint", str(best_t1), "--output", str(initialization),
                    ],
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "w16_inflation.log",
                )
                _mark(state_path, state, "w16_inflation_audit", "completed", initialization=str(initialization), sha256=sha256_file(initialization))

            if not _complete(state, "w16_training_and_validation"):
                _mark(state_path, state, "w16_training_and_validation", "running")
                initialization = state["phases"]["w16_inflation_audit"]["initialization"]
                microbatch = int(state["phases"][smoke_phase]["selected_microbatch"])
                w16_command = [
                    str(accelerate), "launch", "--main_process_port", "0", "--num_processes", str(len(physical_gpus)), "--multi_gpu", "--mixed_precision", "bf16",
                    "-m", "rmdm_hvdit_v4_joint.cli.train_w16",
                    "--config", str(config_path), "--repository-root", str(root),
                    "--initialization", str(initialization), "--per-gpu-batch-size", str(microbatch),
                    "--execution-gpus", visible_all,
                ]
                w16_last = output / "w16_train" / "checkpoints" / "last.pth"
                if w16_last.is_file():
                    w16_command.extend(("--resume-from", str(w16_last)))
                _run_logged(
                    w16_command,
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "w16_training.log",
                )
                selection = output / "w16_train" / "selection.json"
                _mark(
                    state_path,
                    state,
                    "w16_training_and_validation",
                    "completed",
                    selection=str(selection),
                    selection_sha256=sha256_file(selection),
                )

            if not _complete(state, "architecture_comparison"):
                _mark(state_path, state, "architecture_comparison", "running")
                selection = state["phases"]["w16_training_and_validation"]["selection"]
                if sha256_file(selection) != state["phases"]["w16_training_and_validation"]["selection_sha256"]:
                    raise RuntimeError("Frozen W16 selection JSON changed before architecture comparison")
                comparison = output / "comparison" / "factorized_vs_hvdit_v4_joint.json"
                _run_logged(
                    [
                        str(python), "-m", "rmdm_hvdit_v4_joint.cli.compare",
                        "--config", str(config_path), "--repository-root", str(root),
                        "--selection", str(selection), "--output", str(comparison),
                    ],
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "architecture_comparison.log",
                )
                _mark(state_path, state, "architecture_comparison", "completed", artifact=str(comparison), sha256=sha256_file(comparison))

            if through == "formal_test" and not _complete(state, "formal_test"):
                _mark(state_path, state, "formal_test", "running")
                selection_path = Path(state["phases"]["w16_training_and_validation"]["selection"])
                if sha256_file(selection_path) != state["phases"]["w16_training_and_validation"]["selection_sha256"]:
                    raise RuntimeError("Frozen W16 selection JSON changed after validation")
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected_checkpoint = selection["selected"]["checkpoint"]
                if sha256_file(selected_checkpoint) != selection["selected"]["checkpoint_sha256"]:
                    raise RuntimeError("Frozen W16 selected checkpoint changed after validation")
                formal_output = output / "formal_test" / "full100.json"
                _run_logged(
                    [
                        str(accelerate), "launch", "--main_process_port", "0", "--num_processes", str(len(physical_gpus)), "--multi_gpu", "--mixed_precision", "bf16",
                        "-m", "rmdm_hvdit_v4_joint.cli.evaluate",
                        "--config", str(config_path), "--repository-root", str(root),
                        "--variant", "w16", "--checkpoint", str(selected_checkpoint),
                        "--subset-stage", "all", "--split", "test",
                        "--manifest", str(Path(config.evaluation.formal_test_manifest).expanduser().resolve()),
                        "--rates", ",".join(f"{rate:g}" for rate in config.evaluation.formal_test_rates),
                        "--full100", "--output", str(formal_output),
                        "--execution-gpus", visible_all,
                    ],
                    root=root,
                    environment=env_all,
                    log_path=output / "logs" / "formal_test.log",
                )
                _mark(state_path, state, "formal_test", "completed", artifact=str(formal_output), sha256=sha256_file(formal_output))

            state["state"] = "completed"
            state["current_phase"] = None
            write_json_atomic(state_path, state)
        except BaseException as error:
            phase = state.get("current_phase") or "resource_wait"
            _mark(state_path, state, phase, "failed", error=repr(error))
            state["state"] = "failed"
            write_json_atomic(state_path, state)
            raise


if __name__ == "__main__":
    main()
