"""Fail-closed environment, dependency and NATTEN numerical audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch

from rmdm_hvdit_v4_joint.model.attention import _natten_attention, reference_neighborhood_attention
from rmdm_hvdit_v4_joint.model import build_t1_system, build_w16_system
from rmdm_hvdit_v4_joint.provenance import build_dependency_manifest
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from rmdm_hvdit_v4_joint.training.execution import (
    AUTHORIZED_PIPELINE_GPU_PROFILES,
    parse_physical_gpus,
)

from .common import config_argument, load_arguments


PINNED_NATTEN = "0.21.5+torch290cu128"


def _environment_audit(config, physical_gpus: list[int]) -> dict:
    expected_prefix = Path(config.pipeline.environment_path).expanduser().resolve()
    actual_prefix = Path(sys.prefix).expanduser().resolve()
    if actual_prefix != expected_prefix:
        raise RuntimeError(f"Expected Python environment {expected_prefix}, got {actual_prefix}")
    natten_version = importlib.metadata.version("natten")
    if natten_version != PINNED_NATTEN:
        raise RuntimeError(f"Expected natten {PINNED_NATTEN}, got {natten_version}")
    if torch.__version__ != "2.9.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected torch 2.9.0+cu128/CUDA12.8, got {torch.__version__}/{torch.version.cuda}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if tuple(physical_gpus) not in AUTHORIZED_PIPELINE_GPU_PROFILES:
        raise RuntimeError(f"Environment audit placement {physical_gpus} is not authorized")
    expected_visible = ",".join(map(str, physical_gpus))
    if visible.replace(" ", "") != expected_visible:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {expected_visible}, got {visible!r}"
        )
    expected_devices = len(physical_gpus)
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_devices:
        raise RuntimeError(f"The audited process must see exactly {expected_devices} CUDA devices")
    import natten

    if not bool(natten.HAS_LIBNATTEN):
        raise RuntimeError("Pinned NATTEN wheel does not expose libnatten CUDA kernels")
    return {
        "python_prefix": str(actual_prefix),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "natten": natten_version,
        "natten_backend": "cutlass-fna",
        "has_libnatten": bool(natten.HAS_LIBNATTEN),
        "visible_devices": visible,
        "devices": [torch.cuda.get_device_name(index) for index in range(expected_devices)],
    }


def _natten_case(
    *,
    shape: tuple[int, ...],
    kernel: tuple[int, ...],
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    base = [torch.randn(shape, device=device, dtype=torch.float32) for _ in range(3)]

    def run(operation):
        values = [item.detach().clone().requires_grad_(True) for item in base]
        output = operation(*values)
        loss = output.float().square().mean()
        gradients = torch.autograd.grad(loss, values)
        return output.detach(), [gradient.detach() for gradient in gradients]

    reference, reference_gradients = run(
        lambda q, k, v: reference_neighborhood_attention(q, k, v, kernel)
    )
    actual, actual_gradients = run(lambda q, k, v: _natten_attention(q, k, v, kernel))
    output_error = float((reference - actual).abs().max())
    gradient_error = max(float((left - right).abs().max()) for left, right in zip(reference_gradients, actual_gradients))
    if output_error > 2.0e-5 or gradient_error > 5.0e-5:
        raise RuntimeError(
            f"NATTEN {len(kernel)}D reference mismatch: "
            f"output={output_error:.3e}, gradient={gradient_error:.3e}"
        )
    return {"output_max_abs_error": output_error, "gradient_max_abs_error": gradient_error}


def _natten_numerical_audit() -> dict:
    return {
        "na2d": _natten_case(shape=(1, 4, 4, 2, 8), kernel=(3, 3), seed=17),
        "na3d": _natten_case(shape=(1, 3, 4, 4, 2, 8), kernel=(3, 3, 3), seed=18),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-gpus", required=True)
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    physical_gpus = parse_physical_gpus(args.execution_gpus)
    environment = _environment_audit(config, physical_gpus)
    dependency = build_dependency_manifest(config, config_path=config_path, repository_root=root)
    t1 = build_t1_system(config, attention_backend="reference")
    w16 = build_w16_system(config, attention_backend="reference")
    counts = {
        "t1": sum(parameter.numel() for parameter in t1.parameters()),
        "w16": sum(parameter.numel() for parameter in w16.parameters()),
    }
    for name, count in counts.items():
        if not config.model.expected_trainable_parameters_min <= count <= config.model.expected_trainable_parameters_max:
            raise RuntimeError(f"{name} parameter count {count} violates the model contract")
    numerical = _natten_numerical_audit()
    report = {
        "schema": "rmdm_hvdit_v4_joint_preflight_audit_v1",
        "environment": environment,
        "dependency_manifest_sha256": dependency["manifest_sha256"],
        "parameter_counts": counts,
        "gradient_checkpointing": {
            "t1": config.t1_train.gradient_checkpointing,
            "w16": config.model.gradient_checkpointing,
        },
        "natten_numerical": numerical,
        "passed": True,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
