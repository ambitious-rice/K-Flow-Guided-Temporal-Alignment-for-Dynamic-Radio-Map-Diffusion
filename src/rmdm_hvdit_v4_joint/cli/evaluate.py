"""Evaluate an explicit T1 or W16 artifact under the fixed validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration

from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.evaluation import evaluate_stage_a
from rmdm_hvdit_v4_joint.model import build_t1_system, build_w16_system
from rmdm_hvdit_v4_joint.provenance import build_dependency_manifest
from rmdm_hvdit_v4_joint.training.checkpoint import (
    T1_CHECKPOINT_SCHEMA,
    W16_CHECKPOINT_SCHEMA,
    W16_SELECTION_CANDIDATE_SCHEMA,
)
from rmdm_hvdit_v4_joint.training.engine import require_visible_physical_gpus, write_json_atomic
from rmdm_hvdit_v4_joint.training.execution import (
    AUTHORIZED_PIPELINE_GPU_PROFILES,
    parse_physical_gpus,
)

from .common import config_argument, load_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--variant", choices=("t1", "w16"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset-stage", choices=("stage_a", "stage_b_extra", "all"), default="stage_a")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--rates", default="")
    parser.add_argument("--ablate-raw-observations", action="store_true")
    parser.add_argument("--full100", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-gpus", required=True)
    args = parser.parse_args()
    config, config_path, root = load_arguments(args)
    physical_gpus = parse_physical_gpus(args.execution_gpus)
    if tuple(physical_gpus) not in AUTHORIZED_PIPELINE_GPU_PROFILES:
        raise RuntimeError(f"Evaluation placement {physical_gpus} is not authorized")
    require_visible_physical_gpus(physical_gpus)
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    expected_processes = len(physical_gpus)
    if accelerator.num_processes != expected_processes:
        raise RuntimeError(
            f"Evaluation requires {expected_processes} processes on configured physical GPUs, "
            f"got {accelerator.num_processes}"
        )
    model = build_t1_system(config) if args.variant == "t1" else build_w16_system(config)
    payload = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False)
    expected_schemas = (
        {T1_CHECKPOINT_SCHEMA}
        if args.variant == "t1"
        else {W16_CHECKPOINT_SCHEMA, W16_SELECTION_CANDIDATE_SCHEMA}
    )
    dependency = build_dependency_manifest(config, config_path=config_path, repository_root=root)
    if (
        payload.get("schema") not in expected_schemas
        or payload.get("architecture_id") != ARCHITECTURE_ID
        or payload.get("phase") != args.variant
        or payload.get("dependency_manifest_sha256") != dependency["manifest_sha256"]
        or "model" not in payload
    ):
        raise ValueError("Evaluation artifact violates the architecture/phase/dependency contract")
    model.load_state_dict(payload["model"], strict=True)
    model = accelerator.prepare(model)
    result = evaluate_stage_a(
        accelerator,
        model,
        config,
        variant=args.variant,
        subset_stage=args.subset_stage,
        ablate_raw_observations=args.ablate_raw_observations,
        full100=args.full100,
        split=args.split,
        manifest_path=args.manifest or None,
        rates=([float(value) for value in args.rates.split(",") if value.strip()] if args.rates else None),
    )
    if accelerator.is_main_process:
        write_json_atomic(args.output, result)
        print(json.dumps({"output": str(Path(args.output).resolve()), "score": result["macro_full_image_nmse_p1_p2_p3"]}), flush=True)


if __name__ == "__main__":
    main()
