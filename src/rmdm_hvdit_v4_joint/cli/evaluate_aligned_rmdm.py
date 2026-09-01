"""Evaluate T1 or RMDM-SF under the exact same T1 Stage-A protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration

from rmdm.data import SamplingPolicy
from rmdm_hvdit_v4_joint import ARCHITECTURE_ID
from rmdm_hvdit_v4_joint.config import load_config
from rmdm_hvdit_v4_joint.evaluation import evaluate_stage_a
from rmdm_hvdit_v4_joint.evaluation.legacy_rmdm import LegacyRMDMT1ProtocolAdapter
from rmdm_hvdit_v4_joint.model import build_t1_system
from rmdm_hvdit_v4_joint.training.checkpoint import T1_CHECKPOINT_SCHEMA
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from train_sparse_dynamic_rmdm import build_model_config
from utils import build_unet_from_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_t1(config: Any, checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (
        payload.get("schema") != T1_CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != ARCHITECTURE_ID
        or payload.get("phase") != "t1"
        or "model" not in payload
    ):
        raise ValueError("Checkpoint is not an HV-DiT v4 joint T1 artifact")
    model = build_t1_system(config)
    model.load_state_dict(payload["model"], strict=True)
    metadata = {
        "schema": payload.get("schema"),
        "global_step": int(payload.get("global_step", -1)),
        "dependency_manifest_sha256": payload.get("dependency_manifest_sha256"),
    }
    del payload
    return model, metadata


def _load_rmdm(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") != "rmdm_sf_sparse_checkpoint_v1" or "model" not in payload or "args" not in payload:
        raise ValueError("Checkpoint is not an RMDM-SF sparse artifact")
    train_args = argparse.Namespace(**payload["args"])
    model = build_unet_from_config(build_model_config(train_args))
    model.load_state_dict(payload["model"], strict=True)
    metadata = {
        "schema": payload.get("schema"),
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
    }
    del payload
    return LegacyRMDMT1ProtocolAdapter(model), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("t1", "rmdm"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rates",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated sampling percentages evaluated with the unchanged T1 protocol",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=144,
        help="Per-GPU inference batch; 144 is the largest previously verified RMDM value",
    )
    parser.add_argument("--expected-visible-gpus", default="4,5,6,7")
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != args.expected_visible_gpus:
        raise RuntimeError(
            f"Aligned validation requires CUDA_VISIBLE_DEVICES={args.expected_visible_gpus}, got {visible!r}"
        )
    root = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = load_config(config_path)
    subset_manifest = (root / config.evaluation.subset_manifest).resolve()
    rates = [float(value) for value in args.rates.split(",") if value.strip()]
    if not rates or any(value < 1.0 or value > 10.0 for value in rates) or len(set(rates)) != len(rates):
        raise ValueError("--rates must contain unique percentages in [1,10]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    # This is an execution-only override after validating the immutable model
    # configuration. Name-keyed masks/noise make metrics batch-size invariant.
    config.evaluation.t1_evaluation_batch_size = int(args.batch_size)

    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if accelerator.num_processes != 4:
        raise RuntimeError(f"Aligned validation requires four processes, got {accelerator.num_processes}")

    if args.model == "t1":
        model, checkpoint_metadata = _load_t1(config, checkpoint)
    else:
        model, checkpoint_metadata = _load_rmdm(checkpoint)
    model = accelerator.prepare(model)
    result = evaluate_stage_a(
        accelerator,
        model,
        config,
        variant="t1",
        subset_stage="stage_a",
        rates=rates,
        split="val",
        manifest_path=subset_manifest,
    )

    if accelerator.is_main_process:
        subset_payload = json.loads(subset_manifest.read_text(encoding="utf-8"))
        data_index = Path(config.data.root).expanduser().resolve() / "index.json"
        result["evaluated_model"] = args.model
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            **checkpoint_metadata,
        }
        result["aligned_protocol"] = {
            "purpose": "T1-best versus RMDM-epoch9 on the corrected-data T1 Stage-A validation protocol",
            "data_root": str(Path(config.data.root).expanduser().resolve()),
            "data_index_sha256": _sha256(data_index),
            "subset_manifest_sha256": _sha256(subset_manifest),
            "subset_recorded_index_sha256": subset_payload.get("index_sha256"),
            "stable_video_ids_reused_on_corrected_data": True,
            "sampling_policy": SamplingPolicy.SAMPLER_VERSION,
            "sampling_seed": int(config.sampling.seed),
            "sampling_epoch": 0,
            "ddim_noise_policy": "joint-ddim-noise-v1",
            "ddim_steps": int(config.evaluation.ddim_steps),
            "rates": rates,
            "batch_size_per_gpu": int(args.batch_size),
            "visible_physical_gpus": [int(value) for value in visible.split(",")],
        }
        write_json_atomic(output, result)
        summary = {
            "model": args.model,
            "output": str(output),
            "checkpoint": result["checkpoint"],
            "macro_full_image_nmse_p1_p2_p3": result["macro_full_image_nmse_p1_p2_p3"],
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
