"""Variance-gated original and noise-specialist RMDM evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experimental.noise_temporal_rmdm.checkpoint import load
from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.model import build_model
from experimental.noise_temporal_rmdm.validate_original_rmdm import _load_model
from experimental.noise_temporal_rmdm.validation import run_validation


class HybridRMDM(nn.Module):
    """Use the original direct path at low noise and the trained specialist above it."""

    def __init__(self, original: nn.Module, specialist: nn.Module,
                 thresholds: tuple[float, float, float]) -> None:
        super().__init__()
        self.original = original
        self.specialist = specialist
        self.thresholds = thresholds

    def encode_conditions(self, sparse_batch: dict):
        rate = round(float(sparse_batch["sampling_rate"].float().mean()))
        sigma = float(sparse_batch["measurement_standard_deviation"].float().mean())
        use_specialist = sigma >= self.thresholds[rate - 1]
        expert = self.specialist if use_specialist else self.original
        return expert, expert.encode_conditions(sparse_batch)

    def denoise(self, noisy_target: torch.Tensor, diffusion_step: torch.Tensor, cache):
        expert, conditions = cache
        return expert.denoise(noisy_target, diffusion_step, conditions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config.data.root = args.data_root
    config.data.workers = args.workers
    protocol = config.validation if args.split == "val" else config.final_test
    protocol.batch_size = args.batch_size
    original, original_step = _load_model(args.original)
    specialist = build_model(config)
    specialist_payload = load(args.specialist, specialist, expected_phase="t1")
    # Chosen at the midpoints where the two experts exchange rank on validation.
    thresholds = (.04, .055, .07)
    model = HybridRMDM(original, specialist, thresholds)
    summary = run_validation(config, checkpoint_path=args.specialist,
        repository_root=Path(__file__).resolve().parents[2], output_path=args.output,
        evaluation_split=args.split, ddim_steps=args.ddim_steps,
        max_batches=args.max_batches,
        model=model, checkpoint_step=int(specialist_payload["global_step"]),
        evaluated_model="variance_gated_original_and_noise_specialist")
    summary.update({"gate_thresholds": thresholds, "original_step": original_step,
                    "specialist_step": int(specialist_payload["global_step"])})
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({"macro_mse": sum(row["mse"] for row in summary["results"]) /
                      len(summary["results"]), "thresholds": thresholds}), flush=True)


if __name__ == "__main__":
    main()
