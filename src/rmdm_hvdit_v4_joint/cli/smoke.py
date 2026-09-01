"""Three-step CUDA memory/finite-gradient smoke for an explicit model variant."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from rmdm_hvdit_v4_joint.model import build_t1_system, build_w16_system
from rmdm_hvdit_v4_joint.training.engine import make_optimizer, require_visible_physical_gpus, write_json_atomic
from utils import cal_pinn

from .common import config_argument, load_arguments


def _batch(batch_size: int, time: int, size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    shape = (batch_size, time, 1, size, size)
    generator = torch.Generator(device=device).manual_seed(20260718)
    target = torch.rand(shape, generator=generator, device=device)
    mask = (torch.rand(shape, generator=generator, device=device) < 0.03).to(target.dtype)
    return {
        "building": torch.zeros(shape, device=device),
        "tx": torch.rand(shape, generator=generator, device=device),
        "vehicle": torch.zeros(shape, device=device),
        "target": target,
        "observed_rss": target * mask,
        "sampling_mask": mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_argument(parser)
    parser.add_argument("--variant", choices=("t1", "w16"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional smoke-only override; it never mutates the experiment config.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _, _ = load_arguments(args)
    require_visible_physical_gpus([args.physical_gpu])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke command must be isolated to exactly one authorized physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_t1_system(config) if args.variant == "t1" else build_w16_system(config)
    if args.gradient_checkpointing is not None:
        model.denoiser.gradient_checkpointing = bool(args.gradient_checkpointing)
    model = model.to(device).train()
    phase = config.t1_train if args.variant == "t1" else config.w16_train
    optimizer = make_optimizer(
        model,
        learning_rate=phase.learning_rate,
        betas=phase.betas,
        epsilon=phase.epsilon,
        weight_decay=phase.weight_decay,
    )
    frame_count = 1 if args.variant == "t1" else 16
    batch = _batch(args.batch_size, frame_count, config.data.image_size, device=device)
    noisy = torch.randn_like(batch["target"])
    timesteps = torch.arange(args.batch_size, device=device, dtype=torch.long) % config.diffusion.train_timesteps
    losses = []
    step_seconds = []
    gradient_coverage = {}
    for smoke_step in range(3):
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_noise, cal = model(noisy, timesteps, batch)
            diffusion_loss = F.mse_loss(predicted_noise.float(), torch.randn_like(predicted_noise).float())
            calibration_loss = F.mse_loss(cal.float(), batch["target"].float())
            batch_size, frame_count, _, height, width = cal.shape
            pinn_loss = cal_pinn(
                cal.reshape(batch_size * frame_count, height, width),
                ((batch["building"] > 0.5) | (batch["vehicle"] > 0.5))
                .to(cal.dtype)
                .reshape(batch_size * frame_count, height, width),
                batch["tx"].to(cal.dtype).reshape(batch_size * frame_count, height, width),
                k=config.stage1.pinn_k,
            ).mean().float()
            loss = diffusion_loss + calibration_loss + config.stage1.pinn_weight * pinn_loss
        loss.backward()
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        if not torch.isfinite(loss) or not gradients or any(
            not gradient.isfinite().all() for gradient in gradients.values()
        ):
            raise RuntimeError("Smoke produced a non-finite loss or gradient")
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        gradient_coverage[f"step_{smoke_step + 1}"] = len(gradients)
    required_gradient_suffixes = (
        "denoiser.input_stem.dense_projection.weight",
        "denoiser.condition_stem.dense_projection.weight",
        "denoiser.local_encoder.0.attention.qkv.weight",
        "denoiser.global_bottleneck.0.attention.qkv.weight",
        "denoiser.output_head.token_projection.weight",
    )
    missing_gradients = [name for name in required_gradient_suffixes if name not in gradients]
    if missing_gradients:
        raise RuntimeError(f"Three-step smoke did not activate the full backbone: {missing_gradients}")
    zero_gradients = [name for name in required_gradient_suffixes if gradients[name].abs().sum() == 0]
    if zero_gradients:
        raise RuntimeError(f"Three-step smoke left zero backbone gradients: {zero_gradients}")
    hwm_gradients = [gradient for name, gradient in gradients.items() if name.startswith("hwm.")]
    if not hwm_gradients or not any(gradient.abs().sum() > 0 for gradient in hwm_gradients):
        raise RuntimeError("Three-step smoke did not train the HWM/cal branch")
    torch.cuda.synchronize(device)
    report = {
        "schema": "rmdm_hvdit_v4_joint_cuda_smoke_v1",
        "variant": args.variant,
        "batch_size": args.batch_size,
        "physical_gpu": args.physical_gpu,
        "loss": losses[-1],
        "losses": losses,
        "step_seconds": step_seconds,
        "steady_samples_per_second": args.batch_size / step_seconds[-1],
        "optimizer_steps": 3,
        "gradient_coverage": gradient_coverage,
        "required_gradient_l1": {
            name: float(gradients[name].detach().float().abs().sum()) for name in required_gradient_suffixes
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "gradient_checkpointing": model.denoiser.gradient_checkpointing,
        "passed": True,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
