"""Fine-tune the original RMDM's variance input on noisy observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.noise import add_measurement_noise
from experimental.noise_temporal_rmdm.packed_data import PackedFrameReader
from experimental.noise_temporal_rmdm.validation import run_validation
from rmdm.data import SamplingPolicy, WindowDataset
from utils import cal_pinn_without_source

from .model import build_model, load_original_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--val-every", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    config = load_config(args.config)
    config.data.root = args.data_root
    config.data.packed_cache_root = args.cache_root
    config.data.workers = args.workers
    config.data.split_file = str(Path(config.data.split_file).resolve())
    original = torch.load(args.original, map_location="cpu", mmap=True, weights_only=False)
    model = build_model(argparse.Namespace(**original["args"]),
                        reference_variance=config.measurement_noise.reference_variance)
    load_original_weights(model, original["model"])
    # Only the new variance columns move. At sigma=0 this is exactly the old model.
    names = ("unet.input_blocks.0.0.weight", "unet.hwm.conv_blocks_context.0.blocks.0.conv.weight")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, index in zip(names, (5, 5)):
        parameter = dict(model.backbone.named_parameters())[name]
        parameter.requires_grad_(True)
        mask = torch.zeros_like(parameter)
        mask[:, index] = 1
        parameter.register_hook(lambda grad, mask=mask: grad * mask.to(grad.device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0)

    packed_split = json.loads((Path(args.cache_root) / "manifest.json").read_text())["split_file"]
    reader = PackedFrameReader(args.cache_root, source_root=args.data_root,
                               split_file=packed_split)
    dataset = WindowDataset(reader=reader, split="train", window_size=1,
                            seed=config.sampling.seed, include_tx=False,
                            fixed_starts=tuple(range(100)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, drop_last=True)
    sampling = SamplingPolicy(config.sampling, split="train")
    diffusion = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="epsilon")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps({"args": vars(args),
            "original_step": original["global_step"], "world_size": accelerator.num_processes}, indent=2))

    step = 0
    best = float("inf")
    stale = 0
    epoch = 0
    while step < args.max_steps:
        dataset.set_epoch(epoch)
        sampling.set_epoch(epoch)
        model.train()
        for dense in loader:
            dense = {k: v.to(accelerator.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in dense.items()}
            sparse = add_measurement_noise(sampling(dense), config.measurement_noise, epoch=epoch)
            target = sparse["target"][:, 0]
            conditions = accelerator.unwrap_model(model).encode_conditions(sparse)
            t = torch.randint(0, 1000, (target.shape[0],), device=target.device)
            epsilon = torch.randn_like(target)
            noisy = diffusion.add_noise(target, epsilon, t)
            with accelerator.autocast():
                predicted, cal = model(conditions, noisy, t)
                obstacle = ((sparse["building"][:, 0, 0] > .5) |
                            (sparse["vehicle"][:, 0, 0] > .5)).to(target.dtype)
                loss = (F.mse_loss(predicted, epsilon) + F.mse_loss(cal, target)
                        + cal_pinn_without_source(cal[:, 0], obstacle, k=.2).mean())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if accelerator.is_main_process and (step == 1 or step % 100 == 0):
                print(json.dumps({"step": step, "loss": float(loss),
                                  "sigma_mean": float(sparse["measurement_standard_deviation"].mean())}), flush=True)
            if step % args.val_every == 0 or step == args.max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    bare = accelerator.unwrap_model(model)
                    checkpoint = output / f"step_{step:06d}.pth"
                    torch.save({"model": bare.state_dict(), "step": step,
                                "original": args.original}, checkpoint)
                    summary = run_validation(config, checkpoint_path=checkpoint,
                        repository_root=Path(__file__).resolve().parents[2],
                        output_path=output / f"val_{step:06d}.json", model=bare,
                        checkpoint_step=step, max_batches=1 if args.smoke else 0)
                    score = sum(row["mse"] for row in summary["results"]) / len(summary["results"])
                    if score < best:
                        best, stale = score, 0
                        (output / "best.json").write_text(json.dumps({"step": step, "mse": best}))
                    else:
                        stale += 1
                    (output / "status.json").write_text(json.dumps({"step": step,
                        "val_mse": score, "best_mse": best, "stale": stale}))
                    print(json.dumps({"step": step, "val_mse": score, "best": best,
                                      "stale": stale}), flush=True)
                accelerator.wait_for_everyone()
                stop = torch.tensor(int(args.smoke or stale >= args.patience), device=accelerator.device)
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast(stop, src=0)
                if stop.item():
                    return
                model.train()
            if step >= args.max_steps:
                return
        epoch += 1


if __name__ == "__main__":
    main()
