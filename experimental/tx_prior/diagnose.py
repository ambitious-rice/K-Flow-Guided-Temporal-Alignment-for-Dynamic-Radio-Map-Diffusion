"""Small, paired fresh-noise diagnostics; never modifies training state."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rmdm.diffusion import DDIMSampler, DiffusionProcess
from rmdm.evaluation.fixed_sparse_protocol import frame_names_by_sample
from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
from .adapter import build_scene_prior_system
from .config import load_config
from .data import deterministic_prior_noise_like, make_prior_training_batch
from .evaluate import load_model_checkpoint, write_result
from .runner import _dataset, resolve_task_root


def mse_per_frame(a, b):
    return (a.float() - b.float()).square().flatten(1).mean(1)


def x0_from_epsilon(xt, epsilon, alpha):
    return (xt - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()


def epsilon_from_x0(xt, x0, alpha):
    return (xt - alpha.sqrt() * x0) / (1 - alpha).sqrt()


def shuffle_scene(batch):
    """Roll all scene fields together without moving GT/noise/frame identity."""
    result = dict(batch)
    for key in ("building", "tx", "vehicle", "legacy_conditions"):
        if key in result:
            result[key] = result[key].roll(1, 0)
    return result


class DiagnosticFrames(Dataset):
    def __init__(self, dataset, count, include_legacy):
        if count < 2 or count > len(dataset):
            raise ValueError("diagnostics require 2..len(dataset) frames")
        self.dataset = dataset
        self.indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
        self.include_legacy = include_legacy

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        item = self.dataset[source_index]
        if self.include_legacy:
            record, start = self.dataset._resolve_item(source_index)
            reader = self.dataset.reader
            original = reader.dataset.records[record.index]
            frame = reader.dataset.frame_ids_by_record[record.index][start]
            traffic = np.asarray(reader.dataset._load_npz_array(
                original["traffic_grid_path"], "traffic_grid_uint8")[frame], dtype=np.float32)
            if traffic.max(initial=0) > 1:
                traffic = traffic / 255.0
            item["legacy_conditions"] = torch.stack((
                item["building"][0] + 10 * item["tx"][0],
                item["tx"][0], torch.from_numpy(traffic.copy()).unsqueeze(0)), dim=0).squeeze(1)
        return item


class LegacySystem(torch.nn.Module):
    """Original 65k UNet with its native traffic encoding, strict weight load."""
    def __init__(self, checkpoint):
        super().__init__()
        from utils import build_unet_from_config
        self.model = build_unet_from_config(dict(
            image_size=128, in_ch=4, out_ch=1, num_channels=96,
            num_res_blocks=2, attention_resolutions="16", channel_mult="",
            num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
            dropout=0.0, class_cond=False, use_checkpoint=False,
            use_scale_shift_norm=True, resblock_updown=False, use_fp16=False,
            use_new_attention_order=False, learn_sigma=False))
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    def encode_conditions(self, batch):
        return {"scene": batch["legacy_conditions"]}

    def denoise(self, xt, t, cache):
        return self.model(torch.cat((cache["scene"], xt[:, 0]), dim=1), t)[0].unsqueeze(1)

    def forward(self, xt, t, batch):
        eps, cal = self.model(torch.cat((batch["legacy_conditions"], xt[:, 0]), dim=1), t)
        return eps.unsqueeze(1), cal.unsqueeze(1)


@torch.no_grad()
def measure(model, prior, xt, noise, t, alpha, *, check_path):
    epsilon, cal = model(xt, t, prior)
    x0 = x0_from_epsilon(xt, epsilon, alpha)
    shuffled_epsilon, _ = model(xt, t, shuffle_scene(prior))
    shuffled_x0 = x0_from_epsilon(xt, shuffled_epsilon, alpha)
    zero_eps = epsilon_from_x0(xt, torch.zeros_like(xt), alpha)
    cal_eps = epsilon_from_x0(xt, cal, alpha)
    values = {
        "epsilon_mse": mse_per_frame(epsilon, noise),
        "x0_mse": mse_per_frame(x0, prior["target"]),
        "x0_clipped_mse": mse_per_frame(x0.clamp(0, 1), prior["target"]),
        "cal_mse": mse_per_frame(cal, prior["target"]),
        "zero_x0_epsilon_mse": mse_per_frame(zero_eps, noise),
        "copy_cal_epsilon_mse": mse_per_frame(cal_eps, noise),
        "epsilon_delta_from_zero_x0": mse_per_frame(epsilon, zero_eps),
        "epsilon_delta_from_copy_cal": mse_per_frame(epsilon, cal_eps),
        "x0_delta_from_zero": mse_per_frame(x0, torch.zeros_like(x0)),
        "x0_delta_from_cal": mse_per_frame(x0, cal),
        "shuffled_epsilon_mse": mse_per_frame(shuffled_epsilon, noise),
        "shuffled_x0_mse": mse_per_frame(shuffled_x0, prior["target"]),
        "shuffled_epsilon_delta": mse_per_frame(shuffled_epsilon, epsilon),
        "shuffled_x0_delta": mse_per_frame(shuffled_x0, x0),
    }
    if check_path:
        cached = model.denoise(xt, t, model.encode_conditions(prior))
        values["forward_cached_mse"] = mse_per_frame(epsilon, cached)
        values["forward_cached_max_delta"] = (epsilon - cached).abs().flatten(1).max(1).values
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_epsilon, _ = model(xt, t, prior)
        bf16_x0 = x0_from_epsilon(xt, bf16_epsilon.float(), alpha)
        values["bf16_fp32_epsilon_delta"] = mse_per_frame(bf16_epsilon, epsilon)
        values["bf16_epsilon_mse"] = mse_per_frame(bf16_epsilon, noise)
        values["bf16_x0_mse"] = mse_per_frame(bf16_x0, prior["target"])
        values["bf16_x0_clipped_mse"] = mse_per_frame(bf16_x0.clamp(0, 1), prior["target"])
    for key in ("building", "tx", "vehicle"):
        values[f"copy_{key}_x0_mse"] = mse_per_frame(prior[key], prior["target"])
    return {key: value.cpu().tolist() for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/remote.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--old-checkpoint", default="")
    parser.add_argument("--frames-per-split", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 50, 100, 300, 500, 700, 900, 980, 999])
    parser.add_argument("--sample-steps", type=int, nargs="*", default=[20, 50])
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, smoke=False)
    output = Path(args.output).resolve()
    allowed = resolve_task_root(Path.cwd(), config.pipeline.output_root) / "train/diagnostics"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be new and under train/diagnostics")
    if args.batch_size < 2 or args.frames_per_split % args.batch_size:
        raise ValueError("batch must be >=2 and divide frames-per-split for shuffle controls")
    if any(t < 0 or t >= config.diffusion.train_timesteps for t in args.timesteps):
        raise ValueError("invalid timestep")
    if any(s <= 0 or s > config.diffusion.train_timesteps for s in args.sample_steps):
        raise ValueError("invalid sampling steps")
    if torch.cuda.device_count() != 1:
        raise ValueError("select one visible CUDA device")
    started = time.perf_counter()
    model = build_scene_prior_system(config)
    checkpoint = load_model_checkpoint(Path(args.checkpoint).resolve(), model)
    models = {"current": model.cuda().eval()}
    if args.old_checkpoint:
        models["old"] = LegacySystem(args.old_checkpoint).cuda().eval()
    process = DiffusionProcess(config.diffusion)
    records = []
    for split in ("train", "val"):
        ids = manifest_video_ids(config.evaluation.subset_manifest, "stage_a") if split == "val" else None
        dataset = _dataset(config, split=split, fixed_starts=tuple(range(config.data.frames_per_video)), video_ids=ids)
        loader = DataLoader(DiagnosticFrames(dataset, args.frames_per_split, bool(args.old_checkpoint)),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        for batch_index, dense in enumerate(loader):
            prior = make_prior_training_batch({k: v.cuda() if torch.is_tensor(v) else v for k, v in dense.items()})
            names = frame_names_by_sample(prior, batch_size=prior["target"].shape[0], window_size=1)
            noise = deterministic_prior_noise_like(prior["target"], names, seed=args.seed)
            for timestep in args.timesteps:
                t = torch.full((prior["target"].shape[0],), timestep, device="cuda", dtype=torch.long)
                alpha = process.scheduler.alphas_cumprod[timestep].cuda()
                xt = alpha.sqrt() * prior["target"] + (1 - alpha).sqrt() * noise
                for label, system in models.items():
                    values = measure(system, prior, xt, noise, t, alpha, check_path=label == "current")
                    for i, name in enumerate(names):
                        records.append(dict(split=split, model=label, timestep=timestep, frame=name[0],
                                            **{key: value[i] for key, value in values.items()}))
            if split == "val":
                for steps in args.sample_steps:
                    for label, system in models.items():
                        with torch.no_grad():
                            prediction = DDIMSampler(config.diffusion).sample(system, prior, initial_noise=noise.clone(), steps=steps)
                        errors = mse_per_frame(prediction, prior["target"]).cpu().tolist()
                        for i, name in enumerate(names):
                            records.append(dict(split="val_sampling", model=label, timestep=steps,
                                                frame=name[0], mse=errors[i]))
            print(f"{split}: {min((batch_index + 1) * args.batch_size, args.frames_per_split)}/{args.frames_per_split} frames", flush=True)
    groups = {}
    for row in records:
        key = f"{row['split']}/{row['model']}/{row['timestep']}"
        groups.setdefault(key, []).append(row)
    summaries = {key: {name: statistics.fmean(row[name] for row in rows)
                       for name in rows[0] if name not in ("split", "model", "timestep", "frame")}
                 for key, rows in groups.items()}
    write_result(output, dict(schema="tx_prior_path_diagnostics_v1", args=vars(args),
        checkpoint=checkpoint, resolved_config=config.to_dict(), records=records, summaries=summaries,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        device=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        precision="fp32_with_bf16_forward_control", eta=0, elapsed_seconds=time.perf_counter() - started))
    print(f"Saved diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
