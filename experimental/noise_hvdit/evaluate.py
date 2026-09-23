"""Frame-paired W1/W16 evaluation, including unobserved free space."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experimental.noise_temporal_rmdm.noise import add_fixed_measurement_noise
from rmdm.data import SamplingPolicy, WindowDataset, derive_seed
from rmdm.diffusion import DDIMSampler
from rmdm_hvdit_v4_joint.training.engine import write_json_atomic
from .config import load_config
from .model import NoiseHVDiT
from .sampling import sample


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def video_ids(config, split="val", fast=False):
    manifest = config.evaluation.manifest if split == "val" else "configs/manifests/noise_temporal_rmdm_clean16_test.json"
    videos = json.loads(Path(manifest).read_text())["stage_a"]["videos"]
    if fast:
        selected, counts = [], {}
        for video in videos:
            scene = video["scene_id"]
            if counts.get(scene, 0) < 2:
                selected.append(video)
                counts[scene] = counts.get(scene, 0) + 1
        videos = selected
    return [v["video_id"] for v in videos]


def frame_noise(target, videos, starts, seed):
    # The same physical frame gets the same initial noise in W1 and W16.
    return torch.stack([torch.stack([
        torch.randn(target.shape[2:], device=target.device, generator=torch.Generator(device=target.device).manual_seed(
            derive_seed("noise-hvdit-evaluation", seed, video, int(start)+t)))
        for t in range(target.shape[1])]) for video, start in zip(videos, starts)])


def as_single_frames(sparse):
    batch, frames = sparse["target"].shape[:2]
    result = dict(sparse)
    for key, value in sparse.items():
        if torch.is_tensor(value) and value.ndim == 5:
            result[key] = value.reshape(batch*frames, 1, *value.shape[2:])
    result["sampling_rate"] = sparse["sampling_rate"].reshape(batch*frames, 1)
    result["measurement_variance"] = sparse["measurement_variance"].repeat_interleave(frames)
    return result


def summarize(rows):
    result = {}
    for group, subset in {
        "all": rows, "clean": [r for r in rows if r["sigma"] == 0],
        "high_noise": [r for r in rows if r["sigma"] >= 0.05],
    }.items():
        result[group] = {key: sum(r[key] for r in subset)/len(subset)
                         for key in ("mse", "mae", "psnr", "unobserved_mse", "observed_mse", "temporal_delta_mse")}
    return result


@torch.no_grad()
def evaluate(model, config, *, fast=False, ddim_steps=20, split="val", rank=0, world_size=1):
    device = next(model.parameters()).device
    model.eval()
    videos = video_ids(config, split, fast)[rank::world_size]
    if not videos:
        return []
    dataset = WindowDataset(root=config.data.root, split=split, split_file=str(Path(config.data.split_file).resolve()),
                            window_size=16, include_tx=False, video_ids=videos,
                            fixed_starts=config.evaluation.starts[:1] if fast else config.evaluation.starts)
    loader = DataLoader(dataset, batch_size=config.evaluation.batch_size, shuffle=False,
                        num_workers=min(config.data.workers, 4), pin_memory=device.type == "cuda")
    sampling, sampler = SamplingPolicy(config.sampling, split=split), DDIMSampler(config.diffusion)
    rows = []
    for dense in loader:
        dense = to_device(dense, device)
        initial = frame_noise(dense["target"], dense["video_id"], dense["start"], config.train.seed)
        for rate in config.evaluation.rates:
            for sigma in config.evaluation.sigmas:
                sparse = add_fixed_measurement_noise(sampling(dense, fixed_rate=rate), sigma,
                                                     seed=config.measurement_noise.seed)
                inputs, noise = sparse, initial
                if config.data.window_size == 1:
                    inputs, noise = as_single_frames(sparse), initial.flatten(0, 1).unsqueeze(1)
                precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with precision:
                    prediction = sample(sampler, model, inputs, noise, ddim_steps)
                prediction = prediction.reshape_as(dense["target"]).float()
                error = prediction-dense["target"]
                observed = sparse["sampling_mask"]
                unseen = sparse["valid_mask"] * (1-observed)
                frame_mse = error.square().flatten(2).mean(2)
                temporal_error = error[:, 1:]-error[:, :-1]
                for i, video in enumerate(dense["video_id"]):
                    rows.append(dict(video_id=video, start=int(dense["start"][i]), rate=rate, sigma=sigma,
                        mse=float(frame_mse[i].mean()), mae=float(error[i].abs().mean()),
                        psnr=float((-10*frame_mse[i].clamp_min(1e-12).log10()).mean()),
                        unobserved_mse=float((error[i].square()*unseen[i]).sum()/unseen[i].sum().clamp_min(1)),
                        observed_mse=float((error[i].square()*observed[i]).sum()/observed[i].sum().clamp_min(1)),
                        temporal_delta_mse=float(temporal_error[i].square().mean())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=["model", "ema"], default="ema")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from accelerate import Accelerator
    accelerator = Accelerator()
    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NoiseHVDiT(config).to(accelerator.device)
    model.load_state_dict(payload[args.weights])
    rows = evaluate(model, config, fast=args.fast, ddim_steps=args.ddim_steps, split=args.split,
                    rank=accelerator.process_index, world_size=accelerator.num_processes)
    from accelerate.utils import gather_object
    rows = gather_object(rows)
    if accelerator.is_main_process:
        write_json_atomic(args.output, dict(checkpoint=args.checkpoint, weights=args.weights,
                          step=payload["step"], split=args.split, ddim_steps=args.ddim_steps,
                          summary=summarize(rows), rows=rows))
    accelerator.end_training()


if __name__ == "__main__":
    main()
