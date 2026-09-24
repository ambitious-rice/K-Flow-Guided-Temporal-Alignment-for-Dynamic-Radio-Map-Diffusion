"""Train paired noise-injection models from random initialization, never pretrained weights."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from utils import build_unet_from_config, cal_pinn_without_source
from unet import ResBlock
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DDIMSampler, deterministic_noise_like
from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.noise import add_measurement_noise, add_fixed_measurement_noise
from experimental.noise_temporal_rmdm.packed_data import PackedFrameReader
from experimental.noise_temporal_rmdm.validation import validation_video_ids
from .model import DirectNoiseRMDM


ARCHITECTURE = dict(image_size=128, in_ch=6, out_ch=1, num_channels=96,
    num_res_blocks=2, channel_mult='', num_heads=4, num_head_channels=-1,
    num_heads_upsample=-1, attention_resolutions='16', dropout=0., class_cond=False,
    use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=False,
    use_fp16=False, use_new_attention_order=False, learn_sigma=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def expand_conv(old, diffusion_input=False):
    new = nn.Conv2d(old.in_channels + 1, old.out_channels, old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :5].copy_(old.weight[:, :5])
        if diffusion_input:
            new.weight[:, 6].copy_(old.weight[:, 5])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class PairedDropout2d(nn.Dropout2d):
    """Generate channel masks on CPU so GPU model differences do not change RNG."""
    def forward(self, value):
        if not self.training or self.p == 0:
            return value
        mask = (torch.rand(value.shape[0], value.shape[1], 1, 1) >= self.p)
        return value * mask.to(device=value.device, dtype=value.dtype) / (1 - self.p)


def to_cuda(batch):
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def run_validation(config, *, checkpoint_path, repository_root, output_path,
                   model, checkpoint_step, evaluated_model, max_batches=0):
    """Same val protocol, but all masks/noises generated on CPU across machines."""
    protocol = config.validation
    manifest = Path(repository_root) / protocol.subset_manifest
    ids = validation_video_ids(manifest, included_scenes=protocol.included_scenes,
                               excluded_scenes=protocol.excluded_scenes)
    dataset = WindowDataset(root=config.data.root, split='val', split_file=config.data.split_file,
        window_size=1, seed=config.sampling.seed, include_tx=False,
        fixed_starts=protocol.frame_starts, video_ids=ids)
    loader = DataLoader(dataset, batch_size=protocol.batch_size, shuffle=False, num_workers=0)
    sampling = SamplingPolicy(config.sampling, split='val')
    sampler = DDIMSampler(config.diffusion)
    model.eval()
    rows = []
    for rate in protocol.rates:
        for sigma in protocol.noise_standard_deviations:
            mse_sum = 0.
            count = 0
            for i, dense in enumerate(loader):
                if max_batches and i >= max_batches:
                    break
                sparse = add_fixed_measurement_noise(sampling(dense, fixed_rate=rate), sigma,
                                                     seed=config.measurement_noise.seed)
                initial = deterministic_noise_like(sparse['target'], video_ids=list(sparse['video_id']),
                    starts=sparse['start'].tolist(), rate=rate, seed=config.train.seed)
                sparse = to_cuda(sparse)
                generated = sampler.sample(model, sparse, initial_noise=initial.cuda(), steps=protocol.ddim_steps)
                mse = (generated.float() - sparse['target'].float()).flatten(1).square().mean(1)
                mse_sum += float(mse.sum())
                count += len(mse)
            rows.append(dict(sampling_rate=rate, measurement_sigma=sigma, samples=count, mse=mse_sum / count))
            print(json.dumps(rows[-1]), flush=True)
    result = dict(evaluation_split='val', selection_role='checkpoint_selection',
        checkpoint=str(checkpoint_path), checkpoint_step=checkpoint_step,
        evaluated_model=evaluated_model, ddim_steps=protocol.ddim_steps,
        randomness_device='cpu', manifest=str(manifest), results=rows)
    write(Path(output_path), result)
    return result


class ScratchNoiseRMDM(DirectNoiseRMDM):
    def __init__(self, mode, seed=20260924):
        if mode not in ('variance', 'sigma', 'adanorm', 'unconditioned'):
            raise ValueError(mode)
        # Build the shared random backbone BEFORE any mode-specific parameters.
        # No checkpoint path or torch.load occurs in model initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            backbone = build_unet_from_config(dict(ARCHITECTURE))
        super().__init__(backbone, reference_variance=.0081)
        for module in self.backbone.modules():
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Dropout2d):
                    setattr(module, name, PairedDropout2d(child.p, inplace=False))
        self.mode = mode
        self.reported_sigma = None
        self.norm_names = []
        self.noise_heads = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            if mode in ('variance', 'sigma'):
                unet = self.backbone.unet
                unet.input_blocks[0][0] = expand_conv(unet.input_blocks[0][0], True)
                first = unet.hwm.conv_blocks_context[0].blocks[0]
                first.conv = expand_conv(first.conv)
            elif mode == 'adanorm':
                self.noise_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU())
                sites = []
                for name, module in self.backbone.named_modules():
                    if isinstance(module, ResBlock):
                        sites.append((name + '.out_layers.0', module.out_layers[0], module.out_channels))
                    elif name.startswith('unet.hwm.') and isinstance(module, nn.BatchNorm2d):
                        sites.append((name, module, module.num_features))
                for name, norm, channels in sites:
                    head = nn.Linear(64, channels * 2, bias=False)
                    nn.init.zeros_(head.weight)
                    self.noise_heads.append(head)
                    self.norm_names.append(name)
                    norm.register_forward_hook(self._hook(len(self.noise_heads) - 1))

    def _hook(self, index):
        def modulate(module, inputs, output):
            scale, shift = self.noise_heads[index](self._embedding).to(output.dtype).chunk(2, dim=1)
            return output * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return modulate

    def encode_conditions(self, sparse):
        if self.reported_sigma is not None:
            sparse = dict(sparse)
            sparse['measurement_variance'] = torch.full_like(sparse['measurement_variance'], self.reported_sigma ** 2)
        conditions = super().encode_conditions(sparse)
        if self.mode in ('sigma', 'adanorm'):
            conditions = torch.cat((conditions[:, :5], conditions[:, 5:6].sqrt()), dim=1)
        return conditions

    def forward(self, conditions, noisy_target, diffusion_step):
        if self.mode in ('adanorm', 'unconditioned'):
            if self.mode == 'adanorm':
                sigma = conditions[:, 5, 0, 0, None]
                self._embedding = self.noise_embed(sigma) - self.noise_embed(torch.zeros_like(sigma))
            conditions = conditions[:, :5]
        return super().forward(conditions, noisy_target, diffusion_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['variance', 'sigma', 'adanorm'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='experimental/noise_temporal_rmdm/t1.yaml')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config.data.split_file = str(Path(config.data.split_file).resolve())
    config.data.root = args.data_root
    config.data.workers = 0
    config.validation.batch_size = 16
    config.validation.rates = [1., 2., 3.]
    config.validation.noise_standard_deviations = [0., .01, .03, .05, .07, .09]
    if args.smoke:
        config.validation.rates = [1.]
        config.validation.noise_standard_deviations = [0., .09]
    manifest = json.loads((Path(args.cache) / 'manifest.json').read_text())
    reader = PackedFrameReader(args.cache, source_root=config.data.root, split_file=manifest['split_file'])
    # Ensure the packed training cache and evaluation use the SAME split contents,
    # even when the absolute source checkout paths differ between machines.
    configured_split = json.loads(Path(config.data.split_file).read_text())
    if configured_split != json.loads(Path(manifest['split_file']).read_text()):
        raise ValueError('packed and configured split contents differ')
    dataset = WindowDataset(reader=reader, split='train', window_size=1,
                           seed=config.sampling.seed, include_tx=False, fixed_starts=tuple(range(100)))
    model = ScratchNoiseRMDM(args.mode, args.seed).cuda()
    metadata = {'args': vars(args), 'config': config.to_dict(), 'architecture': ARCHITECTURE,
        'initialization': 'random_from_scratch', 'pretrained_checkpoint': None,
        'randomness_device': 'cpu for masks, observation/diffusion noises, timesteps, dropout',
        'bn_running_stats': 'train', 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'modulation_sites': model.norm_names, 'results': {}, 'checks': {}, 'status': 'running'}
    write(output / 'metadata.json', metadata)
    tracked_names = ['backbone.unet.out.2.weight', 'backbone.unet.input_blocks.0.0.weight']
    if args.mode == 'adanorm':
        tracked_names.append('noise_heads.0.weight')
    tracked = {n: p.detach().clone() for n, p in model.named_parameters() if n in tracked_names}
    seed_all(args.seed + 2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2), num_workers=0, drop_last=True)
    sampling = SamplingPolicy(config.sampling, split='train')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    diffusion = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon')
    step, epoch, best = 0, 0, float('inf')
    started = time.monotonic()
    log = (output / 'train.jsonl').open('w')
    while step < args.steps:
        dataset.set_epoch(epoch)
        sampling.set_epoch(epoch)
        for dense in loader:
            model.train()  # BatchNorm running statistics ARE trained from scratch.
            lr_factor = min(1., (step + 1) / max(1, args.warmup))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * lr_factor
            sparse = add_measurement_noise(sampling(dense), config.measurement_noise, epoch=epoch)
            target_cpu = sparse['target'][:, 0]
            t_cpu = torch.randint(0, 1000, (target_cpu.shape[0],))
            epsilon_cpu = torch.randn_like(target_cpu)
            if step == 0:
                metadata['checks']['paired_first_batch'] = dict(video_id=list(sparse['video_id']),
                    start=sparse['start'].tolist(), sigma=sparse['measurement_standard_deviation'].tolist(),
                    mask_counts=sparse['sampling_mask'].flatten(1).sum(1).tolist(),
                    timesteps=t_cpu.tolist(), epsilon_first_values=epsilon_cpu.flatten()[:16].tolist(),
                    observed_sum=float(sparse['observed_rss'].sum()), target_sum=float(target_cpu.sum()),
                    shared_initial_weights=model.backbone.unet.input_blocks[1][0].in_layers[2].weight.detach().flatten()[:16].cpu().tolist())
                write(output / 'metadata.json', metadata)
            sparse = to_cuda(sparse)
            target = sparse['target'][:, 0]
            t, epsilon = t_cpu.cuda(), epsilon_cpu.cuda()
            noisy = diffusion.add_noise(target, epsilon, t)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predicted, cal = model(model.encode_conditions(sparse), noisy, t)
                obstacle = ((sparse['building'][:, 0, 0] > .5) | (sparse['vehicle'][:, 0, 0] > .5)).float()
                ld = F.mse_loss(predicted, epsilon)
                lc = F.mse_loss(cal, target)
                lp = cal_pinn_without_source(cal[:, 0], obstacle, k=.2).mean()
                loss = ld + lc + lp
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            if step in (0, 4):
                metadata['checks'][f'gradient_step_{step+1}'] = {
                    n: None if p.grad is None else float(p.grad.norm()) for n, p in model.named_parameters() if n in tracked}
                metadata['checks']['missing_gradients'] = [n for n, p in model.named_parameters() if p.grad is None]
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 100 == 0:
                row = {'mode': args.mode, 'step': step, 'diffusion': float(ld), 'calibration': float(lc),
                    'pinn': float(lp), 'grad_norm': float(grad_norm), 'lr': optimizer.param_groups[0]['lr'],
                    'seconds': time.monotonic() - started}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
                write(output / 'status.json', row)
            if step % args.val_every == 0 or step == args.steps:
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                checkpoint = output / f'step_{step:06d}.pth'
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                    'mode': args.mode, 'initialization': 'random_from_scratch', 'args': vars(args),
                    'epoch': epoch, 'rng_cpu': cpu_rng, 'rng_cuda': gpu_rng}, checkpoint)
                result = run_validation(config, checkpoint_path=checkpoint, repository_root=Path.cwd(),
                    output_path=output / f'val_{step:06d}.json', model=model, checkpoint_step=step,
                    evaluated_model='scratch_' + args.mode, max_batches=1 if args.smoke else 0)
                score = sum(r['mse'] for r in result['results']) / len(result['results'])
                metadata['results'][str(step)] = score
                if score < best:
                    best = score
                    metadata['best_step'] = step
                metadata['checks']['parameter_max_updates'] = {
                    n: float((dict(model.named_parameters())[n].detach() - p).abs().max()) for n, p in tracked.items()}
                bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
                metadata['checks']['bn_num_batches_tracked'] = int(bn.num_batches_tracked)
                write(output / 'metadata.json', metadata)
                print(json.dumps({'step': step, 'val_mse': score, 'best': best}), flush=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(gpu_rng)
            if step >= args.steps:
                break
        epoch += 1
    log.close()
    metadata['status'] = 'completed'
    metadata['elapsed_seconds'] = time.monotonic() - started
    write(output / 'metadata.json', metadata)
    write(output / 'status.json', {'status': 'completed', 'step': step, 'best_mse': best,
                                  'best_step': metadata['best_step']})


if __name__ == '__main__':
    main()
