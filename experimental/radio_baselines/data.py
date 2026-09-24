"""Shared clean16 frame, mask and validation protocol. No training noise augmentation."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import deterministic_noise_like
from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.packed_data import PackedFrameReader
from experimental.noise_temporal_rmdm.validation import validation_video_ids
from experimental.noise_temporal_rmdm.noise import add_fixed_measurement_noise
from .models import conditions

ROOT = Path(__file__).resolve().parents[2]


def protocol(data_root):
    cfg = load_config(ROOT / 'experimental/noise_temporal_rmdm/t1.yaml')
    cfg.data.root = str(data_root)
    cfg.data.split_file = str(ROOT / cfg.data.split_file)
    cfg.validation.rates = [1., 2., 3.]
    cfg.validation.noise_standard_deviations = [0.]
    cfg.measurement_noise.clean_probability = 1.
    cfg.measurement_noise.nominal_probability = 0.
    cfg.measurement_noise.strong_probability = 0.
    return cfg


def train_dataset(cfg, cache):
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    if json.loads(Path(manifest['split_file']).read_text()) != json.loads(Path(cfg.data.split_file).read_text()):
        raise ValueError('Packed cache and configured scene split differ')
    reader = PackedFrameReader(cache, source_root=cfg.data.root, split_file=manifest['split_file'])
    return WindowDataset(reader=reader, split='train', window_size=1,
                         seed=cfg.sampling.seed, include_tx=False, fixed_starts=tuple(range(100)))


def evaluation_batches(cfg, *, split='val', batch_size=16, smoke=False):
    p = cfg.validation if split == 'val' else cfg.final_test
    ids = validation_video_ids(ROOT / p.subset_manifest, included_scenes=p.included_scenes,
                               excluded_scenes=p.excluded_scenes)
    dataset = WindowDataset(root=cfg.data.root, split=split, split_file=cfg.data.split_file,
                           window_size=1, seed=cfg.sampling.seed, include_tx=False,
                           fixed_starts=p.frame_starts, video_ids=ids)
    result = []
    for dense in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        result.append(dense)
        if smoke:
            break
    return result


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def latent_noise(sparse, rate, seed):
    reference = torch.empty(len(sparse['video_id']), 1, 3, 32, 32)
    return deterministic_noise_like(reference, video_ids=list(sparse['video_id']),
                                    starts=sparse['start'].tolist(), rate=rate, seed=seed)[:, 0]


@torch.no_grad()
def evaluate(model, method, cfg, batches, *, device, split='val', sigmas=(0.,), steps=20):
    model.eval()
    sampling = SamplingPolicy(cfg.sampling, split=split)
    rows = []
    # Evaluation never advances the training random streams, including AE posterior RNG.
    device_ids = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=device_ids):
        for rate in ([1.] if method == 'vae' else [1., 2., 3.]):
            for sigma in ((0.,) if method == 'vae' else sigmas):
                totals = dict(mse=0., mae=0., psnr=0., unobserved_free_space_mse=0.)
                count = 0
                for dense in batches:
                    # CPU draws are paired across machines and GPU models.
                    sparse = add_fixed_measurement_noise(sampling(dense, fixed_rate=rate), sigma,
                                                         seed=cfg.measurement_noise.seed)
                    gpu = to_device(sparse, device)
                    x, target = conditions(gpu), gpu['target'][:, 0]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        if method == 'vae':
                            pred = model(target, sample=False)[0]
                        elif method == 'radiodiff':
                            initial = latent_noise(sparse, rate, cfg.train.seed).to(device)
                            increments = [latent_noise(sparse, rate, cfg.train.seed + 1000 + i).to(device)
                                          for i in range(steps)]
                            pred = model.sample(x, initial, increments)
                        else:
                            pred = model(x)
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError('Nonfinite validation prediction')
                    difference = pred.float().clamp(0, 1) - target.float()
                    mse = difference.square().flatten(1).mean(1)
                    unobserved = gpu['valid_mask'][:, 0] * (1 - gpu['sampling_mask'][:, 0])
                    totals['mse'] += mse.sum().item()
                    totals['mae'] += difference.abs().flatten(1).mean(1).sum().item()
                    totals['psnr'] += (-10 * mse.clamp_min(1e-12).log10()).sum().item()
                    totals['unobserved_free_space_mse'] += ((difference.square() * unobserved).flatten(1).sum(1)
                                                           / unobserved.flatten(1).sum(1).clamp_min(1)).sum().item()
                    count += len(pred)
                if not count:
                    raise RuntimeError('Empty validation set')
                row = dict(sampling_rate=rate, measurement_sigma=float(sigma), samples=count,
                           **{key: value / count for key, value in totals.items()})
                rows.append(row)
                print(json.dumps({'validation': row}), flush=True)
    return dict(evaluation_split=split, selection_role='checkpoint_selection_clean_only' if split == 'val' else 'final_report_only',
                method=method, randomness_device='cpu', tx_input=False, window_size=1,
                prediction_clamp=[0, 1], metric_domain='full_image',
                sampler='constant_sde' if method == 'radiodiff' else 'deterministic',
                sampling_steps=steps if method == 'radiodiff' else 1,
                results=rows, score=sum(row['mse'] for row in rows) / len(rows))
