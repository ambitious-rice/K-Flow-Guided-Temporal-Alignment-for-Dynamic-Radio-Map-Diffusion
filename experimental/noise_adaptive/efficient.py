"""Batched W4 held-out predictions with unchanged physical-frame random streams."""
import time

import numpy as np
import torch
from diffusers import DDIMScheduler

from experimental.noise_estimation_iterative.core import repeat
from experimental.paper_evaluation.randomness import ensemble_initial
from .predict import assign_folds, masked_inputs, short_window_context, validate_public, _synchronize
from .statistics import calibrated_variance, mle


def concatenate(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {k: concatenate([v[k] for v in values]) for k in first}
    if isinstance(first, tuple):
        return tuple(concatenate([v[i] for v in values]) for i in range(len(first)))
    if isinstance(first, list):
        return [concatenate([v[i] for v in values]) for i in range(len(first))]
    raise TypeError(type(first))


@torch.inference_mode()
def collect_w4(model, diffusion, inputs, *, video_index, start, seed, offset=0,
               selected_folds=(0, 1, 2, 3), members=4, steps=50, fold_batch=4):
    """Keep four balanced folds; single-fold mode predicts only fold zero (25%).

    Batch axes combine folds and members; no seed changes, no cross-fold mixing.
    Returned indices address the original complete W16, including later offsets.
    """
    validate_public(inputs)
    if offset not in (0, 4, 8, 12) or members not in (2, 4, 8) or steps < 1:
        raise ValueError('Invalid W4 offset, members, or steps')
    selected_folds = tuple(selected_folds)
    if (not selected_folds or len(set(selected_folds)) != len(selected_folds)
            or any(f not in range(4) for f in selected_folds) or fold_batch not in (1, 2, 4)):
        raise ValueError('Invalid fold selection or fold batch')
    device = inputs['observed_rss'].device
    _synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    labels_full = assign_folds(inputs['sampling_mask'], video_index=video_index,
                               start=start, seed=seed, folds=4)
    visible = torch.zeros_like(labels_full, dtype=torch.bool)
    for fold in selected_folds:
        visible[:, offset:offset+4] |= labels_full[:, offset:offset+4] == fold
    indices = visible.flatten().nonzero().flatten()
    if not len(indices):
        raise ValueError('No held-out observations')
    columns = torch.full_like(labels_full, -1, dtype=torch.long)
    columns.flatten()[indices] = torch.arange(len(indices), device=device)
    chunk = {k: v if k == 'measurement_variance' else v[:, offset:offset+4]
             for k, v in inputs.items()}
    labels = labels_full[:, offset:offset+4]
    samples = torch.full((members, len(indices)), float('nan'), dtype=torch.float32)
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.train_timesteps,
        beta_schedule=diffusion.beta_schedule, prediction_type=diffusion.prediction_type,
        clip_sample=True, set_alpha_to_one=True, steps_offset=0)
    nonempty = [f for f in selected_folds if (labels == f).any()]
    with short_window_context(model, 4):
        for first in range(0, len(nonempty), fold_batch):
            folds = nonempty[first:first+fold_batch]
            # Encode each condition once; then batch both encoding and denoising.
            hidden = concatenate([masked_inputs(chunk, labels, f) for f in folds])
            cache = repeat(model.encode_conditions(hidden), members)
            sample = torch.cat([ensemble_initial(chunk['observed_rss'].shape[1:],
                video_index=video_index, start=start+offset, fold=f,
                members=list(range(members)), seed=seed) for f in folds]).to(device)
            scheduler.set_timesteps(steps, device=device)
            for timestep in scheduler.timesteps:
                times = torch.full((len(sample),), int(timestep), device=device, dtype=torch.long)
                output = model.denoise(scheduler.scale_model_input(sample, timestep), times, cache)
                sample = scheduler.step(output, timestep, sample, eta=0.,
                    use_clipped_model_output=False, return_dict=False)[0]
            for i, fold in enumerate(folds):
                selected = labels == fold
                destinations = columns[:, offset:offset+4][selected].cpu()
                samples[:, destinations] = sample[i*members:(i+1)*members].clamp(0, 1)[:, selected[0]].cpu()
    if not torch.isfinite(samples).all():
        raise ValueError('Incomplete or nonfinite samples')
    result = dict(samples=samples, mean=samples.double().mean(0).float(),
        raw_variance=samples.double().var(0, unbiased=True).float(),
        observed=inputs['observed_rss'][visible].cpu(), flat_indices=indices.cpu(),
        folds=labels_full[visible].cpu())
    _synchronize(device)
    result.update(seconds=time.perf_counter()-tick,
        peak_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.,
        meta=dict(offset=offset, selected_folds=selected_folds, members=members,
                  steps=steps, fold_batch=fold_batch, seed=seed, start=start,
                  video_index=video_index, condition_q=float(inputs['measurement_variance'].item())))
    return result


def pool_update(blocks, qmax=.09**2):
    """Calibrate each disjoint block at its OWN generation q, then pool.

    This is a composite predictive likelihood; adaptive rounds are not independent.
    Exact full-window pixel identities must not be counted twice.
    """
    if not blocks:
        raise ValueError('No prediction blocks')
    ids = np.concatenate([b['moments']['flat_indices'].numpy() for b in blocks])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate held-out pixel identities across rounds')
    means, observed, variances = [], [], []
    for block in blocks:
        u = block['moments']
        means.append(u['mean'].numpy())
        observed.append(u['observed'].numpy())
        variances.append(calibrated_variance(u['raw_variance'].numpy(), block['q'], block['calibration']))
    return mle(np.concatenate(observed), np.concatenate(means), np.concatenate(variances), qmax=qmax)
