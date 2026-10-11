"""Held-out draws with numeric frame seeds and reversible short-window inference."""
from contextlib import contextmanager
import time

import torch
from diffusers import DDIMScheduler

from experimental.paper_evaluation.randomness import balanced_folds, ensemble_initial
from experimental.noise_estimation_iterative.core import repeat

PUBLIC_KEYS = frozenset(('building', 'vehicle', 'observed_rss', 'sampling_mask',
                         'sampling_rate', 'measurement_variance'))


def validate_public(inputs):
    if set(inputs) != PUBLIC_KEYS:
        raise ValueError('Expected exactly the six public conditioning keys')
    reference = inputs['observed_rss']
    if reference.ndim != 5 or reference.shape[:3] != (1, 16, 1):
        raise ValueError('Expected one complete [1,16,1,H,W] window')
    for key in ('building', 'vehicle', 'sampling_mask'):
        if inputs[key].shape != reference.shape:
            raise ValueError(f'{key} shape differs from observations')
    if inputs['sampling_rate'].shape != (1, 16):
        raise ValueError('sampling_rate must be [1,16]')
    q = inputs['measurement_variance']
    if q.numel() != 1 or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError('Expected one finite nonnegative measurement variance')
    if not all(torch.isfinite(value).all() for value in inputs.values()):
        raise ValueError('Nonfinite public input')
    mask = inputs['sampling_mask']
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError('Observation mask must be binary')
    if (reference[mask == 0] != 0).any():
        raise ValueError('Observations outside mask must be zero')


def assign_folds(mask, *, video_index, start, seed, folds=4, spatial=False):
    if folds not in (3, 4) or (spatial and folds != 3):
        raise ValueError('Use 3/4 random folds or exactly 3 spatial bands')
    if not spatial:
        return balanced_folds(mask.cpu(), video_index=video_index, start=start,
                              seed=seed, folds=folds).to(mask.device)
    width = mask.shape[-1]
    if width < 3:
        raise ValueError('Spatial bands require width >= 3')
    labels = (torch.arange(width, device=mask.device) * 3 // width).clamp_max(2)
    return torch.where(mask.bool(), labels.view(1, 1, 1, 1, width), -1).to(torch.int8)


def masked_inputs(inputs, assignment, fold):
    keep = assignment != fold
    result = dict(inputs)
    result['sampling_mask'] = inputs['sampling_mask'] * keep
    result['observed_rss'] = inputs['observed_rss'] * keep
    before = inputs['sampling_mask'].flatten(2).sum(2)
    after = result['sampling_mask'].flatten(2).sum(2)
    result['sampling_rate'] = inputs['sampling_rate'] * after / before.clamp_min(1)
    return result


@contextmanager
def short_window_context(model, window):
    if window not in (4, 16):
        raise ValueError('Only W4 and W16 are supported')
    if window == 16:
        yield
        return
    denoiser = model.denoiser
    attentions = [block.attention for blocks in (denoiser.local_encoder, denoiser.local_decoder)
                  for block in blocks]
    frames = denoiser.frames
    kernels = [attention.kernel_size for attention in attentions]
    if frames != 16 or denoiser.temporal_patch != 2 or denoiser.temporal_factor != 2:
        raise ValueError('Short-window adaptation requires the original W16 hierarchy')
    try:
        denoiser.frames = 4
        for attention, kernel in zip(attentions, kernels):
            if len(kernel) != 3:
                raise ValueError('Expected 3-D local attention')
            attention.kernel_size = (1, *kernel[1:])
        yield
    finally:
        denoiser.frames = frames
        for attention, kernel in zip(attentions, kernels):
            attention.kernel_size = kernel


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def collect(model, diffusion, inputs, *, video_index, start, seed, folds=4,
            members=8, spatial=False, window=16, prefix_only=False, steps=50,
            member_batch=4):
    validate_public(inputs)
    if window not in (4, 16) or (prefix_only and window != 4):
        raise ValueError('prefix_only is available only for W4')
    if members not in (4, 8) or not 1 <= member_batch <= members or steps < 1:
        raise ValueError('Invalid draws, batch size, or DDIM steps')
    observed = inputs['observed_rss']
    device = observed.device
    _synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    assignment = assign_folds(inputs['sampling_mask'], video_index=video_index,
                              start=start, seed=seed, folds=folds, spatial=spatial)
    visible = inputs['sampling_mask'].bool().clone()
    if prefix_only:
        visible[:, 4:] = False
    flat_indices = visible.flatten().nonzero().flatten()
    samples = torch.empty((members, int(visible.sum())), device='cpu', dtype=torch.float32)
    # A full-grid lookup keeps returned columns ordered like full-W16 flatten().
    columns = torch.full((observed.numel(),), -1, device=device, dtype=torch.long)
    columns[flat_indices] = torch.arange(len(flat_indices), device=device)
    columns = columns.reshape_as(observed)
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.train_timesteps,
        beta_schedule=diffusion.beta_schedule, prediction_type=diffusion.prediction_type,
        clip_sample=True, set_alpha_to_one=True, steps_offset=0)
    chunks = [0] if prefix_only else list(range(0, 16, window))
    skipped = []
    batches = []
    with short_window_context(model, window):
        for offset in chunks:
            chunk = {key: value if key == 'measurement_variance' else value[:, offset:offset+window]
                     for key, value in inputs.items()}
            labels = assignment[:, offset:offset+window]
            for fold in range(folds):
                selected = labels == fold
                if not selected.any():
                    skipped.append([offset, fold])
                    continue
                _synchronize(device)
                encode_tick = time.perf_counter()
                cache = model.encode_conditions(masked_inputs(chunk, labels, fold))
                _synchronize(device)
                encode_seconds = time.perf_counter() - encode_tick
                destinations = columns[:, offset:offset+window][selected].cpu()
                for first in range(0, members, member_batch):
                    _synchronize(device)
                    batch_tick = time.perf_counter()
                    member_ids = list(range(first, min(first+member_batch, members)))
                    sample = ensemble_initial(chunk['observed_rss'].shape[1:], video_index=video_index,
                        start=start+offset, fold=fold, members=member_ids, seed=seed).to(device)
                    repeated = repeat(cache, len(member_ids))
                    scheduler.set_timesteps(steps, device=device)
                    for timestep in scheduler.timesteps:
                        times = torch.full((len(member_ids),), int(timestep), device=device, dtype=torch.long)
                        output = model.denoise(scheduler.scale_model_input(sample, timestep), times, repeated)
                        sample = scheduler.step(output, timestep, sample, eta=0.,
                            use_clipped_model_output=False, return_dict=False)[0]
                    samples[first:first+len(member_ids), destinations] = sample.clamp(0, 1)[:, selected[0]].cpu()
                    _synchronize(device)
                    batches.append(dict(chunk_start=offset, fold=fold, member_start=first,
                        seconds=time.perf_counter()-batch_tick + (encode_seconds if first == 0 else 0.)))
    if not torch.isfinite(samples).all():
        raise ValueError('Nonfinite held-out predictions')
    _synchronize(device)
    observed_cpu = observed[visible].cpu()
    flat_indices_cpu = flat_indices.cpu()
    folds_cpu = assignment[visible].cpu()
    _synchronize(device)
    seconds = time.perf_counter() - tick
    shared_setup_seconds = seconds - sum(batch['seconds'] for batch in batches)
    return dict(samples=samples, observed=observed_cpu,
        flat_indices=flat_indices_cpu, folds=folds_cpu,
        seconds=seconds,
        peak_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.,
        meta=dict(batches=batches, shared_setup_seconds=shared_setup_seconds, window=window, prefix_only=prefix_only, chunk_offsets=chunks,
                  fold_count=folds, spatial=spatial, members=members, member_batch=member_batch,
                  steps=steps, seed=seed, seed_stream_stride=8, skipped_empty_folds=skipped,
                  short_window_adaptation=window == 4, retrained=False,
                  local_temporal_kernel=1 if window == 4 else 3,
                  rope='existing per-window normalized [-1,1] token coordinates; no learned position changes'))
