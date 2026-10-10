"""CPU-only random inputs keyed by physical frame, never by model or batch."""
import math

import torch

VERSION = 'paired-rss-v1'


def seed_for(seed, video_index, frame, stream):
    # Fixed numeric slots: physical frame identity, independent of batch/window/model.
    return int(seed) + int(video_index)*1_000_000 + int(frame)*10 + int(stream)


def materialize(dense, *, seed, video_indices, rate, sigma):
    if not (math.isfinite(rate) and 0 < rate <= 100 and math.isfinite(sigma) and sigma >= 0):
        raise ValueError('Require 0 < rate <= 100 and finite sigma >= 0')
    target = dense['target'].cpu().float()
    building, vehicle = dense['building'].cpu().float(), dense['vehicle'].cpu().float()
    valid = ((building <= .5) & (vehicle <= .5)).float()
    mask, measurement, initial = [torch.zeros_like(target) for _ in range(3)]
    for b, video in enumerate(dense['video_id']):
        for t in range(target.shape[1]):
            frame = int(dense['start'][b])+t
            def generator(stream):
                return torch.Generator(device='cpu').manual_seed(seed_for(seed, video_indices[video], frame, stream))
            pixels = valid[b,t].flatten().nonzero().flatten()
            if not len(pixels):
                raise ValueError(f'No free pixels: {video}/{frame}')
            # A shared permutation makes smaller sampling budgets nested in larger ones.
            chosen = pixels[torch.randperm(len(pixels), generator=generator(0))[:max(1, round(rate/100*len(pixels)))]]
            mask[b,t].view(-1)[chosen] = 1
            measurement[b,t] = torch.randn(target.shape[2:], generator=generator(1))
            initial[b,t] = torch.randn(target.shape[2:], generator=generator(2))
    observed = mask*(target+sigma*measurement)
    sparse = dict(target=target, building=building, vehicle=vehicle, valid_mask=valid,
                  sampling_mask=mask, observed_rss=observed,
                  sampling_rate=torch.full(target.shape[:2], float(rate)),
                  measurement_variance=torch.full((len(target),), float(sigma)**2),
                  measurement_standard_deviation=torch.full((len(target),), float(sigma)),
                  video_id=list(dense['video_id']), start=torch.as_tensor(dense['start']).cpu())
    return sparse, initial


def metrics(prediction, sparse):
    error = prediction.float()-sparse['target'].float()
    unseen = sparse['valid_mask']*(1-sparse['sampling_mask'])
    mask = sparse['sampling_mask']
    frame_mse = error.square().flatten(2).mean(2)
    result = []
    for i in range(len(error)):
        result.append(dict(mse=float(frame_mse[i].mean()), mae=float(error[i].abs().mean()),
            psnr=float((-10*frame_mse[i].clamp_min(1e-12).log10()).mean()),
            unobserved_mse=float((error[i].square()*unseen[i]).sum()/unseen[i].sum().clamp_min(1)),
            observed_mse=float((error[i].square()*mask[i]).sum()/mask[i].sum().clamp_min(1)),
            temporal_delta_mse=float((error[i,1:]-error[i,:-1]).square().mean())))
    return result


def summary(rows):
    keys = ('mse','mae','psnr','unobserved_mse','observed_mse','temporal_delta_mse')
    return {group: {k: sum(r[k] for r in subset)/len(subset) for k in keys}
            for group, subset in [('all', rows), ('clean', [r for r in rows if r['sigma']==0]),
                                  ('high_noise', [r for r in rows if r['sigma']>=.05])]
            if subset}
