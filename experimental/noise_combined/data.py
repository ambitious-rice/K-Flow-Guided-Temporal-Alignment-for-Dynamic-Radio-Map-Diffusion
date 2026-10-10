"""Same full data as baseline, with an optional four-noise paired batch."""
import torch
from experimental.noise_scratch_full.data import FullData


def paired_batch(original, seed):
    count = len(original['target'])
    if count % 4:
        raise ValueError('Paired microbatch must be divisible by4')
    groups = count//4
    # Take the first groups of the exact baseline batch, so sample identities
    # are deterministic and the IID arm is bitwise unchanged.
    indices = torch.arange(groups).repeat_interleave(4)
    batch = {k: v[indices].clone() for k, v in original.items()}
    g = torch.Generator().manual_seed(seed+800)
    levels = (torch.arange(4)[None, :] + torch.rand((groups, 4), generator=g)) * (.09/4)
    levels = torch.stack([v[torch.randperm(4, generator=g)] for v in levels]).flatten()
    # Match baseline's underlying Gaussian draws for the retained videos;
    # share them across levels, while diffusion noise remains independent.
    eps = torch.randn(original['target'].shape, generator=torch.Generator().manual_seed(seed+700))[indices]
    batch['measurement_variance'] = levels.square()
    batch['observed_rss'] = batch['sampling_mask']*(batch['target']+levels[:, None, None, None, None]*eps)
    return batch


class CombinedData(FullData):
    def __init__(self, inputs, seed, variant):
        super().__init__(inputs, seed)
        self.paired = variant.endswith('paired')

    def batch(self, step, micro, rank, count):
        original = super().batch(step, micro, rank, count)
        if not self.paired:
            return original
        seed = self.seed+step*100000+micro*10000+rank*1000
        return paired_batch(original, seed)
