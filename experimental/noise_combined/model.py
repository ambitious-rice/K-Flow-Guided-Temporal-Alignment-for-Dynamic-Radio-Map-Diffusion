"""Noise maps plus independent measurement-sigma modulation in DiT and HWM.

The original timestep pathway stays intact. Extra per-block outputs start at
zero, preserving the baseline's initial forward function, not reinitializing
the existing Wan-style blocks as strict DiT adaLN-Zero blocks.
"""
import math
import torch
from torch import nn
from experimental.noise_hvdit.model import mlp
from experimental.noise_paper_injection.model import PaperNoiseDiT
from rmdm_hvdit_v4_joint.model.common import zero_linear

VARIANTS = ('map_paired', 'hybrid_iid', 'hybrid_paired')


class CombinedNoiseDiT(PaperNoiseDiT):
    def __init__(self, config, variant, rollout_gap=20, prox_lambda=7.):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(config, 'noise_map', rollout_gap, prox_lambda)
        self.experiment_variant = variant
        self.hybrid = variant.startswith('hybrid')
        if self.hybrid:
            # Isolate new initialization so shared backbone and RNG match the
            # historical noise-map arm exactly with seed20261009.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261010)
                self.sigma_embedding = mlp(1, config.embedding_width)
                self.sigma_blocks = nn.ModuleDict({
                    name: nn.ModuleList([zero_linear(nn.Linear(config.embedding_width, 6*block.dim))
                                         for block in blocks])
                    for name, blocks in [('encoder', self.denoiser.local_encoder),
                                         ('bottleneck', self.denoiser.global_bottleneck),
                                         ('decoder', self.denoiser.local_decoder)]})
            self.audit_prefixes += ['sigma_embedding', 'sigma_blocks']

    def encode_conditions(self, batch):
        if not self.hybrid:
            return super().encode_conditions(batch)
        raw = {k: batch[k] for k in ('building', 'vehicle', 'observed_rss', 'sampling_mask')}
        raw['tx'] = torch.zeros_like(raw['building'])
        q = batch['measurement_variance'].float().reshape(-1)
        sigma = q.sqrt()
        rate = batch['sampling_rate'].float().reshape(len(q), -1).mean(1)
        er = self.rate_embedding(torch.log(rate.clamp_min(1e-4))[:, None] / math.log(10))
        es = self.sigma_embedding((sigma/.09)[:, None])
        e = er + es
        values = {k: head(e) for k, head in self.heads.items()}
        cache = dict(raw, **self.hwm(raw, e), q=q, rate=rate,
                     local_measurement_modulation=values['local'].reshape(len(q), 6, -1),
                     global_measurement_modulation=values['global'].reshape(len(q), 6, -1),
                     measurement_condition=values['decoder'], measurement_y=raw['observed_rss'])
        cache['input_observation_modulation'] = values['input_observation'].chunk(2, -1)
        cache['condition_observation_modulation'] = values['condition_observation'].chunk(2, -1)
        self.set_noise_map(cache, sigma)
        for name, heads in self.sigma_blocks.items():
            cache[name+'_measurement_modulations'] = tuple(head(es).reshape(len(q), 6, -1) for head in heads)
        cache['condition_high'], cache['condition_low'] = self.denoiser.encode_raw_conditions(cache)
        return cache
