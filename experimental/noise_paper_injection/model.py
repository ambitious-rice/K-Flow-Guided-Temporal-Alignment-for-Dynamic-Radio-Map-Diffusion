"""Paper-inspired sigma paths inside W16; all variants retain full diffusion.

These are RMDM adaptations, not reproductions of the original CNN/prior models.
HWM still sees sparse observations, so proximal updates are discriminative
architecture components, not independent Bayesian posterior updates.
"""
import math
import torch
from torch import nn
from experimental.noise_hvdit.model import NoiseHVDiT
from rmdm.diffusion import DiffusionProcess
from rmdm_hvdit_v4_joint.model.patching import W16DoubleStem
from rmdm_hvdit_v4_joint.model.common import modulate

VARIANTS = ('noise_map', 'usrnet_hyper', 'diffpir_prox')


def masked_prox(prior, observed, mask, rho):
    """argmin ||M*(x-y)||² + rho*||x-prior||², binary M, rho >= 0.

    Stable at rho=0: observed entries equal y, unobserved entries keep prior.
    """
    return prior + mask * (observed - prior) / (1 + rho)


class NoiseMapStem(W16DoubleStem):
    @classmethod
    def from_stem(cls, stem):
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        out.temporal_patch, out.spatial_patch = stem.temporal_patch, stem.spatial_patch
        out.dense_projection, out.fusion = stem.dense_projection, stem.fusion
        volume = out.temporal_patch * out.spatial_patch**2
        dim = stem.observation_projection.out_features
        out.observation_projection = nn.Linear(3 * volume, dim, bias=False)
        # Preserve identical initial RSS/mask weights across all variants.
        # Orthogonal initialization only for the added noise-map channel.
        extra = torch.empty(dim, volume)
        nn.init.orthogonal_(extra)
        with torch.no_grad():
            weight = out.observation_projection.weight.view(dim, out.temporal_patch, 3, out.spatial_patch**2)
            weight[:, :, :2].copy_(stem.observation_projection.weight.view(dim, out.temporal_patch, 2, out.spatial_patch**2))
            weight[:, :, 2].copy_(extra.view(dim, out.temporal_patch, out.spatial_patch**2))
        return out

    def forward(self, dense, observation, modulation):
        sigma = modulation['noise_map'].reshape(-1, 1, 1, 1, 1).to(observation)
        noise_map = sigma.expand_as(observation[:, :, :1])
        obs = self.observation_projection(self._pack(torch.cat((observation, noise_map), dim=2)))
        obs = modulate(obs, *modulation['affine'])
        return self.fusion(torch.cat((self.dense_projection(self._pack(dense)), obs), dim=-1))


class HyperParameters(nn.Module):
    """USRNet HyPaNet: sigma/degradation descriptors -> positive alpha,beta.

    Replace scale factor with sampling rate and add diffusion time for W16.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softplus())
        with torch.no_grad():
            self.net[-2].bias.copy_(torch.tensor([math.log(math.expm1(.01)), math.log(math.expm1(.03))]))
            self.net[-2].weight.mul_(.01)

    def forward(self, sigma, rate, timestep):
        return self.net(torch.stack((sigma / .09, rate / 10, timestep.float() / 999), -1)) + 1e-6


class PaperNoiseDiT(NoiseHVDiT):
    def __init__(self, config, variant, rollout_gap=20, prox_lambda=7.):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(config)
        self.variant, self.rollout_gap, self.prox_lambda = variant, rollout_gap, prox_lambda
        # Keep rate modulation and HWM; remove the old sigma embedding in ALL arms.
        del self.variance_embedding
        self.register_buffer('alpha_bar', DiffusionProcess(config.diffusion).scheduler.alphas_cumprod.clone())
        # Initialize additions without changing RNG for shared backbone weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261009)
            if variant in ('noise_map', 'usrnet_hyper'):
                for key in ['input_stem', 'condition_stem']:
                    setattr(self.denoiser, key, NoiseMapStem.from_stem(getattr(self.denoiser, key)))
            if variant == 'usrnet_hyper':
                self.hypa = HyperParameters()
        self.audit_prefixes = ['denoiser', 'hwm', 'rate_embedding', 'heads']
        if variant == 'usrnet_hyper':
            self.audit_prefixes.append('hypa')

    def encode_conditions(self, batch):
        raw = {k: batch[k] for k in ('building', 'vehicle', 'observed_rss', 'sampling_mask')}
        raw['tx'] = torch.zeros_like(raw['building'])
        q = batch['measurement_variance'].float().reshape(-1)
        rate = batch['sampling_rate'].float().reshape(len(q), -1).mean(1)
        e = self.rate_embedding(torch.log(rate.clamp_min(1e-4))[:, None] / math.log(10))
        values = {k: head(e) for k, head in self.heads.items()}
        hwm = self.hwm(raw, e)
        cache = dict(raw, **hwm, q=q, rate=rate,
                     local_measurement_modulation=values['local'].reshape(len(q), 6, -1),
                     global_measurement_modulation=values['global'].reshape(len(q), 6, -1), measurement_condition=values['decoder'])
        cache['input_observation_modulation'] = values['input_observation'].chunk(2, -1)
        cache['condition_observation_modulation'] = values['condition_observation'].chunk(2, -1)
        cache['measurement_y'] = raw['observed_rss']
        if self.variant != 'noise_map':
            # Direct DiT RSS path is routed through data updates in B/C.
            # Common HWM path remains observation-conditioned as documented.
            cache['observed_rss'] = torch.zeros_like(raw['observed_rss'])
        if self.variant == 'noise_map':
            self.set_noise_map(cache, q.sqrt())
        if self.variant != 'usrnet_hyper':
            cache['condition_high'], cache['condition_low'] = self.denoiser.encode_raw_conditions(cache)
        return cache

    @staticmethod
    def set_noise_map(cache, sigma):
        for name in ['input_observation_modulation', 'condition_observation_modulation']:
            value = cache[name]
            affine = value['affine'] if isinstance(value, dict) else value
            cache[name] = dict(affine=affine, noise_map=sigma)

    def denoise(self, noisy, timesteps, cache):
        a = self.alpha_bar[timesteps].float().reshape(-1, 1, 1, 1, 1)
        if self.variant == 'usrnet_hyper':
            ab = self.hypa(cache['q'].sqrt(), cache['rate'], timesteps).float()
            # USRNet order: data module -> nonblind learned prior module.
            clean_scale = noisy.float() / a.sqrt()
            data = masked_prox(clean_scale, cache['measurement_y'], cache['sampling_mask'], ab[:, 0].reshape(-1, 1, 1, 1, 1))
            local = dict(cache)
            self.set_noise_map(local, ab[:, 1])
            local['condition_high'], local['condition_low'] = self.denoiser.encode_raw_conditions(local)
            return self.denoiser(a.sqrt() * data, timesteps, local)
        prediction = self.denoiser(noisy, timesteps, cache)
        if self.variant == 'diffpir_prox':
            # DiffPIR Eq12: rho_t=lambda*sigma_y²/bar_sigma_t².
            sigma_t_sq = (1 - a) / a
            rho = self.prox_lambda * cache['q'].reshape(-1, 1, 1, 1, 1) / sigma_t_sq.clamp_min(1e-12)
            prediction = masked_prox(prediction.float(), cache['measurement_y'], cache['sampling_mask'], rho)
        return prediction

    def forward(self, noisy, timesteps, batch):
        cache = self.encode_conditions(batch)
        first = self.denoise(noisy, timesteps, cache)
        # Shared two-step differentiable DDIM training in ALL three arms.
        # Supervise the final x0 with the same six original clean losses.
        a = self.alpha_bar[timesteps].float().reshape(-1, 1, 1, 1, 1)
        previous = (timesteps - self.rollout_gap).clamp_min(0)
        ap = self.alpha_bar[previous].float().reshape(-1, 1, 1, 1, 1)
        epsilon = (noisy.float() - a.sqrt() * first.float()) / (1 - a).sqrt()
        state = ap.sqrt() * first.float().clamp(-1, 1) + (1 - ap).sqrt() * epsilon
        prediction = self.denoise(state, previous, cache)
        return prediction, cache['cal']
