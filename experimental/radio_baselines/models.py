"""Official baseline backbones adapted to the five-channel Tx-blind protocol."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.radiounet.modules import RadioWNet
from .vendor.rmegan.encoder_models import Discriminator


def conditions(sparse):
    """Only public observations enter predictors; target and Tx are never read."""
    building = sparse['building'][:, 0]
    return torch.cat((building, torch.zeros_like(building), sparse['vehicle'][:, 0],
                      sparse['observed_rss'][:, 0], sparse['sampling_mask'][:, 0]), 1)


def unet_forward(net, x, prefix=''):
    """Same layer order as official RadioWNet, without evaluating the unused U."""
    def layer(name, value):
        return getattr(net, prefix + name)(value)
    skips = {}
    value = x
    for name in ('00', '0', '1', '10', '2', '20', '3', '4', '5'):
        value = layer('layer' + name, value)
        skips[name] = value
    value = layer('conv_up5', value)
    for up, skip in (('4', '4'), ('3', '3'), ('20', '20'), ('2', '2'),
                     ('10', '10'), ('1', '1'), ('0', '0')):
        value = layer('conv_up' + up, torch.cat((value, skips[skip]), 1))
    value = layer('conv_up00', torch.cat((value, skips['00'], x), 1))
    return layer('conv_up000', torch.cat((value, x), 1))


class RadioUNet(nn.Module):
    def __init__(self, *, refinement=True, phase='first'):
        super().__init__()
        self.net = RadioWNet(inputs=5)
        self.refinement = refinement
        if not refinement:
            for name in list(self.net._modules):
                if name.startswith('W'):
                    delattr(self.net, name)
        self.set_phase(phase)

    def set_phase(self, phase):
        if phase not in ('first', 'second') or (phase == 'second' and not self.refinement):
            raise ValueError(phase)
        self.phase = phase
        for name, parameter in self.net.named_parameters():
            parameter.requires_grad_((name.startswith('W')) == (phase == 'second'))

    def forward(self, x):
        if self.phase == 'first':
            return unet_forward(self.net, x)
        with torch.no_grad():
            first = unet_forward(self.net, x)
        return unet_forward(self.net, torch.cat((first, x), 1), prefix='W')


class RMEGAN(RadioUNet):
    def __init__(self):
        super().__init__(refinement=False)
        # Condition D on the same observations for real and fake examples.
        # Upstream supplies the real/fake label as input and has incorrect real targets.
        self.discriminator = Discriminator(ngpu=1, nc=6)
        self.discriminator.main[-1] = nn.Identity()  # BCEWithLogitsLoss

    def discriminate(self, prediction, x):
        return self.discriminator(torch.cat((prediction, x), 1))


class RadioVAE(nn.Module):
    """RadioDiff's KL autoencoder, 4x spatial compression and three latent channels."""
    def __init__(self):
        super().__init__()
        from .vendor.radiodiff.encoder_decoder import Encoder, Decoder
        cfg = dict(double_z=True, z_channels=3, resolution=128, in_channels=1,
                   out_ch=1, ch=128, ch_mult=(1, 2, 4), num_res_blocks=2,
                   attn_resolutions=(), dropout=0.)
        self.encoder = Encoder(**cfg)
        self.decoder = Decoder(**cfg)
        self.quant_conv = nn.Conv2d(6, 6, 1)
        self.post_quant_conv = nn.Conv2d(3, 3, 1)
        self.logvar = nn.Parameter(torch.zeros(()))

    def encode(self, x):
        from .vendor.radiodiff.encoder_decoder import DiagonalGaussianDistribution
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample=True):
        posterior = self.encode(x)
        return self.decode(posterior.sample() if sample else posterior.mode()), posterior


class Config(dict):
    __getattr__ = dict.__getitem__


class RadioDiff(nn.Module):
    def __init__(self):
        super().__init__()
        from .vendor.radiodiff.mask_cond_unet import Unet
        cfg = Config(cond_net='swin', without_pretrain=True, fix_bb=False,
                     cond_pe=False, input_size=[32, 32])
        self.denoiser = Unet(dim=128, channels=3, dim_mults=(1, 2, 4, 4),
                            window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
                            window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]], cfg=cfg)
        self.denoiser.init_conv_mask.first_coonv[0] = nn.Conv2d(5, 128, 4, stride=4)
        # Unused upstream classification/car branches must not enter the optimizer.
        del self.denoiser.init_conv_carK
        self.denoiser.init_conv_mask.norm = nn.Identity()
        self.denoiser.init_conv_mask.head = nn.Identity()
        self.vae = RadioVAE().requires_grad_(False)
        self.register_buffer('scale_factor', torch.tensor(0.3))

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def forward(self, x, target):
        with torch.no_grad():
            z = self.vae.encode(target).sample() * self.scale_factor
        t = torch.rand(z.shape[0], device=z.device) * (1 - 1e-4) + 1e-4
        epsilon = torch.randn_like(z)
        time = t[:, None, None, None]
        noisy = (1 - time) * z + time.sqrt() * epsilon
        drift, noise = self.denoiser(noisy, t, x)
        # Official decoupled constant-SDE training objective and time weights.
        drift_loss = (drift.float() + z.float()).square().mean((1, 2, 3))
        noise_loss = (noise.float() - epsilon.float()).square().mean((1, 2, 3))
        return (2 * (1 - t).exp() * drift_loss + t.sqrt().exp() * noise_loss).mean()

    @torch.no_grad()
    def sample(self, x, initial_noise, step_noises):
        """Official SDE reverse update, with caller-owned per-frame random draws."""
        value = initial_noise.float()
        count = len(step_noises)
        for index, epsilon in enumerate(step_noises):
            t = 1. - index / count
            next_t = max(0., 1. - (index + 1) / count)
            dt = t - next_t
            times = torch.full((x.shape[0],), t, device=x.device)
            drift, noise = self.denoiser(value, times, x)
            x0 = value - drift.float() * t - noise.float() * t ** .5
            # Official sampler corrects drift to -x0 before the reverse transition.
            value = value + x0 * dt - dt / t ** .5 * noise.float()
            if next_t:
                value = value + (dt * next_t / t) ** .5 * epsilon.to(value)
        return self.vae.decode(value / self.scale_factor).float()


def build_model(method, phase='first'):
    if method == 'radiounet':
        return RadioUNet(phase=phase)
    if method == 'rmegan':
        return RMEGAN()
    if method == 'vae':
        return RadioVAE()
    if method == 'radiodiff':
        return RadioDiff()
    raise ValueError(method)
