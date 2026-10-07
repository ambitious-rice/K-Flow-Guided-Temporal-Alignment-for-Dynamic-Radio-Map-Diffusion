"""Explicit, optional inference calibration; measurement variance stays physical."""
import torch
from experimental.noise_loss_screen.common import public


def calibrate_variance(variance, calibration):
    """Map physical q to an internal embedding coordinate, never a noise estimate."""
    q=variance.float()
    if not torch.isfinite(q).all() or (q<0).any():raise ValueError('Physical variance must be finite and nonnegative')
    knots=q.new_tensor(calibration['external_sigmas']).square()
    values=q.new_tensor(calibration['internal_sigmas']).square()
    if len(knots)!=len(values) or len(knots)<2 or not (knots[1:]>knots[:-1]).all() or not (values[1:]>=values[:-1]).all():raise ValueError('Invalid monotone calibration')
    bounded=q.clamp(knots[0],knots[-1]);hi=torch.searchsorted(knots,bounded.contiguous(),right=True).clamp(1,len(knots)-1);lo=hi-1
    fraction=(bounded-knots[lo])/(knots[hi]-knots[lo])
    return values[lo]+fraction*(values[hi]-values[lo])


@torch.no_grad()
def reconstruct(model,sparse,initial_noise,calibration):
    """A single x0 prediction at t999; this is not default DDIM(steps=1).

    The supplied model must be in eval mode. Ground truth/source fields are
    discarded at the input boundary. Does not modify sparse or its physical q.
    """
    if model.training:raise ValueError('Reconstruction requires model.eval()')
    q=calibrate_variance(sparse['measurement_variance'],calibration)
    cache=model.encode_conditions(public(sparse,q))
    t=torch.full((len(initial_noise),),999,device=initial_noise.device,dtype=torch.long)
    return model.denoise(initial_noise,t,cache).clamp(0,1)
