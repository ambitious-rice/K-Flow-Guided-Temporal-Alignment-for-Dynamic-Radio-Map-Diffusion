"""Measurement-noise-dependent correction inside the existing DDIM chain.

This is an experimental data-consistency module, not an exact posterior update:
the pretrained denoiser is already conditioned on these measurements.
"""
import math
import torch
from torch import nn
from diffusers import DDIMScheduler
from experimental.noise_loss_screen.common import public


class NoiseWeightedFusion(nn.Module):
    def __init__(self,prior_std):
        super().__init__()
        if not math.isfinite(prior_std) or prior_std<=0:raise ValueError('prior_std must be positive')
        self.log_prior_variance=nn.Parameter(torch.tensor(math.log(prior_std**2)))

    def forward(self,prediction,observed,mask,measurement_variance):
        q=measurement_variance.float().reshape(-1,1,1,1,1)
        r=self.log_prior_variance.exp()
        weight=r/(r+q)
        return prediction+mask*weight*(observed-prediction)


def sample(model,config,sparse,initial,*,fusion=None,film_mode='same',steps=50,final_only=False):
    """Caller chooses grad/inference context; no complete target is accessed.

    External physical q enters fusion unchanged. 'zero' is an explicit ablation
    of the original FiLM route, never a calibration of the physical noise.
    """
    if film_mode not in ('same','zero'):raise ValueError(film_mode)
    q=sparse['measurement_variance']
    if not torch.isfinite(q).all() or (q<0).any():raise ValueError('Invalid measurement variance')
    conditions=public(sparse,q if film_mode=='same' else torch.zeros_like(q))
    cache=model.encode_conditions(conditions)
    scheduler=DDIMScheduler(num_train_timesteps=config.train_timesteps,beta_schedule=config.beta_schedule,prediction_type=config.prediction_type,clip_sample=True,set_alpha_to_one=True,steps_offset=0)
    if config.prediction_type!='sample':raise ValueError('Fusion is defined for x0 prediction')
    scheduler.set_timesteps(steps,device=initial.device);state=initial
    for t in scheduler.timesteps:
        tt=torch.full((len(state),),int(t),device=state.device,dtype=torch.long)
        prediction=model.denoise(scheduler.scale_model_input(state,t),tt,cache)
        if fusion is not None and not final_only:prediction=fusion(prediction,sparse['observed_rss'],sparse['sampling_mask'],q)
        state=scheduler.step(prediction,t,state,eta=0,use_clipped_model_output=False,return_dict=False)[0]
    output=state.clamp(0,1)
    if fusion is not None and final_only:output=fusion(output,sparse['observed_rss'],sparse['sampling_mask'],q).clamp(0,1)
    return output
