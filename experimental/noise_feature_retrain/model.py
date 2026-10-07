"""Full W16 diffusion with noise-aware RSS tokens; no output projection/map."""
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from experimental.noise_hvdit.model import NoiseHVDiT,mlp
from rmdm_hvdit_v4_joint.model.common import modulate,zero_linear
from rmdm_hvdit_v4_joint.model.patching import W16DoubleStem

VARIANTS=('blind','film','reliability','reliability_fixed')


class ReliabilityStem(W16DoubleStem):
    """Keep mask/location tokens, attenuate only RSS token features.

    For linear RSS projection W and independent measurement variance q,
    noise variance per token channel is q*sum_p(W_jp²*M_p).
    A learned positive feature scale produces a Wiener-like reliability gate.
    This is feature attenuation, not a claim of an exact Bayesian posterior.
    """
    @classmethod
    def from_stem(cls,stem):
        out=cls.__new__(cls);nn.Module.__init__(out)
        out.temporal_patch=stem.temporal_patch;out.spatial_patch=stem.spatial_patch
        for name in ('dense_projection','observation_projection','fusion'):setattr(out,name,getattr(stem,name))
        out.log_signal_variance=nn.Parameter(torch.full((stem.observation_projection.out_features,),math.log(.03**2)))
        return out

    def forward(self,dense,observation,modulation=None):
        if not isinstance(modulation,dict):return super().forward(dense,observation,modulation)
        q=modulation['variance'].float().reshape(-1,1,1,1,1)
        y,m=observation.split(1,dim=2);zero=torch.zeros_like(y)
        yp=self._pack(torch.cat((y,zero),2))
        mask_y=self._pack(torch.cat((m,zero),2))
        yt=self.observation_projection(yp)
        original_token=self.observation_projection(self._pack(observation))
        # FP32 noise propagation protects small physical variances under BF16.
        with torch.autocast(device_type=dense.device.type,enabled=False):
            noise_variance=q*F.linear(mask_y.float(),self.observation_projection.weight.float().square())
            signal_variance=self.log_signal_variance.float().clamp(-16,4).exp()
            gain=signal_variance/(signal_variance+noise_variance)
        # Subtract only the attenuated RSS component. This is algebraically
        # gain*W[Y,0]+W[0,M], but recovers the original arithmetic exactly at q=0.
        token=original_token-(1-gain).to(yt.dtype)*yt
        affine=modulation.get('affine')
        if affine is not None:token=modulate(token,*affine)
        return self.fusion(torch.cat((self.dense_projection(self._pack(dense)),token),-1))


class FeatureNoiseModel(nn.Module):
    def __init__(self,config,variant):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.base=NoiseHVDiT(config)
        self.config=config

    def initialize(self,state):
        self.base.load_state_dict(state,strict=True)
        # Common initialization equals existing W16 evaluated at sigma0.
        # Freeze its rate/HWM/measurement modulation, retrain all DiT parameters.
        for p in self.base.parameters():p.requires_grad_(False)
        for p in self.base.denoiser.parameters():p.requires_grad_(True)
        if self.variant.startswith('reliability'):
            for name in ('input_stem','condition_stem'):
                setattr(self.base.denoiser,name,ReliabilityStem.from_stem(getattr(self.base.denoiser,name)))
        if self.variant=='film':
            width=self.config.embedding_width
            self.noise_embedding=mlp(2,width)
            self.noise_heads=nn.ModuleDict({name:zero_linear(nn.Linear(width,size)) for name,size in {
              'input_observation':2*self.config.model.local_dim,'condition_observation':2*self.config.model.local_dim,
              'local':6*self.config.model.local_dim,'global':6*self.config.model.global_dim,'decoder':self.config.model.mapping_width}.items()})
        return self

    def train(self,mode=True):
        super().train(mode);self.base.hwm.eval();return self

    def encode_conditions(self,batch):
        raw={k:batch[k] for k in ('building','vehicle','observed_rss','sampling_mask')};raw['tx']=torch.zeros_like(raw['building'])
        q=batch['measurement_variance'].float().reshape(-1)
        features=torch.zeros(len(q),3 if self.base.use_clean_indicator else 2,device=q.device)
        if self.base.use_clean_indicator:features[:,-1]=1
        rate=batch['sampling_rate'].float().reshape(len(q),-1).mean(1)
        with torch.no_grad():
            e=self.base.variance_embedding(features)+self.base.rate_embedding(torch.log(rate.clamp_min(1e-4))[:,None]/math.log(10))
            hwm=self.base.hwm(raw,e)
            old={k:head(e) for k,head in self.base.heads.items()}
        if self.variant=='film':
            noise=self.noise_embedding(torch.stack((q.clamp_min(0).sqrt()/.09,torch.log1p(q/.0009)),-1))
            values={k:old[k]+head(noise) for k,head in self.noise_heads.items()}
        else:values=old
        for key in ('input_observation','condition_observation'):
            affine=values[key].chunk(2,-1)
            raw[key+'_modulation']=dict(affine=affine,variance=torch.full_like(q,.03**2) if self.variant=='reliability_fixed' else q) if self.variant.startswith('reliability') else affine
        high,low=self.base.denoiser.encode_raw_conditions(raw)
        return {**raw,**hwm,'condition_high':high,'condition_low':low,
          'local_measurement_modulation':values['local'].reshape(len(q),6,-1),
          'global_measurement_modulation':values['global'].reshape(len(q),6,-1),
          'measurement_condition':values['decoder']}

    def denoise(self,noisy,t,cache):return self.base.denoiser(noisy,t,cache)
    def forward(self,noisy,t,batch):return self.denoise(noisy,t,self.encode_conditions(batch))
