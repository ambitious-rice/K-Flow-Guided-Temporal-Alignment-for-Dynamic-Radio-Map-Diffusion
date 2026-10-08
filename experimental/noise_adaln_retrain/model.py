"""Independent sigma modulation per block, retaining pretrained RMSNorm."""
import torch
from torch import nn
from experimental.noise_feature_retrain.model import FeatureNoiseModel
from rmdm_hvdit_v4_joint.model.common import zero_linear


class BlockNoiseModel(FeatureNoiseModel):
    def __init__(self,config,variant='adaln'):
        if variant!='adaln':raise ValueError(variant)
        super().__init__(config,'film')
        self.variant=variant

    def initialize(self,state):
        super().initialize(state)
        # New shared local/global heads are identically zero and frozen.
        # The pretrained sigma-zero path and shared stem/output heads remain.
        for name in ('local','global'):
            self.noise_heads[name].requires_grad_(False)
        self.block_noise_heads=nn.ModuleDict({
            name:nn.ModuleList([zero_linear(nn.Linear(self.config.embedding_width,6*block.dim)) for block in blocks])
            for name,blocks in (
                ('encoder',self.base.denoiser.local_encoder),
                ('bottleneck',self.base.denoiser.global_bottleneck),
                ('decoder',self.base.denoiser.local_decoder))})
        return self

    def encode_conditions(self,batch):
        cache=super().encode_conditions(batch)
        q=batch['measurement_variance'].float().reshape(-1)
        embedding=self.noise_embedding(torch.stack((q.clamp_min(0).sqrt()/.09,torch.log1p(q/.0009)),-1))
        for name,heads in self.block_noise_heads.items():
            cache[name+'_measurement_modulations']=tuple(head(embedding).reshape(len(q),6,-1) for head in heads)
        return cache
