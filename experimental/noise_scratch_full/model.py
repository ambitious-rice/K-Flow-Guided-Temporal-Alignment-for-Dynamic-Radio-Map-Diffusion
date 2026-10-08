"""Full HVDiT/HWM initialized from scratch; shared continuous sigma conditioning."""
import math
import torch
from experimental.noise_hvdit.model import NoiseHVDiT


class ScratchNoiseDiT(NoiseHVDiT):
    variant='scratch_shared'

    def encode_conditions(self,batch):
        raw={k:batch[k] for k in ('building','vehicle','observed_rss','sampling_mask')}
        raw['tx']=torch.zeros_like(raw['building'])
        q=batch['measurement_variance'].float().reshape(-1)
        features=torch.stack((q.clamp_min(0).sqrt()/.09,torch.log1p(q/.0009)),-1)
        rate=batch['sampling_rate'].float().reshape(len(q),-1).mean(1)
        e=self.variance_embedding(features)+self.rate_embedding(torch.log(rate.clamp_min(1e-4))[:,None]/math.log(10))
        values={k:head(e) for k,head in self.heads.items()}
        for key in ('input_observation','condition_observation'):raw[key+'_modulation']=values[key].chunk(2,-1)
        high,low=self.denoiser.encode_raw_conditions(raw)
        return {**raw,**self.hwm(raw,e),'condition_high':high,'condition_low':low,
            'local_measurement_modulation':values['local'].reshape(len(q),6,-1),
            'global_measurement_modulation':values['global'].reshape(len(q),6,-1),'measurement_condition':values['decoder']}
