"""Batch independent true-noise/rate cases, preserving four input-sigma outputs."""
import torch
from experimental.noise_loss_screen.common import public
from experimental.noise_video_screen.statistics import SIGMAS


def build_group(data, cases):
    device=data['target'].device
    batches=[];parts=[]
    for rate,sigma in cases:
        b={k:data[k].float() for k in ('target','building','vehicle','valid_mask')}
        mask=data['masks'][rate].to(device=device,dtype=torch.float32)
        b.update(sampling_mask=mask,observed_rss=mask*(b['target']+sigma*data['measurement_noise']),
                 sampling_rate=torch.full(b['target'].shape[:2],float(rate),device=device))
        item=public(b,torch.tensor(SIGMAS,device=device).square())
        assert set(item)=={'building','vehicle','observed_rss','sampling_mask','sampling_rate','measurement_variance'}
        parts.append({k:v if k=='measurement_variance' else v.repeat(4,*([1]*(v.ndim-1))) for k,v in item.items()})
        batches.append(b)
    inputs={k:torch.cat([x[k] for x in parts],dim=0) for k in parts[0]}
    return inputs,data['initial'].repeat(4*len(cases),1,1,1,1),batches
