"""Full9000-video training, discrete uniform rate and continuous uniform sigma."""
import json
from pathlib import Path
import numpy as np
import torch


class FullData:
    def __init__(self,inputs,seed=20261008):
        self.seed=seed;inputs=Path(inputs)
        self.records=json.loads((inputs/'train_videos.json').read_text())
        cache=Path(json.loads((inputs/'protocol.json').read_text())['packed_cache'])
        self.arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ('targets','vehicles','buildings')}
        self.sources=np.load('/dev/shm/noise_hvdit_source_masks.npy',mmap_mode='r')
        assert len(self.records)==len(self.sources)==len(self.arrays['targets'])==9000

    def batch(self,step,micro,rank,count):
        seed=self.seed+step*100000+micro*10000+rank*1000
        def generator(offset):return torch.Generator().manual_seed(seed+offset)
        ids=torch.randint(len(self.records),(count,),generator=generator(0)).tolist()
        starts=torch.randint(85,(count,),generator=generator(1)).tolist()
        rates=torch.randint(1,11,(count,),generator=generator(2))
        sigmas=.09*torch.rand(count,generator=generator(3))
        values={k:[] for k in ('target','vehicle','building','source_label','sampling_mask')}
        for i,(vid,start) in enumerate(zip(ids,starts)):
            rec=self.records[vid];idx=rec['packed_index']
            target=torch.from_numpy(self.arrays['targets'][idx,start:start+16].copy()).float()[:,None]/255
            vehicle=torch.from_numpy(self.arrays['vehicles'][rec['episode_index'],start:start+16].copy()).float()[:,None]
            building=torch.from_numpy(self.arrays['buildings'][rec['scene_index']].copy()).float()[None,None].expand(16,1,-1,-1)/255
            source=torch.from_numpy(self.sources[idx].copy()).float()[None,None].expand(16,1,-1,-1)
            valid=(building<=.5)&(vehicle<=.5);mask=torch.zeros_like(target)
            for t in range(16):
                points=valid[t].flatten().nonzero().flatten();order=points[torch.randperm(len(points),generator=generator(100+i*32+t))]
                mask[t].view(-1)[order[:max(1,round(len(points)*int(rates[i])/100))]]=1
            for key,x in [('target',target),('vehicle',vehicle),('building',building),('source_label',source),('sampling_mask',mask)]:values[key].append(x)
        values={k:torch.stack(v) for k,v in values.items()}
        eps=torch.randn(values['target'].shape,generator=generator(700))
        values.update(observed_rss=values['sampling_mask']*(values['target']+sigmas[:,None,None,None,None]*eps),measurement_variance=sigmas.square(),sampling_rate=rates[:,None].expand(-1,16).float(),
            diffusion_noise=torch.randn(values['target'].shape,generator=generator(701)),timesteps=torch.randint(1000,(count,),generator=generator(702)))
        return values
