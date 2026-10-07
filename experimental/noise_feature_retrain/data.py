"""600 videos, fixed integer seeds, continuous paired noise; no test data."""
import json
from pathlib import Path
import numpy as np
import torch

class PairedData:
    def __init__(self,inputs,seed=20261008):
        self.inputs=Path(inputs);self.seed=seed;self.records=json.loads((self.inputs/'train_videos.json').read_text())
        cache=Path(json.loads((self.inputs/'protocol.json').read_text())['packed_cache'])
        self.arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ('targets','vehicles','buildings')}

    def batch(self,step,micro,rank,base_videos=2):
        # rank and accumulation are part of the explicit arithmetic seed.
        seed=self.seed+step*100000+micro*10000+rank*1000;g=torch.Generator().manual_seed(seed)
        ids=torch.randint(len(self.records),(base_videos,),generator=g).tolist();starts=torch.randint(85,(base_videos,),generator=g).tolist();rates=torch.randint(1,4,(base_videos,),generator=g)
        value={k:[] for k in ('target','vehicle','building','sampling_mask')}
        for index,(vid,start) in enumerate(zip(ids,starts)):
            rec=self.records[vid];target=torch.from_numpy(self.arrays['targets'][rec['packed_index'],start:start+16].copy()).float()[:,None]/255
            vehicle=torch.from_numpy(self.arrays['vehicles'][rec['episode_index'],start:start+16].copy()).float()[:,None]
            building=torch.from_numpy(self.arrays['buildings'][rec['scene_index']].copy()).float()[None,None].expand(16,1,-1,-1)/255
            valid=(building<=.5)&(vehicle<=.5);mask=torch.zeros_like(target)
            for t in range(16):
                points=valid[t].flatten().nonzero().flatten();gg=torch.Generator().manual_seed(seed+100+index*100+t);order=points[torch.randperm(len(points),generator=gg)]
                mask[t].view(-1)[order[:max(1,round(len(points)*int(rates[index])/100))]]=1
            for k,x in [('target',target),('vehicle',vehicle),('building',building),('sampling_mask',mask)]:value[k].append(x)
        value={k:torch.stack(v).repeat_interleave(4,dim=0) for k,v in value.items()}
        # One clean example + independent continuous draws from3 equal ranges.
        u=torch.rand((base_videos,3),generator=g);sigma=torch.cat((torch.zeros(base_videos,1),(u+torch.arange(3)[None])*.03),1).flatten()
        shape=(base_videos,*value['target'].shape[1:]);obsnoise=torch.randn(shape,generator=torch.Generator().manual_seed(seed+700)).repeat_interleave(4,0)
        diffnoise=torch.randn(shape,generator=torch.Generator().manual_seed(seed+701)).repeat_interleave(4,0)
        time=torch.randint(1000,(base_videos,),generator=torch.Generator().manual_seed(seed+702)).repeat_interleave(4)
        value.update(observed_rss=value['sampling_mask']*(value['target']+sigma[:,None,None,None,None]*obsnoise),measurement_variance=sigma.square(),sampling_rate=rates.repeat_interleave(4)[:,None].expand(-1,16).float(),diffusion_noise=diffnoise,timesteps=time)
        return value
