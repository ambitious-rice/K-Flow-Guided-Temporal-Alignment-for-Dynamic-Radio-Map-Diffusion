import json
from pathlib import Path
import numpy as np
import torch
from utils import cal_pinn_components

VARIANTS={}
for name,k in [('k1',1.),('k2',2.),('k3',3.),('off',0.)]:
    for condition in (True,False):
        VARIANTS[name+('' if condition else '_nosigma')]=dict(k=k,observation_weight=0. if name=='off' else 1.,condition=condition)
VARIANTS.update(k4=dict(k=4.,observation_weight=1.,condition=True),k6=dict(k=6.,observation_weight=1.,condition=True))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def public(sparse,variance):
    return {**{k:sparse[k] for k in ('building','vehicle','observed_rss','sampling_mask','sampling_rate')},'measurement_variance':variance}

def observation_loss(pred,sparse,k):
    mask=sparse['sampling_mask'];sigma=sparse['measurement_variance'].sqrt()[:,None,None,None,None]
    excess=((pred.float()-sparse['observed_rss']).abs()-k*sigma).clamp_min(0)
    denom=mask.sum().clamp_min(1)
    return (excess.square()*mask).sum()/denom,((excess>0)*mask).sum()/denom

def loss_terms(pred,cal,sparse,source,variant):
    reconstruction=(pred.float()-sparse['target']).square().mean()
    observation,active=observation_loss(pred,sparse,variant['k'])
    calibration=(cal.float()-sparse['target']).square().mean()
    obstacle=((sparse['building']>.5)|(sparse['vehicle']>.5)).float()
    equation,boundary,anchor=cal_pinn_components(cal.float().flatten(0,2),obstacle.flatten(0,2),source.flatten(0,2),k=.2)
    # Preserve all existing HVDiT auxiliary losses. Only observation k/weight and
    # the model's access to sigma vary; source labels never enter model inputs.
    terms=dict(reconstruction=reconstruction,observation=observation,calibration=calibration,
        equation=equation.mean(),boundary=boundary.mean(),source=anchor.mean())
    loss=reconstruction+variant['observation_weight']*observation+calibration+terms['equation']+terms['boundary']+terms['source']
    return loss,dict(terms,observation_active_fraction=active)

class TrainingData:
    def __init__(self,root):
        root=Path(root);self.spec=json.loads((root/'protocol.json').read_text())
        self.arrays={k:np.load(root/f'train_{k}.npy',mmap_mode='r') for k in ('targets','vehicles','buildings','sources')}
    def batch(self,step):
        seed=self.spec['seed']+step*10000;g=torch.Generator().manual_seed(seed)
        n=self.spec['global_batch'];count=len(self.arrays['targets'])
        vids=torch.randint(count,(n,),generator=g);starts=torch.randint(85,(n,),generator=g)
        rates=torch.randint(1,4,(n,),generator=g).float()
        component=torch.rand(n,generator=g);level=torch.rand(n,generator=g)
        sigmas=torch.where(component<.2,0.,torch.where(component<.85,level*.05,.05+level*.04))
        values={k:[] for k in ('building','vehicle','target','source')}
        for vi,st in zip(vids.tolist(),starts.tolist()):
            values['target'].append(torch.from_numpy(self.arrays['targets'][vi,st:st+16].copy()).float()[:,None]/255)
            values['vehicle'].append(torch.from_numpy(self.arrays['vehicles'][vi,st:st+16].copy()).float()[:,None])
            for key,arr in [('building','buildings'),('source','sources')]:
                x=torch.from_numpy(self.arrays[arr][vi].copy()).float()[None,None].expand(16,1,-1,-1)
                values[key].append(x/255 if key=='building' else x)
        values={k:torch.stack(v) for k,v in values.items()};mask=torch.zeros_like(values['target'])
        for b in range(n):
            for t in range(16):
                valid=((values['building'][b,t]<=.5)&(values['vehicle'][b,t]<=.5)).flatten().nonzero().flatten()
                gg=torch.Generator().manual_seed(seed+100+b*100+t)
                order=valid[torch.randperm(len(valid),generator=gg)]
                mask[b,t].view(-1)[order[:max(1,round(float(rates[b])/100*len(valid)))]]=1
        noise=torch.randn(mask.shape,generator=torch.Generator().manual_seed(seed+9000))
        values.update(sampling_mask=mask,observed_rss=mask*(values['target']+sigmas[:,None,None,None,None]*noise),
            measurement_variance=sigmas.square(),sampling_rate=rates[:,None].expand(-1,16))
        diffusion_generator=torch.Generator().manual_seed(seed+9001)
        values['timesteps']=torch.randint(1000,(n,),generator=diffusion_generator)
        values['diffusion_noise']=torch.randn(mask.shape,generator=diffusion_generator)
        return values

def metrics(pred,sparse):
    error=pred.float()-sparse['target'];unseen=sparse['valid_mask']*(1-sparse['sampling_mask'])
    return dict(mse=float(error.square().mean()),psnr=float((-10*error.square().flatten(2).mean(2).clamp_min(1e-12).log10()).mean()),
        unobserved_mse=float((error.square()*unseen).sum()/unseen.sum().clamp_min(1)),
        temporal_delta_mse=float((error[:,1:]-error[:,:-1]).square().mean()))
