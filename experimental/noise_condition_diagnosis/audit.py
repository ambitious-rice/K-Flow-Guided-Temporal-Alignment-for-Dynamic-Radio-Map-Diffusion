"""Validation-only causal probes. No parameters or production code are changed."""
import argparse,json,time,os
from pathlib import Path
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_hvdit.model import NoiseHVDiT
from experimental.noise_loss_screen.common import TrainingData,public,loss_terms,metrics,write
from rmdm.diffusion import DiffusionProcess,DDIMSampler

ROOT=Path('runs/noise_condition_diagnosis_20261007')
OLD=Path('runs/noise_loss_screen_20261007')
INITIAL=Path('runs/noise_iterative_remote_20261007/inputs/w16_model.pt')

class Cached:
    def __init__(self,model,cache):self.model=model;self.cache=cache
    def encode_conditions(self,_):return self.cache
    def denoise(self,x,t,cache):return self.model.denoise(x,t,cache)

def load():
    cfg=load_config('experimental/noise_hvdit/w16_tolerance.yaml');model=NoiseHVDiT(cfg).cuda()
    state=torch.load(INITIAL,map_location='cpu',weights_only=False)['model'];model.load_state_dict(state)
    return cfg,model,state

def bank():
    return [e for e in json.loads((OLD/'inputs/validation_bank.json').read_text()) if e['group']=='tune']

def fetch(e):
    d=torch.load(OLD/'inputs'/e['file'],weights_only=True)
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()},d['initial'].cuda()

def gradient_probe():
    cfg,model,_=load();data=TrainingData(OLD/'inputs');diff=DiffusionProcess(cfg.diffusion);rows=[]
    groups={name:[p for n,p in model.named_parameters() if n.startswith(prefix)] for name,prefix in [('sigma_embedding','variance_embedding.'),('rate_embedding','rate_embedding.'),('hwm','hwm.'),('dit_heads','heads.')]}
    # Three independent training minibatches, no optimizer or BN updates persist.
    for step in [11,23,37]:
        b={k:v[:2].cuda() for k,v in data.batch(step).items()};model.train()
        noisy=diff.scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
        pred,cal=model(noisy,b['timesteps'],public(b,b['measurement_variance']))
        _,terms=loss_terms(pred,cal,b,b['source'],dict(k=1.,observation_weight=1.,condition=True))
        params=[p for ps in groups.values() for p in ps];vectors={};norms={}
        for loss in ['reconstruction','observation','calibration','equation','boundary','source']:
            gs=torch.autograd.grad(terms[loss],params,retain_graph=True,allow_unused=True)
            offset=0;vectors[loss]={};norms[loss]={}
            for group,ps in groups.items():
                selected=gs[offset:offset+len(ps)];offset+=len(ps)
                norms[loss][group]=sum(float(g.detach().square().sum()) for g in selected if g is not None)**.5
                if group.endswith('embedding'):vectors[loss][group]=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(selected,ps)])
        cos={}
        for group in ['sigma_embedding','rate_embedding']:
            a=vectors['reconstruction'][group];aux=sum(vectors[l][group] for l in ['calibration','equation','boundary','source']);cos[group]=float(torch.nn.functional.cosine_similarity(a,aux,dim=0))
        rows.append(dict(step=step,timesteps=b['timesteps'].tolist(),sigma=b['measurement_variance'].sqrt().tolist(),losses={k:float(v.detach()) for k,v in terms.items()},gradient_norms=norms,auxiliary_vs_reconstruction_cosine=cos))
        write(ROOT/'gradients.json',dict(complete=False,rows=rows))
        del pred,cal,terms,vectors,noisy,b
    write(ROOT/'gradients.json',dict(complete=True,rows=rows))

@torch.no_grad()
def paths():
    cfg,model,_=load();model.eval();sampler=DDIMSampler(cfg.diffusion);diff=DiffusionProcess(cfg.diffusion);rows=[];direct=[]
    for e in bank():
        b,initial=fetch(e);q=b['measurement_variance'];zero=torch.zeros_like(q)
        correct=model.encode_conditions(public(b,q));fixed=model.encode_conditions(public(b,zero))
        caches={'correct':correct,'zero':fixed,'dit_correct_hwm_zero':dict(correct,hwm_gate=fixed['hwm_gate']), 'dit_zero_hwm_correct':dict(fixed,hwm_gate=correct['hwm_gate'])}
        for mode,cache in caches.items():
            pred=sampler.sample(Cached(model,cache),{},initial_noise=initial,steps=20,eta=0)
            rows.append(dict(**e,mode=mode,metrics=metrics(pred,b)))
        for t in [0,100,500,900,999]:
            tt=torch.tensor([t],device='cuda');noisy=diff.scheduler.add_noise(b['target'],initial,tt)
            for mode in ['correct','zero']:
                pred=model.denoise(noisy,tt,caches[mode]);direct.append(dict(**e,t=t,mode=mode,metrics=metrics(pred,b)))
        write(ROOT/'paths.json',dict(complete=False,rows=rows,direct=direct))
    write(ROOT/'paths.json',dict(complete=True,rows=rows,direct=direct))

@torch.no_grad()
def buffers():
    cfg,model,original=load();sampler=DDIMSampler(cfg.diffusion);rows=[]
    bnkeys={k for k in original if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    last=torch.load(OLD/'k1/last.pt',map_location='cpu',weights_only=False)['model']
    drift={k:float((last[k].float()-original[k].float()).norm()) for k in bnkeys}
    for mode in ['initial','trained','trained_original_bn','initial_trained_bn','trained_original_hwm']:
        state=original if mode.startswith('initial') else last
        state=dict(state)
        if mode=='trained_original_bn':state.update({k:original[k] for k in bnkeys})
        if mode=='initial_trained_bn':state.update({k:last[k] for k in bnkeys})
        if mode=='trained_original_hwm':state.update({k:v for k,v in original.items() if k.startswith('hwm.')})
        model.load_state_dict(state);model.eval()
        for e in bank():
            b,initial=fetch(e);pred=sampler.sample(model,public(b,b['measurement_variance']),initial_noise=initial,steps=20,eta=0)
            rows.append(dict(**e,mode=mode,metrics=metrics(pred,b)))
        write(ROOT/'buffers.json',dict(complete=False,rows=rows,bn_buffer_drift=drift))
    write(ROOT/'buffers.json',dict(complete=True,rows=rows,bn_buffer_drift=drift))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['gradients','paths','buffers']);a=p.parse_args()
    torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(ROOT/(a.mode+'_status.json'),dict(state='running',pid=os.getpid(),start=time.time(),device=torch.cuda.get_device_name()))
    {'gradients':gradient_probe,'paths':paths,'buffers':buffers}[a.mode]()
    write(ROOT/(a.mode+'_status.json'),dict(state='complete',pid=os.getpid(),finish=time.time()))
if __name__=='__main__':main()
