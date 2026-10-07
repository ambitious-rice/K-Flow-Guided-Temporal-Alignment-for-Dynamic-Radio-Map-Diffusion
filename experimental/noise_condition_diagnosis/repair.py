"""Small validation-only repairs; existing production model is never modified."""
import argparse,copy,json,math,os,time
from pathlib import Path
import torch
from experimental.noise_hvdit.model import NoiseHVDiT
from experimental.noise_loss_screen.common import TrainingData,public,loss_terms,write,save
from experimental.noise_loss_screen.run import evaluate
from rmdm.diffusion import DiffusionProcess
from .audit import ROOT,OLD,load

VARIANTS={
 'split':dict(freeze_hwm=False,paired=False,adapter=False,mse_only=False,lr=1e-5),
 'frozen_hwm':dict(freeze_hwm=True,paired=False,adapter=False,mse_only=False,lr=1e-5),
 'paired_mse':dict(freeze_hwm=True,paired=True,adapter=False,mse_only=True,lr=1e-5),
 'paired_adapter':dict(freeze_hwm=True,paired=True,adapter=True,mse_only=True,lr=1e-4),
}

class SplitNoiseHVDiT(NoiseHVDiT):
    def split_embeddings(self):
        self.hwm_variance_embedding=copy.deepcopy(self.variance_embedding)
        self.hwm_rate_embedding=copy.deepcopy(self.rate_embedding)

    def encode_conditions(self,batch):
        raw={k:batch[k] for k in ('building','vehicle','observed_rss','sampling_mask')}
        raw['tx']=torch.zeros_like(raw['building'])
        variance=batch['measurement_variance'].float().reshape(-1);ratio=variance/self.reference_variance
        features=[ratio,torch.log1p(ratio)]
        if self.use_clean_indicator:features.append((variance==0).float())
        features=torch.stack(features,-1)
        rate=batch['sampling_rate'].float().reshape(len(variance),-1).mean(1)
        r=torch.log(rate.clamp_min(1e-4))[:,None]/math.log(10)
        e=self.variance_embedding(features)+self.rate_embedding(r)
        h=self.hwm_variance_embedding(features)+self.hwm_rate_embedding(r)
        raw['input_observation_modulation']=self.heads['input_observation'](e).chunk(2,-1)
        raw['condition_observation_modulation']=self.heads['condition_observation'](e).chunk(2,-1)
        high,low=self.denoiser.encode_raw_conditions(raw)
        return {**raw,**self.hwm(raw,h),'condition_high':high,'condition_low':low,
          'local_measurement_modulation':self.heads['local'](e).reshape(len(e),6,-1),
          'global_measurement_modulation':self.heads['global'](e).reshape(len(e),6,-1),
          'measurement_condition':self.heads['decoder'](e)}

def pair_batch(batch,step):
    # Two scenes/windows per batch, each at all four actual measurement sigmas.
    # Same masks, diffusion state, and measurement Gaussian field within a quartet.
    out={k:v[:2].repeat_interleave(4,dim=0) for k,v in batch.items()}
    sigmas=torch.tensor([0,.03,.05,.09]*2)
    noise=torch.randn(batch['target'][:2].shape,generator=torch.Generator().manual_seed(20261007+step*10000+9500)).repeat_interleave(4,0)
    out['measurement_variance']=sigmas.square()
    out['observed_rss']=out['sampling_mask']*(out['target']+sigmas[:,None,None,None,None]*noise)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('variant',choices=list(VARIANTS));p.add_argument('--smoke',action='store_true');a=p.parse_args();v=VARIANTS[a.variant]
    out=ROOT/('smoke_'+a.variant if a.smoke else a.variant);out.mkdir(exist_ok=True)
    torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg,base,state=load();model=SplitNoiseHVDiT(cfg).cuda();model.load_state_dict(state);model.split_embeddings()
    data=TrainingData(OLD/'inputs');diff=DiffusionProcess(cfg.diffusion)
    # End-to-end compatibility and gradient-isolation checks on actual data.
    b={k:x[:1].cuda() for k,x in data.batch(1).items()};noisy=diff.scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
    base.eval();model.eval()
    with torch.no_grad():
        expected,expected_cal=base(noisy,b['timesteps'],public(b,b['measurement_variance']))
        actual,cal=model(noisy,b['timesteps'],public(b,b['measurement_variance']))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(cal,expected_cal,rtol=0,atol=0)
    _,cal=model(noisy,b['timesteps'],public(b,b['measurement_variance']))
    gs=torch.autograd.grad(cal.square().mean(),list(model.variance_embedding.parameters()),allow_unused=True)
    assert all(g is None or g.count_nonzero()==0 for g in gs),'Auxiliary gradient leaked into DiT sigma embedding'
    del base,state,b,noisy,actual,expected,expected_cal,cal,gs
    if v['freeze_hwm']:
        for name,param in model.named_parameters():
            if name.startswith(('hwm.','hwm_variance_embedding.','hwm_rate_embedding.')):param.requires_grad_(False)
    if v['adapter']:
        for name,param in model.named_parameters():param.requires_grad_(name.startswith(('variance_embedding.','heads.')))
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=v['lr'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
    write(out/'protocol.json',dict(variant=v,trainable_parameters=sum(p.numel() for p in params),steps=600,batch=8,seed=20261007,initial_equivalence='bitwise passed FP32',gradient_isolation='passed',training_inputs=str(OLD/'inputs'),selection='tune8 correct-sigma unseen MSE DDIM20 at0,100,300,600; test never used',paired_caveat='Paired variants see2 unique windows per step vs8 in random variants; exploratory bundled change'))
    start=time.time();best=float('inf');best_step=0;steps=2 if a.smoke else 600
    def validate(step):
        nonlocal best,best_step
        rows=evaluate(model,cfg,OLD,dict(condition=True),steps=2 if a.smoke else 20,limit=1 if a.smoke else 0)
        score=sum(r['metrics']['unobserved_mse'] for r in rows)/len(rows)
        write(out/'validation'/f'step{step:04d}.json',dict(step=step,score=score,rows=rows))
        if score<best:
            best,best_step=score,step;save(out/'best.pt',dict(model=model.state_dict(),step=step,score=score,variant=v))
        model.train()
        if v['freeze_hwm']:model.hwm.eval()
    validate(0)
    for step in range(1,steps+1):
        tick=time.time();batch=data.batch(step)
        if v['paired']:batch=pair_batch(batch,step)
        b={k:x.cuda() for k,x in batch.items()};noisy=diff.scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
        lr=v['lr']*min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-50)/550)))
        for pg in optimizer.param_groups:pg['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred,cal=model(noisy,b['timesteps'],public(b,b['measurement_variance']))
            loss,terms=loss_terms(pred,cal,b,b['source'],dict(k=1.,observation_weight=0. if v['mse_only'] else 1.,auxiliary_weight=0. if v['freeze_hwm'] else 1.))
        assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn);optimizer.step()
        if step<=3 or step%20==0:
            row=dict(step=step,loss=float(loss.detach()),reconstruction=float(terms['reconstruction'].detach()),grad_norm=float(gn),lr=lr,seconds=time.time()-tick)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(state='training',pid=os.getpid(),step=step,best_step=best_step,elapsed=time.time()-start))
        del b,batch,noisy,pred,cal,terms,loss
        if step in [100,300,steps]:validate(step)
    save(out/'last.pt',dict(model=model.state_dict(),step=steps,variant=v))
    write(out/'status.json',dict(state='complete',pid=os.getpid(),step=steps,best_step=best_step,best_score=best,elapsed=time.time()-start))
if __name__=='__main__':main()
