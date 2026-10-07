"""Optimize actual five-step reconstruction; no wrong-sigma degradation loss."""
import argparse,json,math,os,time
from pathlib import Path
import torch
from diffusers import DDIMScheduler
from experimental.noise_loss_screen.common import TrainingData,public,write,save,metrics
from .audit import ROOT,OLD,load,fetch
from .repair import SplitNoiseHVDiT,pair_batch

def sample(model,b,initial,sigma=None):
    cache=model.encode_conditions(public(b,b['measurement_variance'] if sigma is None else torch.full_like(b['measurement_variance'],sigma**2)))
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='linear',prediction_type='sample',clip_sample=True,set_alpha_to_one=True,timestep_spacing='trailing')
    scheduler.set_timesteps(5,device=initial.device);x=initial
    for t in scheduler.timesteps:
        pred=model.denoise(x,torch.full((len(x),),int(t),device=x.device,dtype=torch.long),cache)
        # Preserve FP32 scheduler arithmetic even under autocast denoiser training.
        x=scheduler.step(pred.float(),t,x,eta=0,return_dict=False)[0]
    return x.clamp(0,1)

@torch.no_grad()
def evaluate(model,out,step,matrix=False):
    model.eval();rows=[]
    entries=json.loads((OLD/'inputs/validation_bank.json').read_text())
    if not matrix:entries=[e for e in entries if e['group']=='tune']
    for e in entries:
        b,initial=fetch(e)
        for sigma in ([0,.03,.05,.09] if matrix else [e['sigma']]):
            pred=sample(model,b,initial,sigma);rows.append(dict(**e,input_sigma=sigma,metrics=metrics(pred,b)))
    score=sum(r['metrics']['unobserved_mse'] for r in rows)/len(rows)
    write(out/(f'matrix_{step}.json' if matrix else f'validation/step{step:04d}.json'),dict(complete=True,step=step,score=score,rows=rows,sampler='DDIM5 trailing FP32'))
    return score

def main():
    p=argparse.ArgumentParser();p.add_argument('--lr',type=float,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    name=('smoke_' if a.smoke else '')+f'rollout_lr{a.lr:g}';out=ROOT/name;out.mkdir(exist_ok=True)
    torch.manual_seed(20261007);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg,base,state=load();model=SplitNoiseHVDiT(cfg).cuda();model.load_state_dict(state);model.split_embeddings();del base,state
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('variance_embedding.','heads.')))
    model.eval();params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),weight_decay=.01)
    data=TrainingData(OLD/'inputs');steps=2 if a.smoke else 150;started=time.time()
    write(out/'protocol.json',dict(steps=steps,lr=a.lr,batch=8,unique_videos_per_step=2,noise_sigmas=[0,.03,.05,.09],seed=20261007,trainable_parameters=sum(p.numel() for p in params),loss='Dense MSE after differentiable5-step DDIM trailing from pure Gaussian; no auxiliary/observation/ranking loss',sampling='FP32 scheduler, BF16 denoiser training; FP32 inference; same5 trailing steps',selection='tune8 unseen MSE at0,25,75,150 includes initial; confirmation only after selection',source='isolated experimental module; production and old checkpoints unchanged'))
    best=evaluate(model,out,0);best_step=0;save(out/'best.pt',dict(model=model.state_dict(),step=0))
    for step in range(1,steps+1):
        tick=time.time();b={k:v.cuda() for k,v in pair_batch(data.batch(step),step).items()};opt.zero_grad(set_to_none=True)
        for pg in opt.param_groups:pg['lr']=a.lr*min(1,step/10)*(.2+.8*.5*(1+math.cos(math.pi*max(0,step-10)/140)))
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=sample(model,b,b['diffusion_noise']);loss=(pred.float()-b['target']).square().mean()
        assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad);opt.step();torch.cuda.synchronize()
        row=dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),seconds=time.time()-tick,peak_gb=torch.cuda.max_memory_allocated()/1e9)
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(out/'status.json',dict(state='training',pid=os.getpid(),step=step,best_step=best_step,elapsed=time.time()-started,metrics=row))
        del b,pred,loss
        if step in [25,75,steps]:
            score=evaluate(model,out,step)
            if score<best:best,best_step=score,step;save(out/'best.pt',dict(model=model.state_dict(),step=step))
    save(out/'last.pt',dict(model=model.state_dict(),step=steps))
    if best_step>0 and not a.smoke:
        model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model']);evaluate(model,out,'best',matrix=True)
    write(out/'status.json',dict(state='complete',pid=os.getpid(),step=steps,best_step=best_step,best_score=best,elapsed=time.time()-started))
if __name__=='__main__':main()
