import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_hvdit.model import NoiseHVDiT
from rmdm.diffusion import DiffusionProcess,DDIMSampler
from .common import VARIANTS,TrainingData,public,loss_terms,metrics,write,save


@torch.inference_mode()
def evaluate(model,config,root,variant,*,steps=20,matrix=False,limit=0,output=None):
    entries=json.loads((root/'inputs/validation_bank.json').read_text())
    if not matrix:entries=[e for e in entries if e['group']=='tune']
    if limit:entries=entries[:limit]
    model.eval();sampler=DDIMSampler(config.diffusion);rows=[]
    for ei,e in enumerate(entries):
        d=torch.load(root/'inputs'/e['file'],weights_only=True)
        sparse={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()};initial=d['initial'].cuda()
        sigmas=[0,.03,.05,.09] if matrix and variant['condition'] else [e['sigma'] if variant['condition'] else 0.]
        for sigma in sigmas:
            q=torch.tensor([sigma**2],device='cuda');torch.cuda.synchronize();tick=time.perf_counter()
            prediction=sampler.sample(model,public(sparse,q),initial_noise=initial,steps=steps,eta=0)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite evaluation prediction')
            row=dict(e,input_sigma=sigma,metrics=metrics(prediction,sparse),seconds=seconds)
            rows.append(row)
            if output:
                save(output.parent/'predictions'/output.stem/f'{Path(e["file"]).stem}_sigma{sigma:.2f}.pt',prediction.cpu().float())
        if output:write(output,dict(complete=False,steps=steps,variant=variant,rows=rows))
    if output:write(output,dict(complete=True,steps=steps,variant=variant,rows=rows))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--variant',choices=list(VARIANTS),default='k1');p.add_argument('--mode',choices=['train','smoke','baseline'],default='train')
    args=p.parse_args();root=Path(args.root);variant=VARIANTS[args.variant]
    out=root/('baseline' if args.mode=='baseline' else ('smoke_'+args.variant if args.mode=='smoke' else args.variant));out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((root/'inputs/protocol.json').read_text());smoke=args.mode=='smoke'
    contract=dict(spec=spec,variant=variant,initial=str(Path(args.initial).resolve()),mode=args.mode)
    mp=out/'contract.json'
    if mp.exists() and json.loads(mp.read_text())!=contract:raise ValueError('Resume contract mismatch')
    write(mp,contract)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(spec['seed'])
    config=load_config('experimental/noise_hvdit/w16_tolerance.yaml');model=NoiseHVDiT(config)
    model.load_state_dict(torch.load(args.initial,map_location='cpu',weights_only=False)['model'],strict=True)
    model=model.cuda().train();started=time.time();status_base=dict(pid=os.getpid(),variant=variant,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),started_at=started,global_batch=spec['global_batch'])
    if args.mode=='baseline':
        write(out/'status.json',dict(status_base,state='evaluating'))
        evaluate(model,config,root,variant,steps=50,matrix=True,output=out/'matrix.json')
        write(out/'status.json',dict(status_base,state='complete',finished_at=time.time()));return
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
    diffusion=DiffusionProcess(config.diffusion);data=TrainingData(root/'inputs')
    total=3 if smoke else spec['max_steps'];step=0;best=float('inf');best_step=0
    if (out/'last.pt').exists():
        payload=torch.load(out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer']);step=payload['step'];best=payload['best'];best_step=payload['best_step'];del payload
    def validation(step):
        nonlocal best,best_step
        rows=evaluate(model,config,root,variant,steps=2 if smoke else 20,limit=1 if smoke else 0)
        score=float(np.mean([r['metrics']['unobserved_mse'] for r in rows]))
        write(out/'validation'/f'step{step:04d}.json',dict(step=step,score=score,rows=rows))
        if score<best:
            best,best_step=score,step;save(out/'best.pt',dict(model=model.state_dict(),step=step,score=score,variant=variant))
        model.train()
    if step==0:validation(0)
    for step in range(step+1,total+1):
        tick=time.perf_counter();batch={k:v.cuda() for k,v in data.batch(step).items()}
        target=batch['target'];noisy=diffusion.scheduler.add_noise(target,batch['diffusion_noise'],batch['timesteps'])
        q=batch['measurement_variance'] if variant['condition'] else torch.zeros_like(batch['measurement_variance'])
        if step<=spec['warmup']:lr=spec['learning_rate']*step/spec['warmup']
        else:lr=spec['min_learning_rate']+.5*(spec['learning_rate']-spec['min_learning_rate'])*(1+math.cos(math.pi*(step-spec['warmup'])/(spec['max_steps']-spec['warmup'])))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred,cal=model(noisy,batch['timesteps'],public(batch,q))
            loss,terms=loss_terms(pred,cal,batch,batch['source'],variant)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite gradient at {step}')
        optimizer.step();torch.cuda.synchronize()
        if step<=3 or step%20==0:
            row=dict(step=step,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},grad_norm=float(grad),lr=lr,
                sigma_mean=float(batch['measurement_variance'].sqrt().mean()),seconds=time.perf_counter()-tick,elapsed=time.time()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(status_base,state='training',step=step,total=total,updated_at=time.time(),metrics=row));print(json.dumps(row),flush=True)
        # Release graph and batch tensors before FP32 sampling and checkpoint IO.
        del batch,target,noisy,pred,cal,loss,terms,q
        if step%spec['validation_every']==0 or step==total:
            validation(step)
            save(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,best_step=best_step))
    del optimizer
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(payload['model']);del payload
    write(out/'status.json',dict(status_base,state='final_evaluation',step=total,best_step=best_step,updated_at=time.time()))
    evaluate(model,config,root,variant,steps=2 if smoke else 50,matrix=True,limit=1 if smoke else 0,output=out/'matrix.json')
    write(out/'status.json',dict(status_base,state='complete',step=total,best_step=best_step,best_score=best,finished_at=time.time(),elapsed_seconds=time.time()-started))

if __name__=='__main__':main()
