"""DDP matched variants; resumable checkpoints and bounded wall time."""
import argparse,contextlib,copy,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import public,metrics,write,save
from rmdm.diffusion import DiffusionProcess,DDIMSampler
from .data import PairedData
from .model import FeatureNoiseModel,VARIANTS

INITIAL='runs/noise_iterative_remote_20261007/inputs/w16_model.pt'

@torch.inference_mode()
def evaluate(model,cfg,inputs,rank,world,*,matrix=False,limit=0,prediction_dir=None):
    entries=json.loads((inputs/'validation_bank.json').read_text())
    if not matrix:entries=[e for e in entries if e['group']=='tune']
    if limit:entries=entries[:limit]
    model.eval();sampler=DDIMSampler(cfg.diffusion);rows=[]
    for e in entries[rank::world]:
        d=torch.load(Path(e['base'])/e['file'],weights_only=True);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()};initial=d['initial'].cuda()
        sigmas=sorted({0,.03,.05,.09,e['sigma']}) if matrix else [e['sigma']]
        # Blind/fixed models cannot depend on q; evaluate once, report equivalent
        # fixed input rows without rerunning identical deterministic inference.
        predictions={}
        for sigma in sigmas:
            key=sigma if model.variant not in ('blind','reliability_fixed') else 'invariant'
            if key not in predictions:
                tick=time.perf_counter();p=sampler.sample(model,public(b,torch.tensor([sigma**2],device='cuda')),initial_noise=initial,steps=50)
                torch.cuda.synchronize();assert torch.isfinite(p).all();prediction_path=None
                if prediction_dir is not None:
                    prediction_path=prediction_dir/f"{e['bank']}_{Path(e['file']).stem}_sigma{key}.pt";save(prediction_path,p.cpu().float())
                predictions[key]=(metrics(p,b),time.perf_counter()-tick,str(prediction_path) if prediction_path else None)
            m,seconds,prediction_path=predictions[key];rows.append(dict(**e,input_sigma=sigma,metrics=m,seconds=seconds,prediction_path=prediction_path))
    gathered=[None for _ in range(world)];dist.all_gather_object(gathered,rows);return [r for part in gathered for r in part]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--variant',choices=(*VARIANTS,'adaln'),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(local);dist.init_process_group('nccl')
    root=Path(a.root);spec=json.loads((root/'config.json').read_text());inputs=root/'inputs';out=root/(f"smoke_b{spec['microbatch']}_{a.variant}" if a.smoke else a.variant);out.mkdir(exist_ok=True)
    torch.manual_seed(spec['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from experimental.noise_adaln_retrain.model import BlockNoiseModel
    cfg=load_config('experimental/noise_hvdit/w16_tolerance.yaml');factory=BlockNoiseModel if a.variant=='adaln' else FeatureNoiseModel
    if cfg.diffusion.prediction_type!='sample':raise ValueError('Clean-map MSE requires x0/sample prediction')
    if spec['microbatch']%4:raise ValueError('Microbatch must contain complete noise quartets')
    base=factory(cfg,a.variant).initialize(torch.load(INITIAL,map_location='cpu',weights_only=False)['model']).cuda()
    model=DDP(base,device_ids=[local],broadcast_buffers=False);ema=copy.deepcopy(base).eval().requires_grad_(False)
    groups=[dict(params=[p for n,p in base.named_parameters() if p.requires_grad and n.startswith('base.denoiser') and 'log_signal_variance' not in n],lr=spec['backbone_lr'],base_lr=spec['backbone_lr']),dict(params=[p for n,p in base.named_parameters() if p.requires_grad and (not n.startswith('base.denoiser') or 'log_signal_variance' in n)],lr=spec['module_lr'],base_lr=spec['module_lr'])]
    groups=[g for g in groups if g['params']];opt=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    process=DiffusionProcess(cfg.diffusion);data=PairedData(inputs,spec['seed']);steps=8 if a.smoke else spec['max_steps'];accum=spec['accumulation'];basevideos=spec['microbatch']//4
    started=time.time();start_step=0;best=float('inf');best_step=0;best_weights='model';timings=[]
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);start_step=state['step'];best=state['best'];best_step=state['best_step'];best_weights=state['best_weights'];del state
    runinfo=dict(variant=a.variant,pid=os.getpid(),world_size=world,microbatch=spec['microbatch'],accumulation=accum,global_batch=spec['microbatch']*world*accum,unique_videos_per_update=basevideos*world*accum,started_at=started,trainable_parameters=sum(p.numel() for p in base.parameters() if p.requires_grad),initialization='Same pretrained W16 at sigma0; HWM/rate/old FiLM frozen; all DiT weights train; new conditioning initialized fresh; not from scratch')
    def checkpoint(step):
        if rank==0:save(out/'last.pt',dict(model=base.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,best=best,best_step=best_step,best_weights=best_weights,spec=spec,runinfo=runinfo))
        dist.barrier()
    def validation(step):
        nonlocal best,best_step,best_weights
        for name,candidate in [('model',base),('ema',ema)]:
            rows=evaluate(candidate,cfg,inputs,rank,world,limit=2 if a.smoke else 0);score=float(np.mean([r['metrics']['unobserved_mse'] for r in rows]))
            if rank==0:write(out/'validation'/f'step{step:05d}_{name}.json',dict(step=step,score=score,rows=rows))
            if score<best:
                best,best_step,best_weights=score,step,name
                if rank==0:save(out/'best.pt',dict(model=candidate.state_dict(),step=step,weights=name,score=score,variant=a.variant))
        base.train();dist.barrier()
    if rank==0:write(out/'status.json',dict(runinfo,state='initial_validation',step=start_step))
    if start_step==0:validation(0)
    deadline=time.time()+3600*spec['train_hours'] if a.smoke else spec['train_deadline']
    stopped=False;step=start_step
    for step in range(start_step+1,steps+1):
        stop=torch.tensor(int(time.time()>deadline),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():stopped=True;step-=1;break
        tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss_sum=torch.zeros(3,device='cuda')
        if step<=spec['warmup']:factor=step/spec['warmup']
        else:factor=.1+.9*.5*(1+math.cos(math.pi*(step-spec['warmup'])/max(1,spec['max_steps']-spec['warmup'])))
        for group in opt.param_groups:group['lr']=group['base_lr']*factor
        for micro in range(accum):
            b={k:v.cuda() for k,v in data.batch(step,micro,rank,basevideos).items()};noisy=process.scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
            ctx=model.no_sync() if micro<accum-1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(noisy,b['timesteps'],public(b,b['measurement_variance']));error=prediction.float()-b['target'];spatial=error.square().mean();temporal=(error[:,1:]-error[:,:-1]).square().mean();loss=spatial if spec['temporal_weight']==0 else spatial+spec['temporal_weight']*temporal
                if not torch.isfinite(loss):raise FloatingPointError(f'Loss at{step}')
                (loss/accum).backward()
            loss_sum+=torch.stack([loss.detach(),spatial.detach(),temporal.detach()])/accum
            del b,noisy,prediction,error,loss,spatial,temporal
        grad=torch.nn.utils.clip_grad_norm_([p for p in base.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):raise FloatingPointError(f'Gradient at{step}')
        opt.step()
        with torch.no_grad():
            # All parameter lists identical; include frozen weights for fidelity.
            torch._foreach_lerp_(list(ema.parameters()),list(base.parameters()),1-spec['ema_decay'])
            for dst,src in zip(ema.buffers(),base.buffers()):dst.copy_(src)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;timings.append(seconds)
        dist.all_reduce(loss_sum);loss_sum/=world
        if rank==0 and (step<=3 or step%20==0 or a.smoke):
            row=dict(step=step,loss=float(loss_sum[0]),spatial=float(loss_sum[1]),temporal=float(loss_sum[2]),grad_norm=float(grad),step_seconds=seconds,lr=opt.param_groups[0]['lr'],elapsed=time.time()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(runinfo,state='training',step=step,best_step=best_step,best_score=best,metrics=row));print(json.dumps(row),flush=True)
        if step%spec['validation_every']==0 or step==steps:
            validation(step);checkpoint(step)
        elif step%spec['checkpoint_every']==0:checkpoint(step)
    if stopped:checkpoint(step)
    if a.smoke:
        # Exercise loading real saved checkpoint on both ranks.
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(saved['model']);del saved
        report=dict(runinfo,state='complete',step=step,best_step=best_step,median_step_seconds=float(np.median(timings[2:])),peak_gb=torch.cuda.max_memory_allocated()/1e9,checkpoint_reload='passed',elapsed=time.time()-started)
    else:
        if rank==0:write(out/'status.json',dict(runinfo,state='final_evaluation',step=step,best_step=best_step,best_weights=best_weights))
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);base.load_state_dict(saved['model']);del saved
        rows=evaluate(base,cfg,inputs,rank,world,matrix=True,prediction_dir=out/'predictions')
        if rank==0:write(out/'matrix.json',dict(complete=True,best_step=best_step,best_weights=best_weights,rows=rows))
        report=dict(runinfo,state='complete',step=step,best_step=best_step,best_weights=best_weights,best_score=best,reason='time_limit' if stopped else 'max_steps',elapsed=time.time()-started)
    if rank==0:write(out/'status.json',report)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
