"""Four-GPU full-loss random initialization with resume and sigma-alignment monitoring."""
import argparse,contextlib,copy,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import public,write,save
from rmdm.diffusion import DiffusionProcess
from .model import PaperNoiseDiT
from experimental.noise_scratch_full.data import FullData
from experimental.noise_scratch_full.loss import loss_terms
from .evaluate import evaluate,summarize
from experimental.noise_scratch_full.early_stop import EarlyStop


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    root=Path(a.root);spec=json.loads((root/'config.json').read_text());out=root/(f"smoke_b{spec['microbatch']}_a{spec['accumulation']}" if a.smoke else 'training');out.mkdir(exist_ok=True)
    torch.manual_seed(spec['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    cfg=load_config(str(root/'model_config.yaml'));assert cfg.diffusion.prediction_type=='sample' and not cfg.use_clean_indicator
    base=PaperNoiseDiT(cfg,spec['variant'],spec['rollout_gap'],spec['prox_lambda']).cuda();model=DDP(base,device_ids=[local],find_unused_parameters=True,broadcast_buffers=True)
    ema=copy.deepcopy(base).eval().requires_grad_(False)
    opt=torch.optim.AdamW(base.parameters(),lr=spec['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
    process=DiffusionProcess(cfg.diffusion);data=FullData(root/'inputs',spec['seed'])
    torch.manual_seed(spec['seed']+rank);torch.cuda.manual_seed(spec['seed']+rank)
    start=0;best_score=float('inf');best_gain=0.;best_score_meta=None;best_gain_meta=None;best_usable=float('inf')
    monitor=EarlyStop(spec['early_stopping'])
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['world']==world and state['spec']['microbatch']==spec['microbatch'] and state['spec']['accumulation']==spec['accumulation']
        base.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);start=state['step']
        best_score=state['best_score'];best_gain=state['best_gain'];best_score_meta=state['best_score_meta'];best_gain_meta=state['best_gain_meta']
        best_usable=state.get('best_usable',float('inf'))
        monitor=EarlyStop(spec['early_stopping'],state.get('early_stopping'))
        torch.set_rng_state(state['rng'][rank]['cpu']);torch.cuda.set_rng_state(state['rng'][rank]['cuda']);del state
    started=time.time();steps=8 if a.smoke else spec['max_steps'];timings=[];gradient_audit={};step=start
    info=dict(world=world,pid=os.getpid(),variant=spec['variant'],initialization='ALL random initialization; no pretrained weights; full HWM; paper-inspired sigma path; two-step differentiable training',global_batch=world*spec['microbatch']*spec['accumulation'],microbatch=spec['microbatch'],accumulation=spec['accumulation'],trainable_parameters=sum(p.numel() for p in base.parameters()),started_at=started)
    def status(state,**extra):
        if rank==0:write(out/'status.json',dict(info,state=state,step=step,best_score=best_score if math.isfinite(best_score) else None,best_score_meta=best_score_meta,best_alignment_gain=best_gain,best_alignment_meta=best_gain_meta,early_stopping=monitor.state,elapsed=time.time()-started,**extra))
    def checkpoint(archive=False):
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state());states=[None for _ in range(world)];dist.all_gather_object(states,rng)
        if rank==0:
            payload=dict(model=base.state_dict(),ema=ema.state_dict(),step=step,spec=spec,world=world,best_score=best_score,best_gain=best_gain,best_score_meta=best_score_meta,best_gain_meta=best_gain_meta,early_stopping=monitor.state,best_usable=best_usable)
            if archive:save(out/'checkpoints'/f'step{step:06d}.pt',payload)
            save(out/'last.pt',dict(payload,optimizer=opt.state_dict(),rng=states))
        dist.barrier()
    def validation():
        nonlocal best_score,best_gain,best_score_meta,best_gain_meta,best_usable
        status('validation')
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        summaries=[]
        for name,candidate in [('model',base),('ema',ema)]:
            for buffer in candidate.buffers():dist.broadcast(buffer,0)
            rows=evaluate(candidate,cfg,root/'inputs',rank,world,limit=4 if a.smoke else 0)
            stats=summarize(rows);meta=dict(step=step,weights=name,**stats)
            summaries.append(stats)
            if rank==0:write(out/'validation'/f'step{step:06d}_{name}.json',dict(**meta,rows=rows))
            if stats['correct_mse']<best_score:
                best_score=stats['correct_mse'];best_score_meta=meta
                if rank==0:save(out/'best_reconstruction.pt',dict(model=candidate.state_dict(),meta=meta,spec=spec))
            if stats['alignment_gain']>best_gain:
                best_gain=stats['alignment_gain'];best_gain_meta=meta
                if rank==0:save(out/'best_alignment.pt',dict(model=candidate.state_dict(),meta=meta,spec=spec))
            if stats['alignment_gain']>0 and stats['correct_mse']<best_usable:
                best_usable=stats['correct_mse']
                if rank==0:save(out/'best_usable.pt',dict(model=candidate.state_dict(),meta=meta,spec=spec))
            if rank==0:print(json.dumps(dict(validation=meta)),flush=True)
        if not a.smoke:
            monitor.update(step,summaries)
            if rank==0:
                write(out/'validation'/f'step{step:06d}_stopping.json',dict(config=monitor.config,**monitor.state))
                print(json.dumps(dict(early_stopping=monitor.state)),flush=True)
        base.train();torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);dist.barrier()
    status('initializing')
    if start==0 and monitor.state['last_step'] is None:
        if a.smoke or spec.get('initial_validation',True):validation()
        checkpoint(archive=not a.smoke)
    base.train();deadline=float('inf') if a.smoke else spec['train_deadline'];stopped=False;reason='max_steps'
    for step in range(start+1,steps+1):
        stop=torch.tensor(int(time.time()>deadline),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():step-=1;stopped=True;reason='training_time_limit';break
        tick=time.perf_counter();opt.zero_grad(set_to_none=True);sums=None
        factor=step/spec['warmup'] if step<=spec['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-spec['warmup'])/max(1,spec['max_steps']-spec['warmup'])))
        for group in opt.param_groups:group['lr']=spec['learning_rate']*factor
        for micro in range(spec['accumulation']):
            b={k:v.cuda() for k,v in data.batch(step,micro,rank,spec['microbatch']).items()}
            noisy=process.scheduler.add_noise(b['target'],b['diffusion_noise'],b['timesteps'])
            with model.no_sync() if micro+1<spec['accumulation'] else contextlib.nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred,cal=model(noisy,b['timesteps'],public(b,b['measurement_variance']));loss,terms=loss_terms(pred,cal,b)
                if not torch.isfinite(loss):raise FloatingPointError(f'Loss at{step}')
                (loss/spec['accumulation']).backward()
            vals=torch.stack([loss.detach(),*(v.detach() for v in terms.values()),b['measurement_variance'].sqrt().mean(),b['sampling_rate'].mean()])/spec['accumulation']
            sums=vals if sums is None else sums+vals
            del b,noisy,pred,cal,loss,terms
        if a.smoke and step==steps:
            for prefix in base.audit_prefixes:
                gradients=[p.grad for n,p in base.named_parameters() if n.startswith(prefix) and p.grad is not None]
                norm=sum(float(g.float().square().sum()) for g in gradients)**.5;assert math.isfinite(norm) and norm>0,(prefix,norm);gradient_audit[prefix]=norm
        norm=torch.nn.utils.clip_grad_norm_(base.parameters(),1.);assert torch.isfinite(norm);opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()),list(base.parameters()),1-spec['ema_decay'])
            for dst,src in zip(ema.buffers(),base.buffers()):dst.copy_(src)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick;timings.append(seconds);dist.all_reduce(sums);sums/=world
        if rank==0 and (step<=3 or step%20==0 or a.smoke):
            row=dict(zip(['loss','reconstruction','sampled_clean','calibration','equation','obstacle','source','sigma_mean','rate_mean'],[float(x) for x in sums]))
            row.update(step=step,grad_norm=float(norm),seconds=seconds,lr=opt.param_groups[0]['lr'],elapsed=time.time()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status('training',metrics=row);print(json.dumps(row),flush=True)
        due=step==steps or (not a.smoke and (step in spec['validation_milestones'] or step%spec['validation_every']==0))
        if due:
            validation();checkpoint(archive=not a.smoke)
            if not a.smoke and monitor.state['stop']:reason='early_stopping';break
        elif step%spec['checkpoint_every']==0:checkpoint()
    if stopped:checkpoint(archive=True)
    if a.smoke:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False);base.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);del saved
        status('complete',median_step_seconds=float(np.median(timings[2:])),peak_gb=torch.cuda.max_memory_allocated()/1e9,gradient_audit=gradient_audit,checkpoint_reload='model,EMA,optimizer passed')
    else:
        # Confirm both selection criteria; early noise response is not silently
        # replaced by the checkpoint selected solely for reconstruction quality.
        for name in ['best_reconstruction','best_alignment','best_usable']:
            file=out/(name+'.pt')
            if not file.exists():continue
            status('final_evaluation',candidate=name);payload=torch.load(file,map_location='cpu',weights_only=False);base.load_state_dict(payload['model']);meta=payload['meta'];del payload
            rows=evaluate(base,cfg,root/'inputs',rank,world,full=True,prediction_dir=out/'predictions'/name)
            if rank==0:write(out/(name+'_matrix.json'),dict(complete=True,selection=meta,rows=rows))
        chosen='best_usable' if (out/'best_usable.pt').exists() else 'best_alignment'
        if (out/(chosen+'.pt')).exists():
            payload=torch.load(out/(chosen+'.pt'),map_location='cpu',weights_only=False);base.load_state_dict(payload['model']);meta=payload['meta'];del payload
            for repeat in [1,2]:
                status('final_multiseed',candidate=chosen,repeat=repeat)
                rows=evaluate(base,cfg,root/'inputs',rank,world,full=True,confirmation_only=True,repeat=repeat)
                if rank==0:write(out/f'{chosen}_repeat{repeat}.json',dict(complete=True,selection=meta,repeat=repeat,rows=rows))
        status('ready_to_resume' if stopped else 'complete',reason=reason)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
