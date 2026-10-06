"""Resumable remote pilot; truth is used only for calibration or reporting."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_hvdit.model import NoiseHVDiT
from experimental.paper_evaluation.calibration import fit_variance_calibration
from rmdm.diffusion import DDIMSampler
from .core import public_input, collect, calibrated_variance, mle, update


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def metrics(pred,sparse):
    error=pred.float()-sparse['target'];mask=sparse['sampling_mask']
    unseen=sparse['valid_mask']*(1-mask);fm=error.square().flatten(2).mean(2)
    return dict(mse=float(fm.mean()),psnr=float((-10*fm.clamp_min(1e-12).log10()).mean()),
        unobserved_mse=float((error.square()*unseen).sum()/unseen.sum().clamp_min(1)),
        temporal_delta_mse=float((error[:,1:]-error[:,:-1]).square().mean()))


def fit(root):
    spec=json.loads((root/'inputs/protocol.json').read_text())
    entries=json.loads((root/'inputs/calibration_bank.json').read_text())
    nodes=[]
    for node,sigma in enumerate(spec['calibration_sigmas']):
        units=[]
        for e in entries:
            u=torch.load(root/'calibration'/f'{e["id"]}_node{node}.pt',weights_only=True)
            if u['entry']!=e or u['input_sigma']!=sigma:raise ValueError('Calibration identity mismatch')
            units.append(u)
        raw=np.concatenate([u['raw_variance'].numpy() for u in units])
        error=np.concatenate([(u['target'].double()-u['mean']).square().numpy() for u in units])
        c=fit_variance_calibration({'2':raw},{'2':error}).to_dict()
        nodes.append(dict(q=sigma**2,**c))
    write(root/'calibration.json',dict(nodes=nodes,scope='validation only; rate 2%; affine coefficients interpolated in conditioning variance',videos=sorted({e['video_id'] for e in entries}),created_at=time.time()))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['calibration','test','fit','smoke'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    args=p.parse_args();root=Path(args.root)
    if args.stage=='fit':fit(root);return
    spec=json.loads((root/'inputs/protocol.json').read_text())
    smoke=args.stage=='smoke';stage='test' if smoke else args.stage
    entries=json.loads((root/f'inputs/{stage}_bank.json').read_text())
    if smoke:entries=entries[:1]
    else:entries=entries[args.shard::args.shards]
    output=root/(f'smoke_gpu{args.shard}' if smoke else stage)
    output.mkdir(exist_ok=True)
    contract=dict(spec=spec,entries=entries,stage=args.stage,shard=args.shard,shards=args.shards)
    mp=output/f'manifest_{args.shard}.json'
    if mp.exists() and json.loads(mp.read_text())!=contract:raise ValueError('Resume mismatch')
    write(mp,contract)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(spec['seed'])
    config=load_config('experimental/noise_hvdit/w16_tolerance.yaml')
    model=NoiseHVDiT(config)
    model.load_state_dict(torch.load(root/'inputs/w16_model.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    model=model.cuda().eval().requires_grad_(False);sampler=DDIMSampler(config.diffusion)
    steps=2 if smoke else spec['ddim_steps']
    original=json.loads((root/'inputs/original_calibration.json').read_text())['calibrators']['mixed']['variance_calibration']
    if stage=='test':
        calibration=(dict(nodes=[dict(q=s**2,**original) for s in spec['calibration_sigmas']],videos=[])
                     if smoke else json.loads((root/'calibration.json').read_text()))
        if set(calibration['videos'])&{e['video_id'] for e in entries}:raise ValueError('Calibration/test overlap')
    started=time.time();done=0
    def status(state,entry=None):
        write(output/f'status_{args.shard}.json',dict(state=state,pid=os.getpid(),completed=done,total=len(entries),entry=entry,
            updated_at=time.time(),elapsed_seconds=time.time()-started,gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    status('running')
    for e in entries:
        resultfile=output/'cases'/f'{e["id"]}.json'
        if stage=='test' and resultfile.exists():done+=1;continue
        d=torch.load(root/'inputs'/e['data'],weights_only=True)
        sparse={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()}
        initial=d['initial'].cuda();assignment=d['assignment'].cuda()
        def ensemble(q,path):
            if path.exists():
                u=torch.load(path,weights_only=True)
                if u['entry']!=e or abs(u['input_sigma']**2-q)>1e-12:raise ValueError('Cached statistics mismatch')
                return u
            torch.cuda.synchronize();tick=time.perf_counter()
            u=collect(model,config.diffusion,inputs=public_input(sparse,q),assignment=assignment,
                video_index=e['full_data_video_index'],start=e['start'],steps=steps)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            u.update(entry=e,input_sigma=q**.5,seconds=seconds)
            if stage=='calibration':u['target']=sparse['target'][sparse['sampling_mask'].bool()].cpu()
            save(path,u);return u
        def reconstruct(q,name):
            torch.cuda.synchronize();tick=time.perf_counter()
            pred=sampler.sample(model,public_input(sparse,q),initial_noise=initial,steps=steps,eta=0)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite reconstruction')
            path=output/'predictions'/e['id']/f'{name}.pt';save(path,pred.cpu().float())
            return dict(applied_sigma=q**.5,metrics=metrics(pred,sparse),seconds=seconds,prediction=str(path.relative_to(root)))
        if stage=='calibration':
            for ni,sigma in enumerate(spec['calibration_sigmas']):
                ensemble(sigma**2,output/f'{e["id"]}_node{ni}.pt')
        else:
            trajectories=[];baselines={}
            inits=[.03] if smoke else spec['initial_sigmas']
            rounds=2 if smoke else spec['max_iterations']
            for si,init in enumerate(inits):
                q=init**2;rows=[];cumulative=0.;first_stop=None
                for iteration in range(rounds):
                    u=ensemble(q,output/'units'/e['id']/f'init{si}_iter{iteration}.pt')
                    tick=time.perf_counter()
                    v=calibrated_variance(u['raw_variance'].numpy(),q,calibration)
                    estimate=mle(u['observed'].numpy(),u['mean'].numpy(),v)
                    next_q,stop,residual=update(q,estimate['variance'],spec['damping'])
                    update_seconds=time.perf_counter()-tick
                    if stop and first_stop is None:first_stop=iteration+1
                    recon=reconstruct(next_q,f'init{si}_iter{iteration}')
                    cumulative+=u['seconds']+update_seconds+recon['seconds']
                    rows.append(dict(iteration=iteration+1,input_sigma=q**.5,raw_proposal=estimate,
                        updated_sigma=next_q**.5,fixed_point_residual_sigma=residual,stop_eligible=stop,
                        collection_seconds=u['seconds'],update_seconds=update_seconds,cumulative_seconds=cumulative,**recon))
                    if init==.03 and iteration==0:
                        raw=u['raw_variance'].numpy()
                        prior=np.maximum(original['scale']*raw+original['offset'],original['floor'])
                        old=mle(u['observed'].numpy(),u['mean'].numpy(),prior)
                        baselines['original_fixed_reference']=dict(estimate=old,collection_seconds=u['seconds'],
                            **reconstruct(min(old['variance'],.09**2),'original_fixed_reference'))
                        baselines['matched_one_pass']=dict(estimate=estimate,collection_seconds=u['seconds'],
                            **reconstruct(min(estimate['variance'],.09**2),'matched_one_pass'))
                    q=next_q
                    write(output/'progress'/f'{e["id"]}_init{si}.json',dict(entry=e,initial_sigma=init,rows=rows))
                trajectories.append(dict(initial_sigma=init,first_stop_eligible=first_stop,rows=rows))
            # Truth enters only the known-noise reporting reference, after estimation.
            baselines['known_sigma']=reconstruct(e['sigma']**2,'known_sigma')
            baselines['fixed_003']=reconstruct(.03**2,'fixed_003')
            baselines['fixed_zero']=reconstruct(0.,'fixed_zero')
            write(resultfile,dict(entry=e,trajectories=trajectories,baselines=baselines))
        done+=1;status('running',e['id']);print(json.dumps(dict(stage=args.stage,completed=done,total=len(entries),entry=e)),flush=True)
    status('complete')


if __name__=='__main__':main()
