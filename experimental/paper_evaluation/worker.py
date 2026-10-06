"""Resume per case; persist held-out draws, predictions, metrics and timings."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from experimental.paired_evaluation.evaluate import load_model,single_frames
from experimental.paired_evaluation.protocol import metrics
from experimental.noise_estimation_vem.hvdit import FrozenHVDiT,inference_inputs
from experimental.noise_estimation_crossfit.core import audited_mle
from rmdm_noise_estimation.calibration import VarianceCalibration
from .calibration import fit_variance_calibration
from rmdm.diffusion import DDIMSampler
from .ensemble import PaperEnsemble
from .prepare import write_json

ROOT=Path('runs/paper_test200_20261006')
W16=('experimental/noise_hvdit/w16_tolerance.yaml','runs/noise_hvdit_tolerance_20260924/w16_x0/checkpoints/step_016000.pth','model')
W1=('experimental/noise_hvdit/w1_tolerance.yaml','runs/noise_hvdit_tolerance_20260924/w1_x0/checkpoints/step_020000.pth','ema')
ORIGINAL=('experimental/noise_temporal_rmdm/t1.yaml','/data_p6/fzj/tmp/rmdm_noise_injection_probe_20260924/original_epoch_009.pth','model')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save_tensor(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
def fit(root=ROOT):
    units=[torch.load(p,weights_only=True) for p in sorted((root/'calibration/units').glob('*.pt'))]
    expected=json.loads((root/'calibration_bank/bank.json').read_text())['entries']
    if len(units)!=len(expected) or {u['entry']['file'] for u in units}!={e['file'] for e in expected}:raise ValueError('Incomplete calibration')
    result=dict(created_at=now(),primary='mixed',reference_sigma=.03,folds=4,members=8,steps=50,calibrators={})
    for name,selected in [('clean',[u for u in units if u['entry']['sigma']==0]),('mixed',units)]:
        raw,err={},{}
        for rate in [1,2,3]:
            group=[u for u in selected if u['entry']['rate']==rate]
            raw[str(rate)]=np.concatenate([u['raw_variance'].numpy() for u in group])
            err[str(rate)]=np.concatenate([(u['target']-u['mean']).double().square().numpy() for u in group])
        result['calibrators'][name]=dict(variance_calibration=fit_variance_calibration(raw,err).to_dict(),videos=sorted({u['entry']['video_id'] for u in selected}),units=len(selected))
    write_json(root/'calibration.json',result);return result

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibration','main','w1','original'],required=True)
    p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--calibration',default=str(ROOT/'calibration.json'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--member-batch',type=int,default=8)
    args=p.parse_args();bankpath=Path(args.bank);bank=json.loads(bankpath.read_text());entries=bank['entries'][args.shard::args.shards]
    if args.limit: entries=entries[:args.limit]
    if args.mode=='calibration' and bank['spec']['split']!='val':raise ValueError('Calibration must use validation data')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),entries=entries,precision='fp32',projection='none',reference_sigma=.03,members=8,folds=4,seed=20261006)
    mp=root/f'manifest_{args.shard}.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Resume configuration mismatch')
    write_json(mp,manifest)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261006)
    paths=W1 if args.mode=='w1' else ORIGINAL if args.mode=='original' else W16
    model,config=load_model('original_rmdm' if args.mode=='original' else 'hvdit',*paths)
    model=model.cuda().eval().requires_grad_(False);predictor=FrozenHVDiT(model,config.diffusion,steps=args.steps)
    ensemble=PaperEnsemble(model,config.diffusion,args.steps,args.member_batch)
    calibration=json.loads(Path(args.calibration).read_text()) if args.mode=='main' else None
    if calibration and set(calibration['calibrators']['mixed']['videos'])&{e['video_id'] for e in entries}:raise ValueError('Calibration/evaluation overlap')
    begin=time.perf_counter();finished=0
    def status(state):write_json(root/f'status_{args.shard}.json',dict(state=state,pid=os.getpid(),updated_at=now(),completed=finished,total=len(entries),elapsed_seconds=time.perf_counter()-begin,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9))
    status('running')
    for entry in entries:
        stem=Path(entry['file']).stem;record=root/'cases'/f'{stem}.json'
        if record.exists():finished+=1;continue
        data=torch.load(bankpath.parent/entry['file'],weights_only=True)
        sparse={k:v.cuda() if torch.is_tensor(v) else v for k,v in data['sparse'].items()};initial=data['initial'].cuda()
        inp=dict(conditions=inference_inputs(sparse),observed_rss=sparse['observed_rss'],sampling_mask=sparse['sampling_mask'],initial_noise=initial)
        rows=[];unit=None
        if args.mode in ('calibration','main'):
            up=root/'units'/f'{stem}.pt'
            if up.exists():
                unit=torch.load(up,weights_only=True)
                if unit['entry']!=entry:raise ValueError('Unit identity mismatch')
            else:
                assignment=torch.load(bankpath.parent/entry['fold_file'],weights_only=True).cuda()
                unit=ensemble.collect(conditions=inp['conditions'],observed_rss=inp['observed_rss'],sampling_mask=inp['sampling_mask'],assignment=assignment,video_index=entry['full_data_video_index'],start=entry['start'])
                # Clean truth is attached ONLY after all target-free inference.
                unit.update(entry=entry,target=sparse['target'][sparse['sampling_mask'].bool()].cpu());save_tensor(up,unit)
        methods={}
        if args.mode=='main':
            for name,cal in calibration['calibrators'].items():
                tick=time.perf_counter();v=VarianceCalibration(**cal['variance_calibration']).apply(unit['raw_variance'].numpy())
                estimate=audited_mle(unit['observed'].numpy(),unit['mean'].numpy(),v)
                methods['estimated_'+name]=(min(estimate['variance'],.09**2),dict(estimated_sigma=estimate['sigma'],estimated_variance=estimate['variance'],estimation_seconds=unit['seconds']+time.perf_counter()-tick,optimizer=estimate))
            methods.update(known_sigma=(entry['sigma']**2,{}),fixed_003=(.03**2,{}),fixed_zero=(0.,{}))
        elif args.mode in ('w1','original'): methods[args.mode]=(entry['sigma']**2,{})
        for method,(q,extra) in methods.items():
            torch.cuda.synchronize();tick=time.perf_counter()
            if args.mode in ('w1','original'):
                flat,noise=single_frames(sparse,initial)
                # Whitelist inputs for the original sampler as well.
                public={k:flat[k] for k in ('building','vehicle','sampling_rate','observed_rss','sampling_mask')}
                public['measurement_variance']=torch.full((len(noise),),q,device='cuda')
                pred=DDIMSampler(config.diffusion).sample(model,public,initial_noise=noise,steps=args.steps,eta=0).reshape_as(sparse['target'])
            else:pred=predictor(**inp,noise_var=torch.tensor([q],device='cuda'))
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite predictions')
            predpath=root/'predictions'/method/f'{stem}.pt';save_tensor(predpath,pred.cpu().float())
            rows.append(dict(entry,method=method,applied_sigma=q**.5,seconds=seconds,metrics=metrics(pred,sparse)[0],prediction=str(predpath),**extra))
        record.parent.mkdir(exist_ok=True)
        write_json(record,dict(entry=entry,rows=rows,collection_seconds=unit['seconds'] if unit else None))
        finished+=1;status('running');print(json.dumps(dict(mode=args.mode,shard=args.shard,completed=finished,total=len(entries),file=entry['file'],collection_seconds=unit['seconds'] if unit else None,seconds=[r['seconds'] for r in rows])),flush=True)
    status('complete')
if __name__=='__main__':main()
