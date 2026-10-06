"""Standalone paired baseline evaluation on the losslessly packed shared bank."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def pack(bank_path,output):
    bankpath=Path(bank_path);bank=json.loads(bankpath.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    groups={}
    for e in bank['entries']:groups.setdefault((e['video_id'],e['start']),[]).append(e)
    packed=[]
    for wi,entries in enumerate(groups.values()):
        arrays={};file=f'window{wi:04d}.npz'
        for j,e in enumerate(entries):
            s=torch.load(bankpath.parent/e['file'],weights_only=True)['sparse']
            if not j:
                arrays.update(target=s['target'].numpy(),building=s['building'].numpy().astype(np.uint8),vehicle=s['vehicle'].numpy().astype(np.uint8))
            idx=s['sampling_mask'].flatten().nonzero().flatten()
            arrays[f'indices{j}']=idx.numpy().astype(np.int32);arrays[f'observed{j}']=s['observed_rss'].flatten()[idx].numpy()
            packed.append(dict(e,packed_file=file,packed_index=j))
        np.savez_compressed(out/file,**arrays)
        if (wi+1)%50==0:print(f'Packed {wi+1}/{len(groups)} windows',flush=True)
    # Values round trip exactly; uint8 conversion applies only to binary maps.
    for e in packed[::199]:
        s=torch.load(bankpath.parent/e['file'],weights_only=True)['sparse'];z=np.load(out/e['packed_file']);j=e['packed_index']
        for k in ('target','building','vehicle'):
            if not np.array_equal(s[k].numpy(),z[k]):raise ValueError('Packing changed input')
        idx=z[f'indices{j}'];obs=np.zeros(s['observed_rss'].numel(),np.float32);obs[idx]=z[f'observed{j}']
        if not np.array_equal(obs,s['observed_rss'].flatten().numpy()):raise ValueError('Observation packing mismatch')
    save_json(out/'bank.json',dict(spec=bank['spec'],entries=packed,packing='lossless arrays from saved local tensors; no resampling'))

def model_module(source):
    path=Path(source)/'experimental/radio_baselines'
    spec=importlib.util.spec_from_file_location('paper_baseline_models',path/'models.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def latent_noise(entry,stream):
    vi=entry['full_data_video_index'];start=entry['start']
    return torch.stack([torch.randn((3,32,32),generator=torch.Generator().manual_seed(20261006+vi*1000000+(start+t)*100+stream)) for t in range(16)])

def measures(pred,target,building,vehicle,mask):
    err=pred.float()-target.float();valid=((building<=.5)&(vehicle<=.5)).float();unseen=valid*(1-mask)
    fm=err.square().flatten(2).mean(2)
    return dict(mse=float(fm.mean()),psnr=float((-10*fm.clamp_min(1e-12).log10()).mean()),mae=float(err.abs().mean()),
        unobserved_mse=float((err.square()*unseen).sum()/unseen.sum().clamp_min(1)),observed_mse=float((err.square()*mask).sum()/mask.sum().clamp_min(1)),temporal_delta_mse=float((err[:,1:]-err[:,:-1]).square().mean()))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',action='store_true');p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['radiounet','rmegan','radiodiff']);p.add_argument('--checkpoint');p.add_argument('--baseline-source');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--frame-batch',type=int,default=16)
    args=p.parse_args()
    if args.pack:return pack(args.bank,args.output)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261006)
    bankpath=Path(args.bank);bank=json.loads(bankpath.read_text());entries=bank['entries'][args.shard::args.shards]
    if args.limit:entries=entries[:args.limit]
    module=model_module(args.baseline_source);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=module.build_model(args.method,phase=payload.get('args',{}).get('phase','first'));model.load_state_dict(payload['model'],strict=True);model=model.cuda().eval().requires_grad_(False)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);mp=root/f'manifest_{args.shard}.json'
    manifest=dict(args=vars(args),entries=entries,precision='fp32',projection='none',radiodiff_steps=20,step=payload.get('step'),torch=str(torch.__version__))
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Resume configuration changed')
    save_json(mp,manifest);started=time.perf_counter();finished=0;loaded=None;z=None
    def status(state):save_json(root/f'status_{args.shard}.json',dict(state=state,completed=finished,total=len(entries),pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9))
    status('running')
    for e in entries:
        stem=Path(e['file']).stem;record=root/'cases'/f'{stem}.json'
        if record.exists():finished+=1;continue
        if loaded!=e['packed_file']:
            if z is not None:z.close()
            z=np.load(bankpath.parent/e['packed_file']);loaded=e['packed_file']
        target,building,vehicle=[torch.from_numpy(z[k]).float().cuda() for k in ('target','building','vehicle')]
        mask=torch.zeros_like(target);observed=torch.zeros_like(target);j=e['packed_index'];idx=torch.from_numpy(z[f'indices{j}'].astype(np.int64)).cuda()
        mask.view(-1)[idx]=1;observed.view(-1)[idx]=torch.from_numpy(z[f'observed{j}']).cuda()
        # Exactly the public five-channel, no-Tx input used during training.
        x=torch.cat((building[0],torch.zeros_like(building[0]),vehicle[0],observed[0],mask[0]),dim=1)
        if args.method=='radiodiff':
            initial=latent_noise(e,50).cuda();steps=[latent_noise(e,51+k).cuda() for k in range(20)]
        torch.cuda.synchronize();tick=time.perf_counter();predictions=[]
        for start in range(0,16,args.frame_batch):
            end=start+args.frame_batch
            pred=model.sample(x[start:end],initial[start:end],[v[start:end] for v in steps]) if args.method=='radiodiff' else model(x[start:end])
            predictions.append(pred)
        prediction=torch.cat(predictions).unsqueeze(0).float();torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        if not torch.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
        path=root/'predictions'/f'{stem}.pt';path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(prediction.cpu(),tmp);tmp.replace(path)
        row=dict(e,method=args.method,seconds=elapsed,metrics=measures(prediction,target,building,vehicle,mask),prediction=str(path))
        save_json(record,dict(entry=e,rows=[row]));finished+=1;status('running');print(json.dumps(dict(method=args.method,completed=finished,total=len(entries),seconds=elapsed)),flush=True)
    status('complete')
if __name__=='__main__':main()
