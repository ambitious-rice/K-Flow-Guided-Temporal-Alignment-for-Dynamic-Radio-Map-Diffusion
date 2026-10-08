"""Intervene only on sigma for identical observations and DDIM initial noise.

Inspect initialization, early EMA, and a fixed later checkpoint (raw and EMA),
without selecting that later checkpoint by reconstruction quality.
"""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import torch
from experimental.noise_hvdit.config import load_config
from experimental.noise_feature_retrain.model import FeatureNoiseModel
from experimental.noise_feature_retrain.train import INITIAL
from experimental.noise_loss_screen.common import public,metrics,write,save
from rmdm.diffusion import DDIMSampler
from .model import BlockNoiseModel

SIGMAS=[0.,.03,.05,.09]
STAGES=[('initial',None,None),('early_ema','early.pt','model'),('later_raw','later.pt','model'),('later_ema','later.pt','ema'),('final_raw','final.pt','model'),('final_ema','final.pt','ema')]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--variant',choices=['film','adaln'],required=True);a=p.parse_args()
    root=Path(a.root);out=root/'sigma_probe'/a.variant;out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261008);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=load_config('experimental/noise_hvdit/w16_tolerance.yaml')
    factory=BlockNoiseModel if a.variant=='adaln' else FeatureNoiseModel
    initial_state=torch.load(INITIAL,map_location='cpu',weights_only=False)['model']
    model=factory(cfg,a.variant).initialize(initial_state).cuda().eval();del initial_state
    sampler=DDIMSampler(cfg.diffusion);entries=json.loads((root/'sigma_probe/entries.json').read_text());rows=[];started=time.time()
    # Hard-linked snapshots may refer to the same completed training checkpoint.
    # Do not mistake duplicate files for independent training stages.
    stages=STAGES[:4] if (out/'later.pt').samefile(out/'final.pt') else STAGES
    for label,filename,weights in stages:
        if filename:
            payload=torch.load(out/filename,map_location='cpu',weights_only=False);model.load_state_dict(payload[weights]);step=payload['step'];del payload;gc.collect()
        else:step=0
        for index,e in enumerate(entries):
            d=torch.load(Path(e['base'])/e['file'],weights_only=True);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()}
            x=d['initial'].cuda()
            if label=='initial':
                # Zero-initialized added heads imply exact sigma invariance.
                # Verify it on actual model outputs before avoiding duplicates.
                t=torch.tensor([999],device='cuda');u=model(x,t,public(b,torch.tensor([0.],device='cuda')));v=model(x,t,public(b,torch.tensor([.09**2],device='cuda')))
                torch.testing.assert_close(u,v,rtol=0,atol=0)
                predictions=sampler.sample(model,public(b,torch.tensor([0.],device='cuda')),initial_noise=x,steps=50).repeat(4,1,1,1,1)
            else:
                inputs=public(b,torch.tensor(SIGMAS,device='cuda').square())
                inputs={k:(v if k=='measurement_variance' else v.repeat(4,*([1]*(v.ndim-1)))) for k,v in inputs.items()}
                predictions=sampler.sample(model,inputs,initial_noise=x.repeat(4,1,1,1,1),steps=50)
            assert torch.isfinite(predictions).all()
            path=out/'predictions'/f'{label}_{e["bank"]}_{Path(e["file"]).stem}.pt';save(path,predictions.cpu().float())
            valid=b['valid_mask']*(1-b['sampling_mask']);mask=b['sampling_mask']
            for j,sigma in enumerate(SIGMAS):
                pred=predictions[j:j+1];error=pred-predictions[:1]
                rows.append(dict(**e,variant=a.variant,checkpoint=label,step=step,input_sigma=sigma,metrics=metrics(pred,b),
                    output_rmse_vs_sigma0=float(((error.square()*valid).sum()/valid.sum().clamp_min(1)).sqrt()),
                    observed_residual_mse=float(((pred-b['observed_rss']).square()*mask).sum()/mask.sum().clamp_min(1)),prediction_path=str(path)))
            write(out/'status.json',dict(state='running',checkpoint=label,step=step,completed_cases=index+1,cases=len(entries),elapsed=time.time()-started))
        write(out/'rows.json',dict(complete=False,rows=rows))
        print(label,step,'done',flush=True)
    summary=[]
    for label in dict.fromkeys(r['checkpoint'] for r in rows):
        part=[r for r in rows if r['checkpoint']==label]
        matrix=[[float(np.mean([r['metrics']['unobserved_mse'] for r in part if r['sigma']==truth and r['input_sigma']==given])) for given in SIGMAS] for truth in SIGMAS]
        correct=float(np.mean([r['metrics']['unobserved_mse'] for r in part if r['sigma']==r['input_sigma']]))
        fixed=[float(np.mean([r['metrics']['unobserved_mse'] for r in part if r['input_sigma']==s])) for s in SIGMAS]
        mismatch=float(np.mean([r['metrics']['unobserved_mse'] for r in part if r['sigma']!=r['input_sigma']]))
        summary.append(dict(checkpoint=label,step=part[0]['step'],sigma_grid=SIGMAS,matrix=matrix,correct_mse=correct,fixed_mse=fixed,
            mismatched_mse=mismatch,row_minimum_input=[SIGMAS[int(np.argmin(row))] for row in matrix],
            mean_endpoint_response=float(np.mean([r['output_rmse_vs_sigma0'] for r in part if r['input_sigma']==.09]))))
    write(out/'rows.json',dict(complete=True,rows=rows));write(out/'summary.json',dict(variant=a.variant,summary=summary,
        caveats=['4 preselected confirmation videos; exploratory sensitivity/alignment probe, not formal test',
        'Within each case only supplied sigma changes; observations, masks, initial diffusion state, weights fixed',
        'Initialization invariant; early checkpoint is selection-biased, later checkpoint fixed by snapshot time, not by score',
        'Response alone does not establish useful or calibrated noise conditioning; diagonal need not minimize every finite-sample row']))
    write(out/'status.json',dict(state='complete',elapsed=time.time()-started,rows=len(rows)))

if __name__=='__main__':main()
