"""Fixed-observation sigma interventions, full DDIM50 and spatial/temporal metrics."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from rmdm.diffusion import DDIMSampler
from experimental.noise_loss_screen.common import public,metrics,save

SIGMAS=[0.,.03,.05,.09]

@torch.inference_mode()
def evaluate(model,cfg,inputs,rank,world,*,full=False,limit=0,prediction_dir=None):
    entries=json.loads((inputs/'validation_bank.json').read_text())
    if not full:entries=[e for e in entries if e['group']=='tune']
    if limit:entries=entries[:limit]
    model.eval();sampler=DDIMSampler(cfg.diffusion);rows=[]
    for e in entries[rank::world]:
        d=torch.load(Path(e['base'])/e['file'],weights_only=True)
        b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()};sigmas=sorted(set(SIGMAS+[e['sigma']]))
        inp=public(b,torch.tensor(sigmas,device='cuda').square())
        inp={k:v if k=='measurement_variance' else v.repeat(len(sigmas),*([1]*(v.ndim-1))) for k,v in inp.items()}
        tick=time.perf_counter();pred=sampler.sample(model,inp,initial_noise=d['initial'].cuda().repeat(len(sigmas),1,1,1,1),steps=50)
        torch.cuda.synchronize();assert torch.isfinite(pred).all();seconds=time.perf_counter()-tick
        path=None
        if prediction_dir is not None:
            path=prediction_dir/(Path(e['file']).stem+'.pt');save(path,dict(sigmas=sigmas,predictions=pred.cpu().float()))
        unseen=b['valid_mask']*(1-b['sampling_mask'])
        for j,s in enumerate(sigmas):
            response=(((pred[j:j+1]-pred[:1]).float().square()*unseen).sum()/unseen.sum().clamp_min(1)).sqrt()
            rows.append(dict(**e,input_sigma=s,metrics=metrics(pred[j:j+1],b),output_rmse_vs_zero=float(response),seconds_all_inputs=seconds,prediction_path=str(path) if path else None))
    parts=[None for _ in range(world)];dist.all_gather_object(parts,rows);return [r for part in parts for r in part]


def summarize(rows):
    correct=[r for r in rows if r['input_sigma']==r['sigma']]
    fixed={s:float(np.mean([r['metrics']['unobserved_mse'] for r in rows if r['input_sigma']==s])) for s in SIGMAS}
    score=float(np.mean([r['metrics']['unobserved_mse'] for r in correct]));constant=min(fixed,key=fixed.get)
    matrix=[[float(np.mean([r['metrics']['unobserved_mse'] for r in rows if r['sigma']==s and r['input_sigma']==t])) for t in SIGMAS] for s in SIGMAS]
    return dict(correct_mse=score,fixed_mse=fixed,best_fixed_sigma=constant,alignment_gain=1-score/max(fixed[constant],1e-12),matrix=matrix,
        row_min_input=[SIGMAS[int(np.argmin(row))] for row in matrix],mean_response=float(np.mean([r['output_rmse_vs_zero'] for r in rows if r['input_sigma']==.09])),
        correct_metrics={k:float(np.mean([r['metrics'][k] for r in correct])) for k in correct[0]['metrics']})
