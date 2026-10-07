"""Paired original-DDIM50 controls; retain failures and fixed-sigma winners."""
import csv,json
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .run import ROOT,OLD


def main():
 selected=json.loads((ROOT/'selected.json').read_text());r=[]
 for i in range(8):
  d=json.loads((ROOT/f'confirm_part{i}/matrix.json').read_text());assert d['complete'];r+=d['rows']
 assert len(r)==376,len(r)
 base=json.loads((OLD/'baseline/matrix.json').read_text())['rows'];tune=[x for x in base if x['group']=='tune']
 fixed={s:float(np.mean([x['metrics']['unobserved_mse'] for x in tune if x['input_sigma']==s])) for s in [0,.03,.05,.09]};best_base=min(fixed,key=fixed.get)
 old=[dict(x,bank='original') for x in base if x['group']=='confirm']
 for i in [0,1]:
  for x in json.loads(Path(f'runs/noise_condition_diagnosis_20261007/fresh_results{i}.json').read_text())['rows']:
   if x['method']=='original_DDIM50':old.append(dict(x,bank='fresh',input_sigma=x['external_sigma']))
 methods={'fusion_true':[x for x in r if x['input_sigma']==x['sigma']], 'fusion_best_fixed':[x for x in r if x['input_sigma']==selected['best_fixed_sigma']], 'original_true':[x for x in old if x['input_sigma']==x['sigma']], 'original_best_fixed':[x for x in old if x['input_sigma']==best_base]}
 for s in [0,.03,.05,.09]:methods[f'fusion_fixed{s:g}']=[x for x in r if x['input_sigma']==s]
 methods['original_fixed0']=[x for x in old if x['input_sigma']==0];methods['original_fixed003']=[x for x in old if x['input_sigma']==.03]
 summary=[];by_sigma=[];paired=[]
 groups={'original_confirm4':lambda x:x['bank']=='original','fresh4':lambda x:x['bank']=='fresh' and x['sigma'] in [0,.03,.05,.09],'fresh7':lambda x:x['bank']=='fresh','combined4':lambda x:x['sigma'] in [0,.03,.05,.09]}
 for group,keep in groups.items():
  subset={m:[x for x in rows if keep(x)] for m,rows in methods.items()}
  for method,a in subset.items():
   assert a
   summary.append(dict(group=group,method=method,windows=len(a),**{k:float(np.mean([x['metrics'][k] for x in a])) for k in a[0]['metrics']}))
   for sigma in sorted({x['sigma'] for x in a}):
    b=[x for x in a if x['sigma']==sigma];by_sigma.append(dict(group=group,method=method,sigma=sigma,windows=len(b),**{k:float(np.mean([x['metrics'][k] for x in b])) for k in b[0]['metrics']}))
  videos=sorted({x['video_id'] for x in subset['fusion_true']})
  for other in ['fusion_best_fixed','original_true','original_best_fixed']:
   left=subset['fusion_true'];right=subset[other]
   assert {(x['video_id'],x['sigma']) for x in left}=={(x['video_id'],x['sigma']) for x in right}
   delta=np.array([np.mean([x['metrics']['unobserved_mse'] for x in left if x['video_id']==v])-np.mean([x['metrics']['unobserved_mse'] for x in right if x['video_id']==v]) for v in videos]);rng=np.random.default_rng(20261007);boot=delta[rng.integers(0,len(videos),(2000,len(videos)))].mean(1)
   paired.append(dict(group=group,comparison='fusion_true minus '+other,delta_unseen_mse=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),videos=len(videos)))
 result=dict(selected=selected,original_best_fixed_sigma=best_base,original_tune_fixed_mse=fixed,summary=summary,by_sigma=by_sigma,paired=paired,limitations=['Frozen weights; not a trained noise-fusion model','Only2%/start0, reused validation videos across2scenes; exploratory not formal test','Fusion of already-conditioned predictions is not an exact Bayesian posterior','Only full DDIM50 is evaluated; no direct one-step reconstruction or sigma remapping'])
 write(ROOT/'analysis.json',result)
 for name,rows in [('summary',summary),('by_sigma',by_sigma)]:
  with (ROOT/f'{name}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(selected=selected['candidate'],summary=[x for x in summary if x['group']=='combined4']),indent=2))
if __name__=='__main__':main()
