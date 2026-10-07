"""Summarize causal probes without treating an exploratory improvement as a fix."""
import csv,json
from pathlib import Path
import numpy as np
from .audit import ROOT,OLD

def main():
 rows=sum([json.loads((ROOT/f'samplers{i}.json').read_text())['rows'] for i in [0,1]],[])
 old=json.loads((OLD/'baseline/matrix.json').read_text())['rows']
 summary=[]
 for group in ['tune','confirm']:
  for method,data in [('DDIM50_original',old),('one_step_999',[r for r in rows if r['mode']=='one_step_999' and r['seed_offset']==0])]:
   for label in ['correct','fixed_0','fixed_0.03','fixed_0.05','fixed_0.09']:
    selected=[r for r in data if r['group']==group and r['input_sigma']==(r['sigma'] if label=='correct' else float(label[6:]))]
    summary.append(dict(group=group,method=method,input=label,n=len(selected),**{k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}))
 comparison=[]
 for group in ['tune','confirm']:
  one=[r for r in rows if r['mode']=='one_step_999' and r['seed_offset']==0 and r['group']==group];videos=sorted({r['video_id'] for r in one})
  for label,other in [('same_sampler_fixed_003',[r for r in one if r['input_sigma']==.03]),('old_ddim50_fixed_003',[r for r in old if r['group']==group and r['input_sigma']==.03]),('old_ddim50_correct',[r for r in old if r['group']==group and r['input_sigma']==r['sigma']])]:
   correct=[r for r in one if r['input_sigma']==r['sigma']]
   deltas=np.array([np.mean([r['metrics']['unobserved_mse'] for r in correct if r['video_id']==v])-np.mean([r['metrics']['unobserved_mse'] for r in other if r['video_id']==v]) for v in videos])
   rng=np.random.default_rng(20261007);boot=np.mean(deltas[rng.integers(0,len(videos),size=(2000,len(videos)))],axis=1)
   comparison.append(dict(group=group,comparison='one_step_correct minus '+label,video_count=len(videos),delta_unseen_mse=float(deltas.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),caveat='Small exploratory video bootstrap, fixed two validation scenes; no multi-seed training inference'))
 result=dict(summary=summary,paired_comparisons=comparison,conclusions=['The one-step predictor starts from pure Gaussian noise at t999, with no complete truth input. It is not DDIMSampler(steps=1), whose default timestep differs.','One-step does not solve per-noise sigma alignment: .03 still prefers0; .09 prefers intermediate sigma.','Shared auxiliary gradients and sampling changes are observed; neither is proven to be the sole root cause.','Any sampler change invalidates previous ensemble-uncertainty calibration: recalibrate on clean validation before blind estimation comparison.','No test videos used for these probes.'])
 (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 with (ROOT/'sampler_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,8));sigmas=[0,.03,.05,.09]
 for gi,group in enumerate(['tune','confirm']):
  for mi,(method,data) in enumerate([('Original DDIM-50',old),('Pure-noise prediction at t=999',[r for r in rows if r['mode']=='one_step_999' and r['seed_offset']==0])]):
   a=np.array([[np.mean([r['metrics']['unobserved_mse'] for r in data if r['group']==group and r['sigma']==s and r['input_sigma']==inp])*1e4 for inp in sigmas] for s in sigmas]);ax=axes[gi,mi]
   ax.imshow(a,cmap='YlOrRd',vmin=2,vmax=17)
   for y in range(4):
    for x in range(4):ax.text(x,y,f'{a[y,x]:.2f}',ha='center',va='center',fontweight='bold' if x==a[y].argmin() else 'normal')
   ax.set_xticks(range(4),sigmas);ax.set_yticks(range(4),sigmas);ax.set_xlabel('Input sigma');ax.set_ylabel('True measurement sigma');ax.set_title(f'{group} (8 videos): {method}')
 fig.suptitle('Unobserved MSE x 10^4 (lower is better); bold = row minimum')
 fig.tight_layout();fig.savefig(ROOT/'sigma_response.png',dpi=180);plt.close(fig)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
