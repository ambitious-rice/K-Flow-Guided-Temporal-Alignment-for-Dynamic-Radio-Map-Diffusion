import csv,json
import numpy as np
from .audit import ROOT

def main():
 rows=[]
 for part in [0,1]:
  d=json.loads((ROOT/f'fresh_results{part}.json').read_text());assert d['complete'];rows+=d['rows']
 methods={m:[r for r in rows if r['method']==m] for m in sorted({r['method'] for r in rows if r['method']!='original_DDIM50'})}
 for label,select in [('DDIM50_correct',lambda r:r['external_sigma']==r['sigma']),('DDIM50_fixed0',lambda r:r['external_sigma']==0),('DDIM50_fixed003',lambda r:r['external_sigma']==.03)]:methods[label]=[r for r in rows if r['method']=='original_DDIM50' and select(r)]
 summary=[];by_sigma=[];paired=[]
 for name,a in methods.items():
  assert len(a)==56,(name,len(a))
  for group,sigmas in [('all7',[0,.015,.03,.04,.05,.07,.09]),('original4',[0,.03,.05,.09]),('intermediate3',[.015,.04,.07])]:
   selected=[r for r in a if r['sigma'] in sigmas]
   summary.append(dict(method=name,group=group,n=len(selected),**{k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}))
  for sigma in [0,.015,.03,.04,.05,.07,.09]:
   selected=[r for r in a if r['sigma']==sigma];by_sigma.append(dict(method=name,sigma=sigma,**{k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}))
 for other in ['original_correct','original_fixed0.03','DDIM50_correct','DDIM50_fixed0','DDIM50_fixed003']:
  candidate=methods['calibrated_correct'];reference=methods[other];vids=sorted({r['video_id'] for r in candidate})
  for group,sigmas in [('all7',[0,.015,.03,.04,.05,.07,.09]),('original4',[0,.03,.05,.09]),('intermediate3',[.015,.04,.07])]:
   delta=np.array([np.mean([r['metrics']['unobserved_mse'] for r in candidate if r['video_id']==v and r['sigma'] in sigmas])-np.mean([r['metrics']['unobserved_mse'] for r in reference if r['video_id']==v and r['sigma'] in sigmas]) for v in vids]);rng=np.random.default_rng(20261008);boot=np.mean(delta[rng.integers(0,len(vids),size=(2000,len(vids)))],axis=1)
   paired.append(dict(group=group,comparison='calibrated_correct minus '+other,delta_unseen_mse=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),caveat='Exploratory8-video bootstrap across fixed2 validation scenes, not a formal test result'))
 result=dict(summary=summary,by_sigma=by_sigma,paired_comparisons=paired,mapping=json.loads((ROOT/'condition_calibration.json').read_text()),protocol=json.loads((ROOT/'fresh_inputs/protocol.json').read_text()))
 (ROOT/'fresh_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 for filename,data in [('fresh_summary',summary),('fresh_by_sigma',by_sigma)]:
  with (ROOT/(filename+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 for x in summary:
  if x['group']=='original4' and x['method'] in ['calibrated_correct','original_correct','original_fixed0.03','DDIM50_correct','DDIM50_fixed0','DDIM50_fixed003']:print(x)
 print('PER SIGMA: calibrated vs best fixed internal .03')
 for sig in [0,.015,.03,.04,.05,.07,.09]:
  print(sig,{x['method']:round(x['unobserved_mse']*1e4,4) for x in by_sigma if x['sigma']==sig and x['method'] in ['calibrated_correct','original_fixed0.03','original_correct']})
 print('PAIRED',json.dumps(paired))
if __name__=='__main__':main()
