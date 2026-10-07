"""Report full-DDIM50 checkpoints, fixed-sigma sweeps, and matched baselines."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .model import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);summary=[];selected=[];by_sigma=[]
    for variant in VARIANTS:
        status=json.loads((root/variant/'status.json').read_text());d=json.loads((root/variant/'matrix.json').read_text());assert d['complete'];rows=d['rows'];assert len(rows)==504
        tune=[r for r in rows if r['group']=='tune'];fixed={s:float(np.mean([r['metrics']['unobserved_mse'] for r in tune if r['input_sigma']==s])) for s in [0,.03,.05,.09]};best=min(fixed,key=fixed.get)
        selected.append(dict(variant=variant,best_step=status['best_step'],best_weights=status['best_weights'],trained_steps=status['step'],best_fixed_sigma=best,tune_fixed_mse=fixed))
        for group,keep in [('tune',lambda r:r['group']=='tune'),('confirmation4',lambda r:r['group']=='confirm' and r['sigma'] in [0,.03,.05,.09]),('confirmation_intermediate',lambda r:r['group']=='confirm' and r['sigma'] in [.015,.04,.07])]:
            for mode,select in [('correct',lambda r:r['input_sigma']==r['sigma']),('best_fixed',lambda r:r['input_sigma']==best)]+[(f'fixed{s:g}',lambda r,s=s:r['input_sigma']==s) for s in [0,.03,.05,.09]]:
                subset=[r for r in rows if keep(r) and select(r)];assert subset
                summary.append(dict(variant=variant,group=group,input=mode,n=len(subset),best_step=status['best_step'],**{k:float(np.mean([r['metrics'][k] for r in subset])) for k in subset[0]['metrics']}))
                for sigma in sorted({r['sigma'] for r in subset}):
                    part=[r for r in subset if r['sigma']==sigma];by_sigma.append(dict(variant=variant,group=group,input=mode,sigma=sigma,n=len(part),**{k:float(np.mean([r['metrics'][k] for r in part])) for k in part[0]['metrics']}))
    write(root/'analysis.json',dict(selected=selected,summary=summary,by_sigma=by_sigma,caveats=['Warm-start full DiT retraining; frozen HWM/old conditioning, not from scratch','Exploratory reused validation at2%, no formal test used','Best checkpoints can be initialization; trained step count and best_step both retained','Compare correct-input versus best fixed chosen on tune AND separately trained blind/constant-reliability controls']))
    for name,rows in [('summary',summary),('by_sigma',by_sigma)]:
        with (root/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
