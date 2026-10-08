"""Summarize reconstruction and sigma intervention without collapsing the two."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .evaluate import SIGMAS,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'training';tables=[];matrices=[];selections=[]
    for name in ['best_reconstruction','best_alignment']:
        path=out/(name+'_matrix.json')
        if not path.exists():continue
        doc=json.loads(path.read_text());assert doc['complete'];rows=doc['rows'];assert len(rows)==1512
        selections.append(dict(criterion=name,selection=doc['selection']))
        constant=float(doc['selection']['best_fixed_sigma'])
        for group in ['tune','confirm']:
            for rate in [1,2,3]:
                part=[x for x in rows if x['group']==group and x['rate']==rate]
                for mode in ['correct','selected_constant']:
                    selected=[x for x in part if x['input_sigma']==(x['sigma'] if mode=='correct' else constant)]
                    for truth in sorted({x['sigma'] for x in selected}):
                        subset=[x for x in selected if x['sigma']==truth]
                        tables.append(dict(criterion=name,group=group,rate=rate,true_sigma=truth,input_mode=mode,constant=constant,n=len(subset),**{k:float(np.mean([x['metrics'][k] for x in subset])) for k in subset[0]['metrics']}))
                standard=[x for x in part if x['sigma'] in SIGMAS]
                matrices.append(dict(criterion=name,group=group,rate=rate,**summarize(standard)))
    write(root/'analysis.json',dict(selections=selections,sigma_matrices=matrices,by_sigma=tables,caveats=['All weights randomly initialized; original full HWM/DiT trained with six losses','Training sigma uniform0-.09,rate discrete uniform1-10%; no clean mixture component','Validation only,1/2/3% rates; reuse exploratory validation scenes/videos; no formal test','Best-alignment and best-reconstruction criteria are separate; checkpoint step/weight type retained','Best constant chosen from tune selection record for confirmation comparisons; row minima are descriptive diagnostics']))
    if tables:
        with (root/'by_sigma.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    print('Analyzed',len(selections),'checkpoint criteria',flush=True)
if __name__=='__main__':main()
