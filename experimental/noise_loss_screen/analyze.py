"""Local validation-only loss screen; report all variants, never test selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    locations=[p for p in root.iterdir() if p.is_dir() and not p.name.startswith('smoke')]
    if (root/'remote_results').exists():locations+=list((root/'remote_results').iterdir())
    rows=[];summary=[]
    for folder in locations:
        file=folder/'matrix.json'
        if not file.exists():continue
        report=json.loads(file.read_text())
        if not report['complete']:continue
        for r in report['rows']:rows.append(dict(variant=folder.name,group=r['group'],video_id=r['video_id'],true_sigma=r['sigma'],input_sigma=r['input_sigma'],**r['metrics']))
        for group in ('tune','confirm'):
            for sigma in (0,.03,.05,.09):
                for inp in sorted({r['input_sigma'] for r in report['rows']}):
                    selected=[r for r in report['rows'] if r['group']==group and r['sigma']==sigma and r['input_sigma']==inp]
                    if selected:summary.append(dict(variant=folder.name,group=group,true_sigma=sigma,input_sigma=inp,windows=len(selected),**{k:float(np.mean([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}))
    out=root/'analysis';out.mkdir(exist_ok=True)
    for name,data in [('rows',rows),('summary',summary)]:
        if data:
            with (out/f'{name}.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(variants=sorted({r['variant'] for r in rows}),rows=len(rows),output=str(out))))

if __name__=='__main__':main()
