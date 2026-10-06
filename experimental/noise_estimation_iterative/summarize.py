"""Local analysis of paired trajectories; no method selection on test data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    cases=[json.loads(p.read_text()) for p in sorted((root/'test/cases').glob('*.json'))]
    expected=json.loads((root/'inputs/test_bank.json').read_text())
    if {c['entry']['id'] for c in cases}!={e['id'] for e in expected}:raise ValueError('Incomplete test coverage')
    rows=[];summary=[]
    for c in cases:
        e=c['entry'];ref=c['baselines']['original_fixed_reference']['metrics']['psnr']
        for trajectory in c['trajectories']:
            for row in trajectory['rows']:
                rows.append(dict(video=e['video_id'],sigma=e['sigma'],initial_sigma=trajectory['initial_sigma'],iteration=row['iteration'],
                    estimated_sigma=row['updated_sigma'],sigma_error=row['updated_sigma']-e['sigma'],
                    raw_proposal_sigma=row['raw_proposal']['sigma'],psnr=row['metrics']['psnr'],mse=row['metrics']['mse'],
                    psnr_delta_original=row['metrics']['psnr']-ref,
                    psnr_delta_matched=row['metrics']['psnr']-c['baselines']['matched_one_pass']['metrics']['psnr'],
                    seconds=row['cumulative_seconds'],stop_eligible=row['stop_eligible']))
    for sigma in (0,.03,.05,.09):
        for initial in (0,.03,.09):
            for iteration in range(1,5):
                g=[r for r in rows if r['sigma']==sigma and r['initial_sigma']==initial and r['iteration']==iteration]
                errors=np.array([r['sigma_error'] for r in g])
                summary.append(dict(sigma=sigma,initial_sigma=initial,iteration=iteration,n=len(g),
                    sigma_mae=float(abs(errors).mean()),sigma_rmse=float(np.sqrt((errors**2).mean())),
                    **{k:float(np.mean([r[k] for r in g])) for k in ('estimated_sigma','psnr','mse','psnr_delta_original','psnr_delta_matched','seconds','stop_eligible')}))
    for name,data in [('paired_rows',rows),('summary',summary)]:
        with (root/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (root/'analysis_complete.json').write_text(json.dumps(dict(cases=len(cases),rows=len(rows),scope='8 videos, one rate and one window; pilot, not a full-paper claim'),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),rows=len(rows),output=str(root))))


if __name__=='__main__':main()
