"""Paired paper tables; refuses missing rows unless explicitly asked for progress."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/paper_test200_20261006');p.add_argument('--remote');p.add_argument('--partial',action='store_true');args=p.parse_args()
    root=Path(args.root);bank=json.loads((root/'bank/bank.json').read_text());expected={e['file'] for e in bank['entries']}
    methods={};seen=set()
    locations=[root/m for m in ('main','w1','original')]
    if args.remote:locations += [Path(args.remote)/m for m in ('radiounet','rmegan','radiodiff')]
    for folder in locations:
        for file in sorted((folder/'cases').glob('*.json')):
            for row in json.loads(file.read_text())['rows']:
                key=(row['method'],row['file'])
                if key in seen:raise ValueError(f'Duplicate row: {key}')
                seen.add(key);methods.setdefault(row['method'],[]).append(row)
    required={'estimated_mixed','estimated_clean','known_sigma','fixed_003','fixed_zero','w1','original','radiounet','rmegan','radiodiff'}
    complete=set(methods)==required and all({r['file'] for r in rows}==expected for rows in methods.values())
    if not complete and not args.partial:raise ValueError('Incomplete suite; use --partial only for progress tables')
    out=root/('partial_tables' if not complete else 'tables');out.mkdir(exist_ok=True)
    noise=[];control=[]
    for rate in (1,2,3):
        main=[]
        for sigma in (0,.03,.05,.09):
            for method,rows in sorted(methods.items()):
                group=[r for r in rows if r['rate']==rate and r['sigma']==sigma]
                if not group:continue
                result=dict(rate=rate,true_sigma=sigma,true_variance=sigma**2,method=method,windows=len(group),videos=len({r['video_id'] for r in group}))
                result.update({k:float(np.mean([r['metrics'][k] for r in group])) for k in group[0]['metrics']})
                result['reconstruction_seconds_mean']=float(np.mean([r['seconds'] for r in group]));main.append(result)
                if method.startswith('estimated_'):
                    error=np.array([r['estimated_sigma']-sigma for r in group]);qe=np.array([r['estimated_variance']-sigma**2 for r in group])
                    noise.append(dict(rate=rate,true_sigma=sigma,method=method,windows=len(group),sigma_mae=float(np.abs(error).mean()),sigma_rmse=float(np.sqrt((error**2).mean())),sigma_bias=float(error.mean()),variance_mae=float(np.abs(qe).mean()),variance_rmse=float(np.sqrt((qe**2).mean())),variance_bias=float(qe.mean()),above_training_range_fraction=float(np.mean([r['estimated_sigma']>.09 for r in group])),estimation_seconds_mean=float(np.mean([r['estimation_seconds'] for r in group]))))
                if method in required-{'known_sigma','w1','original','radiounet','rmegan','radiodiff'}:
                    reference={r['file']:r for r in methods.get('known_sigma',[])}
                    paired=[r for r in group if r['file'] in reference]
                    if paired:control.append(dict(rate=rate,true_sigma=sigma,method=method,paired_windows=len(paired),psnr_delta_vs_known=float(np.mean([r['metrics']['psnr']-reference[r['file']]['metrics']['psnr'] for r in paired])),mse_delta_vs_known=float(np.mean([r['metrics']['mse']-reference[r['file']]['metrics']['mse'] for r in paired]))))
        write_csv(out/f'reconstruction_rate{rate}.csv',main)
    write_csv(out/'noise_accuracy.csv',noise);write_csv(out/'noise_condition_comparison.csv',control)
    (out/'coverage.json').write_text(json.dumps(dict(complete=complete,expected_per_method=len(expected),rows={m:len(r) for m,r in methods.items()},metrics='whole-image frame-average PSNR and whole-window MSE; unobserved free-space MSE separately',projection='none',primary='estimated_mixed'),indent=2)+'\n')
    print(json.dumps(dict(complete=complete,output=str(out),rows={m:len(r) for m,r in methods.items()})),flush=True)
if __name__=='__main__':main()
