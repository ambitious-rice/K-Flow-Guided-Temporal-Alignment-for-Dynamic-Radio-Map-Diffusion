"""Completed-suite means and paired episode bootstrap, stratified by scene."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    root=Path('runs/paper_test200_20261006');expected={e['file'] for e in json.loads((root/'bank/bank.json').read_text())['entries']}
    methods=defaultdict(dict)
    locations=[root/m for m in ('main','w1','original')]+[root/'remote_results'/m for m in ('radiounet','rmegan','radiodiff')]
    for folder in locations:
        for path in sorted((folder/'cases').glob('*.json')):
            for row in json.loads(path.read_text())['rows']:
                assert row['file'] not in methods[row['method']]
                assert np.isfinite(list(row['metrics'].values())).all()
                methods[row['method']][row['file']]=row
    assert len(methods)==10 and all(set(m)==expected for m in methods.values())
    videos=sorted({r['video_id'] for r in methods['estimated_mixed'].values()})
    scenes=sorted({v.split('/')[0] for v in videos});assert len(videos)==200 and len(scenes)==2
    # Exactly one transmitter per episode: videos are episode clusters here.
    assert len({v.rsplit('/',1)[0] for v in videos})==200
    rng=np.random.default_rng(20261007)
    indices=[]
    for scene in scenes:
        members=np.array([i for i,v in enumerate(videos) if v.startswith(scene+'/')]);assert len(members)==100
        indices.append(members[rng.integers(0,100,size=(2000,100))])
    resampled=np.concatenate(indices,axis=1)
    summary={'coverage':{m:len(rows) for m,rows in methods.items()},'metrics_finite':True,
             'bootstrap':{'seed':20261007,'replicates':2000,'unit':'episode (one video each), keep both windows and all conditions together','strata':'two fixed test scenes; resample 100 episodes within each','interval':'95% percentile; test-episode uncertainty conditional on fixed checkpoints and scenes, not training-seed or unseen-scene uncertainty'},'overall':{},'noise':[]}
    for m,byfile in methods.items():
        rows=list(byfile.values());summary['overall'][m]={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
        summary['overall'][m]['reconstruction_seconds']=float(np.mean([r['seconds'] for r in rows]))
    for sigma in (0,.03,.05,.09):
        group=[r for r in methods['estimated_mixed'].values() if r['sigma']==sigma]
        error=np.array([r['estimated_sigma']-sigma for r in group]);qerror=np.array([r['estimated_variance']-sigma**2 for r in group])
        summary['noise'].append(dict(sigma=sigma,n=len(group),mean_estimate=float(np.mean([r['estimated_sigma'] for r in group])),sigma_mae=float(abs(error).mean()),sigma_rmse=float(np.sqrt((error**2).mean())),variance_mae=float(abs(qerror).mean()),above_range=float(np.mean([r['estimated_sigma']>.09 for r in group])),estimation_seconds=float(np.mean([r['estimation_seconds'] for r in group]))))
    intervals=[]
    for method in ('known_sigma','w1','original','radiounet','rmegan','radiodiff','fixed_003','fixed_zero'):
        byvideo=defaultdict(list)
        for file,row in methods['estimated_mixed'].items():
            ref=methods[method][file];assert (row['video_id'],row['start'],row['rate'],row['sigma'])==(ref['video_id'],ref['start'],ref['rate'],ref['sigma'])
            byvideo[row['video_id']].append(row['metrics']['psnr']-ref['metrics']['psnr'])
        assert all(len(v)==24 for v in byvideo.values())
        values=np.array([np.mean(byvideo[v]) for v in videos]);boot=values[resampled].mean(1)
        low,high=np.quantile(boot,[.025,.975]);intervals.append(dict(method='estimated_mixed',reference=method,psnr_delta=float(values.mean()),ci95_low=float(low),ci95_high=float(high)))
    out=root/'tables';(out/'complete_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'paired_episode_bootstrap.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(intervals[0]));writer.writeheader();writer.writerows(intervals)
    print(json.dumps(summary,indent=2));print(json.dumps(intervals,indent=2))


if __name__=='__main__':main()
