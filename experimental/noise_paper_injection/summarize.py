"""Video-paired, tune-frozen sigma controls and reference-model comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write
from .model import VARIANTS
from .evaluate import SIGMAS


def comparison(rows, fixed):
    rows=[r for r in rows if r['group']=='confirm' and r['sigma'] in SIGMAS]
    videos=sorted({r['video_id'] for r in rows});pairs=[]
    for video in videos:
        part=[r for r in rows if r['video_id']==video]
        pairs.append([np.mean([r['metrics']['unobserved_mse'] for r in part if r['input_sigma']==r['sigma']]),
                      np.mean([r['metrics']['unobserved_mse'] for r in part if r['input_sigma']==fixed])])
    pairs=np.array(pairs);rng=np.random.default_rng(20261009);draw=pairs[rng.integers(len(pairs),size=(10000,len(pairs)))].mean(1)
    gain=1-draw[:,0]/draw[:,1]
    return dict(videos=len(videos),correct_mse=float(pairs[:,0].mean()),fixed_mse=float(pairs[:,1].mean()),
                gain=float(1-pairs[:,0].mean()/pairs[:,1].mean()),gain_video_bootstrap95=np.quantile(gain,[.025,.975]).tolist(),
                video_wins=int((pairs[:,0]<pairs[:,1]).sum()),fixed_sigma=fixed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root);records=[]
    for variant in VARIANTS:
        for path in sorted((root/variant/'training').glob('*_matrix.json')):
            d=json.loads(path.read_text());constant=d['selection']['best_fixed_sigma']
            records.append(dict(variant=variant,criterion=path.stem,selection=d['selection'],rates={str(rate):comparison([r for r in d['rows'] if rate is None or r['rate']==rate],constant) for rate in [None,1,2,3]}))
        for path in sorted((root/variant/'training').glob('*_repeat*.json')):
            d=json.loads(path.read_text());records.append(dict(variant=variant,criterion=path.stem,repeat=d['repeat'],confirmation=comparison(d['rows'],d['selection']['best_fixed_sigma'])))
    old=Path('runs/noise_scratch_full_20261008');references={}
    for name,path in [('paper_step16000',old/'paper_model_comparison/matrix.json'),('scratch_step7000_ema',old/'training/best_reconstruction_matrix.json'),('scratch_step2000_model',old/'training/best_alignment_matrix.json')]:
        d=json.loads(path.read_text());references[name]={str(rate):comparison([r for r in d['rows'] if rate is None or r['rate']==rate],0.) for rate in [None,1,2,3]}
    write(root/'comparison.json',dict(results=records,references=references,notes=['References use identical original bank and DDIM50; sigma gain reference here is versus fixed0, not reselected fixed','New candidates select fixed sigma on tune only; confirmation bootstrap over videos is exploratory','Initialization seeds,2-step training and global batch differ from prior scratch reference; causal comparison limited to current three matched arms','Check alternate diffusion seeds before declaring stable conditioning; no test-set evaluation']))


if __name__=='__main__':main()
