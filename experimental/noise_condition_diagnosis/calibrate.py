"""Fit monotone internal conditioning on tune8; verify on fresh validation videos."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from rmdm.data import WindowDataset
from rmdm.diffusion import DDIMSampler
from experimental.paper_evaluation.randomness import materialize
from experimental.noise_loss_screen.common import public,metrics,write
from .audit import ROOT,OLD,load

LEVELS=[0,.03,.05,.09]

def fit():
 rows=sum([json.loads((ROOT/f'samplers{i}.json').read_text())['rows'] for i in [0,1]],[])
 a=np.array([[np.mean([r['metrics']['unobserved_mse'] for r in rows if r['group']=='tune' and r['mode']=='one_step_999' and r['seed_offset']==0 and r['sigma']==s and r['input_sigma']==v]) for v in LEVELS] for s in LEVELS])
 options=[idx for idx in itertools.product(range(4),repeat=4) if idx[0]==0 and all(idx[i]<=idx[i+1] for i in range(3))]
 best=min(options,key=lambda ids:np.mean([a[i,j] for i,j in enumerate(ids)]))
 return dict(external_sigmas=LEVELS,internal_sigmas=[LEVELS[j] for j in best],interpolation='Piecewise linear in variance q=sigma²; clip to training range[0,.09²]',fit_videos='Original tune8 only; seed0; minimum aggregate unseen MSE subject to monotonicity and f(0)=0',fit_mse=float(np.mean([a[i,j] for i,j in enumerate(best)])),sampler='Pure Gaussian at t999; one x0 prediction',limitations='Internal conditioning is calibrated, not the actual physical measurement variance; original estimator can be retained independently.')

def prepare():
 out=ROOT/'fresh_inputs';out.mkdir(exist_ok=True)
 if (out/'bank.json').exists():raise FileExistsError('Fresh validation bank already exists')
 mapping=fit();write(ROOT/'condition_calibration.json',mapping)
 manifest=json.loads(Path('/dev/shm/noise_temporal_clean16_train_v2/manifest.json').read_text());data_root=Path(manifest['source_root']);splitfile=Path('configs/splits/m20_formal075_clean16_scene_split.json').resolve();split=json.loads(splitfile.read_text())
 all_samples=sorted(json.loads((data_root/'index.json').read_text())['samples'],key=lambda s:(s['scene_id'],s['episode_id'],s['tx_id']))
 indices={f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}":i for i,s in enumerate(all_samples)}
 old=list(json.loads((OLD/'inputs/protocol.json').read_text())['validation_groups'])+json.loads(Path('runs/paper_test200_20261006/calibration.json').read_text())['calibrators']['mixed']['videos']
 excluded={v.rsplit('/',1)[0] for v in old};videos=[]
 for si,scene in enumerate(sorted(split['val'])):
  samples=[s for s in all_samples if s['scene_id']==scene and f"{scene}/{s['episode_id']}" not in excluded]
  episodes=sorted({s['episode_id'] for s in samples});rng=np.random.default_rng(20261008+si*100000)
  for episode in rng.choice(episodes,4,replace=False).tolist():
   candidates=[s for s in samples if s['episode_id']==episode];s=candidates[int(rng.integers(len(candidates)))];videos.append(f"{scene}/{episode}/{s['tx_id']}")
 write(out/'protocol.json',dict(seed=20261008,videos=videos,excluded_episodes=sorted(excluded),true_sigmas=[0,.015,.03,.04,.05,.07,.09],rate=2,start=0,test_used=False,selection='Four distinct new episodes per validation scene; all previous tune/confirm/calibration episodes excluded',frozen_mapping=mapping,observation_rng='Existing paper materialize integer-seed protocol; exact sparse inputs and initial noise saved'))
 ds=WindowDataset(root=str(data_root),split='val',split_file=str(splitfile),window_size=16,include_tx=False,video_ids=sorted(videos),fixed_starts=[0]);entries=[]
 for dense in DataLoader(ds,batch_size=1,num_workers=2):
  video=dense['video_id'][0]
  for sigma in [0,.015,.03,.04,.05,.07,.09]:
   sparse,initial=materialize(dense,video_index=indices[video],rate=2,sigma=sigma)
   filename=f'fresh_{len(entries):04d}.pt';torch.save(dict(sparse=sparse,initial=initial),out/filename)
   entries.append(dict(file=filename,video_id=video,sigma=sigma,rate=2,start=0,group='fresh_validation'))
 write(out/'bank.json',entries)

@torch.no_grad()
def evaluate(part):
 cfg,model,_=load();model.eval();sampler=DDIMSampler(cfg.diffusion);mapping=json.loads((ROOT/'condition_calibration.json').read_text());out=ROOT/'fresh_inputs';rows=[];start=time.time()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def transform(sigma):return float(np.sqrt(np.interp(sigma*sigma,np.square(mapping['external_sigmas']),np.square(mapping['internal_sigmas']))))
 for e in json.loads((out/'bank.json').read_text())[part::2]:
  d=torch.load(out/e['file'],weights_only=True);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['sparse'].items()};initial=d['initial'].cuda()
  for label,external,internal in [('calibrated_correct',e['sigma'],transform(e['sigma'])),('original_correct',e['sigma'],e['sigma'])]+[(f'calibrated_fixed{s}',s,transform(s)) for s in LEVELS]+[(f'original_fixed{s}',s,s) for s in LEVELS]:
   cache=model.encode_conditions(public(b,torch.tensor([internal**2],device='cuda')))
   pred=model.denoise(initial,torch.tensor([999],device='cuda'),cache).clamp(0,1)
   rows.append(dict(**e,method=label,external_sigma=external,internal_sigma=internal,metrics=metrics(pred,b)))
  for inp in sorted({0,.03,e['sigma']}):
   pred=sampler.sample(model,public(b,torch.tensor([inp**2],device='cuda')),initial_noise=initial,steps=50,eta=0)
   rows.append(dict(**e,method='original_DDIM50',external_sigma=inp,internal_sigma=inp,metrics=metrics(pred,b)))
  write(ROOT/f'fresh_results{part}.json',dict(complete=False,rows=rows,elapsed=time.time()-start))
 write(ROOT/f'fresh_results{part}.json',dict(complete=True,rows=rows,elapsed=time.time()-start))

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--part',type=int,choices=[0,1]);a=p.parse_args()
 if a.prepare:prepare()
 else:evaluate(a.part)
if __name__=='__main__':main()
