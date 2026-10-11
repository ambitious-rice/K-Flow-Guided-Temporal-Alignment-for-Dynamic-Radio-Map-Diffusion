"""Prepare fixed video inputs locally; run remote interventions; summarize locally."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experimental.noise_loss_screen.common import public, save, write
from experimental.noise_video_screen.statistics import SIGMAS, assess, merge_stats, sufficient_stats
from experimental.paired_evaluation.protocol import materialize, seed_for

SEED = 20261011


def video_ids(value, scenes):
    if isinstance(value, dict):
        for item in value.values():
            yield from video_ids(item, scenes)
    elif isinstance(value, list):
        for item in value:
            yield from video_ids(item, scenes)
    elif isinstance(value, str):
        parts = value.split('/')
        if len(parts) == 3 and parts[0] in scenes and parts[1].startswith('episode_'):
            yield value


def prepare(root):
    from rmdm.data import WindowDataset
    root.mkdir(parents=True, exist_ok=True)
    reference = Path('runs/noise_combined_20261010/hybrid_paired')
    formal = Path('runs/noise_trajectory_finetune_20261010/fast20_allparams_from2500')
    splitfile = Path('configs/splits/m20_formal075_clean16_scene_split.json').resolve()
    split = json.loads(splitfile.read_text())
    data_root = Path(json.loads((reference/'inputs/protocol.json').read_text())['source_root'])
    samples = json.loads((data_root/'index.json').read_text())['samples']
    exclusions = sorted(set(Path('runs').glob('*/inputs/validation_bank.json')) | {
        reference/'inputs/validation_bank.json',
        Path('runs/noise_paper_injection_20261009/expanded100/protocol.json'),
        Path('runs/paper_test200_20261006/calibration_bank/bank.json'),
    })
    excluded = set()
    for p in exclusions:
        excluded.update(v.rsplit('/', 1)[0] for v in video_ids(json.loads(p.read_text()), split['val']))
    chosen = []
    for si, scene in enumerate(sorted(split['val'])):
        eligible = [s for s in samples if s['scene_id'] == scene and
                    f"{scene}/{s['episode_id']}" not in excluded]
        episodes = sorted({s['episode_id'] for s in eligible})
        rng = np.random.default_rng(SEED + si*100000)
        assert len(episodes) >= 10
        for episode in sorted(rng.choice(episodes, 10, replace=False).tolist()):
            candidates = sorted((s for s in eligible if s['episode_id'] == episode), key=lambda s: str(s['tx_id']))
            s = candidates[int(rng.integers(len(candidates)))]
            chosen.append(f"{scene}/{episode}/{s['tx_id']}")
    videos = sorted(chosen)
    assert len(videos) == len(set(videos)) == len({v.rsplit('/', 1)[0] for v in videos}) == 20
    assert not {v.rsplit('/', 1)[0] for v in videos} & excluded
    checkpoint = formal/'training/checkpoints/step004400.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['step'] == 4400
    save(root/'checkpoint.pt', dict(model=payload['model'], step=4400, weights='model', source=str(checkpoint.resolve())))
    del payload
    shutil.copyfile(formal/'model_config.yaml', root/'model_config.yaml')
    shutil.copyfile(splitfile, root/'scene_split.json')
    protocol = dict(seed=SEED, split='val', videos=videos, starts=[0, 48], window_size=16,
        rates=[1, 2, 3], sigmas=SIGMAS, steps=50, weights='model', checkpoint_step=4400,
        checkpoint=str(checkpoint.resolve()), source_data_root=str(data_root), excluded_episodes=sorted(excluded),
        exclusion_sources=[str(p.resolve()) for p in exclusions],
        selection='10 random episodes per validation scene without replacement; one random transmitter per episode; frozen before inference',
        rng='NumPy PCG64 seed20261011+scene_index*100000 for episode/Tx selection; CPU torch seed20261011+video_index*1000000+absolute_frame*10+stream(mask0,measurement1,initial2)',
        pairing='Nested masks across rates; one Gaussian field and initial state per window; identical observations and initial state across input-sigma interventions',
        inference='FP32 DDIM50 eta0; batch4 input-sigma conditions; no observation projection; no sigma remapping; explicit target-free model input whitelist',
        aggregation='Pool window SSE/count within video, then equally average videos/rates; also report each window separately; temporal differences never cross disjoint windows',
        fixed_sigma_reference=.05, fixed_reference_source='Frozen from original tune8 for finetune4400; not selected on these20 videos',
        evaluation_scope='New episodes in the existing validation scenes; this is independent-video confirmation, not unseen-scene validation; no checkpoint tuning from this run',
        expected_cases=480, expected_rows=1920, gpus=[0, 1])
    if (root/'protocol.json').exists():
        assert json.loads((root/'protocol.json').read_text()) == protocol, 'Existing preparation protocol differs'
    write(root/'protocol.json', protocol)
    write(root/'manifest.json', [dict(video_index=i, video_id=v, starts=protocol['starts']) for i, v in enumerate(videos)])
    (root/'videos.csv').write_text('video_index,video_id,start1,start2\n'+''.join(f'{i},{v},0,48\n' for i,v in enumerate(videos)))
    dataset = WindowDataset(root=str(data_root), split='val', split_file=str(splitfile),
        window_size=16, include_tx=False, video_ids=videos, fixed_starts=protocol['starts'])
    assert len(dataset) == 40
    indices = {v: i for i,v in enumerate(videos)}
    for count, dense in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0), 1):
        video = dense['video_id'][0]; vi = indices[video]; start = int(dense['start'][0])
        masks = {}
        for rate in protocol['rates']:
            b, initial = materialize(dense, seed=SEED, video_indices=indices, rate=rate, sigma=0.)
            masks[rate] = b['sampling_mask'].bool()
        eps = torch.stack([torch.randn(dense['target'].shape[2:], generator=torch.Generator().manual_seed(
            seed_for(SEED, vi, start+t, 1))) for t in range(16)])[None]
        assert torch.all(masks[1] <= masks[2]) and torch.all(masks[2] <= masks[3])
        data = {k: b[k] for k in ('target', 'building', 'vehicle', 'valid_mask')}
        data.update(masks=masks, measurement_noise=eps, initial=initial, video_id=video, start=start)
        save(root/'inputs'/f'v{vi:03d}_t{start:02d}.pt', data)
        write(root/'prepare_status.json', dict(state='complete' if count==40 else 'preparing', windows=count, total=40))
        print(json.dumps(dict(prepared_windows=count, total=40)), flush=True)


@torch.inference_mode()
def worker(root, rank, limit=0, steps=50):
    from experimental.noise_combined.model import CombinedNoiseDiT
    from experimental.noise_hvdit.config import load_config
    from rmdm.diffusion import DDIMSampler
    torch.cuda.set_device(0); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    spec = json.loads((root/'protocol.json').read_text())
    assert limit or steps == spec['steps']
    allocation = spec.get('worker_video_indices', [list(range(0,20,2)),list(range(1,20,2))])
    assert sorted(i for part in allocation for i in part) == list(range(20))
    out = root/('smoke' if limit else 'results')/f'worker{rank}'
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(root/'model_config.yaml'))
    model = CombinedNoiseDiT(cfg, 'hybrid_paired', 20, 7.).cuda().eval()
    payload = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    assert payload['step'] == spec['checkpoint_step'] and payload['weights'] == 'model'
    model.load_state_dict(payload['model'], strict=True); del payload
    sampler = DDIMSampler(cfg.diffusion)
    total = limit or len(allocation[rank])*len(spec['starts'])*12
    done = 0; started = time.time()
    def status(state, **extra):
        value = dict(state=state, pid=os.getpid(), completed_cases=done, total_cases=total,
                     elapsed=time.time()-started, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                     steps=steps, peak_gb=torch.cuda.max_memory_allocated()/1e9, **extra)
        write(out/'status.json', value)
        return value
    status('running')
    for vi in allocation[rank]:
        for start in spec['starts']:
            data = torch.load(root/'inputs'/f'v{vi:03d}_t{start:02d}.pt', weights_only=True)
            data = {k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()}
            for rate in spec['rates']:
                for sigma in SIGMAS:
                    dest = out/'cases'/f'v{vi:03d}_t{start:02d}_r{rate}_s{sigma:g}.json'
                    seconds = None
                    if not dest.exists():
                        b = {k:data[k].float() for k in ('target','building','vehicle','valid_mask')}
                        mask = data['masks'][rate].cuda().float()
                        b.update(sampling_mask=mask, observed_rss=mask*(b['target']+sigma*data['measurement_noise']),
                                 sampling_rate=torch.full(b['target'].shape[:2], float(rate), device='cuda'))
                        inp = public(b, torch.tensor(SIGMAS,device='cuda').square())
                        assert set(inp) == {'building','vehicle','observed_rss','sampling_mask','sampling_rate','measurement_variance'}
                        inp = {k:v if k=='measurement_variance' else v.repeat(4,*([1]*(v.ndim-1))) for k,v in inp.items()}
                        torch.cuda.synchronize(); tick=time.perf_counter()
                        pred = sampler.sample(model, inp, initial_noise=data['initial'].repeat(4,1,1,1,1), steps=steps, eta=0)
                        torch.cuda.synchronize(); seconds=time.perf_counter()-tick
                        assert torch.isfinite(pred).all()
                        rows = [dict(video_id=spec['videos'][vi], video_index=vi, start=start, rate=rate,
                            sigma=sigma, input_sigma=s, stats=sufficient_stats(pred[j:j+1],b)) for j,s in enumerate(SIGMAS)]
                        write(dest, dict(rows=rows, seconds=seconds, steps=steps))
                        if not limit and vi in (0,10) and start==0 and rate==2 and sigma in (0.,.09):
                            save(root/'predictions'/dest.with_suffix('.pt').name,dict(sigmas=SIGMAS,predictions=pred.cpu()))
                        del pred, inp, b
                    done += 1
                    state = status('running', last_case_seconds=seconds)
                    if done%12==0 or limit: print(json.dumps(state),flush=True)
                    if done==total:
                        print(json.dumps(status('complete')),flush=True)
                        return
    raise AssertionError((done,total))


def summarize(root):
    spec=json.loads((root/'protocol.json').read_text()); files=sorted((root/'results').glob('worker*/cases/*.json'))
    assert len(files)==spec['expected_cases']
    windows={}; grouped=defaultdict(list)
    for p in files:
        d=json.loads(p.read_text()); assert d['steps']==50 and len(d['rows'])==4
        for r in d['rows']:
            key=(r['video_id'],r['start'],r['rate'],r['sigma'],r['input_sigma'])
            assert key not in windows
            windows[key]=r['stats']; grouped[(key[0],*key[2:])].append(r)
    merged={}
    for key,rows in grouped.items():
        assert {r['start'] for r in rows}==set(spec['starts']) and len(rows)==len(spec['starts'])
        merged[key]=merge_stats([r['stats'] for r in rows])
    out=root/'tables';out.mkdir(exist_ok=True)
    long=[dict(video_id=k[0],rate=k[1],sigma=k[2],input_sigma=k[3],**m) for k,m in sorted(merged.items())]
    with (out/'merged_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(long[0]));w.writeheader();w.writerows(long)
    def report(start=None):
        def value(v,r,s,t):
            return (merged[(v,r,s,t)] if start is None else merge_stats([windows[(v,start,r,s,t)]]))['unobserved_mse']
        arr=np.array([[[[value(v,r,s,t) for t in SIGMAS] for s in SIGMAS] for r in spec['rates']] for v in spec['videos']])
        def stats(a):
            mean=a.mean(0); assessments=[assess(m) for m in a]; fixed=SIGMAS.index(spec['fixed_sigma_reference'])
            correct=np.diagonal(a,axis1=1,axis2=2).mean(1);constant=a[:,:,fixed].mean(1)
            rng=np.random.default_rng(SEED+900);ix=rng.integers(len(a),size=(2000,len(a)))
            gains=100*(1-correct[ix].mean(1)/constant[ix].mean(1))
            comparison=[]
            for i,s in enumerate(SIGMAS):
                j=min((j for j in range(4) if j!=i),key=lambda j:mean[i,j])
                comparison.append(dict(sigma=s,correct_mse=float(mean[i,i]),best_wrong_sigma=SIGMAS[j],
                    best_wrong_mse=float(mean[i,j]),gain_percent=float(100*(1-mean[i,i]/mean[i,j]))))
            return dict(mean_matrix=mean.tolist(),mean_assessment=assess(mean),
                all_four_videos=sum(a['all_four'] for a in assessments),
                correct_minimum_videos_by_sigma={str(s):sum(a['pass_by_sigma'][i] for a in assessments) for i,s in enumerate(SIGMAS)},
                comparison=comparison,fixed_sigma=spec['fixed_sigma_reference'],
                fixed_sigma_gain_percent=float(100*(1-correct.mean()/constant.mean())),
                fixed_sigma_gain_bootstrap95=np.quantile(gains,[.025,.975]).tolist())
        return dict(combined=stats(arr.mean(1)),by_rate={str(r):stats(arr[:,i]) for i,r in enumerate(spec['rates'])},
            all12_videos=sum(all(assess(m)['all_four'] for m in matrices) for matrices in arr),
            per_video={v:{str(r):assess(arr[i,j]) for j,r in enumerate(spec['rates'])} for i,v in enumerate(spec['videos'])})
    result=dict(complete=True,videos=len(spec['videos']),cases=len(files),checkpoint_step=spec['checkpoint_step'],
        weights=spec['weights'],combined_windows=report(),by_window={str(s):report(s) for s in spec['starts']},
        correct_sigma_metrics={key:float(np.mean([r[key] for r in long if r['sigma']==r['input_sigma']])) for key in next(iter(merged.values()))},
        notes=[spec['evaluation_scope'],'All20 videos retained; no selection by sigma success','Bootstrap resamples videos and keeps windows/rates paired; percentile95%,2000 replicates','Best wrong sigma and best fixed sigma on confirmation are descriptive; fixed0.05 reference frozen before inference'])
    write(out/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['combined_windows','by_window']},ensure_ascii=False),flush=True)


def launch(root, cloud):
    spec=json.loads((root/'protocol.json').read_text());jobs=[]
    for rank,gpu in enumerate(spec['gpus']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-u','-m',__package__+'.run','worker','--root',str(root),'--rank',str(rank)]
        log=root/f'worker{rank}.log';handle=log.open('a')
        proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
        jobs.append(dict(proc=proc,handle=handle,pid=proc.pid,gpu=gpu,command=cmd,log=str(log)))
    def snapshot():
        return [{**{k:v for k,v in j.items() if k not in ['proc','handle']},'exit_code':j['proc'].poll()} for j in jobs]
    write(root/'jobs.json',snapshot())
    while any(j['proc'].poll() is None for j in jobs):time.sleep(10)
    write(root/'jobs.json',snapshot())
    assert all(j['proc'].returncode==0 for j in jobs),snapshot()
    for j in jobs:j['handle'].close()
    files=list((root/'results').glob('worker*/cases/*.json'));assert len(files)==spec['expected_cases']
    archive=root/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['protocol.json','manifest.json','videos.csv','model_config.yaml','scene_split.json','jobs.json','results','predictions','worker0.log','worker1.log']:
            if (root/name).exists():tar.add(root/name,arcname=name)
    subprocess.run(['aliyunpan','upload',str(archive),cloud],check=True)
    write(root/'export.json',dict(state='uploaded',cloud=cloud+'/results.tar.gz',bytes=archive.stat().st_size,cases=len(files)))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summarize','launch'])
    p.add_argument('--root',required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--steps',type=int,default=50);p.add_argument('--cloud')
    a=p.parse_args();torch.set_num_threads(4);root=Path(a.root).resolve()
    if a.mode=='prepare':prepare(root)
    elif a.mode=='worker':worker(root,a.rank,a.limit,a.steps)
    elif a.mode=='summarize':summarize(root)
    else:launch(root,a.cloud)


if __name__=='__main__':main()
