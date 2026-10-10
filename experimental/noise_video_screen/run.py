import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from rmdm.data import WindowDataset
from rmdm.diffusion import DDIMSampler
from experimental.noise_combined.model import CombinedNoiseDiT
from experimental.noise_hvdit.config import load_config
from experimental.noise_loss_screen.common import public,write,save
from experimental.paired_evaluation.protocol import materialize,seed_for
from .statistics import SIGMAS,sufficient_stats

SEED=20261012
GPUS=[0,1,4,5]


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    reference=Path('runs/noise_combined_20261010/hybrid_paired')
    splitfile=Path('configs/splits/m20_formal075_clean16_scene_split.json').resolve()
    split=json.loads(splitfile.read_text())
    data_root=Path(json.loads((reference/'inputs/protocol.json').read_text())['source_root'])
    samples=json.loads((data_root/'index.json').read_text())['samples']
    chosen=[]
    for si,scene in enumerate(sorted(split['test'])):
        population=sorted(f"{scene}/{s['episode_id']}/{s['tx_id']}" for s in samples if s['scene_id']==scene)
        assert len(population)==750
        chosen.extend(np.random.default_rng(SEED+si).choice(population,300,replace=False).tolist())
    videos=sorted(chosen);assert len(set(videos))==600
    checkpoint=reference/'training/checkpoints/step007000.pt'
    d=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);assert d['step']==7000
    save(root/'checkpoint.pt',dict(model=d['ema'],source=str(checkpoint.resolve()),step=7000,weights='ema'));del d
    shutil.copy2(reference/'model_config.yaml',root/'model_config.yaml')
    protocol=dict(seed=SEED,selection='300 videos uniform without replacement per test scene;sorted population,NumPy PCG64 seed20261012+sceneindex;no hashes',
        videos=videos,data_root=str(data_root),split_file=str(splitfile),split='test',starts=[0,48],window_size=16,
        rates=[1,2,3],sigmas=SIGMAS,steps=50,checkpoint=str(checkpoint.resolve()),weights='ema',gpus=GPUS,
        expected_windows=1200,expected_cases=14400,expected_window_rows=57600,expected_merged_rows=28800,
        pairing='CPU torch per physical frame seed20261012+video_index*1000000+frame*10+stream(mask0,measurement1,initial2);same observed RSS andinitial noise across4input sigmas; nested masks acrossrates',
        metric='Merge two windows by SSE/count for each video/rate/true/input sigma;PSNR from pooled MSE,also retainmean-framePSNR;do not formtemporal edge between windows',
        selection_metric='valid-unobserved MSE; diagonal is tied minimum within1e-12;report perrate4/4,mean-of-three-rate matrix4/4,andall12/12',
        artifacts='All per-window sufficient statistics;all mergedvideo matrices;selectedlists;600video summary;randominput bundles;predictions forpreselected videoindices0,100,200,300,400,500')
    write(root/'protocol.json',protocol)
    write(root/'manifest.json',dict(seed=SEED,split='test',videos=[dict(video_index=i,video_id=v,scene=v.split('/')[0],episode=v.rsplit('/',1)[0],starts=[0,48]) for i,v in enumerate(videos)]))
    (root/'videos.csv').write_text('video_index,video_id,start1,start2\n'+''.join(f'{i},{v},0,48\n' for i,v in enumerate(videos)))


def bundle(dense,vi,video,root):
    start=int(dense['start'][0]);dest=root/'inputs'/f'v{vi:03d}_t{start:02d}.pt'
    if dest.exists():return torch.load(dest,weights_only=False)
    masks={};initial=None
    for rate in [1,2,3]:
        b,initial=materialize(dense,seed=SEED,video_indices={video:vi},rate=rate,sigma=0)
        masks[rate]=b['sampling_mask'].bool()
    eps=torch.stack([torch.randn(dense['target'].shape[2:],generator=torch.Generator().manual_seed(seed_for(SEED,vi,start+t,1))) for t in range(16)])[None]
    assert torch.all(masks[1]<=masks[2]) and torch.all(masks[2]<=masks[3])
    data=dict(target=b['target'],building=b['building'],vehicle=b['vehicle'],valid_mask=b['valid_mask'].bool(),
              masks=masks,measurement_noise=eps,initial=initial,video_id=video,start=start)
    save(dest,data);return data


@torch.inference_mode()
def worker(root,rank,limit):
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    spec=json.loads((root/'protocol.json').read_text());videos=spec['videos'];indices={v:i for i,v in enumerate(videos)}
    workers=len(spec.get('gpus',GPUS));assigned=videos[rank::workers]
    if 'worker_video_indices' in spec:
        allocation=spec['worker_video_indices']
        flat=[i for part in allocation for i in part]
        assert sorted(flat)==list(range(len(videos)))
        assigned=[videos[i] for i in allocation[rank]]
    out=root/('smoke' if limit else 'results')/f'worker{rank}';out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(str(root/'model_config.yaml'));model=CombinedNoiseDiT(cfg,'hybrid_paired',20,7.).cuda().eval()
    d=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(d['model']);del d
    sampler=DDIMSampler(cfg.diffusion)
    dataset=WindowDataset(root=spec['data_root'],split='test',split_file=spec['split_file'],window_size=16,include_tx=False,video_ids=assigned,fixed_starts=spec['starts'])
    assert len(dataset)==len(assigned)*len(spec['starts'])
    done=0;started=time.time();total=limit or len(dataset)*12
    write(out/'status.json',dict(state='running',pid=os.getpid(),completed_cases=0,total_cases=total))
    for dense in DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0):
        video=dense['video_id'][0];vi=indices[video];start=int(dense['start'][0])
        data=None
        for rate in [1,2,3]:
            for sigma in SIGMAS:
                dest=out/'cases'/f'v{vi:03d}_t{start:02d}_r{rate}_s{sigma:g}.json'
                if not dest.exists():
                    if data is None:
                        data=bundle(dense,vi,video,root)
                        data={k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()}
                    mask=data['masks'][rate].cuda().float()
                    b={k:data[k].float() for k in ['target','building','vehicle','valid_mask']}
                    b.update(sampling_mask=mask,observed_rss=mask*(b['target']+sigma*data['measurement_noise']),sampling_rate=torch.full(b['target'].shape[:2],float(rate),device='cuda'))
                    inp=public(b,torch.tensor(SIGMAS,device='cuda').square())
                    inp={k:v if k=='measurement_variance' else v.repeat(4,*([1]*(v.ndim-1))) for k,v in inp.items()}
                    tick=time.time();pred=sampler.sample(model,inp,initial_noise=data['initial'].repeat(4,1,1,1,1),steps=50)
                    torch.cuda.synchronize();assert torch.isfinite(pred).all();seconds=time.time()-tick
                    rows=[dict(video_id=video,video_index=vi,start=start,scene=video.split('/')[0],rate=rate,sigma=sigma,input_sigma=s,stats=sufficient_stats(pred[j:j+1],b)) for j,s in enumerate(SIGMAS)]
                    if vi%100==0 and not limit:save(root/'predictions'/dest.with_suffix('.pt').name,dict(sigmas=SIGMAS,predictions=pred.cpu()))
                    write(dest,dict(rows=rows,seconds=seconds))
                    del pred,inp,b
                done+=1
                status=dict(state='running',pid=os.getpid(),completed_cases=done,total_cases=total,elapsed=time.time()-started,last_video=video,start=start,peak_gb=torch.cuda.max_memory_allocated()/1e9)
                write(out/'status.json',status)
                if done%12==0 or limit:print(json.dumps(status),flush=True)
                if limit and done>=limit:break
            if limit and done>=limit:break
        if limit and done>=limit:break
    assert done==total
    write(out/'status.json',dict(state='complete',pid=os.getpid(),completed_cases=done,total_cases=total,elapsed=time.time()-started))


def launch(root,limit):
    record=Path('.agents/runs')/(root.name+'.yaml');jobs={};handles=[];started=time.time()
    spec=json.loads((root/'protocol.json').read_text())
    for rank,gpu in enumerate(spec.get('gpus',GPUS)):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-u','-m','experimental.noise_video_screen.run','worker','--root',str(root),'--rank',str(rank),'--limit',str(limit)]
        log=root/f"{'smoke' if limit else 'worker'}{rank}.log";handle=log.open('a');handles.append(handle)
        proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[str(rank)]=dict(process=proc,pid=proc.pid,gpu=gpu,command=cmd,log=str(log),state='running')
    def snapshot():return {n:{**{k:v for k,v in j.items() if k!='process'},'exit_code':j['process'].poll()} for n,j in jobs.items()}
    d=json.loads(record.read_text());d.update(status='smoke_running' if limit else 'running',started_at=started,jobs=snapshot());write(record,d)
    while any(j['state']=='running' for j in jobs.values()):
        for j in jobs.values():
            if j['state']!='running':continue
            if j['process'].poll() is not None:j['state']='complete' if j['process'].returncode==0 else 'failed'
            elif time.time()-started>(1800 if limit else spec.get('budget_hours',7)*3600):
                os.killpg(j['pid'],signal.SIGTERM)
                try:j['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(j['pid'],signal.SIGKILL);j['process'].wait()
                j['state']='time_limit'
        write(root/('smoke_jobs.json' if limit else 'jobs.json'),snapshot())
        if any(j['state']=='running' for j in jobs.values()):time.sleep(10)
    for h in handles:h.close()
    complete=all(j['state']=='complete' for j in jobs.values())
    if complete and not limit and spec['starts']==[0,48]:
        result=subprocess.run([sys.executable,'-m','experimental.noise_video_screen.summarize','--root',str(root)])
        complete=result.returncode==0
    d=json.loads(record.read_text());d.update(status=('smoke_complete' if limit else 'complete') if complete else 'incomplete',finished_at=time.time(),jobs=snapshot(),next='Read local tables or synchronize supplemental results for three-window aggregation' if complete and not limit else 'Inspect worker states');write(record,d)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','launch']);p.add_argument('--root',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int,default=0);a=p.parse_args();root=Path(a.root).resolve()
    if a.mode=='prepare':prepare(root)
    elif a.mode=='worker':worker(root,a.rank,a.limit)
    else:launch(root,a.limit)


if __name__=='__main__':main()
