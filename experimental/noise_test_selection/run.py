"""Local preparation/analysis, shared local/remote inference, durable reuse artifacts."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shlex
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
from .ranking import ranked, score_video

SEED=20261011
MODULE='experimental.noise_test_selection.run'


def input_name(vi,start):return f'v{vi:04d}_t{start:03d}.pt'


def case_name(vi,start,rate,sigma):return f'v{vi:04d}_t{start:03d}_r{rate}_s{sigma:g}'


def prepare(root,project):
    from rmdm.data import WindowDataset
    root.mkdir(parents=True,exist_ok=True)
    reference=project/'runs/noise_combined_20261010/hybrid_paired'
    formal=project/'runs/noise_trajectory_finetune_20261010/fast20_allparams_from2500'
    splitfile=project/'configs/splits/m20_formal075_clean16_scene_split.json'
    split=json.loads(splitfile.read_text())
    data_root=Path(json.loads((reference/'inputs/protocol.json').read_text())['source_root'])
    samples=json.loads((data_root/'index.json').read_text())['samples']
    population=sorted(f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}" for s in samples if s['scene_id'] in split['test'])
    assert len(population)==len(set(population))==1500
    videos=sorted(np.random.default_rng(SEED).choice(population,500,replace=False).tolist())
    dataset=WindowDataset(root=str(data_root),split='test',split_file=str(splitfile),window_size=16,
                          include_tx=False,video_ids=videos,fixed_starts=[0,80])
    counts={r.video_id:dataset.reader.frame_count(r) for r in dataset.reader.records}
    assert set(counts)==set(videos) and set(counts.values())=={100},'Head/tail windows must match actual video length'
    checkpoint=formal/'training/selected.pt'
    d=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert d['meta']['step']==7000 and d['meta']['weights']=='model'
    if not (root/'checkpoint.pt').exists():
        save(root/'checkpoint.pt',dict(model=d['model'],step=7000,weights='model',source=str(checkpoint)))
    del d
    shutil.copy2(formal/'model_config.yaml',root/'model_config.yaml')
    shutil.copy2(splitfile,root/'scene_split.json')
    # Random assignment independent of video content; keep exactly one owner per video.
    order=np.random.default_rng(SEED+1).permutation(500).tolist(); sizes=[58]*8+[22,14]
    allocations=[];offset=0
    for size in sizes:
        allocations.append(sorted(order[offset:offset+size]));offset+=size
    assert sorted(i for part in allocations for i in part)==list(range(500))
    spec=dict(protocol_id=root.name+'_v1',seed=SEED,population_count=1500,videos=videos,starts=[0,80],
        rates=[1,2,3],sigmas=SIGMAS,window_size=16,steps=50,eta=0,precision='float32',weights='model',checkpoint_step=7000,
        checkpoint=str(checkpoint),source_data_root=str(data_root),split='test',selected_count=200,
        selection='Uniform500withoutreplacement from sorted1500testvideoIDs; NumPy PCG64 seed20261011; no stratification or outcome filtering of candidates',
        video_unit='scene/episode/transmitter as in original1500-video test population',
        windows='0-based[0,16) and[80,96); first and last16frames within the96frame evaluation horizon',protocol_revision=2,evaluation_frames=96,
        rng='CPUtorch seed20261011+candidate_video_index*1000000+absolute_frame*10+stream(mask0,measurement1,initial2)',
        pairing='Nested1/2/3percent free-space masks; same Gaussian measurement field and diffusion initial state across all sigma interventions; no clipping of noisy RSS',
        ranking='Pool two windows SSE/count per video/rate/true/input sigma; G=(min_wrong_MSE-correct_MSE)/max(min_wrong_MSE,1e-12); rank descending meanG over12rate/true_sigma cells; tie lower mean correctMSE then videoID; select200 regardless of all-cells pass',
        ranking_metric='valid-unobserved MSE',
        inference='FP32 DDIM50 eta0, four input-sigmas batched; no projection or sigma remapping; six public model input fields only',
        retention='All500inputs and all48000FP32W16predictions; per-window sufficient stats, merged tables, all500ranking, selected200split and reuse index',
        worker_video_indices=allocations,worker_hosts=['local']*8+['remote']*2,
        local_ranks=list(range(8)),remote_ranks=[8,9],local_gpus=list(range(8)),remote_gpus=[0,1],
        expected_windows=1000,expected_cases=12000,expected_prediction_rows=48000,expected_merged_rows=24000)
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==spec
    write(root/'protocol.json',spec)
    write(root/'population1500.json',dict(seed=SEED,split='test',videos=population))
    write(root/'candidate500.json',dict(seed=SEED,split='test',videos=videos))
    write(root/'manifest.json',[dict(video_index=i,video_id=v,frame_count=counts[v],starts=spec['starts']) for i,v in enumerate(videos)])
    write(root/'allocation.json',[dict(rank=i,host=spec['worker_hosts'][i],gpu=i if i<8 else i-8,video_indices=indices) for i,indices in enumerate(allocations)])
    index={v:i for i,v in enumerate(videos)}
    for count,dense in enumerate(DataLoader(dataset,batch_size=1,num_workers=8,shuffle=False),1):
        video=dense['video_id'][0];vi=index[video];start=int(dense['start'][0]);dest=root/'inputs'/input_name(vi,start)
        if not dest.exists():
            masks={}
            for rate in spec['rates']:
                b,initial=materialize(dense,seed=SEED,video_indices=index,rate=rate,sigma=0.)
                masks[rate]=b['sampling_mask'].bool()
            eps=torch.stack([torch.randn(dense['target'].shape[2:],generator=torch.Generator().manual_seed(seed_for(SEED,vi,start+t,1))) for t in range(16)])[None]
            assert torch.all(masks[1]<=masks[2]) and torch.all(masks[2]<=masks[3])
            data={k:b[k] for k in ('target','building','vehicle')}
            data.update(valid_mask=b['valid_mask'].bool(),masks=masks,measurement_noise=eps,initial=initial,
                        video_id=video,video_index=vi,start=start,frame_names=dense['frame_names'],protocol_id=spec['protocol_id'])
            save(dest,data)
        if count%20==0 or count==1000:
            state=dict(state='complete' if count==1000 else 'preparing',windows=count,total=1000)
            write(root/'prepare_status.json',state);print(json.dumps(state),flush=True)
    with tarfile.open(root/'remote_inputs_w96.tar','w') as tar:
        for name in ['protocol.json','manifest.json','candidate500.json','population1500.json','allocation.json','model_config.yaml','scene_split.json','checkpoint.pt']:
            tar.add(root/name,arcname=name)
        for rank in spec['remote_ranks']:
            for vi in allocations[rank]:
                for start in spec['starts']:
                    name='inputs/'+input_name(vi,start);tar.add(root/name,arcname=name)
    write(root/'prepare_status.json',dict(state='complete',windows=1000,total=1000,remote_archive_bytes=(root/'remote_inputs_w96.tar').stat().st_size))


@torch.inference_mode()
def worker(root,rank,limit):
    from experimental.noise_combined.model import CombinedNoiseDiT
    from experimental.noise_hvdit.config import load_config
    from rmdm.diffusion import DDIMSampler
    torch.cuda.set_device(0);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert limit in (0,1,2), 'Smoke covers head-clean and tail-high-noise'
    spec=json.loads((root/'protocol.json').read_text());assigned=spec['worker_video_indices'][rank]
    out=root/('smoke' if limit else 'results')/f'worker{rank}';out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(str(root/'model_config.yaml'));model=CombinedNoiseDiT(cfg,'hybrid_paired',20,7.).cuda().eval()
    d=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    assert d['step']==7000 and d['weights']=='model';model.load_state_dict(d['model'],strict=True);del d
    sampler=DDIMSampler(cfg.diffusion);done=0;started=time.time();total=limit or len(assigned)*24
    def status(state,**extra):
        data=dict(state=state,pid=os.getpid(),rank=rank,completed_cases=done,total_cases=total,elapsed=time.time()-started,
            gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),peak_gb=torch.cuda.max_memory_allocated()/1e9,**extra)
        write(out/'status.json',data);return data
    status('running')
    for vi in assigned:
        for start in spec['starts']:
            data=torch.load(root/'inputs'/input_name(vi,start),weights_only=True)
            assert data['video_id']==spec['videos'][vi] and data['protocol_id']==spec['protocol_id'] and data['start']==start
            data={k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()}
            for rate in spec['rates']:
                for sigma in SIGMAS:
                    if limit and (vi!=assigned[0] or (start,rate,sigma) not in [(0,1,0.),(80,3,.09)][:limit]):
                        continue
                    stem=case_name(vi,start,rate,sigma);dest=out/'cases'/(stem+'.json');predpath=out/'predictions'/(stem+'.pt');seconds=None
                    if not (dest.exists() and predpath.exists()):
                        b={k:data[k].float() for k in ('target','building','vehicle','valid_mask')};mask=data['masks'][rate].cuda().float()
                        b.update(sampling_mask=mask,observed_rss=mask*(b['target']+sigma*data['measurement_noise']),sampling_rate=torch.full(b['target'].shape[:2],float(rate),device='cuda'))
                        inp=public(b,torch.tensor(SIGMAS,device='cuda').square())
                        assert set(inp)=={'building','vehicle','observed_rss','sampling_mask','sampling_rate','measurement_variance'}
                        inp={k:v if k=='measurement_variance' else v.repeat(4,*([1]*(v.ndim-1))) for k,v in inp.items()}
                        torch.cuda.synchronize();tick=time.perf_counter()
                        pred=sampler.sample(model,inp,initial_noise=data['initial'].repeat(4,1,1,1,1),steps=50,eta=0)
                        torch.cuda.synchronize();seconds=time.perf_counter()-tick
                        assert torch.isfinite(pred).all()
                        rows=[dict(video_id=spec['videos'][vi],video_index=vi,start=start,rate=rate,sigma=sigma,input_sigma=s,
                            prediction_index=j,stats=sufficient_stats(pred[j:j+1],b)) for j,s in enumerate(SIGMAS)]
                        save(predpath,dict(predictions=pred.cpu(),input_sigmas=SIGMAS,true_sigma=sigma,rate=rate,video_index=vi,
                            video_id=spec['videos'][vi],start=start,steps=50,weights='model',checkpoint_step=7000,protocol_id=spec['protocol_id']))
                        write(dest,dict(rows=rows,seconds=seconds,steps=50,protocol_id=spec['protocol_id'],prediction_file=str(predpath.relative_to(root))))
                        del pred,inp,b
                    done+=1;state=status('running',last_case_seconds=seconds)
                    if done%12==0 or limit:print(json.dumps(state),flush=True)
                    if done==total:
                        print(json.dumps(status('complete')),flush=True);return
    raise AssertionError((done,total))


def launch(root,host,limit,cloud):
    spec=json.loads((root/'protocol.json').read_text());jobs=[]
    for rank,gpu in zip(spec[host+'_ranks'],spec[host+'_gpus']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(root),'--rank',str(rank),'--limit',str(limit)]
        log=root/f"{'smoke' if limit else 'worker'}{rank}.log";handle=log.open('a')
        process=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(pid=process.pid,rank=rank,gpu=gpu,command=cmd,log=str(log),process=process,handle=handle))
    def snapshot():return [{**{k:v for k,v in j.items() if k not in ('process','handle')},'exit_code':j['process'].poll()} for j in jobs]
    jobfile=root/f"{'smoke_' if limit else ''}{host}_jobs.json"
    while True:
        write(jobfile,snapshot())
        if all(j['process'].poll() is not None for j in jobs):break
        time.sleep(10)
    for j in jobs:j['handle'].close()
    assert all(j['process'].returncode==0 for j in jobs),snapshot()
    if host=='remote' and not limit:
        for label,predictions in [('metrics',False),('predictions',True)]:
            archive=root/f'remote_{label}.tar'
            with tarfile.open(archive,'w') as tar:
                for rank in spec['remote_ranks']:
                    worker_root=root/'results'/f'worker{rank}'
                    for name in (['predictions'] if predictions else ['cases','status.json']):
                        path=worker_root/name;tar.add(path,arcname=str(path.relative_to(root)))
            with (root/f'upload_{label}.log').open('a') as log:
                subprocess.run(['aliyunpan','upload',str(archive),cloud],check=True,stdout=log,stderr=subprocess.STDOUT)
            write(root/f'export_{label}.json',dict(state='uploaded',cloud=cloud+'/'+archive.name,bytes=archive.stat().st_size))
    write(root/f"{'smoke_' if limit else ''}{host}_complete.json",dict(state='complete',finished_at=time.time(),jobs=snapshot()))


def summarize(root):
    spec=json.loads((root/'protocol.json').read_text());files=sorted((root/'results').glob('worker*/cases/*.json'))
    assert len(files)==spec['expected_cases'],(len(files),spec['expected_cases'])
    grouped=defaultdict(list);window_rows=[];reuse=[];seen=set()
    for p in files:
        d=json.loads(p.read_text());assert d['protocol_id']==spec['protocol_id'] and d['steps']==50 and len(d['rows'])==4
        for r in d['rows']:
            assert spec['videos'][r['video_index']]==r['video_id']
            key=(r['video_id'],r['start'],r['rate'],r['sigma'],r['input_sigma']);assert key not in seen;seen.add(key)
            grouped[(key[0],*key[2:])].append(r)
            window_rows.append({k:v for k,v in r.items() if k!='stats'}|merge_stats([r['stats']]))
            reuse.append({k:v for k,v in r.items() if k!='stats'}|dict(input_file='inputs/'+input_name(r['video_index'],r['start']),
                prediction_file=d['prediction_file'],case_file=str(p.relative_to(root)),protocol_id=spec['protocol_id'],checkpoint_step=7000))
    merged={}
    for key,rows in grouped.items():
        assert len(rows)==2 and {r['start'] for r in rows}==set(spec['starts'])
        merged[key]=merge_stats([r['stats'] for r in rows])
    assert len(merged)==spec['expected_merged_rows']
    ranking=[]
    for vi,video in enumerate(spec['videos']):
        matrices=[[[merged[(video,rate,s,t)]['unobserved_mse'] for t in SIGMAS] for s in SIGMAS] for rate in spec['rates']]
        ranking.append(dict(video_id=video,video_index=vi,**score_video(matrices)))
    ranking=ranked(ranking)
    for i,r in enumerate(ranking,1):r.update(rank=i,selected=i<=spec['selected_count'])
    selected=ranking[:spec['selected_count']];out=root/'tables';out.mkdir(exist_ok=True)
    def csvfile(name,rows):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csvfile('window_metrics.csv',window_rows)
    long=[dict(video_id=k[0],rate=k[1],sigma=k[2],input_sigma=k[3],**v) for k,v in sorted(merged.items())]
    csvfile('merged_metrics.csv',long)
    csvfile('ranking500.csv',[{k:v for k,v in r.items() if k not in ('matrices','gains_percent')} for r in ranking])
    csvfile('reuse_index.csv',reuse)
    selected_ids={r['video_id'] for r in selected}
    csvfile('selected200_metrics.csv',[r for r in long if r['video_id'] in selected_ids])
    csvfile('selected200_reuse_index.csv',[r for r in reuse if r['video_id'] in selected_ids])
    write(out/'ranking500.json',ranking)
    split=dict(schema='rmdm_ranked_test_video_split_v1',protocol_id=spec['protocol_id'],seed=spec['seed'],
        source_split='test',source_population_count=1500,candidate_count=500,selection_count=200,
        selection_rule=spec['ranking'],selection_checkpoint_step=7000,selection_checkpoint_weights='model',
        starts=spec['starts'],window_size=16,rates=spec['rates'],sigmas=SIGMAS,
        videos=[r['video_id'] for r in selected],video_indices={r['video_id']:r['video_index'] for r in selected},
        reuse_index='tables/selected200_reuse_index.csv',all_candidates='candidate500.json')
    write(root/'selected200_split.json',split)
    def summary(rs):
        ids={r['video_id'] for r in rs};a=np.asarray([r['matrices'] for r in rs]);means=a.mean(0)
        return dict(videos=len(rs),score_percent_mean=float(np.mean([r['score_percent'] for r in rs])),
            score_percent_range=[min(r['score_percent'] for r in rs),max(r['score_percent'] for r in rs)],
            all12_videos=sum(r['correct_minimum_cells']==12 for r in rs),
            mean_matrices_by_rate={str(rate):means[j].tolist() for j,rate in enumerate(spec['rates'])},
            mean_assessments_by_rate={str(rate):assess(means[j]) for j,rate in enumerate(spec['rates'])},
            correct_sigma_metrics={k:float(np.mean([r[k] for r in long if r['video_id'] in ids and r['sigma']==r['input_sigma']])) for k in next(iter(merged.values()))},
            scene_counts={scene:sum(v.startswith(scene+'/') for v in ids) for scene in sorted({v.split('/')[0] for v in ids})})
    write(out/'report.json',dict(complete=True,protocol_id=spec['protocol_id'],checkpoint_step=7000,
        cases=len(files),all500=summary(ranking),selected200=summary(selected),selection_rule=spec['ranking'],
        split_path=str(root/'selected200_split.json'),artifact_root=str(root)))
    print(json.dumps(dict(state='selection_complete',report=str(out/'report.json')),ensure_ascii=False),flush=True)


def collect(root):
    config=json.loads((root/'collection_config.json').read_text());started=time.time()
    def remote_ready(label):
        cmd='cat '+shlex.quote(config['remote_root']+'/export_'+label+'.json')
        r=subprocess.run(config['ssh']+[cmd],capture_output=True,text=True,timeout=30)
        return r.returncode==0 and json.loads(r.stdout).get('state')=='uploaded'
    try:
        for label in ['metrics','predictions']:
            while not remote_ready(label):
                if time.time()-started>12*3600:raise TimeoutError('Remote export not complete within12h')
                write(root/'collection_status.json',dict(state='waiting_remote_'+label,updated_at=time.time()))
                time.sleep(45)
            dest=root/'downloads'/label;dest.mkdir(parents=True,exist_ok=True)
            with (root/f'download_{label}.log').open('a') as log:
                subprocess.run(['aliyunpan','download','--np','--saveto',str(dest),config['cloud']+'/remote_'+label+'.tar'],check=True,stdout=log,stderr=subprocess.STDOUT)
            archives=list(dest.rglob('remote_'+label+'.tar'));assert len(archives)==1
            with tarfile.open(archives[0]) as tar:tar.extractall(root,filter='data')
            if label=='metrics':
                while not (root/'local_complete.json').exists():
                    if time.time()-started>12*3600:raise TimeoutError('Local evaluation not complete within12h')
                    write(root/'collection_status.json',dict(state='waiting_local',updated_at=time.time()));time.sleep(30)
                summarize(root)
        spec=json.loads((root/'protocol.json').read_text());predictions=list((root/'results').glob('worker*/predictions/*.pt'))
        assert len(predictions)==spec['expected_cases'],len(predictions)
        state=dict(state='complete',finished_at=time.time(),prediction_files=len(predictions),report=str(root/'tables/report.json'),split=str(root/'selected200_split.json'))
        write(root/'collection_status.json',state)
        import yaml
        path=Path(config['record']);record=yaml.safe_load(path.read_text());record.update(status='complete',finished_at=time.time(),result=state,next='Frozen selected200split and all paired evaluation artifacts ready for reuse.')
        tmp=path.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(record,sort_keys=False,allow_unicode=True));tmp.replace(path)
        print(json.dumps(state),flush=True)
    except Exception as exc:
        write(root/'collection_status.json',dict(state='failed',error=repr(exc),updated_at=time.time()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','launch','summarize','collect'])
    p.add_argument('--root',required=True);p.add_argument('--project',default='/data_p6/fzj/projects/RMDM')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--host',choices=['local','remote'],default='local');p.add_argument('--cloud')
    args=p.parse_args();torch.set_num_threads(4);root=Path(args.root).resolve()
    if args.mode=='prepare':prepare(root,Path(args.project).resolve())
    elif args.mode=='worker':worker(root,args.rank,args.limit)
    elif args.mode=='launch':launch(root,args.host,args.limit,args.cloud)
    elif args.mode=='summarize':summarize(root)
    else:collect(root)


if __name__=='__main__':main()
