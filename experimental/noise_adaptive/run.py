"""Frozen-model pilot; labels are available only to calibration and scoring.

Outer iterations refit a predictive Gaussian likelihood, not an exact EM
objective. Timing separates estimator work from diagnostic reconstructions.
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from experimental.noise_loss_screen.common import public, save, write
from experimental.noise_video_screen.statistics import merge_stats, sufficient_stats
from experimental.paper_evaluation.calibration import fit_variance_calibration
from .predict import collect
from .statistics import calibrated_variance, mle, update


def ordered_variants(spec):
    # Parents precede derived variants, allowing exact first-round reuse.
    names = ['w16_f4_m8', 'w16_f4_m4', 'w16_f3_m4', 'w16_spatial3_m4',
             'w4_chunks_f4_m4', 'w4_prefix_f4_m4', 'w16_f4_m4_ddim25']
    by_name = {v['name']: v for v in spec['variants']}
    return [by_name[n] for n in names]


def load_model(root):
    from experimental.noise_combined.model import CombinedNoiseDiT
    from experimental.noise_hvdit.config import load_config
    from rmdm.diffusion import DDIMSampler
    torch.cuda.set_device(0)
    torch.manual_seed(20261012)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = load_config(str(root/'model_config.yaml'))
    model = CombinedNoiseDiT(cfg, 'hybrid_paired', 20, 7.).cuda().eval()
    p = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    assert p['step'] == 7000 and p['weights'] == 'model'
    model.load_state_dict(p['model'], strict=True)
    return model, cfg, DDIMSampler(cfg.diffusion)


def sparse_case(root, entry, sigma, rate):
    data = torch.load(root/entry['input_file'], map_location='cpu', weights_only=True)
    assert data['video_id'] == entry['video_id'] and data['start'] == entry['start']
    b = {k: data[k].cuda().float() for k in ('target', 'building', 'vehicle', 'valid_mask', 'sampling_mask')}
    b['observed_rss'] = b['sampling_mask']*(b['target']+sigma*data['measurement_noise'].cuda())
    b['sampling_rate'] = torch.full(b['target'].shape[:2], float(rate), device='cuda')
    return b, data['initial'].cuda()


def cache_name(variant, q):
    return variant['name'] + '_q' + format(q, '.17g') + '.pt'


def moments(u, members=None, prefix=None):
    samples = u['samples'] if members is None else u['samples'][:members]
    chosen = torch.ones(len(u['flat_indices']), dtype=torch.bool)
    if prefix is not None:
        chosen = u['flat_indices'] < prefix
    samples = samples[:, chosen]
    assert samples.shape[0] >= 2 and samples.shape[1] > 0
    return dict(samples=samples, mean=samples.double().mean(0).float(),
                raw_variance=samples.double().var(0, unbiased=True).float(),
                observed=u['observed'][chosen], flat_indices=u['flat_indices'][chosen],
                folds=u['folds'][chosen])


def predict(root, spec, model, cfg, b, entry, variant, q, cache, *, allow_derive=True):
    path = cache/cache_name(variant, q)
    identity = dict(protocol_version=1, checkpoint_step=7000, weights='model',
                    video_id=entry['video_id'], start=entry['start'], variant=variant,
                    q=q, seed=spec['seed'], rate=spec['rate'])
    if path.exists():
        p = torch.load(path, weights_only=False)
        assert p['identity'] == identity
        return p
    parents = {'w16_f4_m4': 'w16_f4_m8', 'w4_prefix_f4_m4': 'w4_chunks_f4_m4'}
    parent = next((v for v in spec['variants'] if v['name'] == parents.get(variant['name'])), None)
    parent_path = cache/cache_name(parent, q) if parent else None
    if allow_derive and parent_path and parent_path.exists():
        p = torch.load(parent_path, weights_only=False)
        assert p['identity'] == dict(identity, variant=parent)
        prefix = 4*b['target'].shape[-2]*b['target'].shape[-1] if variant['prefix_only'] else None
        tick=time.perf_counter()
        u = moments(p, members=variant['members'], prefix=prefix)
        moment_seconds=time.perf_counter()-tick
        # Actual component timings, not a proportional FLOP approximation.
        meta = p['meta']
        selected = [t for t in meta['batches'] if t['member_start'] < variant['members']
                    and (not variant['prefix_only'] or t['chunk_start'] == 0)]
        seconds = sum(t['seconds'] for t in selected)+meta['shared_setup_seconds']+moment_seconds
        meta = dict(meta, derived_from=str(parent_path), batches=selected,
                    moment_seconds=moment_seconds,peak_memory_scope='parent collection; not standalone variant',
                    timing='sum of measured selected component batches plus shared setup and moments; conservative for prefix')
        u.update(seconds=seconds, meta=meta, peak_gb=p['peak_gb'], new_compute_seconds=0.)
    else:
        inp = public(b, torch.tensor([q], device='cuda'))
        u = collect(model, cfg.diffusion, inp, video_index=entry['video_index'], start=entry['start'],
                    seed=spec['seed'], **{k: variant[k] for k in ('folds', 'members', 'spatial', 'window', 'prefix_only', 'steps')}, member_batch=4)
        tick=time.perf_counter();stats=moments(u);moment_seconds=time.perf_counter()-tick
        u.update(stats);u['seconds']+=moment_seconds
        u['new_compute_seconds']=u['seconds'];u['meta']['moment_seconds']=moment_seconds
    u['identity'] = identity
    save(path, u)
    return u


@torch.inference_mode()
def reconstruction(sampler, model, b, initial, q, path):
    meta = path.with_suffix('.json')
    if meta.exists():
        r = json.loads(meta.read_text()); assert r['q'] == q
        return r
    torch.cuda.synchronize(); tick = time.perf_counter()
    pred = sampler.sample(model, public(b, torch.tensor([q], device='cuda')), initial_noise=initial, steps=50, eta=0.)
    torch.cuda.synchronize(); seconds = time.perf_counter()-tick
    if not torch.isfinite(pred).all(): raise ValueError('Nonfinite reconstruction')
    r = dict(q=q, sigma=q**.5, seconds=seconds, stats=sufficient_stats(pred,b))
    save(path, dict(prediction=pred.cpu(), q=q, steps=50))
    write(meta, r)
    return r


def jobs(spec, phase):
    entries = [e for e in spec['manifest'] if e['role'] == ('calibration' if phase == 'calibration' else 'evaluation')]
    result = []
    for e in entries:
        for si, sigma in enumerate(spec['sigmas']):
            if phase == 'calibration':
                for qi, node in enumerate(spec['calibration_sigmas']):
                    result.append(dict(id=f"v{e['video_index']:05d}_s{si}_q{qi}", entry=e, sigma=sigma, q=node**2))
            else:
                result.append(dict(id=f"v{e['video_index']:05d}_s{si}", entry=e, sigma=sigma))
    return result


@torch.inference_mode()
def worker(root, rank, phase):
    spec = json.loads((root/'protocol.json').read_text())
    model, cfg, sampler = load_model(root)
    queue = jobs(spec, phase); started = time.time(); completed = 0
    out = root/phase; (out/'claims').mkdir(parents=True, exist_ok=True)
    def status(state, **kw):
        value = dict(state=state, rank=rank, phase=phase, pid=os.getpid(), completed=completed,
                     total=len(queue), elapsed=time.time()-started, gpu=torch.cuda.get_device_name(),
                     torch=str(torch.__version__), updated_at=time.time(), **kw)
        write(out/f'worker{rank}_status.json',value)
    status('running')
    for job in queue:
        dest = out/'cases'/job['id']; done = dest/'result.json'
        if done.exists(): continue
        try: (out/'claims'/job['id']).mkdir()
        except FileExistsError: continue
        tick = time.time(); entry = job['entry']; sigma = job['sigma']
        status('running', current_job=job['id'])
        b, initial = sparse_case(root, entry, sigma, spec['rate'])
        if phase == 'calibration':
            rows=[]
            for variant in ordered_variants(spec):
                u = predict(root,spec,model,cfg,b,entry,variant,job['q'],dest/'moments')
                # Clean targets are used only on the independent calibration partition.
                target = b['target'].flatten()[u['flat_indices'].cuda()].cpu()
                save(dest/(variant['name']+'_cal.pt'),dict(raw_variance=u['raw_variance'],
                    squared_clean_error=(target.double()-u['mean'].double()).square(),
                    q=job['q'], true_sigma=sigma, identity=u['identity']))
                rows.append(dict(variant=variant['name'],seconds=u['seconds']))
            write(done,dict(job=job,seconds=time.time()-tick,rows=rows,gpu=torch.cuda.get_device_name()))
        else:
            calibrations=json.loads((root/'calibration.json').read_text())
            baselines={name:reconstruction(sampler,model,b,initial,q,dest/'reconstruction'/f'{name}.pt')
                       for name,q in [('known_sigma',sigma**2),('fixed_003',.03**2),('fixed_zero',0.)]}
            rows=[]
            for variant in ordered_variants(spec):
                name=variant['name']; q=spec['initial_sigma']**2; cumulative=0.; trajectory=[]; consecutive=0
                for r in range(1,spec['max_rounds']+1):
                    u=predict(root,spec,model,cfg,b,entry,variant,q,dest/'moments')
                    t=time.perf_counter()
                    v=calibrated_variance(u['raw_variance'].numpy(),q,calibrations[name])
                    estimate=mle(u['observed'].numpy(),u['mean'].numpy(),v,qmax=spec['qmax'])
                    next_q,stop,residual=update(q,estimate['variance'],damping=1.)
                    update_seconds=time.perf_counter()-t
                    collection_seconds=0. if q in trajectory else u['seconds']
                    cumulative+=collection_seconds+update_seconds
                    consecutive=consecutive+1 if stop else 0
                    rec=reconstruction(sampler,model,b,initial,next_q,dest/'reconstruction'/f'{name}_r{r}.pt')
                    row=dict(variant=name,round=r,input_sigma=q**.5,estimated_sigma=next_q**.5,
                        estimate=estimate,stop_eligible=bool(stop),two_consecutive_stops=consecutive>=2,
                        fixed_point_residual=residual,collection_seconds=collection_seconds,
                        reused_same_condition=q in trajectory,
                        update_seconds=update_seconds,cumulative_estimator_seconds=cumulative,
                        reconstruction=rec,deployment_seconds=cumulative+rec['seconds'],
                        q_history=trajectory+[q],timing_meta=u['meta'])
                    rows.append(row);trajectory.append(q);q=next_q
                    write(dest/'partial.json',dict(job=job,rows=rows,baselines=baselines))
            write(done,dict(job=job,rows=rows,baselines=baselines,seconds=time.time()-tick,gpu=torch.cuda.get_device_name()))
        completed+=1;status('running',last_job=job['id'],last_seconds=time.time()-tick)
        print(json.dumps(dict(phase=phase,rank=rank,completed=completed,job=job['id'],seconds=time.time()-tick)),flush=True)
    status('complete')


def fit(root):
    spec=json.loads((root/'protocol.json').read_text()); result={}
    assert len(list((root/'calibration/cases').glob('*/result.json')))==len(jobs(spec,'calibration'))
    for variant in spec['variants']:
        nodes=[]
        for sigma in spec['calibration_sigmas']:
            arrays=[]
            for p in (root/'calibration/cases').glob('*/'+variant['name']+'_cal.pt'):
                u=torch.load(p,weights_only=False)
                if u['q']==sigma**2:arrays.append(u)
            assert len(arrays)==len([e for e in spec['manifest'] if e['role']=='calibration'])*len(spec['sigmas'])
            raw=np.concatenate([u['raw_variance'].numpy() for u in arrays])
            error=np.concatenate([u['squared_clean_error'].numpy() for u in arrays])
            c=fit_variance_calibration({'2':raw},{'2':error}).to_dict()
            nodes.append(dict(q=sigma**2,**c,points=len(raw),raw_mean=float(raw.mean()),error_mean=float(error.mean())))
        result[variant['name']]=dict(nodes=nodes,fit='clean mean prediction error; nonnegative affine Gaussian NLL',
            calibration_videos=[e['video_id'] for e in spec['manifest'] if e['role']=='calibration'])
    write(root/'calibration.json',result)


@torch.inference_mode()
def smoke(root):
    spec=json.loads((root/'protocol.json').read_text());model,cfg,sampler=load_model(root)
    entry=next(e for e in spec['manifest'] if e['role']=='calibration')
    b,initial=sparse_case(root,entry,.05,spec['rate']); rows=[]
    for variant in ordered_variants(spec):
        small=dict(variant,steps=2)
        u=predict(root,spec,model,cfg,b,entry,small,.03**2,root/'smoke/moments',allow_derive=False)
        assert torch.equal(u['observed'],b['observed_rss'].flatten()[u['flat_indices'].cuda()].cpu())
        rows.append(dict(variant=variant['name'],points=len(u['observed']),seconds=u['seconds'],meta=u['meta']))
    rec=reconstruction(sampler,model,b,initial,.03**2,root/'smoke/reconstruction.pt')
    # One full-DDIM standalone member4 pass for timing and production equivalence.
    v=next(v for v in spec['variants'] if v['name']=='w16_f4_m4')
    u=predict(root,spec,model,cfg,b,entry,v,.03**2,root/'smoke/full50',allow_derive=False)
    # Exercise production cache derivation (including component timings) as well.
    parent=next(v for v in spec['variants'] if v['name']=='w16_f4_m8')
    p=predict(root,spec,model,cfg,b,entry,parent,.03**2,root/'smoke/reuse50',allow_derive=False)
    derived=predict(root,spec,model,cfg,b,entry,v,.03**2,root/'smoke/reuse50')
    difference=float((u['samples']-derived['samples']).abs().max())
    assert difference<=1e-6,difference
    # Verify scalar calibration/update and W16 reconstruction remain compatible.
    target=b['target'].flatten()[u['flat_indices'].cuda()].cpu().double().numpy()
    raw=u['raw_variance'].numpy();error=(target-u['mean'].numpy())**2
    affine=fit_variance_calibration({'2':raw},{'2':error}).to_dict()
    cv=np.maximum(affine['scale']*raw+affine['offset'],affine['floor'])
    estimate=mle(u['observed'].numpy(),u['mean'].numpy(),cv,qmax=spec['qmax'])
    # This smoke uses a calibration video only and never supplies evaluation calibration.
    result=dict(state='passed',gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),variants=rows,
                full50_collection_seconds=u['seconds'],full50_m8_seconds=p['seconds'],
                derived_m4_seconds=derived['seconds'],derived_max_abs_difference=difference,
                smoke_scalar_estimate=estimate,reconstruction_seconds=rec['seconds'],peak_gb=u['peak_gb'])
    write(root/'smoke/result.json',result);print(json.dumps(result),flush=True)


def summarize(root):
    spec=json.loads((root/'protocol.json').read_text());files=sorted((root/'evaluation/cases').glob('*/result.json'))
    assert len(files)==len(jobs(spec,'evaluation'))
    long=[]; baseline=[]
    for p in files:
        d=json.loads(p.read_text());job=d['job'];true=job['sigma']
        for row in d['rows']:
            m=merge_stats([row['reconstruction']['stats']]);err=row['estimated_sigma']-true
            long.append(dict(video_id=job['entry']['video_id'],gpu=d['gpu'],true_sigma=true,variant=row['variant'],round=row['round'],
                estimated_sigma=row['estimated_sigma'],sigma_error=err,absolute_sigma_error=abs(err),
                squared_sigma_error=err**2,variance_error=row['estimated_sigma']**2-true**2,
                estimator_seconds=row['cumulative_estimator_seconds'],deployment_seconds=row['deployment_seconds'],
                stop_eligible=row['stop_eligible'],two_consecutive_stops=row['two_consecutive_stops'],
                boundary=row['estimate']['boundary'],**m))
        for name,row in d['baselines'].items():
            baseline.append(dict(video_id=job['entry']['video_id'],true_sigma=true,variant=name,
                                 seconds=row['seconds'],**merge_stats([row['stats']])))
    out=root/'tables';out.mkdir(exist_ok=True)
    for name,rows in [('iterations',long),('baselines',baseline)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in long:
        groups[(r['variant'],r['round'],'all')].append(r)
        groups[(r['variant'],r['round'],str(r['true_sigma']))].append(r)
    report=[]
    for (variant,iteration,sigma),rows in sorted(groups.items()):
        r=dict(variant=variant,round=iteration,sigma=sigma,cases=len(rows))
        for k in ('absolute_sigma_error','sigma_error','mse','unobserved_mse','psnr','estimator_seconds','deployment_seconds'):
            r[k]=float(np.mean([v[k] for v in rows]))
        r['sigma_rmse']=float(np.sqrt(np.mean([v['squared_sigma_error'] for v in rows])))
        r['boundary_rate']=float(np.mean([v['boundary'] is not None for v in rows]))
        r['stop_eligible_rate']=float(np.mean([v['stop_eligible'] for v in rows]))
        r['two_consecutive_stop_rate']=float(np.mean([v['two_consecutive_stops'] for v in rows]))
        report.append(r)
    write(out/'report.json',dict(state='complete',cases=len(files),rows=report,notes=[
        'Four calibration and eight independent evaluation videos in validation scenes; exploratory pilot.',
        'Rate2 only; all four true sigmas retained; no selection by success.',
        'Adaptive cross-fitting is not exact EM and does not ensure cross-round independence or monotone likelihood.',
        'All three rounds are measured; stop eligibility is diagnostic and is not truth based.',
        'W4 changes local temporal attention kernel from3 to1; original normalized RoPE coordinates retained.',
        'Every estimated sigma is evaluated by original W16 DDIM50 reconstruction.',
        'Deployment time excludes reconstructions at previous diagnostic rounds; derived draws use measured component times.',
        'Sigma search is bounded to the declared training/calibration domain [0,.09]; boundary frequency reported.']))


def launch(root, cloud):
    for phase in ('calibration','evaluation'):
        if phase=='evaluation':fit(root)
        handles=[];processes=[]
        for rank,gpu in enumerate([0,1]):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src:.',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd=[sys.executable,'-u','-m',__package__+'.run','worker','--root',str(root),'--rank',str(rank),'--phase',phase]
            log=(root/f'{phase}_worker{rank}.log').open('a');handles.append(log)
            processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
        write(root/'jobs.json',dict(state='running',phase=phase,pids=[p.pid for p in processes],started_at=time.time()))
        codes=[p.wait() for p in processes]
        for h in handles:h.close()
        if any(codes):
            write(root/'jobs.json',dict(state='failed',phase=phase,exit_codes=codes,finished_at=time.time()))
            raise RuntimeError(f'{phase} workers failed: {codes}; inspect logs before reclaiming jobs')
        spec=json.loads((root/'protocol.json').read_text())
        assert len(list((root/phase/'cases').glob('*/result.json')))==len(jobs(spec,phase))
    # Metric aggregation runs locally after artifact transfer.
    archive=root/'results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['protocol.json','manifest.json','model_config.yaml','calibration.json','jobs.json','smoke',
                     'calibration','evaluation','calibration_worker0.log','calibration_worker1.log',
                     'evaluation_worker0.log','evaluation_worker1.log']:
            if (root/name).exists():tar.add(root/name,arcname=name)
    write(root/'jobs.json',dict(state='exporting',finished_at=time.time()))
    subprocess.run(['aliyunpan','upload',str(archive),cloud],check=True)
    write(root/'export.json',dict(state='uploaded',cloud=cloud+'/results.tar.gz',bytes=archive.stat().st_size,finished_at=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','fit','smoke','summarize','launch'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--phase',choices=['calibration','evaluation']);p.add_argument('--cloud')
    a=p.parse_args();torch.set_num_threads(4);root=a.root.resolve()
    if a.mode=='worker':worker(root,a.rank,a.phase)
    elif a.mode=='launch':
        try:launch(root,a.cloud)
        except Exception as exc:
            previous=json.loads((root/'jobs.json').read_text()) if (root/'jobs.json').exists() else {}
            write(root/'jobs.json',dict(previous,state='failed',error=repr(exc),failed_at=time.time()))
            raise
    else:globals()[a.mode](root)


if __name__=='__main__':main()
