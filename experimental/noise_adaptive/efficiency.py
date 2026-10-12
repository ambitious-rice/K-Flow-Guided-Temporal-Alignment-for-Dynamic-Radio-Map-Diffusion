"""Same-GPU W4 efficiency pilot; no intermediate deployment reconstructions.

Input bank and frozen model are inherited unchanged from the original pilot.
Only independent calibration cases access clean labels before final scoring.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from experimental.noise_loss_screen.common import public, save, write
from experimental.noise_video_screen.statistics import sufficient_stats, merge_stats
from experimental.paper_evaluation.calibration import fit_variance_calibration
from .efficient import collect_w4, pool_update
from .predict import collect
from .run import load_model, sparse_case, jobs
from .statistics import calibrated_variance, mle


def variants():
    def item(name, steps, folds=(0, 1, 2, 3), members=4, offsets=(0,)):
        return dict(name=name, steps=steps, selected_folds=list(folds), members=members,
                    offsets=list(offsets), fold_batch=4)
    return [item('full_d50', 50), item('full_d20', 20), item('full_d10', 10),
            item('holdout25_d50', 50, (0,)), item('holdout25_d20', 20, (0,)),
            item('holdout25_d10', 10, (0,)), item('holdout25_m2_d20', 20, (0,), 2),
            item('rotate_full_d20', 20, offsets=(0, 4)),
            item('rotate_holdout25_d20', 20, (0,), offsets=(0, 4))]


def calibration_key(variant, offset):
    return f"d{variant['steps']}_m{variant['members']}_f{''.join(map(str,variant['selected_folds']))}_t{offset}"


def configurations():
    return {calibration_key(v, o): (v, o) for v in variants() for o in v['offsets']}


def collect_case(model, cfg, b, entry, spec, v, offset, q):
    return collect_w4(model, cfg.diffusion, public(b, torch.tensor([q], device='cuda')),
        video_index=entry['video_index'], start=entry['start'], seed=spec['seed'], offset=offset,
        **{k: v[k] for k in ('steps', 'selected_folds', 'members', 'fold_batch')})


def environment():
    return dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                cuda=torch.version.cuda, device=str(torch.cuda.get_device_properties(0)))


@torch.inference_mode()
def reconstruct(model, sampler, b, initial, q):
    torch.cuda.synchronize(); tick = time.perf_counter()
    pred = sampler.sample(model, public(b, torch.tensor([q], device='cuda')),
                          initial_noise=initial, steps=50, eta=0.)
    torch.cuda.synchronize(); seconds = time.perf_counter()-tick
    if not torch.isfinite(pred).all():
        raise ValueError('Nonfinite W16 reconstruction')
    return dict(q=q, sigma=q**.5, seconds=seconds, stats=sufficient_stats(pred, b))


@torch.inference_mode()
def smoke(parent, out):
    spec = json.loads((parent/'protocol.json').read_text())
    model, cfg, sampler = load_model(parent)
    entry = next(e for e in spec['manifest'] if e['role'] == 'calibration')
    b, initial = sparse_case(parent, entry, .03, spec['rate'])
    inp = public(b, torch.tensor([.0009], device='cuda'))
    original = collect(model, cfg.diffusion, inp, video_index=entry['video_index'],
        start=entry['start'], seed=spec['seed'], window=4, prefix_only=True, members=4, steps=2)
    checks = []
    for batch in (1, 2, 4):
        v = dict(variants()[0], steps=2, fold_batch=batch)
        u = collect_case(model, cfg, b, entry, spec, v, 0, .0009)
        assert torch.equal(u['flat_indices'], original['flat_indices'])
        error = float((u['samples']-original['samples']).abs().max())
        if error > 1e-4:
            raise ValueError(f'Batch {batch} numerical mismatch {error}')
        checks.append(dict(fold_batch=batch, max_abs_error=error, peak_gb=u['peak_gb']))
    later = collect_case(model, cfg, b, entry, spec, dict(variants()[6], steps=2), 4, .0009)
    assert len(set(later['flat_indices'].tolist()) & set(original['flat_indices'].tolist())) == 0
    # Full W16 end-to-end smoke; few steps only, not a timing observation.
    pred = sampler.sample(model, inp, initial_noise=initial, steps=2, eta=0.)
    assert torch.isfinite(pred).all() and pred.shape == b['target'].shape
    write(out/'smoke.json', dict(state='passed', checks=checks, environment=environment(),
                               reconstruction_stats=sufficient_stats(pred, b)))
    print('smoke passed', flush=True)


@torch.inference_mode()
def benchmark(parent, out, repeats=3):
    spec = json.loads((parent/'protocol.json').read_text())
    calibration = json.loads((parent/'calibration.json').read_text())['w4_prefix_f4_m4']
    model, cfg, sampler = load_model(parent)
    # Fixed in advance: first calibration video in each scene, sigma .03 and .09.
    entries = []; scenes = set()
    for entry in spec['manifest']:
        if entry['role'] == 'calibration' and entry['scene_id'] not in scenes:
            entries.append(entry); scenes.add(entry['scene_id'])
    rows = []
    for entry in entries:
        for sigma in (.03, .09):
            b, initial = sparse_case(parent, entry, sigma, spec['rate'])
            inp = public(b, torch.tensor([.0009], device='cuda'))
            kw = dict(video_index=entry['video_index'], start=entry['start'], seed=spec['seed'])
            reference = collect(model, cfg.diffusion, inp, window=4, prefix_only=True,
                                members=4, steps=50, **kw)
            def estimate(u):
                mu=u['samples'].double().mean(0).float().numpy()
                raw=u['samples'].double().var(0,unbiased=True).float().numpy()
                return mle(u['observed'].numpy(), mu, calibrated_variance(raw,.0009,calibration))['sigma']
            reference_sigma=estimate(reference)
            for batch in (1, 2, 4):
                collect_w4(model,cfg.diffusion,inp,fold_batch=batch,**kw) # full warmup per shape
            reconstruct(model,sampler,b,initial,.0009) # full W16 warmup
            for repeat in range(repeats):
                # Counterbalance ordering, always measure actual standalone executions.
                order = (1,2,4) if repeat%2 == 0 else (4,2,1)
                for batch in order:
                    u=collect_w4(model,cfg.diffusion,inp,fold_batch=batch,**kw)
                    assert torch.equal(u['flat_indices'],reference['flat_indices'])
                    diff=float((u['samples']-reference['samples']).abs().max())
                    sigma_diff=abs(estimate(u)-reference_sigma)
                    if diff>1e-4 or sigma_diff>1e-5:
                        raise ValueError(f'Parallel equivalence failed: batch={batch}, max={diff}, sigma={sigma_diff}')
                    rows.append(dict(video_index=entry['video_index'],true_sigma=sigma,repeat=repeat,
                        batch=batch*4,seconds=u['seconds'],peak_gb=u['peak_gb'],
                        max_abs_prediction_difference=diff,absolute_sigma_difference=sigma_diff))
                rows.append(dict(video_index=entry['video_index'],true_sigma=sigma,repeat=repeat,
                    batch='W16',seconds=reconstruct(model,sampler,b,initial,.0009)['seconds']))
            write(out/'benchmark.json',dict(rows=rows,environment=environment(),repeats=repeats,
                tolerance=dict(prediction_atol=1e-4,sigma_atol=1e-5),state='running'))
            print(f'benchmark {entry["video_index"]} sigma={sigma}',flush=True)
    groups={str(batch):dict(median_seconds=float(np.median([r['seconds'] for r in rows if r['batch']==batch])),
                           min_seconds=min(r['seconds'] for r in rows if r['batch']==batch),
                           max_seconds=max(r['seconds'] for r in rows if r['batch']==batch)) for batch in (4,8,16,'W16')}
    write(out/'benchmark.json',dict(rows=rows,groups=groups,environment=environment(),repeats=repeats,
          tolerance=dict(prediction_atol=1e-4,sigma_atol=1e-5),state='complete'))


@torch.inference_mode()
def calibrate(parent,out,model,cfg,spec):
    configs=configurations()
    for job in jobs(spec,'calibration'):
        dest=out/'calibration'/job['id']
        if (dest/'complete.json').exists():
            continue
        b,_=sparse_case(parent,job['entry'],job['sigma'],spec['rate'])
        for key,(v,offset) in configs.items():
            path=dest/(key+'.pt')
            if path.exists():
                continue
            u=collect_case(model,cfg,b,job['entry'],spec,v,offset,job['q'])
            target=b['target'].flatten()[u['flat_indices'].cuda()].cpu()
            save(path,dict(raw_variance=u['raw_variance'],
                squared_clean_error=(target.double()-u['mean'].double()).square(),
                q=job['q'],true_sigma=job['sigma'],meta=u['meta'],video_id=job['entry']['video_id'],
                flat_indices=u['flat_indices']))
        write(dest/'complete.json',dict(job=job,state='complete'))
        print(f'calibration {job["id"]}',flush=True)
    result={}
    for key in configs:
        nodes=[]
        for sigma in spec['calibration_sigmas']:
            arrays=[torch.load(out/'calibration'/j['id']/(key+'.pt'),weights_only=False)
                    for j in jobs(spec,'calibration') if j['q']==sigma**2]
            raw=np.concatenate([a['raw_variance'].numpy() for a in arrays])
            error=np.concatenate([a['squared_clean_error'].numpy() for a in arrays])
            fitted=fit_variance_calibration({'2':raw},{'2':error}).to_dict()
            nodes.append(dict(q=sigma**2,**fitted,points=len(raw)))
        result[key]=dict(nodes=nodes,source_key=key,
            calibration_videos=[e['video_id'] for e in spec['manifest'] if e['role']=='calibration'])
    write(out/'calibration.json',result)
    return result


@torch.inference_mode()
def evaluate(parent,out,model,cfg,sampler,spec,calibrations):
    for job in jobs(spec,'evaluation'):
        dest=out/'evaluation'/job['id']
        if (dest/'result.json').exists():
            continue
        b,initial=sparse_case(parent,job['entry'],job['sigma'],spec['rate'])
        # Warm each denoiser shape and W16; all timing is same-device and synchronized.
        for folds,members in (((0,1,2,3),4),((0,),4),((0,),2)):
            v=dict(variants()[0],selected_folds=folds,members=members,steps=2)
            collect_case(model,cfg,b,job['entry'],spec,v,0,.0009)
        reconstruct(model,sampler,b,initial,.0009)
        baselines={name:reconstruct(model,sampler,b,initial,q) for name,q in
                   [('known_sigma',job['sigma']**2),('fixed_003',.0009),('fixed_zero',0.)]}
        rows=[]
        # Counterbalance variant order over cases; no prediction cache reuse in timing.
        ordered=variants() if len(list((out/'evaluation').glob('*/result.json')))%2==0 else variants()[::-1]
        for v in ordered:
            q=spec['initial_sigma']**2; blocks=[]; trajectory=[]
            torch.cuda.synchronize(); tick=time.perf_counter()
            for offset in v['offsets']:
                u=collect_case(model,cfg,b,job['entry'],spec,v,offset,q)
                key=calibration_key(v,offset)
                blocks.append(dict(q=q,moments=u,calibration=calibrations[key]))
                estimate=pool_update(blocks,qmax=spec['qmax'])
                trajectory.append(dict(offset=offset,condition_q=q,estimated_sigma=estimate['sigma'],
                    calibration_key=key,points=len(u['flat_indices']),collection_seconds=u['seconds'],
                    flat_indices=u['flat_indices'].tolist(),peak_gb=u['peak_gb']))
                q=estimate['variance']
            torch.cuda.synchronize(); estimator_seconds=time.perf_counter()-tick
            rec=reconstruct(model,sampler,b,initial,q)
            rows.append(dict(variant=v['name'],estimated_sigma=q**.5,estimate=estimate,
                estimator_seconds=estimator_seconds,reconstruction=rec,
                deployment_seconds=estimator_seconds+rec['seconds'],trajectory=trajectory,
                peak_gb=max(t['peak_gb'] for t in trajectory)))
            write(dest/'partial.json',dict(job=job,rows=rows,baselines=baselines))
        write(dest/'result.json',dict(job=job,rows=rows,baselines=baselines,environment=environment()))
        print(f'evaluation {job["id"]}',flush=True)


def summarize(out):
    files=list((out/'evaluation').glob('*/result.json'))
    groups=defaultdict(list)
    for path in files:
        case=json.loads(path.read_text())
        for row in case['rows']:
            for sigma in ('all',str(case['job']['sigma'])):
                groups[(row['variant'],sigma)].append((row,case))
    summary=[]
    for (variant,sigma),pairs in sorted(groups.items()):
        rows=[r for r,c in pairs]; errors=[r['estimated_sigma']-c['job']['sigma'] for r,c in pairs]
        rec=merge_stats([r['reconstruction']['stats'] for r in rows])
        known=merge_stats([c['baselines']['known_sigma']['stats'] for r,c in pairs])
        fixed=merge_stats([c['baselines']['fixed_003']['stats'] for r,c in pairs])
        summary.append(dict(variant=variant,true_sigma=sigma,cases=len(rows),
            videos=len({c['job']['entry']['video_id'] for r,c in pairs}),
            sigma_mae=float(np.mean(np.abs(errors))),sigma_rmse=float(np.sqrt(np.mean(np.square(errors)))),
            sigma_bias=float(np.mean(errors)),estimated_sigma_mean=float(np.mean([r['estimated_sigma'] for r in rows])),
            lower_hits=sum(r['estimate']['boundary']=='lower' for r in rows),
            upper_hits=sum(r['estimate']['boundary']=='upper' for r in rows),
            estimator_seconds=float(np.mean([r['estimator_seconds'] for r in rows])),
            final_w16_seconds=float(np.mean([r['reconstruction']['seconds'] for r in rows])),
            deployment_seconds=float(np.mean([r['deployment_seconds'] for r in rows])),
            reconstruction_fraction=float(sum(r['reconstruction']['seconds'] for r in rows)/sum(r['deployment_seconds'] for r in rows)),
            peak_gb=max(r['peak_gb'] for r in rows),reconstruction=rec,
            mse_change_vs_known_percent=100*(rec['mse']/known['mse']-1),
            mse_change_vs_fixed003_percent=100*(rec['mse']/fixed['mse']-1)))
    report=dict(completed_cases=len(files),expected_cases=32,summary=summary,
        caveats=['8 independent evaluation videos, existing scenes, rate2%, 7000 model weights.',
                 'Sigma search [0,.09]; endpoint accuracy benefits from bounds.',
                 'Rotating blocks calibrated at their generation q; no cross-round independence claim.',
                 'Calibration and I/O excluded from deployment latency; only one final W16 reconstruction.'])
    write(out/'report.json',report)
    import csv
    if summary:
        with (out/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['smoke','benchmark','run','summarize'])
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='smoke':
        smoke(args.parent,args.out)
    elif args.mode=='benchmark':
        benchmark(args.parent,args.out)
    elif args.mode=='summarize':
        summarize(args.out)
    else:
        spec=json.loads((args.parent/'protocol.json').read_text())
        protocol=dict(schema='w4-efficiency-v1',parent=str(args.parent),variants=variants(),
            seed=spec['seed'],manifest=spec['manifest'],sigmas=spec['sigmas'],rate=spec['rate'],
            checkpoint_step=7000,weights='model',calibration_q_nodes=spec['calibration_q_nodes'])
        path=args.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:
            raise ValueError('Existing efficiency protocol differs')
        write(path,protocol)
        model,cfg,sampler=load_model(args.parent)
        calibrations=calibrate(args.parent,args.out,model,cfg,spec)
        evaluate(args.parent,args.out,model,cfg,sampler,spec,calibrations)
        report=summarize(args.out)
        assert report['completed_cases']==report['expected_cases']
        write(args.out/'complete.json',dict(state='complete',environment=environment(),finished_at=time.time()))


if __name__=='__main__':
    main()
