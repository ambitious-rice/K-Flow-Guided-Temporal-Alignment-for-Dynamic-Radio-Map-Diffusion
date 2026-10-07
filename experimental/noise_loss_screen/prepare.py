"""Fixed seed, scene-disjoint small train set and tuning/confirmation validation."""
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from rmdm.data import WindowDataset
from experimental.paper_evaluation.randomness import materialize
from .common import write

def main():
    root=Path('runs/noise_loss_screen_20261007');out=root/'inputs';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Prepared inputs already exist')
    seed=20261007;cache=Path('/dev/shm/noise_temporal_clean16_train_v2')
    manifest=json.loads((cache/'manifest.json').read_text());records=manifest['records']
    splitfile=Path('configs/splits/m20_formal075_clean16_scene_split.json').resolve();split=json.loads(splitfile.read_text())
    arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ('targets','vehicles','buildings')}
    sources=np.load('/dev/shm/noise_hvdit_source_masks.npy',mmap_mode='r');chosen=[]
    for si,scene in enumerate(sorted(split['train'])):
        episodes=sorted({r['episode_id'] for r in records if r['scene_id']==scene})
        rng=np.random.default_rng(seed+100000*si)
        for episode in sorted(rng.choice(episodes,10,replace=False).tolist()):
            candidates=[i for i,r in enumerate(records) if r['scene_id']==scene and r['episode_id']==episode]
            chosen.append(int(rng.choice(candidates)))
    for dest,src in [('targets','targets'),('vehicles','vehicles'),('buildings','buildings'),('sources',None)]:
        parts=[]
        for i in chosen:
            rec=records[i];index=i if dest in ('targets','sources') else rec['episode_index'] if dest=='vehicles' else rec['scene_index']
            parts.append(sources[index] if src is None else arrays[src][index])
        np.save(out/f'train_{dest}.npy',np.stack(parts))
    train=[dict(records[i],packed_index=i) for i in chosen];write(out/'train_videos.json',train)
    data_root=Path(manifest['source_root']);all_samples=sorted(json.loads((data_root/'index.json').read_text())['samples'],key=lambda s:(s['scene_id'],s['episode_id'],s['tx_id']))
    indices={f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}":i for i,s in enumerate(all_samples)}
    old=set(json.loads(Path('runs/paper_test200_20261006/calibration.json').read_text())['calibrators']['mixed']['videos'])
    excluded_episodes={v.rsplit('/',1)[0] for v in old};groups={}
    for si,scene in enumerate(sorted(split['val'])):
        samples=[s for s in all_samples if s['scene_id']==scene and f"{scene}/{s['episode_id']}" not in excluded_episodes]
        episodes=sorted({s['episode_id'] for s in samples});rng=np.random.default_rng(seed+2000000+si*100000)
        for ei,episode in enumerate(rng.choice(episodes,8,replace=False).tolist()):
            candidates=[s for s in samples if s['episode_id']==episode];s=candidates[int(rng.integers(len(candidates)))];video=f"{scene}/{episode}/{s['tx_id']}"
            groups[video]='tune' if ei<4 else 'confirm'
    dataset=WindowDataset(root=str(data_root),split='val',split_file=str(splitfile),window_size=16,include_tx=False,video_ids=sorted(groups),fixed_starts=[0])
    entries=[]
    for dense in DataLoader(dataset,batch_size=1,num_workers=2):
        video=dense['video_id'][0]
        for sigma in [0,.03,.05,.09]:
            sparse,initial=materialize(dense,video_index=indices[video],rate=2,sigma=sigma)
            file=f'val_{len(entries):04d}.pt';torch.save(dict(sparse=sparse,initial=initial),out/file)
            entries.append(dict(file=file,video_id=video,group=groups[video],sigma=sigma,rate=2,start=0))
    write(out/'validation_bank.json',entries)
    write(out/'protocol.json',dict(seed=seed,train_videos=120,train_scenes=sorted(split['train']),validation_groups=groups,
        test_used=False,global_batch=8,max_steps=1500,learning_rate=1e-5,min_learning_rate=1e-6,warmup=50,
        validation_every=500,selection='Minimum tune unobserved MSE at correct sigma (or zero for no-sigma models), DDIM20; includes step0',
        final_evaluation='Frozen selected weights, tune and confirmation groups separate; true sigma x input sigma matrix at DDIM50',
        initial_sigmas=[0,.03,.05,.09],training_rates=[1,2,3],validation_rate=2,
        noise_distribution='20% zero;65% U(0,.05);15% U(.05,.09)',
        initialization='same W16 step016000 model weights; warm-start fine-tuning, not training from scratch',
        source_supervision='Keep existing HVDiT calibration/PDE/obstacle/source losses; no source/Tx labels in model inputs',
        randomness='CPU generators; seed+step*10000; mask +100+batch_index*100+frame; observation noise +9000; diffusion +9001',
        comparisons='k=1,2,3,off each sigma-on/off locally; k=4,6 sigma-on remotely; all other training settings matched'))
    with tarfile.open(root/'inputs.tar.gz','w:gz',compresslevel=1) as archive:archive.add(out,arcname='inputs')
    print(json.dumps(dict(root=str(root),training_videos=len(train),validation_cases=len(entries),archive_bytes=(root/'inputs.tar.gz').stat().st_size)),flush=True)

if __name__=='__main__':main()
