"""Freeze 200 videos and materialize common observations before model testing."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmdm.data import WindowDataset
from .randomness import MASTER_SEED, materialize, balanced_folds


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def select_videos(samples,scenes,*,seed=MASTER_SEED,per_scene=100):
    full=sorted(samples,key=lambda s:(s['scene_id'],s['episode_id'],s['tx_id']))
    indices={f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}":i for i,s in enumerate(full)}
    selected=[]
    for scene_number,scene in enumerate(sorted(scenes)):
        candidates=[s for s in full if s['scene_id']==scene]
        episodes=sorted({s['episode_id'] for s in candidates})
        rng=np.random.Generator(np.random.PCG64(seed+100_000*scene_number))
        chosen=sorted(rng.choice(len(episodes),size=per_scene,replace=False).tolist())
        for episode_number in chosen:
            episode=episodes[episode_number]
            transmitters=[s for s in candidates if s['episode_id']==episode]
            tx_seed=seed+100_000*scene_number+1_000+episode_number
            tx_rng=np.random.Generator(np.random.PCG64(tx_seed))
            item=transmitters[int(tx_rng.integers(len(transmitters)))]
            video=f"{scene}/{episode}/{item['tx_id']}"
            selected.append(dict(scene_id=scene,episode_id=episode,tx_id=item['tx_id'],video_id=video,
                full_data_video_index=indices[video],frame_count=100,selection_seed=seed+100_000*scene_number,
                tx_selection_seed=tx_seed))
    return sorted(selected,key=lambda s:s['video_id'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--split-file',default='configs/splits/m20_formal075_clean16_scene_split.json')
    p.add_argument('--seed',type=int,default=MASTER_SEED)
    p.add_argument('--manifest-only',action='store_true')
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    index=json.loads((Path(args.root)/'index.json').read_text())
    split=json.loads(Path(args.split_file).read_text())
    selected=select_videos(index['samples'],split['test'],seed=args.seed)
    manifest=dict(schema='paper-test200-episode-stratified-v1',seed=args.seed,
        source_index=str(Path(args.root).resolve()/'index.json'),split_file=str(Path(args.split_file).resolve()),
        selection='Per test scene: uniform 100/150 episodes without replacement; one uniformly selected transmitter per episode',
        stage_a=dict(videos=selected,per_scene_video_count=100),starts=[0,48],window_size=16,
        rates=[1,2,3],sigmas=[0,.03,.05,.09],bootstrap_cluster='scene + episode',
        diagnostic_videos_per_scene=10,diagnostic_selection='First 10 selected episodes per scene in an independent seeded permutation')
    diagnostic=[]
    for i,scene in enumerate(sorted(split['test'])):
        candidates=[v['video_id'] for v in selected if v['scene_id']==scene]
        rng=np.random.Generator(np.random.PCG64(args.seed+300_000+i))
        diagnostic.extend([candidates[j] for j in rng.permutation(len(candidates))[:10]])
    manifest['diagnostic_videos']=sorted(diagnostic)
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Existing sampling manifest differs; use a new output directory')
    write_json(manifest_path,manifest)
    with (out/'videos.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    spec=dict(version='paper-paired-rss-v1',seed=args.seed,split='test',window_size=16,
        starts=[0,48],rates=[1,2,3],sigmas=[0,.03,.05,.09],videos=[v['video_id'] for v in selected],
        manifest=str(manifest_path.resolve()),split_file=str(Path(args.split_file).resolve()),smoke=False,
        torch_version=str(torch.__version__),numpy_version=str(np.__version__),
        seed_rule='seed + full_data_video_index * 1000000 + absolute_frame * 100 + stream',
        streams=dict(mask=0,observation_noise=1,final_ddim_initial=2,fold_assignment=3,ensemble='10 + fold * 8 + member (fold=0..3, member=0..7)',radiodiff_latent=50),
        noise='CPU float32 independent Gaussian; Y=M*(X+sigma*epsilon), no clipping',
        mask='uniform exact free-space budget; rate masks nested; shared across sigma',
        full_data_video_indices={v['video_id']:v['full_data_video_index'] for v in selected})
    write_json(out/'protocol.json',spec)
    if args.manifest_only:
        print(json.dumps(dict(videos=len(selected),windows=400,cases=4800,manifest=str(manifest_path))),flush=True)
        return
    dataset=WindowDataset(root=args.root,split='test',split_file=str(Path(args.split_file).resolve()),window_size=16,
        include_tx=False,video_ids=spec['videos'],fixed_starts=[0,48],seed=args.seed)
    if len(dataset)!=400:
        raise ValueError('Expected 400 W16 windows')
    bank_dir=out/'bank';bank_dir.mkdir(exist_ok=True)
    folds_dir=bank_dir/'folds';folds_dir.mkdir(exist_ok=True)
    entries=[];counts=[]
    started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    for window_index,dense in enumerate(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)):
        video=dense['video_id'][0];start=int(dense['start'][0]);video_index=spec['full_data_video_indices'][video]
        for rate in spec['rates']:
            fold_file=f'folds/window{window_index:04d}_rate{rate}.pt'
            for sigma in spec['sigmas']:
                file=f'{len(entries):06d}.pt'
                sparse,initial=materialize(dense,video_index=video_index,rate=rate,sigma=sigma,seed=args.seed)
                if sigma==0:
                    assignment=balanced_folds(sparse['sampling_mask'],video_index=video_index,start=start,seed=args.seed)
                    torch.save(assignment,bank_dir/fold_file)
                    for t in range(16):
                        counts.append(dict(video_id=video,frame=start+t,rate=rate,
                            valid_pixels=int(sparse['valid_mask'][0,t].sum()),observed_pixels=int(sparse['sampling_mask'][0,t].sum())))
                path=bank_dir/file
                entry=dict(file=file,video_id=video,start=start,rate=rate,sigma=sigma,
                    full_data_video_index=video_index,fold_file=fold_file)
                # CPU-only deterministic generation; write atomically. Reruns
                # compare existing tensors and keep them instead of overwriting.
                if path.exists():
                    saved=torch.load(path,weights_only=True)
                    if not torch.equal(saved['initial'],initial) or any(not torch.equal(saved['sparse'][k],v)
                        for k,v in sparse.items() if torch.is_tensor(v)):
                        raise ValueError(f'Existing bank entry differs: {file}')
                else:
                    temporary=path.with_suffix('.tmp');torch.save(dict(sparse=sparse,initial=initial),temporary);temporary.replace(path)
                entries.append(entry)
        status=dict(state='preparing',started_at=started,updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            windows=window_index+1,total_windows=400,entries=len(entries),total_entries=4800)
        write_json(out/'preparation_status.json',status)
        if (window_index+1)%10==0:
            print(json.dumps(status),flush=True)
    write_json(bank_dir/'bank.json',dict(spec=spec,entries=entries))
    with (out/'observation_counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    status.update(state='complete',finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    write_json(out/'preparation_status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    main()
