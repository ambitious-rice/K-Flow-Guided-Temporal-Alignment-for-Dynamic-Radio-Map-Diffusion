"""Reuse the preselected validation videos with the paper integer RNG protocol."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from rmdm.data import WindowDataset
from .randomness import materialize,balanced_folds
from .prepare import write_json

ROOT=Path('runs/paper_test200_20261006')
def main():
    spec=json.loads((ROOT/'protocol.json').read_text())
    old=json.loads(Path('runs/noise_estimation_crossfit_20261006/calibration_bank/val_seed20261006/bank.json').read_text())
    videos=sorted({e['video_id'] for e in old['entries']})
    data_root=Path(json.loads((ROOT/'manifest.json').read_text())['source_index']).parent
    samples=sorted(json.loads((data_root/'index.json').read_text())['samples'],key=lambda s:(s['scene_id'],s['episode_id'],s['tx_id']))
    indices={f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}":i for i,s in enumerate(samples)}
    dataset=WindowDataset(root=str(data_root),split='val',split_file=spec['split_file'],window_size=16,include_tx=False,video_ids=videos,fixed_starts=[0],seed=spec['seed'])
    out=ROOT/'calibration_bank';(out/'folds').mkdir(parents=True,exist_ok=True)
    entries=[]
    for wi,dense in enumerate(DataLoader(dataset,batch_size=1,num_workers=2)):
        video=dense['video_id'][0];vi=indices[video];start=int(dense['start'][0])
        for rate in [1,2,3]:
            ff=f'folds/window{wi:04d}_rate{rate}.pt'
            for sigma in [0,.03,.09]:
                sparse,initial=materialize(dense,video_index=vi,rate=rate,sigma=sigma)
                if sigma==0: torch.save(balanced_folds(sparse['sampling_mask'],video_index=vi,start=start),out/ff)
                file=f'{len(entries):06d}.pt';torch.save(dict(sparse=sparse,initial=initial),out/file)
                entries.append(dict(file=file,video_id=video,start=start,rate=rate,sigma=sigma,full_data_video_index=vi,fold_file=ff))
    spec.update(split='val',videos=videos,starts=[0],sigmas=[0,.03,.09],full_data_video_indices={v:indices[v] for v in videos})
    write_json(out/'bank.json',dict(spec=spec,entries=entries))
    print(f'Prepared {len(entries)} calibration cases',flush=True)
if __name__=='__main__': main()
