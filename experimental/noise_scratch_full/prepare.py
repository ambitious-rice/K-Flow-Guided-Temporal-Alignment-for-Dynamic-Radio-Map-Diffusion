"""Freeze full training manifest and nested1/2/3% validation observations."""
import json
from pathlib import Path
import torch
from experimental.noise_loss_screen.common import write,save

ROOT=Path('runs/noise_scratch_full_20261008')

def main():
    root=ROOT;out=root/'inputs';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Inputs already frozen')
    packed=Path('/dev/shm/noise_temporal_clean16_train_v2');manifest=json.loads((packed/'manifest.json').read_text())
    split=json.loads(Path('configs/splits/m20_formal075_clean16_scene_split.json').read_text())
    records=[dict(r,packed_index=i) for i,r in enumerate(manifest['records'])]
    assert len(records)==9000 and {r['scene_id'] for r in records}==set(split['train'])
    write(out/'train_videos.json',records);write(out/'packed_manifest.json',manifest)
    old=json.loads(Path('runs/noise_adaln_retrain_20261008/inputs/validation_bank.json').read_text());videos=sorted({e['video_id'] for e in old});entries=[]
    assert all(v.split('/')[0] in split['val'] for v in videos)
    for vi,video in enumerate(videos):
        items=[e for e in old if e['video_id']==video];anchor=items[0];d=torch.load(Path(anchor['base'])/anchor['file'],weights_only=True);b=d['sparse'];seed=20261008+vi*10000
        eps=torch.randn(b['target'].shape,generator=torch.Generator().manual_seed(seed+900))
        initial=torch.randn(b['target'].shape,generator=torch.Generator().manual_seed(seed+901))
        for rate in [1,2,3]:
            mask=torch.zeros_like(b['target'])
            for t in range(16):
                valid=b['valid_mask'][0,t].flatten().nonzero().flatten();g=torch.Generator().manual_seed(seed+100+t)
                order=valid[torch.randperm(len(valid),generator=g)];mask[0,t].view(-1)[order[:max(1,round(len(valid)*rate/100))]]=1
            for e in items:
                sigma=e['sigma'];sparse=dict(b,sampling_mask=mask,observed_rss=mask*(b['target']+sigma*eps),sampling_rate=torch.full_like(b['sampling_rate'],rate),measurement_variance=torch.tensor([sigma**2]),measurement_standard_deviation=torch.tensor([sigma]))
                file=f'video{vi:02d}_rate{rate}_sigma{sigma:g}.pt';save(out/file,dict(sparse=sparse,initial=initial))
                entries.append(dict(e,file=file,base=str(out.resolve()),bank='scratch_fixed',rate=rate))
    assert len(entries)==360
    write(out/'validation_bank.json',entries)
    write(out/'protocol.json',dict(seed=20261008,packed_cache=str(packed),training_videos=9000,train_scenes=split['train'],validation_videos=len(videos),validation_cases=len(entries),rates=[1,2,3],test_used=False,training_rates=list(range(1,11)),training_sigma='Uniform standard deviation[0,0.09), no clean component',source_masks='/dev/shm/noise_hvdit_source_masks.npy',training_rng='FullData: seed+step*100000+micro*10000+rank*1000; IDs+0,start+1,rate+2,sigma+3,mask+100+index*32+frame,observation+700,diffusion+701,time+702; uniform full9000 video/window draws with replacement',validation_rng='sorted video IDs; seed20261008+video_index*10000; mask+100+frame,measurement+900,diffusion+901',validation_pairing='Nested rate masks; same measurement Gaussian field and initial diffusion noise for all rates and sigmas per video; clean target only for supervision/evaluation',source_root=manifest['source_root']))
    print('Prepared9000 train videos and360 validation cases',flush=True)

if __name__=='__main__':main()
