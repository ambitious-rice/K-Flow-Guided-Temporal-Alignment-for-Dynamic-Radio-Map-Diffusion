import json
from pathlib import Path
import numpy as np
from experimental.noise_loss_screen.common import write

ROOT=Path('runs/noise_feature_retrain_20261008')

def main():
    out=ROOT/'inputs';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Inputs already frozen')
    cache=Path('/dev/shm/noise_temporal_clean16_train_v2');records=json.loads((cache/'manifest.json').read_text())['records'];split=json.loads(Path('configs/splits/m20_formal075_clean16_scene_split.json').read_text());chosen=[]
    for si,scene in enumerate(sorted(split['train'])):
        episodes=sorted({r['episode_id'] for r in records if r['scene_id']==scene});rng=np.random.default_rng(20261008+si*100000)
        for episode in sorted(rng.choice(episodes,50,replace=False).tolist()):
            ids=[i for i,r in enumerate(records) if r['scene_id']==scene and r['episode_id']==episode];i=int(rng.choice(ids));chosen.append(dict(records[i],packed_index=i))
    write(out/'train_videos.json',chosen)
    old=Path('runs/noise_loss_screen_20261007/inputs');entries=[dict(e,base=str(old.resolve()),bank='original') for e in json.loads((old/'validation_bank.json').read_text())]
    fresh=Path('runs/noise_condition_diagnosis_20261007/fresh_inputs');entries += [dict(e,base=str(fresh.resolve()),group='confirm',bank='fresh') for e in json.loads((fresh/'bank.json').read_text())]
    write(out/'validation_bank.json',entries)
    write(out/'protocol.json',dict(seed=20261008,packed_cache=str(cache),training_videos=len(chosen),train_scenes=sorted(split['train']),selection='50 distinct episodes/scene, one Tx each; integer RNG seeds; one saved list shared by all groups',rates=[1,2,3],noise='pairedquartets:0,U(0,.03),U(.03,.06),U(.06,.09),same mask/Gaussian field/diffusion state within quartet',test_used=False,tune='8 videos/32 cases',confirmation='16 distinct videos/88 cases including intermediate noise levels; reused exploratory validation only',rng='seed+step*100000+micro*10000+rank*1000; masks+100+video*100+frame; obs+700; diffusion+701; time+702'))
    print('prepared',len(chosen),'train videos',len(entries),'validation cases')
if __name__=='__main__':main()
