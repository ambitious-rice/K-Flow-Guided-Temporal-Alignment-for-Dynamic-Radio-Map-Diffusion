import json

import pytest

from .run import SIGMAS, summarize, video_ids


def test_video_exclusions_ignore_nonvideo_paths():
    value={'videos':['scene/episode_1/tx_1','/some/root/scene/episode_1/tx_1'],
           'nested':[{'video_id':'scene/episode_2/tx_2'}]}
    assert set(video_ids(value,['scene']))=={'scene/episode_1/tx_1','scene/episode_2/tx_2'}


def test_confirmation_pools_windows_and_preserves_video_counts(tmp_path):
    videos=['scene/episode_1/tx_1','scene/episode_2/tx_1']
    spec=dict(videos=videos,starts=[0,48],rates=[1,2,3],expected_cases=48,
              checkpoint_step=4400,weights='model',fixed_sigma_reference=.05,evaluation_scope='fixture')
    (tmp_path/'protocol.json').write_text(json.dumps(spec))
    dest=tmp_path/'results/worker0/cases';dest.mkdir(parents=True)
    for vi,video in enumerate(videos):
        for start,count in [(0,10),(48,30)]:
            for rate in spec['rates']:
                for i,sigma in enumerate(SIGMAS):
                    rows=[]
                    for j,inp in enumerate(SIGMAS):
                        mse=(1 if i==j else 2)+(0 if start==0 else 1)
                        stats=dict(sse=mse*count,sae=count,count=count,unseen_sse=mse*count,
                            unseen_count=count,observed_sse=mse*count,observed_count=count,
                            frame_psnr_sum=0.,frames=16,temporal_sse=mse*count,temporal_count=count)
                        rows.append(dict(video_id=video,start=start,rate=rate,sigma=sigma,input_sigma=inp,stats=stats))
                    (dest/f'{vi}_{start}_{rate}_{i}.json').write_text(json.dumps(dict(rows=rows,steps=50)))
    summarize(tmp_path)
    result=json.loads((tmp_path/'tables/report.json').read_text())
    overall=result['combined_windows']
    assert overall['all12_videos']==2
    assert overall['combined']['mean_matrix'][0][0]==pytest.approx(1.75)
    assert overall['combined']['all_four_videos']==2
    assert result['by_window']['0']['combined']['mean_matrix'][0][0]==1.
    assert overall['combined']['fixed_sigma_gain_percent']==pytest.approx(30.)
    assert result['correct_sigma_metrics']['unobserved_mse']==pytest.approx(1.75)
