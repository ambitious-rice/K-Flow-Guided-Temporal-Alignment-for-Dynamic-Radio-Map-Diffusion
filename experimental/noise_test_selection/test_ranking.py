import numpy as np
import pytest
from .ranking import score_video, ranked
from experimental.noise_video_screen.statistics import merge_stats


def matrices(diag, off):
    a=np.full((3,4,4),float(off));a[:,np.arange(4),np.arange(4)]=diag
    return a


def test_signed_gain_and_no_all_cells_gate():
    a=matrices(.8,1.);a[0,0,0]=1.2
    score=score_video(a)
    assert score['correct_minimum_cells']==11
    assert score['score']==pytest.approx((11*.2-.2)/12)
    b=score_video(matrices(.9,1.))
    assert ranked([dict(video_id='b',**b),dict(video_id='a',**score)])[0]['video_id']=='a'


def test_scale_invariance_and_tie_break():
    a=matrices(1.,2.)
    assert score_video(a)['score']==score_video(a*100)['score']
    items=[dict(video_id=v,**score_video(a)) for v in ['z','a']]
    assert [x['video_id'] for x in ranked(items)]==['a','z']


def test_window_pooling_uses_counts():
    def st(sse,n):
        return dict(sse=sse,sae=1,count=n,unseen_sse=sse,unseen_count=n,observed_sse=0,observed_count=1,
                    frame_psnr_sum=1,frames=1,temporal_sse=0,temporal_count=1)
    result=merge_stats([st(2,2),st(8,4)])
    assert result['unobserved_mse']==pytest.approx(10/6)
    with pytest.raises(ValueError):score_video(np.full((3,4,4),np.nan))


def test_summary_preserves_candidate_indices_and_reuse(tmp_path):
    import json
    from .run import summarize,case_name
    from experimental.noise_loss_screen.common import write
    videos=['scene/episode0/tx0','scene/episode1/tx0']
    spec=dict(protocol_id='unit',videos=videos,starts=[0,84],rates=[1,2,3],sigmas=[0.,.03,.05,.09],
              expected_cases=48,expected_merged_rows=96,selected_count=1,seed=42,ranking='mean signed relative gain')
    write(tmp_path/'protocol.json',spec)
    for vi,video in enumerate(videos):
        for start in spec['starts']:
            for rate in spec['rates']:
                for sigma in spec['sigmas']:
                    rows=[]
                    for j,inp in enumerate(spec['sigmas']):
                        mse=(.8 if vi==1 else .9) if sigma==inp else 1.
                        n=2 if start==0 else 3
                        st=dict(sse=mse*n,sae=n,count=n,unseen_sse=mse*n,unseen_count=n,
                                observed_sse=0,observed_count=1,frame_psnr_sum=1,frames=1,temporal_sse=0,temporal_count=1)
                        rows.append(dict(video_id=video,video_index=vi,start=start,rate=rate,sigma=sigma,input_sigma=inp,prediction_index=j,stats=st))
                    stem=case_name(vi,start,rate,sigma)
                    write(tmp_path/'results/worker0/cases'/f'{stem}.json',dict(protocol_id='unit',steps=50,rows=rows,prediction_file=f'results/worker0/predictions/{stem}.pt'))
    summarize(tmp_path)
    split=json.loads((tmp_path/'selected200_split.json').read_text())
    assert split['videos']==[videos[1]] and split['video_indices'][videos[1]]==1
    assert len((tmp_path/'tables/selected200_reuse_index.csv').read_text().splitlines())==97
