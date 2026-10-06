import torch

from .prepare import select_videos
from .randomness import materialize, balanced_folds, ensemble_initial, frame_seed


def test_episode_stratification_is_reproducible_and_order_independent():
    samples=[dict(scene_id=s,episode_id=f'episode_{e:06d}',tx_id=f'tx_{t:02d}')
             for s in ['scene_a','scene_b'] for e in range(150) for t in range(5)]
    a=select_videos(samples,['scene_a','scene_b'])
    b=select_videos(list(reversed(samples)),['scene_b','scene_a'])
    assert a==b and len(a)==200
    assert len({(x['scene_id'],x['episode_id']) for x in a})==200
    for s in ['scene_a','scene_b']:
        assert sum(v['scene_id']==s for v in a)==100


def dense():
    x=torch.full((1,2,1,32,32),.4)
    return dict(target=x,building=torch.zeros_like(x),vehicle=torch.zeros_like(x),
                video_id=['scene/episode/tx'],start=torch.tensor([0]))


def test_masks_noise_and_initials_are_paired_and_exactly_reproducible():
    a,initial=materialize(dense(),video_index=12,rate=1,sigma=.03)
    b,initial2=materialize(dense(),video_index=12,rate=3,sigma=.09)
    again,initial3=materialize(dense(),video_index=12,rate=1,sigma=.03)
    assert torch.equal(initial,initial2) and torch.equal(initial,initial3)
    assert torch.equal(a['observed_rss'],again['observed_rss'])
    assert (a['sampling_mask']<=b['sampling_mask']).all()
    selected=a['sampling_mask'].bool()
    torch.testing.assert_close((a['observed_rss']-a['target'])[selected]/.03,
                               (b['observed_rss']-b['target'])[selected]/.09,atol=2e-6,rtol=2e-6)
    assert a['sampling_mask'].sum()==20


def test_saved_folds_and_ensemble_members_do_not_depend_on_batch_order():
    a,_=materialize(dense(),video_index=12,rate=3,sigma=0)
    fold=balanced_folds(a['sampling_mask'],video_index=12,start=0)
    assert (fold[a['sampling_mask']==0]==-1).all()
    for t in range(2):
        counts=[int((fold[0,t]==f).sum()) for f in range(4)]
        assert max(counts)-min(counts)<=1
    args=dict(video_index=12,start=0,fold=1)
    full=ensemble_initial((2,1,4,4),members=[0,1,2,3],**args)
    separate=ensemble_initial((2,1,4,4),members=[2,3],**args)
    assert torch.equal(full[2:],separate)
    assert not torch.equal(full[0],full[1])
    seeds=[frame_seed(v,t,s) for v in [0,1,14999] for t in range(100) for s in range(100)]
    assert len(set(seeds))==len(seeds)
