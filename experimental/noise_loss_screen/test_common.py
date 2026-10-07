import torch
from .common import observation_loss,public,VARIANTS

def test_k_relaxes_noisy_observation_constraint():
    pred=torch.tensor([.5,.5]).reshape(2,1,1,1,1)
    s=dict(observed_rss=torch.zeros_like(pred),sampling_mask=torch.ones_like(pred),measurement_variance=torch.tensor([.01,.04]))
    loss1,_=observation_loss(pred,s,1);loss2,_=observation_loss(pred,s,2)
    assert loss2<loss1
    s['measurement_variance'].zero_()
    assert observation_loss(pred,s,1)[0]==observation_loss(pred,s,6)[0]

def test_sigma_ablation_only_changes_model_input():
    x=torch.ones(2,1,1,1,1)
    s=dict(building=x,vehicle=x,observed_rss=x,sampling_mask=x,sampling_rate=torch.ones(2,1),measurement_variance=torch.tensor([.001,.004]),target=x*7,source=x*9)
    b=public(s,torch.zeros(2))
    assert 'target' not in b and 'source' not in b
    assert torch.equal(s['measurement_variance'],torch.tensor([.001,.004]))
    assert b['measurement_variance'].count_nonzero()==0
    assert VARIANTS['k2']['k']==VARIANTS['k2_nosigma']['k']
