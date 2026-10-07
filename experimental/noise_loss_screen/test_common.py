import torch
from .common import observation_loss,public,VARIANTS,loss_terms

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

def test_auxiliary_weight_scales_only_auxiliary_gradient():
    x=torch.full((1,2,1,4,4),.3)
    pred=x.clone().requires_grad_();cal=(x+.1).requires_grad_()
    s=dict(target=x,observed_rss=x,sampling_mask=torch.ones_like(x),measurement_variance=torch.tensor([.0025]),building=torch.zeros_like(x),vehicle=torch.zeros_like(x))
    source=torch.ones_like(x)
    full,_=loss_terms(pred,cal,s,source,VARIANTS['k2'])
    small,_=loss_terms(pred,cal,s,source,VARIANTS['k2_aux01'])
    gf=torch.autograd.grad(full,cal,retain_graph=True)[0]
    gs=torch.autograd.grad(small,cal)[0]
    torch.testing.assert_close(gs,.1*gf)
