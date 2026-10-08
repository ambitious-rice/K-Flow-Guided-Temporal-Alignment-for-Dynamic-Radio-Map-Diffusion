import torch
from experimental.noise_scratch_full.loss import sampled_clean_loss,loss_terms


def test_sampled_loss_uses_clean_target_at_nonzero_sigma():
    target=torch.tensor([[[[[.2,.4],[.6,.8]]]]]);mask=torch.tensor([[[[[1.,0.],[1.,0.]]]]])
    pred=target.clone().requires_grad_(True)
    assert sampled_clean_loss(pred,target,mask)==0
    prediction=target+torch.tensor([[[[[.1,99.],[.3,99.]]]]])
    torch.testing.assert_close(sampled_clean_loss(prediction,target,mask),torch.tensor(.05))


def test_all_six_terms_and_no_noisy_target_dependency():
    target=torch.rand(2,16,1,8,8);pred=torch.rand_like(target).requires_grad_(True);cal=torch.rand_like(target).requires_grad_(True)
    b=dict(target=target,sampling_mask=torch.ones_like(target),building=torch.zeros_like(target),vehicle=torch.zeros_like(target),source_label=torch.zeros_like(target),measurement_variance=torch.tensor([.03**2,.09**2]),observed_rss=target+3)
    loss,terms=loss_terms(pred,cal,b);assert len(terms)==6;torch.testing.assert_close(loss,sum(terms.values()))
    other,_=loss_terms(pred,cal,dict(b,observed_rss=target-100));torch.testing.assert_close(loss,other)
    loss.backward();assert pred.grad.abs().sum()>0 and cal.grad.abs().sum()>0
