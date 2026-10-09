import torch
from experimental.noise_paper_injection.model import masked_prox, NoiseMapStem, HyperParameters
from rmdm_hvdit_v4_joint.model.patching import W16DoubleStem


def test_prox_limits_and_stationary_equation():
    z=torch.tensor([.2,.4,.6],requires_grad=True);y=torch.tensor([.8,0.,.1]);mask=torch.tensor([1.,0.,1.])
    torch.testing.assert_close(masked_prox(z,y,mask,0.),torch.tensor([.8,.4,.1]))
    rho=torch.tensor(.7,requires_grad=True);x=masked_prox(z,y,mask,rho)
    torch.testing.assert_close(mask*(x-y)+rho*(x-z),torch.zeros(3),atol=1e-7,rtol=0)
    x.square().sum().backward();assert z.grad.abs().sum()>0 and rho.grad.abs()>0
    torch.testing.assert_close(masked_prox(z,y,mask,1e9),z)


def test_noise_map_channel_preserves_zero_map_and_is_spatially_effective():
    torch.manual_seed(3);old=W16DoubleStem(4,16,2,4);new=NoiseMapStem.from_stem(old)
    dense=torch.randn(2,16,4,8,8);obs=torch.randn(2,16,2,8,8);affine=(torch.zeros(2,16),torch.zeros(2,16))
    a=old(dense,obs,affine);b=new(dense,obs,dict(affine=affine,noise_map=torch.zeros(2)))
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    sigma=torch.tensor([.03,.09],requires_grad=True);c=new(dense,obs,dict(affine=affine,noise_map=sigma))
    assert not torch.allclose(b,c);c.square().mean().backward();assert sigma.grad.abs().min()>0


def test_hyperparameters_positive_and_differentiable_in_noise():
    torch.manual_seed(3);model=HyperParameters();s=torch.tensor([0.,.03,.05,.09],requires_grad=True)
    out=model(s,torch.ones(4)*2,torch.tensor([0,100,500,999]));assert (out>0).all() and torch.isfinite(out).all()
    out.sum().backward();assert s.grad.abs().min()>0
