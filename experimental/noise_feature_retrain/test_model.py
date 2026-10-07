import math
import torch
from experimental.noise_feature_retrain.model import ReliabilityStem
from rmdm_hvdit_v4_joint.model.patching import W16DoubleStem
from experimental.noise_feature_retrain.data import PairedData


def test_noise_propagation_and_mask_preservation():
    base=W16DoubleStem(4,3,2,2);stem=ReliabilityStem.from_stem(base)
    with torch.no_grad():
        stem.observation_projection.weight.fill_(1);stem.dense_projection.weight.zero_();stem.fusion.weight.zero_();stem.fusion.weight[:,3:]=torch.eye(3);stem.log_signal_variance.fill_(0)
    dense=torch.zeros(3,16,4,4,4);obs=torch.ones(3,16,2,4,4);q=torch.tensor([0,.1,1.])
    result=stem(dense,obs,dict(variance=q));expected=8+8/(1+8*q)
    torch.testing.assert_close(result[:,0,0,0,0],expected)
    result.sum().backward();assert torch.isfinite(stem.log_signal_variance.grad).all();assert stem.log_signal_variance.grad.sum()>0


def test_paired_generator_repeats_reproducibly_without_sigma_leakage():
    data=PairedData('runs/noise_feature_retrain_20261008/inputs');a=data.batch(10,0,0);b=data.batch(10,0,0)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k in ['target','building','vehicle','sampling_mask','diffusion_noise','timesteps']:
        for offset in [1,2,3]:torch.testing.assert_close(a[k][::4],a[k][offset::4],rtol=0,atol=0)
    sigma=a['measurement_variance'].sqrt().reshape(-1,4);assert (sigma[:,0]==0).all()
    for i in [1,2,3]:assert ((sigma[:,i]>=(i-1)*.03)&(sigma[:,i]<=i*.03)).all()
    for offset in [2,3]:
        residual=(a['observed_rss'][offset::4]-a['sampling_mask'][offset::4]*a['target'][offset::4])*sigma[:,1,None,None,None,None]
        reference=(a['observed_rss'][1::4]-a['sampling_mask'][1::4]*a['target'][1::4])*sigma[:,offset,None,None,None,None]
        torch.testing.assert_close(residual,reference,atol=1e-8,rtol=2e-5)
