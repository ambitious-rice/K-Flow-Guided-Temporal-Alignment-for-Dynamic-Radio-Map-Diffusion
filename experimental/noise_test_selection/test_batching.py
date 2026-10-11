import torch
from .batching import build_group


def test_cases_keep_true_and_input_sigma_separate():
    shape=(1,16,1,2,2);target=torch.ones(shape)*.5
    data=dict(target=target,building=torch.zeros(shape),vehicle=torch.zeros(shape),valid_mask=torch.ones(shape),
              masks={1:torch.ones(shape),3:torch.ones(shape)},measurement_noise=torch.ones(shape)*2,initial=torch.ones(shape)*7)
    inputs,initial,metrics=build_group(data,[(1,.03),(3,.09)])
    assert set(inputs)=={'building','vehicle','observed_rss','sampling_mask','sampling_rate','measurement_variance'}
    assert initial.shape==(8,16,1,2,2) and torch.all(initial==7)
    assert torch.allclose(inputs['observed_rss'][:4],torch.full_like(inputs['observed_rss'][:4],.56))
    assert torch.allclose(inputs['observed_rss'][4:],torch.full_like(inputs['observed_rss'][4:],.68))
    assert torch.allclose(inputs['measurement_variance'],torch.tensor([0.,.03,.05,.09]*2).square())
    assert torch.all(inputs['sampling_rate'][:4]==1) and torch.all(inputs['sampling_rate'][4:]==3)
    assert len(metrics)==2 and torch.equal(metrics[0]['target'],target)
