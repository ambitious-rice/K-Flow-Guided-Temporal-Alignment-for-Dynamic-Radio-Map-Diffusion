import torch
from experimental.noise_condition_diagnosis.reconstruct import calibrate_variance,reconstruct

CAL=dict(external_sigmas=[0,.03,.05,.09],internal_sigmas=[0,0,.03,.05])

def test_variance_interpolation_and_monotonicity():
    q=torch.linspace(0,.09**2,101);out=calibrate_variance(q,CAL)
    assert (out[1:]>=out[:-1]).all()
    torch.testing.assert_close(calibrate_variance(torch.tensor([0,.03**2,.05**2,.09**2]),CAL),torch.tensor([0,0,.03**2,.05**2]))
    torch.testing.assert_close(calibrate_variance(torch.tensor([.04**2]),CAL),torch.tensor([.00039375]))

def test_no_physical_variance_mutation_or_ground_truth_input():
    class Spy:
        training=False
        def encode_conditions(self,b):
            assert set(b)=={'building','vehicle','observed_rss','sampling_mask','sampling_rate','measurement_variance'}
            self.q=b['measurement_variance'].clone();return b
        def denoise(self,z,t,cache):
            assert int(t[0])==999
            return cache['observed_rss']+self.q[:,None,None,None,None]
    model=Spy();x=torch.ones(1,16,1,2,2)*.5
    b={k:x.clone() for k in ['building','vehicle','observed_rss','sampling_mask']};b.update(sampling_rate=torch.tensor([2.]),measurement_variance=torch.tensor([.09**2]),target=x*99,source_label=x*55)
    before=b['measurement_variance'].clone();a=reconstruct(model,b,torch.zeros_like(x),CAL)
    b['target']*=0;b['source_label']*=0;c=reconstruct(model,b,torch.zeros_like(x),CAL)
    torch.testing.assert_close(a,c);torch.testing.assert_close(b['measurement_variance'],before);torch.testing.assert_close(model.q,torch.tensor([.05**2]))
