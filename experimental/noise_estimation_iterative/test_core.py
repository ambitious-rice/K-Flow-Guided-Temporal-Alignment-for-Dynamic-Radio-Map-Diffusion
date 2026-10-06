import numpy as np
import torch
from .core import public_input, hide, calibrated_variance, update, mle


def test_hidden_measurements_and_labels_do_not_enter_prediction():
    x=torch.ones(1,4,1,2,2)
    s=dict(building=x,vehicle=x,sampling_rate=torch.ones(1,4)*2,
           observed_rss=x.clone(),sampling_mask=x,target=x*100,measurement_variance=torch.tensor([9.]))
    folds=(torch.arange(x.numel()).reshape_as(x)%4)
    baseline=hide(public_input(s,.0009),folds,0)
    s['target']*=3; s['measurement_variance']*=3; s['observed_rss'][folds==0]=999
    changed=hide(public_input(s,.0009),folds,0)
    assert all(torch.equal(v,changed[k]) for k,v in baseline.items())
    assert baseline['observed_rss'][folds==0].count_nonzero()==0


def test_calibration_interpolation_and_fixed_point():
    c={'nodes':[dict(q=0,scale=2,offset=.001),dict(q=.0081,scale=4,offset=.003)]}
    np.testing.assert_allclose(calibrated_variance(np.array([.01]),.00405,c),[.032])
    q,stop,residual=update(.0025,.0025)
    assert q==.0025 and stop and residual==0
    q,stop,residual=update(0,.25)
    assert q==.75*.0081 and not stop


def test_mle_recovers_homoscedastic_variance():
    # Exactly known second moment, testing a statistical identity, not random luck.
    observed=np.array([-1.,1.])*np.sqrt(.0025+.0004)
    result=mle(observed,np.zeros(2),np.full(2,.0004))
    assert abs(result['variance']-.0025)<1e-8
