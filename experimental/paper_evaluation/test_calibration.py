import numpy as np
from .calibration import fit_variance_calibration

def test_dimensionless_fit_recovers_affine_variance_at_small_scale():
    raw=np.linspace(1e-9,2e-6,1000)
    error=400*raw+1e-4
    fit=fit_variance_calibration({'1':raw},{'1':error})
    np.testing.assert_allclose([fit.scale,fit.offset],[400,1e-4],rtol=2e-4)

def test_constant_error_allows_boundary_solution():
    raw=np.linspace(1e-9,2e-6,1000)
    fit=fit_variance_calibration({'1':raw},{'1':np.full_like(raw,2e-4)})
    np.testing.assert_allclose(fit.apply(raw),2e-4,rtol=2e-4)
