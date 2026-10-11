import numpy as np
import pytest

from experimental.noise_adaptive.statistics import calibrated_variance, mle, update


def test_constant_variance_matches_analytic_solution():
    y = np.array([-.04, .04, -.08, .08])
    v = np.full_like(y, .0004)
    result = mle(y, np.zeros_like(y), v)
    expected = np.mean(y**2) - v[0]
    assert result['variance'] == pytest.approx(expected, abs=1e-10)
    assert result['sigma'] == pytest.approx(np.sqrt(expected), abs=1e-8)
    assert result['boundary'] is None
    assert mle(y * 0, y * 0, v)['boundary'] == 'lower'
    assert mle(y * 10, y * 0, v)['boundary'] == 'upper'
    assert mle(y, y * 0, v, qmax=0)['variance'] == 0


def test_nonconvex_fit_optimizes_inside_domain_instead_of_clipping():
    v = np.array([1.5789196841094525e-7, .0035207162054289435,
                  .0777984418979235, .0492679447406491])
    squared = np.array([4.876835890078771e-7, .04958227516272376,
                        .00036140311026671595, .06705125472588883])
    constrained = mle(np.sqrt(squared), np.zeros(4), v)
    wide = mle(np.sqrt(squared), np.zeros(4), v, qmax=.25)
    assert wide['variance'] > .0081
    assert 0 < constrained['variance'] < .0001
    clipped_nll = .5 * np.mean(np.log(v + .0081) + squared / (v + .0081))
    assert constrained['nll'] < clipped_nll - 1e-4
    grid = np.r_[0., np.geomspace(1e-12, .0081, 20000)]
    values = .5 * (np.log(v[:, None] + grid) + squared[:, None] / (v[:, None] + grid)).mean(0)
    assert constrained['nll'] <= values.min() + 1e-10


@pytest.mark.parametrize('y,mu,v,qmax', [
    ([], [], [], .01), ([1], [0, 0], [1], .01),
    ([np.nan], [0], [1], .01), ([1], [np.inf], [1], .01),
    ([1], [0], [0], .01), ([1], [0], [-1], .01),
    ([1], [0], [np.inf], .01), ([1], [0], [1], -1),
    ([1], [0], [1], np.nan), ([1], [0], [1], [1]),
])
def test_invalid_mle_inputs(y, mu, v, qmax):
    with pytest.raises(ValueError):
        mle(y, mu, v, qmax=qmax)


def calibration():
    return {'nodes': [dict(q=.01, scale=4., offset=.003, floor=.004),
                      dict(q=0., scale=2., offset=.001, floor=.002)]}


def test_calibration_interpolates_coefficients_and_floor():
    result = calibrated_variance(np.array([0., .01]), .005, calibration())
    np.testing.assert_allclose(result, [.003, .032])
    np.testing.assert_allclose(calibrated_variance([0.], 0., calibration()), [.002])


@pytest.mark.parametrize('raw,q', [([-1], .005), ([np.nan], .005),
                                  ([], .005), ([0], -.001), ([0], .011)])
def test_invalid_calibration_inputs(raw, q):
    with pytest.raises(ValueError):
        calibrated_variance(raw, q, calibration())


def test_invalid_calibration_nodes():
    for key, value in [('q', 0.), ('scale', -1.), ('offset', np.nan), ('floor', 0.)]:
        c = calibration()
        c['nodes'][0][key] = value
        with pytest.raises(ValueError):
            calibrated_variance([0.], .005, c)


def test_damping_does_not_fake_convergence_or_clip_proposal():
    next_q, stop, residual = update(0., .25, damping=.001)
    assert next_q == .00025
    assert residual == .5
    assert not stop
    assert update(.25, .25) == (.25, True, 0.)
    assert update(0., .25)[0] == .25


@pytest.mark.parametrize('kwargs', [dict(q=-1), dict(proposal=np.inf),
    dict(damping=0), dict(damping=1.1), dict(abs_tol=-1), dict(rel_tol=np.nan)])
def test_invalid_update_inputs(kwargs):
    arguments = dict(q=.001, proposal=.002)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        update(**arguments)
