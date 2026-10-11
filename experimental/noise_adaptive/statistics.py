"""Bounded Gaussian marginal fitting and damped fixed-point updates."""
import numpy as np
from scipy.optimize import minimize_scalar


def _scalar(value, name, *, positive=False):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite scalar')
    value = float(array)
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return value


def mle(observed, mean, variance, qmax=.09**2):
    """Search the declared domain, including boundaries and sampled local minima.

    The heteroscedastic objective need not be convex. A dense global grid plus
    bounded refinements is used; this numerical search is not a formal proof
    of global optimality for arbitrary input arrays.
    """
    y, mu, v = [np.asarray(x, dtype=np.float64) for x in (observed, mean, variance)]
    if y.size == 0 or y.shape != mu.shape or y.shape != v.shape:
        raise ValueError('observed, mean and variance must be nonempty and aligned')
    if not all(np.isfinite(x).all() for x in (y, mu, v)) or np.any(v <= 0):
        raise ValueError('Inputs must be finite and variance strictly positive')
    qmax = _scalar(qmax, 'qmax')
    with np.errstate(over='ignore'):
        residual = (y - mu)**2
    if not np.isfinite(residual).all():
        raise ValueError('Squared residual overflow')

    def objective(q):
        total = v + q
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            result = .5 * np.mean(np.log(total) + residual / total)
        return float(result)

    if qmax == 0:
        grid = np.array([0.])
    else:
        grid = np.unique(np.r_[0., np.linspace(0., qmax, 513),
                               qmax * np.geomspace(1e-12, 1., 513)])
    values = np.array([objective(q) for q in grid])
    candidates = [(float(q), float(value)) for q, value in zip(grid, values)]
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            fit = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]),
                                  method='bounded', options={'xatol': max(qmax*1e-12, 1e-18)})
            if fit.success and np.isfinite(fit.fun):
                candidates.append((float(fit.x), float(fit.fun)))
    q, nll = min(candidates, key=lambda item: item[1])
    if not np.isfinite(nll):
        raise ValueError('No finite marginal objective in requested domain')
    boundary = 'lower' if q == 0 else ('upper' if q == qmax else None)
    return dict(variance=q, sigma=float(np.sqrt(q)), nll=nll, boundary=boundary)


def calibrated_variance(raw, q, calibration):
    """Interpolate independently fitted nonnegative affine calibrators in q."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Raw variances must be nonempty, finite and nonnegative')
    q = _scalar(q, 'q')
    nodes = calibration.get('nodes', [])
    if not nodes:
        raise ValueError('Calibration needs at least one node')
    coefficients = []
    for node in nodes:
        coefficients.append([_scalar(node[key], key, positive=key == 'floor')
                             for key in ('q', 'scale', 'offset', 'floor')])
    coefficients = np.array(sorted(coefficients, key=lambda row: row[0]))
    grid = coefficients[:, 0]
    if np.any(np.diff(grid) <= 0):
        raise ValueError('Calibration q nodes must be unique')
    if q < grid[0] or q > grid[-1]:
        raise ValueError('Outside calibration domain')
    scale, offset, floor = [float(np.interp(q, grid, coefficients[:, i]))
                            for i in (1, 2, 3)]
    with np.errstate(over='ignore'):
        result = np.maximum(scale * raw + offset, floor)
    if not np.isfinite(result).all():
        raise ValueError('Calibrated variance overflow')
    return result


def update(q, proposal, damping=1., abs_tol=.001, rel_tol=.02):
    """Damp in variance space; test the undamped residual in sigma units."""
    q = _scalar(q, 'q')
    proposal = _scalar(proposal, 'proposal')
    damping = _scalar(damping, 'damping', positive=True)
    abs_tol = _scalar(abs_tol, 'abs_tol')
    rel_tol = _scalar(rel_tol, 'rel_tol')
    if damping > 1:
        raise ValueError('damping must lie in (0, 1]')
    next_q = (1 - damping) * q + damping * proposal
    residual = abs(np.sqrt(proposal) - np.sqrt(q))
    return float(next_q), bool(residual <= abs_tol + rel_tol * np.sqrt(q)), float(residual)
