"""No labels enter the adaptive update; cross-round independence is not assumed."""
import numpy as np
import torch
from diffusers import DDIMScheduler
from scipy.optimize import minimize_scalar
from experimental.paper_evaluation.randomness import ensemble_initial
from rmdm_noise_estimation.folds import hide_fold
from rmdm_noise_estimation.statistics import marginal_mle, marginal_nll


def public_input(sparse, q):
    if not np.isfinite(q) or q < 0:
        raise ValueError('Invalid conditioning variance')
    return {**{k: sparse[k] for k in ('building', 'vehicle', 'sampling_rate', 'observed_rss', 'sampling_mask')},
            'measurement_variance': torch.tensor([q], device=sparse['observed_rss'].device)}


def hide(batch, assignment, fold):
    result = hide_fold(batch, assignment, fold)
    before = batch['sampling_mask'].flatten(2).sum(2)
    after = result['sampling_mask'].flatten(2).sum(2)
    result['sampling_rate'] = batch['sampling_rate'] * after / before.clamp_min(1)
    return result


def repeat(value, n):
    if torch.is_tensor(value): return value.repeat_interleave(n, dim=0)
    if isinstance(value, dict): return {k: repeat(v,n) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(repeat(v,n) for v in value)
    if isinstance(value, list): return [repeat(v,n) for v in value]
    raise TypeError(type(value))


@torch.inference_mode()
def collect(model, diffusion, *, inputs, assignment, video_index, start, steps=50):
    """Same 4x8 draws at every iteration; q enters only through inputs."""
    observed = inputs['observed_rss']; visible = inputs['sampling_mask'].bool()
    if not torch.equal(assignment >= 0, visible): raise ValueError('Invalid folds')
    scheduler = DDIMScheduler(num_train_timesteps=diffusion.train_timesteps,
        beta_schedule=diffusion.beta_schedule, prediction_type=diffusion.prediction_type,
        clip_sample=True, set_alpha_to_one=True, steps_offset=0)
    samples = torch.empty((8,int(visible.sum())),device=observed.device)
    point_folds = assignment[visible]
    for fold in range(4):
        cache = repeat(model.encode_conditions(hide(inputs, assignment, fold)),8)
        sample = ensemble_initial(observed.shape[1:],video_index=video_index,start=start,
                                  fold=fold,members=list(range(8))).to(observed.device)
        scheduler.set_timesteps(steps,device=observed.device)
        for timestep in scheduler.timesteps:
            times = torch.full((8,),int(timestep),device=observed.device,dtype=torch.long)
            output = model.denoise(scheduler.scale_model_input(sample,timestep),times,cache)
            sample = scheduler.step(output,timestep,sample,eta=0.,use_clipped_model_output=False,return_dict=False)[0]
        selected = assignment == fold
        samples[:,point_folds == fold] = sample.clamp(0,1)[:,selected[0]]
    samples = samples.cpu()
    if not torch.isfinite(samples).all(): raise ValueError('Nonfinite samples')
    return dict(samples=samples,mean=samples.double().mean(0).float(),
        raw_variance=samples.double().var(0,unbiased=True).float(),observed=observed[visible].cpu(),
        flat_indices=visible.flatten().nonzero().flatten().cpu(),folds=point_folds.cpu())


def calibrated_variance(raw, q, calibration):
    """Interpolate nonnegative affine coefficients in conditioning variance."""
    nodes = sorted(calibration['nodes'],key=lambda n:n['q'])
    grid = np.array([n['q'] for n in nodes])
    if q < grid[0]-1e-12 or q > grid[-1]+1e-12: raise ValueError('Outside calibration grid')
    a = float(np.interp(q,grid,[n['scale'] for n in nodes]))
    b = float(np.interp(q,grid,[n['offset'] for n in nodes]))
    return np.maximum(a*np.asarray(raw,dtype=np.float64)+b,1e-8)


def mle(observed, mean, variance):
    residual = (np.asarray(observed,dtype=np.float64)-np.asarray(mean,dtype=np.float64))**2
    historical = marginal_mle(residual,variance,maximum_variance=.25)
    grid = np.unique(np.r_[0.,np.geomspace(1e-10,.25,160),np.linspace(0,.25,80)])
    values = [marginal_nll(q,residual,variance) for q in grid]
    candidates = [historical,(0.,values[0]),(.25,values[-1])]
    for i in range(1,len(grid)-1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            fit = minimize_scalar(lambda q:marginal_nll(q,residual,variance),
                bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-14})
            candidates.append((float(fit.x),float(fit.fun)))
    q,nll = min(candidates,key=lambda x:x[1])
    return dict(variance=q,sigma=q**.5,nll=nll)


def update(q, proposal, damping=.75):
    if not 0 < damping <= 1 or not np.isfinite([q,proposal]).all() or min(q,proposal)<0:
        raise ValueError('Invalid update')
    # Projection is a fixed training-domain constraint, never the test noise label.
    projected = min(proposal,.09**2)
    next_q = (1-damping)*q+damping*projected
    residual = abs(projected**.5-q**.5)
    return next_q, residual <= .001 + .02*q**.5, residual
