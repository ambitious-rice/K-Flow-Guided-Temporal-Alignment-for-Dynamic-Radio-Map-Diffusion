"""DDIM sampling with exact consistency for noiseless observations."""
import torch


@torch.no_grad()
def project_clean_observations(output, sparse):
    clean = (sparse["measurement_variance"] == 0).reshape(-1, 1, 1, 1, 1)
    mask = sparse["sampling_mask"] * clean
    return output*(1-mask) + sparse["observed_rss"]*mask


@torch.no_grad()
def sample(sampler, model, sparse, initial_noise, steps):
    output = sampler.sample(model, sparse, initial_noise=initial_noise, steps=steps)
    return project_clean_observations(output, sparse)
