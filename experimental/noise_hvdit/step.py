"""Paired prediction targets and identical clean auxiliary supervision."""
import torch
import torch.nn.functional as F

from experimental.noise_temporal_rmdm.noise import add_measurement_noise
from experimental.noise_temporal_rmdm.step import _diffusion_seeds
from utils import cal_pinn_components


def training_step(model, dense, sampling, diffusion, config, epoch):
    sparse = add_measurement_noise(sampling(dense), config.measurement_noise, epoch=epoch)
    target = sparse["target"]
    batch = diffusion.training_batch(target, seeds=_diffusion_seeds(sparse, training_seed=config.train.seed, epoch=epoch))
    prediction, cal = model(batch.noisy_target, batch.timesteps, sparse)
    expected = target if config.diffusion.prediction_type == "sample" else batch.noise
    reconstruction = F.mse_loss(prediction.float(), expected.float())
    calibration = F.mse_loss(cal.float(), target.float())
    obstacle = ((sparse["building"] > 0.5) | (sparse["vehicle"] > 0.5)).float()
    equation, boundary, source = cal_pinn_components(
        cal.float().flatten(0, 2), obstacle.flatten(0, 2),
        dense["source_label"].flatten(0, 2), k=config.loss.pinn_k,
    )
    equation, boundary, source = equation.mean(), boundary.mean(), source.mean()
    loss = (reconstruction + config.loss.calibration * calibration + config.loss.equation * equation
            + config.loss.obstacle * boundary + config.loss.source * source)
    with torch.no_grad():
        x0 = prediction.float() if config.diffusion.prediction_type == "sample" else diffusion.predict_x0(
            batch.noisy_target.float(), prediction.float(), batch.timesteps)
        error = (x0-target).square().flatten(1).mean(1)
        metrics = dict(loss=loss.detach(), diffusion=reconstruction.detach(), calibration=calibration.detach(),
                       equation=equation.detach(), obstacle=boundary.detach(), source=source.detach(),
                       x0_mse=error.mean(), sigma=sparse["measurement_standard_deviation"].mean())
        # Sums/counts, rather than an average of means with different support.
        for index in range(4):
            selected = (batch.timesteps >= index*250) & (batch.timesteps < (index+1)*250)
            metrics[f"x0_t{index}_sum"] = error[selected].sum()
            metrics[f"x0_t{index}_count"] = selected.sum().float()
    return loss, metrics
