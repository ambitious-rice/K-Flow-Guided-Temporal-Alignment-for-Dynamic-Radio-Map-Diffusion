"""Late-step x0 data assimilation with a per-window noise floor."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler

from rmdm_hvdit_v4_x0_w16_ratebalanced.inverse_sampling import observation_gradient_update


class NoiseAwareDDIMSampler:
    def __init__(self, diffusion_config: Any) -> None:
        self.prediction_type = str(diffusion_config.prediction_type)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_config.train_timesteps,
            beta_schedule=diffusion_config.beta_schedule,
            prediction_type=diffusion_config.prediction_type,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )

    def _timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        self.scheduler.set_timesteps(steps, device=device)
        return self.scheduler.timesteps

    @torch.no_grad()
    def baseline(
        self,
        model: Any,
        cache: dict[str, torch.Tensor],
        initial_noise: torch.Tensor,
        *,
        steps: int,
        accelerator: Any,
    ) -> torch.Tensor:
        sample = initial_noise
        for timestep in self._timesteps(steps, sample.device):
            times = torch.full(
                (sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long
            )
            with accelerator.autocast():
                output = model.denoise(
                    self.scheduler.scale_model_input(sample, timestep), times, cache
                )
            sample = self.scheduler.step(
                output,
                timestep,
                sample,
                eta=0.0,
                use_clipped_model_output=False,
                return_dict=False,
            )[0]
        return sample.clamp(0.0, 1.0)

    def guided(
        self,
        model: Any,
        cache: dict[str, torch.Tensor],
        sparse_batch: dict[str, Any],
        initial_noise: torch.Tensor,
        *,
        steps: int,
        guided_steps: int,
        strength: float,
        max_update: float,
        noise_variance: float,
        accelerator: Any,
    ) -> torch.Tensor:
        sample = initial_noise
        visible = (sparse_batch["sampling_mask"] > 0.5).float()
        observed = sparse_batch["observed_rss"].float()
        guided_from = steps - guided_steps
        for step_index, timestep in enumerate(self._timesteps(steps, sample.device)):
            times = torch.full(
                (sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long
            )
            if step_index < guided_from:
                with torch.no_grad(), accelerator.autocast():
                    output = model.denoise(
                        self.scheduler.scale_model_input(sample, timestep), times, cache
                    )
                    sample = self.scheduler.step(
                        output,
                        timestep,
                        sample,
                        eta=0.0,
                        use_clipped_model_output=False,
                        return_dict=False,
                    )[0]
                continue

            parent = sample.detach().requires_grad_(True)
            with accelerator.autocast():
                output = model.denoise(
                    self.scheduler.scale_model_input(parent, timestep), times, cache
                )
            if self.prediction_type == "sample":
                predicted_x0 = output.float()
            else:
                alpha = self.scheduler.alphas_cumprod[int(timestep)].to(
                    device=sample.device, dtype=torch.float32
                )
                predicted_x0 = (parent.float() - (1.0 - alpha).sqrt() * output.float()) / alpha.sqrt()
            residual = (predicted_x0 - observed) * visible
            count = visible.flatten(1).sum(1).clamp_min(1.0)
            loss = residual.square().flatten(1).sum(1) / count
            gradient = torch.autograd.grad(loss.sum(), parent, only_inputs=True)[0].float()
            correction = observation_gradient_update(
                gradient,
                loss,
                strength=strength,
                normalization="rms",
                max_update=max_update,
                observation_noise_variance=noise_variance,
            )
            with torch.no_grad():
                previous = self.scheduler.step(
                    output.detach(),
                    timestep,
                    parent.detach(),
                    eta=0.0,
                    use_clipped_model_output=False,
                    return_dict=False,
                )[0]
                sample = (previous.float() - correction).detach()
        return sample.clamp(0.0, 1.0)


__all__ = ["NoiseAwareDDIMSampler"]
