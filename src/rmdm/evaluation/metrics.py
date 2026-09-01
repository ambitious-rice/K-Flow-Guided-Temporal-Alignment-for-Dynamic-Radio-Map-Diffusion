"""Additive metrics for the two fixed W16 evaluation domains."""

from __future__ import annotations

import math

import torch


DOMAIN_NAMES = ("full_image", "unobserved_free_space")
STAT_NAMES = ("sum_squared_error", "sum_target_energy", "sum_absolute_error", "pixel_count")


class MetricAccumulator:
    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.sums = torch.zeros((len(DOMAIN_NAMES), len(STAT_NAMES)), dtype=torch.float64, device=device)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        building: torch.Tensor,
        vehicle: torch.Tensor,
        sampling_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if any(value.shape != target.shape for value in (building, vehicle, sampling_mask)):
            raise ValueError("metric masks must match target shape")
        error = prediction.to(torch.float64) - target.to(torch.float64)
        target64 = target.to(torch.float64)
        domains = (
            torch.ones_like(target, dtype=torch.bool),
            (building <= 0.5) & (vehicle <= 0.5) & (sampling_mask <= 0.5),
        )
        for index, domain in enumerate(domains):
            mask = domain.to(torch.float64)
            self.sums[index, 0] += (error.square() * mask).sum()
            self.sums[index, 1] += (target64.square() * mask).sum()
            self.sums[index, 2] += (error.abs() * mask).sum()
            self.sums[index, 3] += mask.sum()

    def add_raw(self, raw: dict[str, dict[str, float]]) -> None:
        for domain_index, domain in enumerate(DOMAIN_NAMES):
            for stat_index, stat in enumerate(STAT_NAMES):
                self.sums[domain_index, stat_index] += float(raw[domain][stat])

    def raw(self) -> dict[str, dict[str, float]]:
        values = self.sums.detach().cpu().tolist()
        return {
            domain: {stat: float(values[domain_index][stat_index]) for stat_index, stat in enumerate(STAT_NAMES)}
            for domain_index, domain in enumerate(DOMAIN_NAMES)
        }

    def compute(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for domain_index, domain in enumerate(DOMAIN_NAMES):
            squared, energy, absolute, count = [float(value) for value in self.sums[domain_index].detach().cpu()]
            if count <= 0:
                raise RuntimeError(f"Metric domain {domain} contains no pixels")
            mse = squared / count
            result[domain] = {
                "mse": mse,
                "nmse": squared / max(energy, 1.0e-12),
                "mae": absolute / count,
                "psnr": float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse),
            }
        return result

