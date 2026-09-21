"""CPU-safe synthetic forward/backward smoke; this does not start training."""

from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .model import build_model, parameter_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/noise_temporal_rmdm/smoke.yaml")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, smoke=True)
    model = build_model(config)
    batch, time, height, width = 1, 1, 32, 32
    shape = (batch, time, 1, height, width)
    target = torch.rand(shape)
    mask = (torch.rand(shape) < 0.05).float()
    conditions = {
        "building": (torch.rand(shape) < 0.2).float(),
        "vehicle": (torch.rand(shape) < 0.05).float(),
        "observed_rss": mask * target,
        "sampling_mask": mask,
        "measurement_variance": torch.tensor([0.03 ** 2]),
    }
    prediction, cal = model(torch.randn(shape), torch.tensor([500]), conditions)
    loss = prediction.square().mean() + cal.square().mean()
    if args.backward:
        loss.backward()
    trainable, total = parameter_counts(model)
    print(json.dumps({
        "mode": "synthetic_cpu_smoke", "prediction_shape": list(prediction.shape),
        "cal_shape": list(cal.shape), "finite": bool(torch.isfinite(loss)),
        "backward": args.backward, "trainable_parameters": trainable, "total_parameters": total,
    }, indent=2))


if __name__ == "__main__":
    main()
