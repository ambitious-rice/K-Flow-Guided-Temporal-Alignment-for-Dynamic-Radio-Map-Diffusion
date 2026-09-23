"""Original HVDiT topology with explicit measurement conditioning."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from rmdm_hvdit_v4_joint.model.common import zero_linear
from rmdm_hvdit_v4_joint.model.hwm import build_hwm_from_scratch
from rmdm_hvdit_v4_joint.model.joint import JointTokenDenoiser
from rmdm_hvdit_v4_joint.transfer.inflate_t1_to_w16 import _inflate_state


def mlp(inputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())


class ConditionedHWM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = build_hwm_from_scratch(base_features=32)
        encoder = self.network.conv_blocks_context
        decoder = self.network.conv_blocks_localization
        channels = [block.output_channels for block in encoder[:-1]] + [encoder[-1][-1].output_channels]
        self.encoder_modulation = nn.ModuleList([zero_linear(nn.Linear(width, 2*c)) for c in channels])
        self.decoder_modulation = nn.ModuleList([
            zero_linear(nn.Linear(width, 2*block[-1].output_channels)) for block in decoder
        ])

    def forward(self, raw, embedding):
        inputs = torch.cat([raw[k] for k in ("building", "tx", "vehicle", "observed_rss", "sampling_mask")], 2)
        batch, frames, _, height, width = inputs.shape
        flat = inputs.flatten(0, 1)
        condition = embedding[:, None].expand(-1, frames, -1).flatten(0, 1)
        gates, calibrations = [], []
        for start in range(0, len(flat), 16):
            e = condition[start:start+16]
            modulations = tuple([
                head(e).unsqueeze(-1).unsqueeze(-1).chunk(2, 1) for head in heads
            ] for heads in (self.encoder_modulation, self.decoder_modulation))
            anchors, cal = self.network(flat[start:start+16], feature_modulations=modulations)
            gate = sum(F.interpolate(a, (height, width), mode="bilinear", align_corners=False).mean(1, keepdim=True)
                       for a in anchors[:2]).sigmoid().detach()
            gates.append(gate)
            calibrations.append(cal)
        return {
            "hwm_gate": torch.cat(gates).reshape(batch, frames, 1, height, width),
            "cal": torch.cat(calibrations).reshape(batch, frames, 1, height, width),
        }


class NoiseHVDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reference_variance = config.measurement_noise.reference_variance
        width = config.embedding_width
        self.variance_embedding = mlp(3, width)
        self.rate_embedding = mlp(1, width)
        self.hwm = ConditionedHWM(width)
        self.denoiser = JointTokenDenoiser(config.model, frames=config.data.window_size)
        self.heads = nn.ModuleDict({
            name: zero_linear(nn.Linear(width, size)) for name, size in {
                "input_observation": 2 * config.model.local_dim,
                "condition_observation": 2 * config.model.local_dim,
                "local": 6 * config.model.local_dim,
                "global": 6 * config.model.global_dim,
                "decoder": config.model.mapping_width,
            }.items()
        })

    def encode_conditions(self, batch):
        # Source labels are intentionally absent from this input boundary.
        raw = {k: batch[k] for k in ("building", "vehicle", "observed_rss", "sampling_mask")}
        raw["tx"] = torch.zeros_like(raw["building"])
        variance = batch["measurement_variance"].float().reshape(-1)
        ratio = variance / self.reference_variance
        features = torch.stack([ratio, torch.log1p(ratio), (variance == 0).float()], -1)
        rate = batch["sampling_rate"].float().reshape(len(variance), -1).mean(1)
        e = self.variance_embedding(features) + self.rate_embedding(torch.log(rate.clamp_min(1e-4))[:, None] / math.log(10))
        raw["input_observation_modulation"] = self.heads["input_observation"](e).chunk(2, -1)
        raw["condition_observation_modulation"] = self.heads["condition_observation"](e).chunk(2, -1)
        high, low = self.denoiser.encode_raw_conditions(raw)
        return {
            **raw, **self.hwm(raw, e), "condition_high": high, "condition_low": low,
            "local_measurement_modulation": self.heads["local"](e).reshape(len(e), 6, -1),
            "global_measurement_modulation": self.heads["global"](e).reshape(len(e), 6, -1),
            "measurement_condition": self.heads["decoder"](e),
        }

    def denoise(self, noisy, timesteps, cache):
        return self.denoiser(noisy, timesteps, cache)

    def forward(self, noisy, timesteps, batch):
        cache = self.encode_conditions(batch)
        return self.denoise(noisy, timesteps, cache), cache["cal"]


def initialize_w16(model, payload):
    inflated, _ = _inflate_state(payload["model"], model.state_dict())
    model.load_state_dict(inflated)
