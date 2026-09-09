"""Small CPU tests of the actual fixed protocol, loader, sampler, and metrics."""

from types import SimpleNamespace

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import prepare_data_loader
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts import evaluate_observation_balance_cell as evaluator
from rmdm.data import WindowDataset
from rmdm.evaluation.fixed_sparse_protocol import (
    add_fixed_observation_noise, apply_fixed_sparse_observations,
    deterministic_frame_noise_like, frame_names_by_sample,
)
from rmdm_hvdit_v4_joint.config import ExperimentConfig
from rmdm_noise_estimation.assimilation import NoiseAwareDDIMSampler


class TinyDenoiser(torch.nn.Module):
    def encode_conditions(self, batch):
        return {}

    def denoise(self, sample, times, cache):
        return sample * 0.2 + 0.4


class TinyReader:
    records = ["a/video_0", "a/video_1", "a/video_2"]

    def frame_count(self, record):
        return 16

    def read_window(self, record, start, length):
        shape = (length, 8, 8)
        return {
            "building": np.zeros(shape, dtype=np.float32),
            "vehicle": np.zeros(shape, dtype=np.float32),
            "tx": np.zeros(shape, dtype=np.float32),
            "target": np.full(shape, 0.6, dtype=np.float32),
            "frame_names": [f"{record}/{frame:03d}" for frame in range(start, start + length)],
        }


def tiny_dataset(**kwargs):
    return WindowDataset(reader=TinyReader(), **kwargs)


def tiny_case(variant):
    config = ExperimentConfig()
    config.data.frames_per_video = 16
    config.data.workers = 0
    config.evaluation.t1_evaluation_batch_size = 2
    config.evaluation.w16_evaluation_batch_size = 2
    config.diffusion.prediction_type = "sample"
    cell = {
        "variant": variant, "starts": "all" if variant == "w1" else [0],
        "split": "val", "video_ids": TinyReader.records,
        "seeds": {"global": 1, "dataset": 2, "mask": 3, "observation_noise": 4, "ddim_noise": 5},
        "rate": 3, "noise_std": 0.05, "ddim_steps": 3,
        "guidance": {"guided_steps": 2, "strength": 0.5, "max_update": 0.25},
        "methods": ["no_da", "known_noise_da"],
    }
    return config, cell


@pytest.mark.parametrize("variant", ["w1", "w16"])
def test_cell_results_repeat_and_methods_do_not_interfere(monkeypatch, variant):
    monkeypatch.setattr(evaluator, "WindowDataset", tiny_dataset)
    accelerator = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(even_batches=False))
    config, cell = tiny_case(variant)
    model = TinyDenoiser()
    result = evaluator.evaluate(accelerator, model, config, cell)
    assert result["scored_frames"] == 48
    assert result == evaluator.evaluate(accelerator, model, config, cell)
    for method in cell["methods"]:
        alone = evaluator.evaluate(accelerator, model, config, {**cell, "methods": [method]})
        assert alone["methods"][method] == result["methods"][method]
        assert np.isfinite(result["methods"][method]["metrics"]["unobserved_free_space"]["nmse"])


def test_w1_w16_share_physical_frame_masks_and_noise():
    outputs = []
    for window_size in (1, 16):
        dataset = tiny_dataset(window_size=window_size, fixed_starts=range(0, 16, window_size))
        by_frame = {}
        for batch in DataLoader(dataset, batch_size=2):
            sparse = apply_fixed_sparse_observations(batch, rate=3, split="val", manifest_seed=3)
            sparse = add_fixed_observation_noise(sparse, standard_deviation=0.05, rate=3, seed=4)
            names = frame_names_by_sample(batch, batch_size=batch["target"].shape[0], window_size=window_size)
            noise = deterministic_frame_noise_like(batch["target"], names, rate=3, seed=5)
            for i, frames in enumerate(names):
                for j, name in enumerate(frames):
                    by_frame[name] = (sparse["sampling_mask"][i, j], sparse["observed_rss"][i, j], noise[i, j])
        outputs.append(by_frame)
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        assert all(torch.equal(a, b) for a, b in zip(outputs[0][name], outputs[1][name]))


@pytest.mark.parametrize("size", [1, 3, 5, 17])
def test_four_rank_loader_scores_every_sample_once(size):
    seen = []
    for rank in range(4):
        loader = prepare_data_loader(
            DataLoader(list(range(size)), batch_size=2), device=torch.device("cpu"),
            num_processes=4, process_index=rank, even_batches=False,
        )
        seen.extend(value for batch in loader for value in batch.tolist())
    assert sorted(seen) == list(range(size))


@pytest.mark.parametrize("prediction_type", ["sample", "epsilon"])
def test_sampler_preserves_noise_and_disabled_guidance_matches_baseline(prediction_type):
    config = SimpleNamespace(prediction_type=prediction_type, train_timesteps=20, beta_schedule="linear")
    sampler = NoiseAwareDDIMSampler(config)
    accelerator = Accelerator(cpu=True)
    noise = torch.randn((2, 1, 1, 4, 4), generator=torch.Generator().manual_seed(1))
    original = noise.clone()
    batch = {"sampling_mask": torch.ones_like(noise), "observed_rss": torch.full_like(noise, 0.2)}
    baseline = sampler.baseline(TinyDenoiser(), {}, noise, steps=4, accelerator=accelerator)
    for settings in (
        {"strength": 0.0, "guided_steps": 2, "noise_variance": 0.0},
        {"strength": 0.5, "guided_steps": 0, "noise_variance": 0.0},
        {"strength": 0.5, "guided_steps": 4, "noise_variance": 1e6},
    ):
        guided = sampler.guided(
            TinyDenoiser(), {}, batch, noise, steps=4, max_update=0.25,
            accelerator=accelerator, **settings,
        )
        assert torch.equal(guided, baseline)
        assert torch.equal(noise, original)


def test_guidance_reduces_observed_error_for_simple_denoiser():
    config = SimpleNamespace(prediction_type="sample", train_timesteps=20, beta_schedule="linear")
    sampler = NoiseAwareDDIMSampler(config)
    accelerator = Accelerator(cpu=True)
    noise = torch.full((1, 1, 1, 4, 4), 0.8)
    batch = {"sampling_mask": torch.ones_like(noise), "observed_rss": torch.full_like(noise, 0.2)}
    baseline = sampler.baseline(TinyDenoiser(), {}, noise, steps=1, accelerator=accelerator)
    guided = sampler.guided(
        TinyDenoiser(), {}, batch, noise, steps=1, guided_steps=1,
        strength=0.5, max_update=0.25, noise_variance=0, accelerator=accelerator,
    )
    assert ((guided - 0.2)**2).mean() < ((baseline - 0.2)**2).mean()
