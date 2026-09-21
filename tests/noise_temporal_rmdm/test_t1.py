from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experimental.noise_temporal_rmdm import ARCHITECTURE_ID
from experimental.noise_temporal_rmdm.checkpoint import T1_SCHEMA, build_payload, load
from experimental.noise_temporal_rmdm.config import ExperimentConfig, MeasurementNoiseConfig
from experimental.noise_temporal_rmdm.data_order import VideoBlockShuffleSampler
from experimental.noise_temporal_rmdm.model import build_model
from experimental.noise_temporal_rmdm.noise import add_fixed_measurement_noise, add_measurement_noise
from experimental.noise_temporal_rmdm.packed_data import PACKED_SCHEMA, PackedFrameReader, _sha256
from experimental.noise_temporal_rmdm.step import training_step
from experimental.noise_temporal_rmdm.validation import validation_video_ids
from rmdm.data import SamplingPolicy, WindowDataset
from rmdm.diffusion import DDIMSampler, DiffusionProcess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def tiny_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.image_size = 16
    config.data.workers = 0
    config.model.model_channels = 32
    config.model.channel_multipliers = [1, 1]
    config.model.residual_blocks_per_level = 1
    config.model.attention_levels = [1]
    config.model.attention_heads = 4
    config.model.hwm_base_features = 8
    config.model.hwm_channel_multipliers = [1, 2]
    config.model.hwm_blocks_per_level = 1
    config.model.variance_embedding_dim = 32
    config.model.variance_mlp_width = 32
    config.model.gradient_checkpointing = False
    config.model.expected_trainable_parameters_min = 0
    config.model.expected_trainable_parameters_max = 10_000_000
    return config


def dense_batch(batch: int = 2, size: int = 8) -> dict:
    shape = (batch, 1, 1, size, size)
    return {
        "building": torch.zeros(shape),
        "vehicle": torch.zeros(shape),
        "target": torch.linspace(0.1, 0.9, steps=torch.tensor(shape).prod().item()).reshape(shape),
        "video_id": [f"scene/video_{index}" for index in range(batch)],
        "start": torch.arange(batch),
        "frame_names": [[f"frame_{index}.png"] for index in range(batch)],
    }


def test_video_block_sampler_is_complete_deterministic_and_cache_friendly() -> None:
    sampler = VideoBlockShuffleSampler(size=10, frames_per_video=4, seed=17)
    first = list(sampler)
    assert sorted(first) == list(range(10))
    assert first == list(VideoBlockShuffleSampler(size=10, frames_per_video=4, seed=17))
    blocks = [index // 4 for index in first]
    assert sum(left != right for left, right in zip(blocks, blocks[1:])) == 2

    sampler.set_epoch(1)
    second = list(sampler)
    assert sorted(second) == list(range(10))
    assert second != first


def test_packed_reader_preserves_values_and_frame_identity(tmp_path: Path) -> None:
    split_file = tmp_path / "split.json"
    split_file.write_text('{"train": ["scene"]}\n', encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "building_uint8.npy", np.array([[[0, 255], [255, 0]]], dtype=np.uint8))
    np.save(cache / "vehicle_uint8.npy", np.array([[[0, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8))
    np.save(cache / "target_uint8.npy", np.array([[[0, 51], [102, 153]], [[255, 204], [153, 102]]], dtype=np.uint8))
    metadata = {
        "schema": PACKED_SCHEMA,
        "split_sha256": _sha256(split_file),
        "frames_per_video": 2,
        "image_size": 2,
        "records": [{"index": 0, "scene_id": "scene", "episode_id": "episode", "tx_id": "tx"}],
    }
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    reader = PackedFrameReader(cache, split_file=split_file)
    result = reader.read_window(reader.records[0], 1, 1)
    assert result["building"].tolist() == [[[0.0, 1.0], [1.0, 0.0]]]
    assert result["vehicle"].tolist() == [[[1.0, 0.0], [0.0, 0.0]]]
    assert np.allclose(result["target"], [[[1.0, 0.8], [0.6, 0.4]]])
    assert result["frame_names"] == ["scene/episode/tx/frame_000001.png"]


def condition_batch(size: int = 16) -> dict:
    shape = (1, 1, 1, size, size)
    mask = (torch.rand(shape) < 0.1).float()
    return {
        "building": (torch.rand(shape) < 0.2).float(),
        "vehicle": (torch.rand(shape) < 0.05).float(),
        "observed_rss": mask * torch.rand(shape),
        "sampling_mask": mask,
        "measurement_variance": torch.tensor([0.03**2]),
    }


def test_clean16_split_is_disjoint_complete_and_sized() -> None:
    split = json.loads(
        (REPOSITORY_ROOT / "configs/splits/m20_formal075_clean16_scene_split.json").read_text()
    )
    groups = [set(split[name]) for name in ("train", "val", "test", "excluded")]
    assert [len(group) for group in groups] == [12, 2, 2, 4]
    assert sum(len(group) for group in groups) == len(set().union(*groups)) == 20
    index_path = Path(
        "/data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/"
        "M20_Formal075_RadioMapSeerPack/index.json"
    )
    source = json.loads(index_path.read_text())
    counts = {}
    for item in source["samples"]:
        counts[item["scene_id"]] = counts.get(item["scene_id"], 0) + 1
    assert set(counts) == set().union(*groups)
    assert sum(counts[scene] for scene in split["train"]) == 9000
    assert sum(counts[scene] for scene in split["val"]) == 1500
    assert sum(counts[scene] for scene in split["test"]) == 1500
    assert sum(counts[scene] for scene in split["excluded"]) == 3000


def test_validation_manifest_uses_only_clean16_val_scenes() -> None:
    manifest = REPOSITORY_ROOT / "configs/manifests/noise_temporal_rmdm_clean16_val.json"
    selected = validation_video_ids(
        manifest,
        included_scenes=["town02_opt_junction_0298", "town10_junction_0532"],
        excluded_scenes=[
            "town01_opt_junction_0087", "town05_opt_junction_0053",
            "town05_opt_junction_0838", "town05_opt_junction_1427",
        ],
    )
    assert len(selected) == 20
    assert {value.split("/", 1)[0] for value in selected} == {
        "town02_opt_junction_0298", "town10_junction_0532"
    }


def test_final_test_manifest_is_disjoint_and_report_only() -> None:
    config = ExperimentConfig()
    manifest = REPOSITORY_ROOT / "configs/manifests/noise_temporal_rmdm_clean16_test.json"
    selected = validation_video_ids(
        manifest,
        included_scenes=config.final_test.included_scenes,
        excluded_scenes=config.final_test.excluded_scenes,
    )
    assert len(selected) == 20
    assert {value.split("/", 1)[0] for value in selected} == {
        "town04_opt_junction_0053", "town05_opt_junction_0396"
    }
    assert set(config.validation.included_scenes).isdisjoint(config.final_test.included_scenes)
    assert config.validation.ddim_steps == config.final_test.ddim_steps == 20


def test_measurement_noise_is_deterministic_and_masked() -> None:
    batch = dense_batch()
    mask = torch.zeros_like(batch["target"])
    mask[..., 1, 2] = 1
    sparse = {**batch, "sampling_mask": mask}
    config = MeasurementNoiseConfig()
    first = add_measurement_noise(sparse, config, epoch=3)
    second = add_measurement_noise(sparse, config, epoch=3)
    assert torch.equal(first["observed_rss"], second["observed_rss"])
    assert torch.equal(first["measurement_standard_deviation"], second["measurement_standard_deviation"])
    assert torch.count_nonzero(first["observed_rss"] * (1 - mask)) == 0
    assert torch.equal(
        first["measurement_variance"], first["measurement_standard_deviation"].square()
    )


def test_noise_mixture_probabilities_and_ranges() -> None:
    batch = dense_batch(batch=2000, size=1)
    sparse = {**batch, "sampling_mask": torch.ones_like(batch["target"])}
    result = add_measurement_noise(sparse, MeasurementNoiseConfig(), epoch=0)
    components = result["measurement_noise_component"]
    sigma = result["measurement_standard_deviation"]
    fractions = {name: components.count(name) / len(components) for name in set(components)}
    assert fractions["clean"] == pytest.approx(0.20, abs=0.035)
    assert fractions["nominal"] == pytest.approx(0.65, abs=0.04)
    assert fractions["strong"] == pytest.approx(0.15, abs=0.03)
    for index, component in enumerate(components):
        if component == "clean":
            assert sigma[index] == 0
        elif component == "nominal":
            assert 0 <= sigma[index] <= 0.05
        else:
            assert 0.05 <= sigma[index] <= 0.09


def test_sigma_zero_is_exact_and_fixed_noise_is_deterministic() -> None:
    batch = dense_batch()
    mask = torch.ones_like(batch["target"])
    sparse = {**batch, "sampling_mask": mask}
    clean = add_fixed_measurement_noise(sparse, 0.0, seed=5)
    assert torch.equal(clean["observed_rss"], batch["target"])
    first = add_fixed_measurement_noise(sparse, 0.07, seed=5)
    second = add_fixed_measurement_noise(sparse, 0.07, seed=5)
    assert torch.equal(first["observed_rss"], second["observed_rss"])
    assert torch.equal(first["measurement_variance"], torch.full((2,), 0.07**2))


def test_tx_is_absent_from_dataset_and_irrelevant_to_model() -> None:
    class Record:
        video_id = "scene/episode/tx"

    class Reader:
        records = [Record()]

        @staticmethod
        def frame_count(record) -> int:
            return 1

        @staticmethod
        def read_window(record, start: int, length: int) -> dict:
            import numpy as np
            value = np.zeros((1, 4, 4), dtype=np.float32)
            return {"building": value, "vehicle": value, "target": value, "frame_names": ["frame.png"]}

    assert "tx" not in WindowDataset(window_size=1, reader=Reader())[0]

    model = build_model(tiny_config()).eval()
    conditions = condition_batch()
    noisy = torch.randn(1, 1, 1, 16, 16)
    timestep = torch.tensor([10])
    with torch.no_grad():
        baseline = model(noisy, timestep, conditions)
        poisoned = model(noisy, timestep, {**conditions, "tx": torch.randn_like(noisy) * 1e6})
    assert torch.equal(baseline[0], poisoned[0])
    assert torch.equal(baseline[1], poisoned[1])


def test_variance_conditions_hwm_and_receives_gradient() -> None:
    model = build_model(tiny_config())
    conditions = condition_batch()
    variance = torch.tensor([0.02**2], requires_grad=True)
    conditions["measurement_variance"] = variance
    cal = model.encode_conditions(conditions)["cal"]
    cal.mean().backward()
    assert variance.grad is not None
    assert torch.isfinite(variance.grad).all()
    assert torch.count_nonzero(variance.grad) > 0


def test_forward_backward_shapes_without_tx() -> None:
    model = build_model(tiny_config())
    conditions = condition_batch()
    prediction, cal = model(torch.randn(1, 1, 1, 16, 16), torch.tensor([500]), conditions)
    assert prediction.shape == cal.shape == (1, 1, 1, 16, 16)
    (prediction.square().mean() + cal.square().mean()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_short_ddim_path_is_finite() -> None:
    config = tiny_config()
    model = build_model(config).eval()
    conditions = condition_batch()
    with torch.no_grad():
        generated = DDIMSampler(config.diffusion).sample(
            model, conditions, initial_noise=torch.randn(1, 1, 1, 16, 16), steps=2
        )
    assert generated.shape == (1, 1, 1, 16, 16)
    assert torch.isfinite(generated).all()
    assert generated.min() >= 0 and generated.max() <= 1


def test_training_step_has_no_tx_or_source_requirement() -> None:
    config = tiny_config()
    config.loss.pinn_weight = 1.0
    dense = dense_batch(batch=1, size=8)

    class StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))

        def forward(self, noisy, timesteps, sparse):
            assert "tx" not in sparse
            return torch.zeros_like(noisy) + self.value, torch.sigmoid(torch.zeros_like(noisy) + self.value)

    model = StubModel()
    result = training_step(
        model, dense, SamplingPolicy(config.sampling), DiffusionProcess(config.diffusion), config, epoch=0
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.value.grad is not None


def test_checkpoint_schema_is_strict(tmp_path: Path) -> None:
    config = tiny_config()
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = build_payload(
        model, optimizer, scheduler, config, global_step=2, epoch=1,
        microbatches_consumed_in_epoch=3,
        source_provenance={"commit": "abc"},
    )
    assert payload["schema"] == T1_SCHEMA
    assert payload["architecture_id"] == ARCHITECTURE_ID
    assert payload["source"]["commit"] == "abc"
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    loaded = load(path, model, optimizer, scheduler)
    assert loaded["global_step"] == 2
    payload["architecture_id"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema or architecture"):
        load(path, model)
