from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch import nn

from experimental.tx_prior.adapter import ScenePriorSystem, scene_prior_batch
from experimental.tx_prior.data import deterministic_prior_noise_like, make_prior_training_batch
from experimental.tx_prior.runner import allow_incomplete_restart


class FakeHWM(nn.Module):
    def forward(self, batch):
        scene = batch["building"] + 2 * batch["tx"] + 3 * batch["vehicle"]
        observation = 5 * batch["observed_rss"] + 7 * batch["sampling_mask"]
        return {"hwm_gate": scene + observation, "cal": scene + observation}


class FakeDenoiser(nn.Module):
    def encode_raw_conditions(self, raw):
        scene = raw["building"] + 11 * raw["tx"] + 13 * raw["vehicle"]
        observation = 17 * raw["observed_rss"] + 19 * raw["sampling_mask"]
        high = scene + observation
        return high, 2 * high

    def forward(self, noisy_target, diffusion_step, cache):
        observation = 23 * cache["observed_rss"] + 29 * cache["sampling_mask"]
        return noisy_target + cache["hwm_gate"] + cache["condition_high"] + observation


def batch(*, tx_value=1.0, observed_value=0.0, mask_value=0.0):
    shape = (2, 1, 1, 4, 4)
    return {
        "building": torch.ones(shape),
        "tx": torch.full(shape, tx_value),
        "vehicle": torch.full(shape, 2.0),
        "observed_rss": torch.full(shape, observed_value),
        "sampling_mask": torch.full(shape, mask_value),
    }


def test_scene_prior_batch_does_not_mutate_input():
    original = batch(observed_value=4, mask_value=1)
    sanitized = scene_prior_batch(original)
    assert torch.count_nonzero(sanitized["observed_rss"]) == 0
    assert torch.count_nonzero(sanitized["sampling_mask"]) == 0
    assert torch.count_nonzero(original["observed_rss"]) > 0


def test_observations_cannot_change_cache_or_output():
    model = ScenePriorSystem(FakeHWM(), FakeDenoiser())
    first = batch(observed_value=-3, mask_value=0)
    second = batch(observed_value=91, mask_value=1)
    first_cache = model.encode_conditions(first)
    second_cache = model.encode_conditions(second)
    for key in ("observed_rss", "sampling_mask", "hwm_gate", "cal", "condition_high", "condition_low"):
        assert torch.equal(first_cache[key], second_cache[key]), key
    noisy = torch.randn_like(first["building"])
    timestep = torch.tensor([3, 9])
    assert torch.equal(model(noisy, timestep, first)[0], model(noisy, timestep, second)[0])
    # Even a caller-modified cache is sanitized immediately before the input stem.
    second_cache["observed_rss"].fill_(100)
    second_cache["sampling_mask"].fill_(1)
    assert torch.equal(model.denoise(noisy, timestep, first_cache), model.denoise(noisy, timestep, second_cache))


def test_tx_remains_an_effective_condition():
    model = ScenePriorSystem(FakeHWM(), FakeDenoiser())
    without_tx = model.encode_conditions(batch(tx_value=0))
    with_tx = model.encode_conditions(batch(tx_value=1))
    assert not torch.equal(without_tx["hwm_gate"], with_tx["hwm_gate"])
    assert not torch.equal(without_tx["condition_high"], with_tx["condition_high"])


def test_training_batch_is_built_without_sampling():
    dense = batch(observed_value=9, mask_value=1)
    prior = make_prior_training_batch(dense)
    assert torch.count_nonzero(prior["observed_rss"]) == 0
    assert torch.count_nonzero(prior["sampling_mask"]) == 0
    assert torch.count_nonzero(prior["sampling_rate"]) == 0
    assert prior["sampling_mode"] == ["scene_prior", "scene_prior"]


def test_prior_noise_has_no_rate_input_and_is_frame_deterministic():
    target = torch.zeros(1, 1, 1, 4, 4)
    names = [["scene/frame.png"]]
    first = deterministic_prior_noise_like(target, names, seed=7)
    second = deterministic_prior_noise_like(target, names, seed=7)
    changed_seed = deterministic_prior_noise_like(target, names, seed=8)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed_seed)


def test_train_path_does_not_import_or_instantiate_sampling_policy():
    source = Path(__file__).with_name("runner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SamplingPolicy" not in imported
    assert "SamplingPolicy" not in called


def test_configs_keep_single_output_and_formal_machine_paths():
    import yaml

    directory = Path(__file__).parent
    formal = yaml.safe_load((directory / "train.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load((directory / "smoke.yaml").read_text(encoding="utf-8"))
    assert formal["pipeline"]["output_root"] == smoke["pipeline"]["output_root"] == "runs/tx_prior"
    assert formal["pipeline"]["allowed_physical_gpus"] == [4, 5, 6, 7]
    assert formal["t1_train"]["max_steps"] == 10_000
    assert Path(formal["data"]["root"]).is_dir()
    assert Path(formal["data"]["split_file"]).is_file()
    assert Path(formal["evaluation"]["subset_manifest"]).is_file()


def test_runner_has_exact_eval_and_stage_output_contracts():
    source = Path(__file__).with_name("runner.py").read_text(encoding="utf-8")
    assert 'accelerator.even_batches = False' in source
    assert '"smoke" if smoke else "train"' in source
    assert 'expected_frames = len(video_ids) * config.data.frames_per_video' in source
    assert 'epoch_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset' in source
    assert "accelerator.wait_for_everyone()\n    accelerator.end_training()" in source


def test_incomplete_restart_accepts_only_step_zero_without_checkpoint(tmp_path):
    stage = tmp_path / "smoke"
    stage.mkdir()
    assert allow_incomplete_restart(stage)[0] is False
    (stage / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (stage / "status.json").write_text(
        '{"state":"failed","global_step":0}\n', encoding="utf-8"
    )
    assert allow_incomplete_restart(stage)[0] is True
    (stage / "status.json").write_text(
        '{"state":"failed","global_step":1}\n', encoding="utf-8"
    )
    assert allow_incomplete_restart(stage)[0] is False
    (stage / "status.json").write_text(
        '{"state":"training","global_step":0}\n', encoding="utf-8"
    )
    checkpoint = stage / "checkpoints/last.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert allow_incomplete_restart(stage)[0] is False
