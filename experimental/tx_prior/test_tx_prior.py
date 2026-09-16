from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experimental.tx_prior.adapter import (
    ScenePriorSystem,
    T1SceneStem,
    build_scene_prior_system,
    scene_prior_batch,
)
from experimental.tx_prior.checkpoint import SCHEMA, restore_cuda_rng
from experimental.tx_prior.data import (
    deterministic_prior_noise_like,
    make_prior_training_batch,
)
from experimental.tx_prior.runner import (
    allow_incomplete_restart,
    initial_early_stop_state,
    resolve_task_root,
    resume_microbatch_offset,
    update_early_stop_state,
    validation_due,
    validation_result_path,
    write_step_validation,
)
from experimental.tx_prior.packed import (
    FORMAT,
    MARKER,
    PackedFrameReader,
    build_cache,
    remove_cache,
    verify_cache,
)
from experimental.tx_prior.config import load_config
from experimental.tx_prior.step import training_step
from rmdm.diffusion.process import DiffusionTrainingBatch
from rmdm.legacy import LegacyVideoRecord
from rmdm_hvdit_v4_joint.model.hwm import TrainableHWM


class FakeHWM(nn.Module):
    def forward(self, batch):
        scene = batch["building"] + 2 * batch["tx"] + 3 * batch["vehicle"]
        return {"hwm_gate": scene, "cal": scene}


class FakeDenoiser(nn.Module):
    def encode_raw_conditions(self, raw):
        scene = raw["building"] + 11 * raw["tx"] + 13 * raw["vehicle"]
        return scene, 2 * scene

    def forward(self, noisy_target, diffusion_step, cache):
        return noisy_target + cache["hwm_gate"] + cache["condition_high"]


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
    assert "observed_rss" not in first_cache
    assert "sampling_mask" not in first_cache
    for key in ("hwm_gate", "cal", "condition_high", "condition_low"):
        assert torch.equal(first_cache[key], second_cache[key]), key
    noisy = torch.randn_like(first["building"])
    timestep = torch.tensor([3, 9])
    assert torch.equal(
        model(noisy, timestep, first)[0], model(noisy, timestep, second)[0]
    )


def test_tx_remains_an_effective_condition():
    model = ScenePriorSystem(FakeHWM(), FakeDenoiser())
    without_tx = model.encode_conditions(batch(tx_value=0))
    with_tx = model.encode_conditions(batch(tx_value=1))
    assert not torch.equal(without_tx["hwm_gate"], with_tx["hwm_gate"])
    assert not torch.equal(without_tx["condition_high"], with_tx["condition_high"])


def test_scene_stem_has_no_observation_projection_or_fusion():
    stem = T1SceneStem(dense_channels=3, dim=8, patch_size=2)
    assert set(dict(stem.named_parameters())) == {"dense_projection.weight"}
    dense = torch.randn(2, 1, 3, 4, 4)
    first = stem(dense, torch.randn(2, 1, 2, 4, 4))
    second = stem(dense, torch.randn(2, 1, 2, 4, 4))
    assert torch.equal(first, second)
    assert first.shape == (2, 1, 2, 2, 8)


def test_scene_hwm_uses_exactly_three_channels():
    hwm = TrainableHWM(nn.Identity(), chunk_size=1, input_channels=3)
    first = hwm.conditions(batch(observed_value=-3, mask_value=0))
    second = hwm.conditions(batch(observed_value=91, mask_value=1))
    assert first.shape[2] == 3
    assert torch.equal(first, second)


def test_built_prior_has_no_observation_stem_and_three_channel_hwm():
    config = load_config(Path(__file__).with_name("smoke.yaml"), smoke=True)
    model = build_scene_prior_system(config, attention_backend="torch")
    denoiser_names = set(dict(model.denoiser.named_parameters()))
    assert not any("observation_projection" in name for name in denoiser_names)
    assert not any("input_stem.fusion" in name for name in denoiser_names)
    assert not any("condition_stem.fusion" in name for name in denoiser_names)
    first_hwm_conv = model.hwm.hwm.conv_blocks_context[0].blocks[0].conv
    assert isinstance(first_hwm_conv, nn.Conv2d)
    assert model.hwm.input_channels == 3
    assert first_hwm_conv.in_channels == 3


def test_training_step_regresses_true_epsilon_not_x0():
    shape = (2, 1, 1, 2, 2)
    dense = {
        **batch(observed_value=0, mask_value=0),
        "target": torch.full(shape, 17.0),
        "sampling_rate": torch.zeros(2),
        "video_id": ["a", "b"],
        "start": torch.tensor([0, 1]),
    }
    # Match all condition shapes to this test's compact target.
    for key in ("building", "tx", "vehicle", "observed_rss", "sampling_mask"):
        dense[key] = dense[key][..., :2, :2]

    class FakeDiffusion:
        def training_batch(self, target, *, seeds):
            assert len(seeds) == target.shape[0]
            return DiffusionTrainingBatch(
                noisy_target=torch.zeros_like(target),
                noise=torch.full_like(target, 2.0),
                timesteps=torch.tensor([3, 5]),
            )

        def predict_x0(self, noisy_target, predicted_noise, timesteps):
            return torch.zeros_like(noisy_target)

    class EpsilonModel(nn.Module):
        def forward(self, noisy_target, timesteps, sparse_batch):
            return torch.full_like(noisy_target, 3.0), torch.zeros_like(noisy_target)

    result = training_step(
        EpsilonModel(),
        dense,
        lambda value: value,
        FakeDiffusion(),
        training_seed=7,
        epoch=0,
        pinn_k=1.0,
        pinn_weight=0.0,
    )
    # (predicted epsilon 3 - true epsilon 2)^2 == 1. If this regressed x0=17,
    # the value would instead be 196.
    assert result.diffusion_loss.item() == pytest.approx(1.0)
    assert torch.allclose(result.epsilon_mse_per_sample, torch.ones(2))


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
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SamplingPolicy" not in imported
    assert "SamplingPolicy" not in called


def test_configs_keep_single_output_and_formal_machine_paths():
    import yaml

    directory = Path(__file__).parent
    formal = yaml.safe_load((directory / "train.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load((directory / "smoke.yaml").read_text(encoding="utf-8"))
    remote = yaml.safe_load((directory / "remote.yaml").read_text(encoding="utf-8"))
    assert (
        formal["pipeline"]["output_root"]
        == smoke["pipeline"]["output_root"]
        == remote["pipeline"]["output_root"]
        == "runs/tx_prior"
    )
    assert formal["pipeline"]["allowed_physical_gpus"] == [4, 5, 6, 7]
    assert formal["t1_train"]["max_steps"] == 80_000
    assert formal["t1_train"]["validation_every_steps"] == 5_000
    assert formal["t1_train"]["early_stop_min_step"] == 25_000
    assert formal["t1_train"]["patience_validations"] == 2
    assert formal["data"]["root"].startswith("/data_p6/")
    assert formal["data"]["split_file"].startswith("/data_p6/")
    assert formal["evaluation"]["subset_manifest"].startswith("/data_p6/")
    assert remote["data"]["root"].startswith("/data_16T_137/")
    assert remote["data"]["split_file"].startswith("/data_16T_137/")
    assert remote["evaluation"]["subset_manifest"].startswith("/data_16T_137/")
    assert remote["pipeline"]["allowed_physical_gpus"] == [0, 1]
    assert remote["t1_train"]["per_gpu_batch_size"] == 64
    assert remote["t1_train"]["gradient_accumulation_steps"] == 2
    assert remote["t1_train"]["effective_global_batch_size"] == 256
    assert remote["t1_train"]["max_steps"] == 80_000
    assert formal["diffusion"]["prediction_type"] == "epsilon"
    assert smoke["diffusion"]["prediction_type"] == "epsilon"
    assert remote["diffusion"]["prediction_type"] == "epsilon"


def test_early_stop_improvement_and_patience_after_minimum_step():
    class Train:
        max_steps = 80_000
        validation_first_step = 10_000
        validation_every_steps = 5_000
        early_stop_min_step = 25_000
        patience_validations = 2

    state = initial_early_stop_state()
    assert update_early_stop_state(
        state, score=0.4, step=10_000, early_stop_min_step=25_000
    )
    assert state == {"best_score": 0.4, "best_step": 10_000, "stale_validations": 0}
    assert not update_early_stop_state(
        state, score=0.5, step=15_000, early_stop_min_step=25_000
    )
    assert not update_early_stop_state(
        state, score=0.4, step=20_000, early_stop_min_step=25_000
    )
    assert state["stale_validations"] == 0
    assert not (20_000 >= Train.early_stop_min_step)
    assert not update_early_stop_state(
        state, score=0.45, step=25_000, early_stop_min_step=25_000
    )
    assert state["stale_validations"] == 1
    assert not update_early_stop_state(
        state, score=0.45, step=30_000, early_stop_min_step=25_000
    )
    assert state["stale_validations"] == 2
    assert validation_due(Train, 10_000)
    assert validation_due(Train, 25_000)
    assert validation_due(Train, 80_000)
    assert not validation_due(Train, 12_000)


def test_old_checkpoint_early_stop_state_is_safe_and_new_state_restores():
    assert initial_early_stop_state({"schema": "tx_prior_w1_checkpoint_v1"}) == {
        "best_score": float("inf"), "best_step": 0, "stale_validations": 0
    }
    payload = {"early_stop": {"best_score": 0.25, "best_step": 15_000,
                               "stale_validations": 1}}
    assert initial_early_stop_state(payload) == payload["early_stop"]


def test_checkpoint_schema_marks_epsilon_scene_architecture():
    assert SCHEMA == "tx_prior_epsilon_scene_checkpoint_v1"


def test_step_validation_path_and_foreign_step_protection(tmp_path):
    path = validation_result_path(tmp_path, 10_000)
    assert path.name == "step_010000.json"
    write_step_validation(path, {"metrics": {"full_image": {"nmse": 0.3}}}, 10_000)
    assert json.loads(path.read_text())["global_step"] == 10_000
    write_step_validation(path, {"metrics": {"full_image": {"nmse": 0.2}}}, 10_000)
    path.write_text('{"global_step":15000}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="step mismatch"):
        write_step_validation(path, {}, 10_000)


def test_runner_has_exact_eval_and_stage_output_contracts():
    source = Path(__file__).with_name("runner.py").read_text(encoding="utf-8")
    assert "accelerator.even_batches = False" in source
    assert '"smoke" if smoke else "train"' in source
    assert "expected_frames = len(video_ids) * config.data.frames_per_video" in source
    assert (
        "epoch_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset"
        in source
    )
    assert "accelerator.wait_for_everyone()\n    accelerator.end_training()" in source
    assert (
        "if not accelerator.sync_gradients:\n"
        "                fetch_started = time.perf_counter()\n"
        "                continue"
    ) in source
    validation_tail = source.index("# Validation and checkpoint I/O")
    assert source.index("last_time = time.perf_counter()", validation_tail) > validation_tail
    assert source.index("last_timed_step = global_step", validation_tail) > validation_tail
    assert source.index("data_wait_seconds = 0.0", validation_tail) > validation_tail
    assert source.index("fetch_started = last_time", validation_tail) > validation_tail


def test_task_root_allows_only_fixed_lexical_path_and_task_symlink(tmp_path):
    repository = tmp_path / "project"
    external = tmp_path / "results" / "tx_prior"
    (repository / "runs").mkdir(parents=True)
    external.mkdir(parents=True)
    (repository / "runs" / "tx_prior").symlink_to(external, target_is_directory=True)
    assert resolve_task_root(repository, "runs/tx_prior") == external.resolve()
    for invalid in ("runs/other", "../runs/tx_prior", str(external)):
        with pytest.raises(ValueError, match="exactly runs/tx_prior"):
            resolve_task_root(repository, invalid)


def test_resume_cursor_converts_four_gpu_accum1_to_two_gpu_accum2_exactly():
    class Train:
        per_gpu_batch_size = 64
        gradient_accumulation_steps = 2
        effective_global_batch_size = 256

    class Pipeline:
        allowed_physical_gpus = [0, 1]

    class Config:
        t1_train = Train()
        pipeline = Pipeline()

    payload = {
        "microbatches_consumed_in_epoch": 1798,
        "resolved_config": {
            "t1_train": {
                "per_gpu_batch_size": 64,
                "gradient_accumulation_steps": 1,
                "effective_global_batch_size": 256,
            },
            "pipeline": {"allowed_physical_gpus": [4, 5, 6, 7]},
        },
    }
    assert resume_microbatch_offset(payload, Config()) == 3596


def test_resume_cursor_rejects_global_batch_change_or_inexact_cursor():
    class Train:
        per_gpu_batch_size = 48
        gradient_accumulation_steps = 2
        effective_global_batch_size = 192

    class Pipeline:
        allowed_physical_gpus = [0, 1]

    class Config:
        t1_train = Train()
        pipeline = Pipeline()

    payload = {
        "microbatches_consumed_in_epoch": 1,
        "resolved_config": {
            "t1_train": {
                "per_gpu_batch_size": 64,
                "gradient_accumulation_steps": 1,
                "effective_global_batch_size": 256,
            },
            "pipeline": {"allowed_physical_gpus": [4, 5, 6, 7]},
        },
    }
    with pytest.raises(ValueError, match="effective global batch"):
        resume_microbatch_offset(payload, Config())


def test_cuda_rng_restore_is_full_only_when_visible_device_counts_match(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: calls.append(("all", len(states))))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: calls.append(("one", device)))
    states = [torch.zeros(4, dtype=torch.uint8) for _ in range(2)]
    result = restore_cuda_rng(states)
    assert result == {
        "state": "full",
        "saved_visible_devices": 2,
        "current_visible_devices": 2,
        "restored_devices": 2,
    }
    assert calls == [("all", 2)]


def test_cuda_rng_restore_marks_cross_world_size_as_partial(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: calls.append(("all", len(states))))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: calls.append(("one", device)))
    states = [torch.zeros(4, dtype=torch.uint8) for _ in range(4)]
    result = restore_cuda_rng(states)
    assert result == {
        "state": "partial_due_to_world_size_change",
        "saved_visible_devices": 4,
        "current_visible_devices": 2,
        "restored_devices": 2,
    }
    assert calls == [("one", 0), ("one", 1)]


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


def _write_packed_fixture(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    root = tmp_path / "packed"
    root.mkdir()
    (root / MARKER).write_text(FORMAT + "\n", encoding="utf-8")
    split_file = tmp_path / "split.json"
    split_file.write_text("{}", encoding="utf-8")
    target_u8 = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    vehicle_u8 = (target_u8 > 8).astype(np.uint8)
    building = np.ones((4, 4), dtype=np.float32)
    tx = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    np.save(root / "targets.npy", target_u8[None])
    np.save(root / "vehicles.npy", vehicle_u8[None])
    np.save(root / "tx.npy", tx[None])
    np.save(root / "buildings.npy", building[None])
    np.save(root / "frame_ids.npy", np.array([[3, 7]], dtype=np.int32))
    names = ["s1/e1/t1/frame_000003.png", "s1/e1/t1/frame_000007.png"]
    manifest = {
        "format": FORMAT,
        "state": "complete",
        "source_root": str(source),
        "split_file": str(split_file),
        "tx_heatmap_sigma_px": 1.5,
        "split": "train",
        "videos": 1,
        "episodes": 1,
        "scenes": 1,
        "frames_per_video": 2,
        "shape": [2, 4, 4],
        "target_dtype": "uint8",
        "vehicle_dtype": "uint8",
        "tx_dtype": "float32",
        "building_dtype": "float32",
        "frame_id_dtype": "int32",
        "records": [
            {
                "scene_id": "s1",
                "episode_id": "e1",
                "tx_id": "t1",
                "scene_index": 0,
                "episode_index": 0,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, source, split_file, manifest, target_u8, vehicle_u8, building, tx


def test_packed_reader_preserves_legacy_w1_contract(tmp_path):
    root, source, split_file, _, target_u8, vehicle_u8, building, tx = (
        _write_packed_fixture(tmp_path)
    )
    reader = PackedFrameReader(
        root,
        source_root=str(source),
        split_file=str(split_file),
        tx_heatmap_sigma_px=1.5,
    )
    window = reader.read_window(reader.records[0], 0, 2)
    assert set(window) == {"building", "tx", "vehicle", "target", "frame_names"}
    np.testing.assert_array_equal(
        window["target"], target_u8.astype(np.float32) / 255.0
    )
    np.testing.assert_array_equal(window["vehicle"], vehicle_u8.astype(np.float32))
    np.testing.assert_array_equal(
        window["building"], np.broadcast_to(building, (2, 4, 4))
    )
    np.testing.assert_array_equal(window["tx"], np.broadcast_to(tx, (2, 4, 4)))
    assert window["frame_names"] == [
        "s1/e1/t1/frame_000003.png",
        "s1/e1/t1/frame_000007.png",
    ]
    for name in ("building", "tx", "vehicle", "target"):
        assert window[name].shape == (2, 4, 4)
        assert window[name].dtype == np.float32


@pytest.mark.parametrize(
    ("change", "kwargs"),
    [
        ({"state": "building"}, {}),
        ({"split": "validation"}, {}),
        ({"tx_heatmap_sigma_px": 2.0}, {}),
        ({"shape": [2, 5, 4]}, {}),
        ({"target_dtype": "float32"}, {}),
        ({"records": [{"scene_id": "s1", "episode_id": "e1", "tx_id": "t1", "scene_index": 1, "episode_index": 0}]}, {}),
        ({}, {"source_root": "different"}),
        ({}, {"split_file": "different.json"}),
    ],
)
def test_packed_reader_strictly_rejects_bad_cache(tmp_path, change, kwargs):
    root, source, split_file, manifest, *_ = _write_packed_fixture(tmp_path)
    manifest.update(change)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        PackedFrameReader(
            root,
            source_root=kwargs.get("source_root", source),
            split_file=kwargs.get("split_file", split_file),
            tx_heatmap_sigma_px=1.5,
        )


def test_packed_reader_rejects_missing_marker_and_array(tmp_path):
    root, source, split_file, *_ = _write_packed_fixture(tmp_path)
    (root / MARKER).unlink()
    with pytest.raises(ValueError, match="marker"):
        PackedFrameReader(
            root,
            source_root=source,
            split_file=split_file,
            tx_heatmap_sigma_px=1.5,
        )


def test_packed_reader_rejects_actual_array_dtype(tmp_path):
    root, source, split_file, *_ = _write_packed_fixture(tmp_path)
    np.save(root / "targets.npy", np.zeros((1, 2, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="targets dtype"):
        PackedFrameReader(
            root,
            source_root=source,
            split_file=split_file,
            tx_heatmap_sigma_px=1.5,
        )
    (root / MARKER).write_text(FORMAT + "\n", encoding="utf-8")
    (root / "targets.npy").unlink()
    with pytest.raises(ValueError, match="targets"):
        PackedFrameReader(
            root,
            source_root=source,
            split_file=split_file,
            tx_heatmap_sigma_px=1.5,
        )


def test_remove_cache_only_removes_marked_cache_paths(tmp_path):
    root = tmp_path / "packed"
    building = tmp_path / "packed.building"
    root.mkdir()
    building.mkdir()
    (root / MARKER).write_text(FORMAT + "\n", encoding="utf-8")
    (building / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unmarked"):
        remove_cache(root, confirm=True)
    assert root.is_dir()
    assert (building / "user-file").is_file()
    (building / MARKER).write_text(FORMAT + "\n", encoding="utf-8")
    remove_cache(root, confirm=True)
    assert not root.exists()
    assert not building.exists()


def test_remove_cache_requires_confirmation(tmp_path):
    with pytest.raises(ValueError, match="confirmation"):
        remove_cache(tmp_path / "packed")


def test_packed_public_cli_has_no_partial_limit_option():
    source = Path(__file__).with_name("packed_cli.py").read_text(encoding="utf-8")
    assert '"--limit"' not in source


def test_build_round_trip_matches_legacy_w1(monkeypatch, tmp_path):
    frames, height, width = 100, 2, 3
    target = np.arange(frames * height * width, dtype=np.uint16).reshape(
        frames, height, width
    ) % 256
    legacy_window = {
        "target": target.astype(np.float32) / 255.0,
        "vehicle": (target % 3 == 0).astype(np.float32),
        "building": np.broadcast_to(
            np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32),
            (frames, height, width),
        ).copy(),
        "tx": np.broadcast_to(
            np.linspace(0, 1, height * width, dtype=np.float32).reshape(
                height, width
            ),
            (frames, height, width),
        ).copy(),
        "frame_names": [f"s/e/t/frame_{index:06d}.png" for index in range(frames)],
    }

    class FakeLegacyReader:
        def __init__(self, *args, **kwargs):
            self.records = [LegacyVideoRecord(0, "s", "e", "t")]

        def frame_count(self, record):
            return frames

        def read_window(self, record, start, length):
            stop = start + length
            return {
                key: value[start:stop] if key != "frame_names" else value[start:stop]
                for key, value in legacy_window.items()
            }

    monkeypatch.setattr(
        "experimental.tx_prior.packed.LegacyFrameReader", FakeLegacyReader
    )
    source = tmp_path / "source"
    source.mkdir()
    split_file = tmp_path / "split.json"
    split_file.write_text("{}", encoding="utf-8")
    root = tmp_path / "cache"
    progress = []
    build_cache(
        root=root,
        source_root=source,
        split_file=split_file,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(1, 1)]
    assert root.is_dir()
    assert not Path(str(root) + ".building").exists()
    reader = PackedFrameReader(
        root,
        source_root=source,
        split_file=split_file,
        tx_heatmap_sigma_px=1.5,
    )
    packed = reader.read_window(reader.records[0], 0, frames)
    assert set(packed) == set(legacy_window)
    for name in ("target", "vehicle", "building", "tx"):
        np.testing.assert_array_equal(packed[name], legacy_window[name])
        assert packed[name].shape == legacy_window[name].shape
        assert packed[name].dtype == legacy_window[name].dtype
    assert packed["frame_names"] == legacy_window["frame_names"]
    verification = verify_cache(
        root,
        source_root=source,
        split_file=split_file,
        sample_videos=1,
    )
    assert verification["compared_frames"] == 3
    remove_cache(root, confirm=True)
    build_cache(root=root, source_root=source, split_file=split_file)
    assert (root / "manifest.json").is_file()
