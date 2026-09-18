import json

import pytest
import torch

from experimental.tx_prior.checkpoint import SCHEMA
from experimental.tx_prior.evaluate import load_model_checkpoint, parse_args, result_paths, write_result


def test_cli():
    args = parse_args(["--checkpoint", "best.pth"])
    assert args.steps == [20, 50] and args.batch_size == 4 and not args.smoke
    assert parse_args(["--checkpoint", "best.pth", "--steps", "50", "--smoke"]).smoke
    assert parse_args(["--checkpoint", "best.pth", "--expected-frames", "2000"]).expected_frames == 2000
    for extra in (["--steps", "10"], ["--steps", "20", "20"], ["--batch-size", "0"]):
        with pytest.raises(SystemExit):
            parse_args(["--checkpoint", "best.pth", *extra])


def test_two_scene_protocol_preserves_training_configuration():
    from pathlib import Path
    from experimental.tx_prior.config import load_config
    from rmdm_hvdit_v4_joint.evaluation.evaluator import manifest_video_ids
    directory = Path(__file__).parent
    original = load_config(directory / "remote.yaml", smoke=False).to_dict()
    filtered = load_config(directory / "remote_two_scene.yaml", smoke=False).to_dict()
    assert filtered["evaluation"].pop("subset_manifest") == "experimental/tx_prior/val_two_scene.json"
    original["evaluation"].pop("subset_manifest")
    assert original == filtered
    ids = manifest_video_ids(directory / "val_two_scene.json", "stage_a")
    assert len(ids) == len(set(ids)) == 20
    assert {video.split("/")[0] for video in ids} == {"town05_opt_junction_0053", "town05_opt_junction_1427"}


def test_checkpoint_schema_and_strict_weights(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.pth"
    torch.save({"schema": "tx_prior_x0_scene_checkpoint_v1", "model": model.state_dict()}, path)
    with pytest.raises(ValueError, match="schema"):
        load_model_checkpoint(path, model)
    for step in (None, -1, True, "10000"):
        torch.save({"schema": SCHEMA, "global_step": step, "model": model.state_dict()}, path)
        with pytest.raises(ValueError, match="global_step"):
            load_model_checkpoint(path, model)
    torch.save({"schema": SCHEMA, "global_step": 10000, "model": {}}, path)
    with pytest.raises(RuntimeError):
        load_model_checkpoint(path, model)
    weights = {name: torch.ones_like(value) for name, value in model.state_dict().items()}
    torch.save({"schema": SCHEMA, "global_step": 10000, "model": weights,
                "optimizer": {"ignored": True}, "rng": {"ignored": True}}, path)
    metadata = load_model_checkpoint(path, model)
    assert metadata == {"path": str(path), "global_step": 10000, "schema": SCHEMA, "resolved_config": None}
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())


def test_results_exclusive_metadata_smoke_isolation(tmp_path):
    checkpoint = {"global_step": 10000, "schema": SCHEMA, "path": "best.pth"}
    formal = result_paths(tmp_path, checkpoint, [20, 50], smoke=False)
    smoke = result_paths(tmp_path, checkpoint, [20, 50], smoke=True)
    assert set(formal).isdisjoint(smoke)
    result = {"checkpoint": checkpoint, "resolved_config": {"batch": 4},
              "source": {"head": "abc", "dirty": ""}, "ddim": {"steps": 20, "eta": 0.0}}
    write_result(formal[0], result)
    assert json.loads(formal[0].read_text()) == result
    with pytest.raises(FileExistsError):
        write_result(formal[0], {"wrong": True})
    with pytest.raises(FileExistsError):
        result_paths(tmp_path, checkpoint, [20, 50], smoke=False)
    assert result_paths(tmp_path, checkpoint, [20, 50], smoke=True) == smoke
