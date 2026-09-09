from dataclasses import asdict
import sys

import pytest
import yaml

from scripts import run_w1_observation_balance_training as controller
from rmdm_hvdit_v4_x0_observation_balance.config import ObservationBalanceConfig


def test_validation_interruptions_do_not_repeat_training(monkeypatch, tmp_path):
    config = ObservationBalanceConfig()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    baseline = {"rows": [
        {"variant": "w1", "stage": "fast_w1", "noise_std": sigma,
         "method": "no_da", "rate": rate, "unobserved_free_space_nmse": 1.0}
        for sigma in (0.0, 0.05) for rate in (1.0, 3.0)
    ]}
    controller.write_json_atomic(
        tmp_path / config.evaluation.output_root / config.evaluation.baseline_id / "summary.json", baseline,
    )
    monkeypatch.setattr(sys, "argv", ["controller", "--config", str(config_path),
                                     "--repository-root", str(tmp_path)])
    launches = []
    checkpoint = tmp_path / "step_001000.pth"

    def train(command, **kwargs):
        launches.append(command)
        controller.write_json_atomic(tmp_path / config.output_dir / "status.json", {
            "branch_step": 1000, "checkpoint": str(checkpoint), "state": "awaiting_fast_w1",
        })

    attempt = 0

    def validate(**kwargs):
        if (attempt == 0 and not kwargs["compare"]) or (attempt == 1 and kwargs["compare"]):
            raise RuntimeError("interrupted validation")
        if kwargs["compare"]:
            return {"passed": True}
        return {"rows": [{**row, "unobserved_free_space_nmse": 0.9} for row in baseline["rows"]]}

    monkeypatch.setattr(controller, "_run", train)
    monkeypatch.setattr(controller, "_validation", validate)
    for attempt in (0, 1):
        with pytest.raises(RuntimeError, match="interrupted validation"):
            controller.main()
    attempt = 2
    controller.main()
    state = controller.read_json(tmp_path / config.output_dir / "controller_state.json")
    assert len(launches) == 1
    assert len(state["fast_history"]) == 1
    assert state["state"] == "complete"
    assert state["selected_checkpoint"] == str(checkpoint)
    controller.main()
    assert len(launches) == 1


def test_missing_baseline_is_generated_before_training(monkeypatch, tmp_path):
    config = ObservationBalanceConfig()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    source = tmp_path / config.source_checkpoint
    source.parent.mkdir(parents=True)
    source.touch()
    model_config = tmp_path / config.model_config
    model_config.parent.mkdir(parents=True)
    model_config.touch()
    manifest = tmp_path / config.evaluation.manifest
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    suite = tmp_path / config.evaluation.suite
    suite.parent.mkdir(parents=True)
    suite.touch()
    baseline = {"rows": [
        {"variant": "w1", "stage": "fast_w1", "noise_std": sigma,
         "method": "no_da", "rate": rate, "unobserved_free_space_nmse": 1.0}
        for sigma in (0.0, 0.05) for rate in (1.0, 3.0)
    ]}
    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        if kwargs["candidate_id"] == config.evaluation.baseline_id:
            path = tmp_path / config.evaluation.output_root / config.evaluation.baseline_id
            controller.write_json_atomic(path / "summary.json", baseline)
            return baseline
        if kwargs["compare"]:
            return {"passed": True}
        return {"rows": [{**row, "unobserved_free_space_nmse": 0.9} for row in baseline["rows"]]}

    checkpoint = tmp_path / "step_001000.pth"

    def train(command, **kwargs):
        controller.write_json_atomic(tmp_path / config.output_dir / "status.json", {
            "branch_step": 1000, "checkpoint": str(checkpoint), "state": "awaiting_fast_w1",
        })

    monkeypatch.setattr(controller, "_validation", validate)
    monkeypatch.setattr(controller, "_run", train)
    monkeypatch.setattr(sys, "argv", ["controller", "--config", str(config_path),
                                     "--repository-root", str(tmp_path)])
    controller.main()
    assert calls[0]["candidate_id"] == config.evaluation.baseline_id
    assert calls[0]["stages"] == "fast_w1,full_w1,w1_da_gate"
