from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import yaml

SUITE = Path("configs/evaluation/observation_balance_validation_v1.yaml")
MODULE_PATH = Path("src/rmdm/evaluation/observation_balance_suite.py")
SPEC = importlib.util.spec_from_file_location("observation_balance_suite", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
balanced_video_ids = MODULE.balanced_video_ids
expand_cells = MODULE.expand_cells
load_suite = MODULE.load_suite


def test_fixed_suite_expands_to_stable_unique_cells() -> None:
    suite = load_suite(SUITE)
    first = expand_cells(deepcopy(suite))
    second = expand_cells(deepcopy(suite))

    assert first == second
    assert len(first) == 42
    assert len({cell["cell_id"] for cell in first}) == len(first)
    assert len([cell for cell in first if cell["variant"] == "w1"]) == 17
    assert len([cell for cell in first if cell["variant"] == "w16"]) == 25


def test_clean_cells_do_not_run_known_noise_da() -> None:
    suite = load_suite(SUITE)
    clean = [cell for cell in expand_cells(suite) if cell["noise_std"] == 0]

    assert clean
    assert all("known_noise_da" not in cell["methods"] for cell in clean)


def test_every_cell_reuses_the_same_random_seeds() -> None:
    suite = load_suite(SUITE)
    cells = expand_cells(suite)

    assert all(cell["seeds"] == suite["seeds"] for cell in cells)


def test_balanced_video_selection_is_independent_of_manifest_order() -> None:
    entries = [
        {"scene_id": "b", "video_id": "b/video_02"},
        {"scene_id": "a", "video_id": "a/video_03"},
        {"scene_id": "a", "video_id": "a/video_01"},
        {"scene_id": "b", "video_id": "b/video_01"},
        {"scene_id": "a", "video_id": "a/video_02"},
        {"scene_id": "b", "video_id": "b/video_03"},
    ]
    manifest = {"stage_a": {"videos": entries}}
    reversed_manifest = {"stage_a": {"videos": list(reversed(entries))}}

    expected = ["a/video_01", "a/video_02", "b/video_01", "b/video_02"]
    assert balanced_video_ids(manifest, "stage_a", 2) == expected
    assert balanced_video_ids(reversed_manifest, "stage_a", 2) == expected


def test_suite_file_contains_only_numeric_fixed_settings() -> None:
    payload = yaml.safe_load(SUITE.read_text(encoding="utf-8"))

    assert payload["seeds"] == {
        "global": 20260909,
        "dataset": 20260717,
        "mask": 20260714,
        "ddim_noise": 20260717,
        "observation_noise": 20260805,
    }
    assert payload["guidance"] == {
        "strength": 0.5,
        "guided_steps": 7,
        "max_update": 0.25,
    }
