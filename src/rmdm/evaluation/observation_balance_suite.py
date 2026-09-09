"""Configuration and cell expansion for the observation-balance test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


SUITE_SCHEMA = "rmdm_observation_balance_suite_v1"
CELL_SCHEMA = "rmdm_observation_balance_cell_v1"
RESULT_SCHEMA = "rmdm_observation_balance_result_v1"


def load_suite(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict) or suite.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"test suite must use schema {SUITE_SCHEMA!r}")
    return suite


def _number_key(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def expand_cells(
    suite: dict[str, Any],
    *,
    stages: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = set(stages or suite["stages"])
    unknown = selected - set(suite["stages"])
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    cells: list[dict[str, Any]] = []
    for stage_name, stage in suite["stages"].items():
        if stage_name not in selected:
            continue
        for rate in stage["rates"]:
            for sigma in stage["noise_stds"]:
                methods = list(stage["methods"])
                if float(sigma) == 0:
                    methods = [method for method in methods if method != "known_noise_da"]
                if not methods:
                    continue
                cell_id = (
                    f"{stage_name}__{stage['variant']}__p{_number_key(rate)}"
                    f"__s{_number_key(sigma)}__{'-'.join(methods)}"
                )
                cells.append(
                    {
                        "schema": CELL_SCHEMA,
                        "suite_id": suite["suite_id"],
                        "cell_id": cell_id,
                        "stage": stage_name,
                        "variant": stage["variant"],
                        "rate": float(rate),
                        "noise_std": float(sigma),
                        "methods": methods,
                        "split": suite["split"],
                        "subset_stage": stage["subset_stage"],
                        "videos_per_scene": int(stage["videos_per_scene"]),
                        "starts": stage["starts"],
                        "ddim_steps": int(stage["ddim_steps"]),
                        "world_size": int(stage["world_size"]),
                        "seeds": dict(suite["seeds"]),
                        "guidance": dict(suite["guidance"]),
                    }
                )
    return cells


def balanced_video_ids(manifest: dict[str, Any], stage: str, videos_per_scene: int) -> list[str]:
    entries = manifest.get(stage, {}).get("videos", [])
    if not entries:
        raise ValueError(f"manifest contains no videos for {stage!r}")
    groups: dict[str, list[str]] = {}
    for entry in entries:
        video_id = str(entry["video_id"])
        scene_id = str(entry.get("scene_id") or video_id.split("/", 1)[0])
        groups.setdefault(scene_id, []).append(video_id)
    selected: list[str] = []
    for scene_id in sorted(groups):
        videos = sorted(groups[scene_id])
        if len(videos) < videos_per_scene:
            raise ValueError(f"scene {scene_id!r} has fewer than {videos_per_scene} videos")
        selected.extend(videos[:videos_per_scene])
    return selected


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, destination)


__all__ = [
    "CELL_SCHEMA",
    "RESULT_SCHEMA",
    "SUITE_SCHEMA",
    "balanced_video_ids",
    "expand_cells",
    "load_suite",
    "read_json",
    "write_json_atomic",
]
