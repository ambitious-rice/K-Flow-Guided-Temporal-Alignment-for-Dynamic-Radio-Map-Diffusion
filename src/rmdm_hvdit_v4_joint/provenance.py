"""Immutable dependency manifests for isolated HV-DiT v4 joint runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from . import ARCHITECTURE_ID
from .config import ExperimentConfig


READ_ONLY_BOUNDARY_FILES = (
    "src/rmdm/data/window_dataset.py",
    "src/rmdm/data/sampling.py",
    "src/rmdm/diffusion/process.py",
    "src/rmdm/diffusion/ddim.py",
    "src/rmdm/evaluation/metrics.py",
    "unet.py",
    "utils/losses.py",
)

PROTOCOL_FILES = (
    ".agent/HANDOFF.md",
    "docs/dynamic_sparse_observation_rmdm_baseline_plan.md",
    "docs/dynamic_sparse_motivation_test_results.md",
    "docs/dynamic_sparse_joint_denoising_w16_pilot_protocol.md",
    "reports/dynamic_sparse_motivation_explainer_20260715/README.md",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def build_dependency_manifest(
    config: ExperimentConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    sf_reference_checkpoint = Path(config.evaluation.sf_reference_checkpoint).expanduser().resolve()
    subset = Path(config.evaluation.subset_manifest).expanduser().resolve()
    formal_test_subset = Path(config.evaluation.formal_test_manifest).expanduser().resolve()
    isolated_paths = ("src/rmdm_hvdit_v4_joint", "configs/hvdit_v4_joint", "tests/hvdit_v4_joint", "docs/hvdit_v4_joint")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", *isolated_paths)
    status = "\n".join(
        line for line in status.splitlines() if "__pycache__" not in line and not line.endswith(".pyc")
    )
    tracked_diff = _git(root, "diff", "--binary", "HEAD", "--", *isolated_paths)
    files: dict[str, str] = {}
    for relative in (*READ_ONLY_BOUNDARY_FILES, *PROTOCOL_FILES):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required dependency file is absent: {path}")
        files[relative] = sha256_file(path)
    isolated_files: dict[str, str] = {}
    for relative_root in isolated_paths:
        directory = root / relative_root
        if directory.is_dir():
            for path in sorted(
                item
                for item in directory.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix in {".py", ".yaml", ".yml", ".md", ".txt"}
            ):
                relative = str(path.relative_to(root))
                isolated_files[relative] = sha256_file(path)
    manifest = {
        "schema": "rmdm_hvdit_v4_joint_dependency_manifest_v1",
        "architecture_id": ARCHITECTURE_ID,
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_sha256": sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": sha256_bytes(tracked_diff.encode("utf-8")),
        },
        "config": {
            "path": str(config_file),
            "sha256": sha256_file(config_file),
            "resolved_sha256": canonical_hash(config.to_dict()),
        },
        "hwm": {
            "initialization": "from_scratch",
            "base_features": config.stage1.base_features,
            "trainable": config.stage1.trainable,
        },
        "sf_reference_checkpoint": {
            "path": str(sf_reference_checkpoint),
            "sha256": sha256_file(sf_reference_checkpoint),
        },
        "evaluation_subset": {"path": str(subset), "sha256": sha256_file(subset)},
        "formal_test_subset": {
            "path": str(formal_test_subset),
            "sha256": sha256_file(formal_test_subset),
        },
        "files": files,
        "isolated_files": isolated_files,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "natten": _version("natten"),
            "accelerate": _version("accelerate"),
            "diffusers": _version("diffusers"),
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def assert_dependency_match(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    expected_hash = expected.get("manifest_sha256")
    actual_hash = actual.get("manifest_sha256")
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            "Dependency manifest drift detected; refusing resume: "
            f"checkpoint={expected_hash!r}, current={actual_hash!r}"
        )
