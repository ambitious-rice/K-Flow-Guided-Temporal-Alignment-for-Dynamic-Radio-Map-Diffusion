"""Dependency manifest for the isolated x0 pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import ARCHITECTURE_ID


def sha256_file(path: str | Path, *, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes(root: Path, relative_roots: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_root in relative_roots:
        directory = root / relative_root
        if not directory.is_dir():
            raise FileNotFoundError(f"Required source directory is absent: {directory}")
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".yaml", ".yml"}
            ):
                result[str(path.relative_to(root))] = sha256_file(path)
    return result


def build_dependency_manifest(
    config: Any,
    *,
    config_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    shared_files = (
        "src/rmdm/data/window_dataset.py",
        "src/rmdm/data/sampling.py",
        "src/rmdm/diffusion/process.py",
        "src/rmdm/diffusion/ddim.py",
        "src/rmdm/evaluation/metrics.py",
        "unet.py",
        "utils/losses.py",
        "manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json",
    )
    shared_hashes: dict[str, str] = {}
    for relative in shared_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot dependency is absent: {path}")
        shared_hashes[relative] = sha256_file(path)
    manifest = {
        "schema": "rmdm_hvdit_v4_x0_dependency_manifest_v1",
        "architecture_id": ARCHITECTURE_ID,
        "git_commit": commit,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "resolved_config_sha256": canonical_hash(config.to_dict()),
        "isolated_sources": _source_hashes(
            root,
            (
                "src/rmdm_hvdit_v4_x0",
                "configs/hvdit_v4_x0",
                "tests/hvdit_v4_x0",
            ),
        ),
        "frozen_v4_model_sources": _source_hashes(root, ("src/rmdm_hvdit_v4_joint/model",)),
        "shared_dependencies": shared_hashes,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest
