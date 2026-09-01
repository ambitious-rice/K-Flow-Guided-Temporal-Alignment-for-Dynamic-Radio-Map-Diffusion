"""Dependency manifest for the 10k-to-50k x0 continuation."""

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


def _canonical_hash(payload: Any) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _tree(root: Path, relative: str) -> dict[str, str]:
    directory = root / relative
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".yaml", ".yml"}
    }


def build_dependency_manifest(
    config: Any,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    source_checkpoint: str | Path | None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    source = Path(source_checkpoint).expanduser().resolve() if source_checkpoint else None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    manifest = {
        "schema": "rmdm_hvdit_v4_x0_continue_dependency_manifest_v1",
        "architecture_id": ARCHITECTURE_ID,
        "git_commit": commit,
        "config": {
            "path": str(config_file),
            "sha256": sha256_file(config_file),
            "resolved_sha256": _canonical_hash(config.to_dict()),
        },
        "source_checkpoint": (
            {"path": str(source), "sha256": sha256_file(source), "used_for_initialization": True}
            if source is not None
            else {"path": "", "sha256": "", "used_for_initialization": False}
        ),
        "continuation_sources": {
            **_tree(root, "src/rmdm_hvdit_v4_x0_continue"),
            **_tree(root, "configs/hvdit_v4_x0_continue"),
            **_tree(root, "tests/hvdit_v4_x0_continue"),
        },
        "frozen_x0_pilot_sources": _tree(root, "src/rmdm_hvdit_v4_x0"),
        "frozen_v4_model_sources": _tree(root, "src/rmdm_hvdit_v4_joint/model"),
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    return manifest
