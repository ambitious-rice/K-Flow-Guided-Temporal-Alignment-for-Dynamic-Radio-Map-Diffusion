"""Dependency manifest for the 10k-to-50k x0 continuation."""

from __future__ import annotations

import hashlib
import json
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
    manifest = {
        "schema": "rmdm_hvdit_v4_x0_continue_run_metadata_v2",
        "architecture_id": ARCHITECTURE_ID,
        "repository_root": str(root),
        "config": {
            "path": str(config_file),
            "sha256": sha256_file(config_file),
            "resolved_sha256": _canonical_hash(config.to_dict()),
        },
        "source_checkpoint": str(source) if source is not None else "",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    return manifest
