#!/usr/bin/env python3
"""Train one segment of the isolated W1 observation-balance branch."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from rmdm_hvdit_v4_joint.config import load_config
from rmdm_hvdit_v4_x0_observation_balance.config import load_observation_balance_config
from rmdm_hvdit_v4_x0_observation_balance.runner import run_training_segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    training_config = load_observation_balance_config(args.config)
    model_config_path = Path(training_config.model_config).expanduser()
    if not model_config_path.is_absolute():
        model_config_path = root / model_config_path
    model_config = load_config(model_config_path)
    run_training_segment(
        model_config,
        training_config,
        repository_root=root,
        resume_from=args.resume_from or None,
    )


if __name__ == "__main__":
    main()
