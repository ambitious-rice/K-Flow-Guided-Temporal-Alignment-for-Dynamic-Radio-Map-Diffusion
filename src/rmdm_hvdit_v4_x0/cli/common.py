"""Shared CLI configuration loader."""

from __future__ import annotations

import argparse
from pathlib import Path

from rmdm_hvdit_v4_x0.config import ExperimentConfig, load_config


def config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")


def load_arguments(args: argparse.Namespace) -> tuple[ExperimentConfig, Path, Path]:
    config_path = Path(args.config).expanduser().resolve()
    root = Path(args.repository_root).expanduser().resolve()
    return load_config(config_path), config_path, root
