"""Independent clean-data-prediction pilot built on the frozen V4-W1 architecture."""

from .config import ExperimentConfig, load_config

ARCHITECTURE_ID = "rmdm_hvdit_v4_x0_ddpm_sample_pilot_v1"

__all__ = ["ARCHITECTURE_ID", "ExperimentConfig", "load_config"]
