"""Independent HV-DiT v4 joint implementation for dynamic sparse radio maps."""

from .config import ExperimentConfig, load_config

ARCHITECTURE_ID = "rmdm_hvdit_v4_joint_scale_stem_pixel_decoder_v2"

__all__ = ["ARCHITECTURE_ID", "ExperimentConfig", "load_config"]
