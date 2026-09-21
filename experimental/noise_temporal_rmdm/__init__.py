"""Noise-aware RMDM, with a stable T1-to-T16 model boundary."""

ARCHITECTURE_ID = "noise_temporal_rmdm_t1_no_tx_v2"

from .model import NoiseAwareRMDM, build_model

__all__ = ["ARCHITECTURE_ID", "NoiseAwareRMDM", "build_model"]
