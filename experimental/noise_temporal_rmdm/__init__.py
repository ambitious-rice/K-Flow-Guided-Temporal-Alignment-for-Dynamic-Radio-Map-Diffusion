"""Noise-aware RMDM, with a stable T1-to-T16 model boundary."""

T1_ARCHITECTURE_ID = "noise_temporal_rmdm_t1_no_tx_v2"
T16_ARCHITECTURE_ID = "noise_temporal_rmdm_t16_no_tx_v1"
ARCHITECTURE_ID = T1_ARCHITECTURE_ID

from .model import NoiseAwareRMDM, build_model

__all__ = [
    "ARCHITECTURE_ID", "T1_ARCHITECTURE_ID", "T16_ARCHITECTURE_ID",
    "NoiseAwareRMDM", "build_model",
]
