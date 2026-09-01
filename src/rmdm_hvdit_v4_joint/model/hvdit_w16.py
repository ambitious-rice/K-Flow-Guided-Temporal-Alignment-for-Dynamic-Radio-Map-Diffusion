"""Explicit W16 V4 model with 3-D local and joint global attention."""

from __future__ import annotations

from rmdm_hvdit_v4_joint.config import ModelConfig

from .joint import JointTokenDenoiser


class HvditW16(JointTokenDenoiser):
    def __init__(self, config: ModelConfig, *, attention_backend: str | None = None) -> None:
        super().__init__(config, frames=16, attention_backend=attention_backend)
