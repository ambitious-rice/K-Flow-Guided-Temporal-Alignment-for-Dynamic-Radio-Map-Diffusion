"""Explicit single-frame V4 joint pretraining model."""

from __future__ import annotations

from rmdm_hvdit_v4_joint.config import ModelConfig

from .joint import JointTokenDenoiser


class HvditT1(JointTokenDenoiser):
    def __init__(
        self,
        config: ModelConfig,
        *,
        attention_backend: str | None = None,
        gradient_checkpointing: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            frames=1,
            attention_backend=attention_backend,
            gradient_checkpointing=gradient_checkpointing,
        )
