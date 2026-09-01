"""Immutable W16 scoring domains for first96 and formal full100 evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredWindow:
    start: int
    local_start: int
    local_stop: int


FIRST96_WINDOWS = tuple(ScoredWindow(start, 0, 16) for start in (0, 16, 32, 48, 64, 80))
FULL100_WINDOWS = FIRST96_WINDOWS + (ScoredWindow(84, 12, 16),)


def scoring_windows(*, full100: bool) -> tuple[ScoredWindow, ...]:
    return FULL100_WINDOWS if full100 else FIRST96_WINDOWS


def validate_scoring_domain(windows: tuple[ScoredWindow, ...], *, expected_frames: int) -> None:
    frame_indices = [
        window.start + local
        for window in windows
        for local in range(window.local_start, window.local_stop)
    ]
    if frame_indices != list(range(expected_frames)):
        raise ValueError(f"Scoring windows do not cover exactly frames [0,{expected_frames - 1}]")
