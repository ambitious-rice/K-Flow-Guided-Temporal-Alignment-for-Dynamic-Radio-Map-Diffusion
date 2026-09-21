"""Cache-friendly deterministic ordering for frame-expanded video datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class VideoBlockShuffleSampler(Sampler[int]):
    """Shuffle videos per epoch while keeping every video's frames together.

    The training dataset is laid out as contiguous fixed-frame blocks. Keeping
    each block intact lets a DataLoader worker reuse the decoded building and
    traffic arrays instead of reopening the same NPZ files for every frame.
    """

    def __init__(self, size: int, frames_per_video: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("sampler size must be positive")
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be positive")
        self.size = int(size)
        self.frames_per_video = int(frames_per_video)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        block_count = (self.size + self.frames_per_video - 1) // self.frames_per_video
        for block in torch.randperm(block_count, generator=generator).tolist():
            start = block * self.frames_per_video
            stop = min(start + self.frames_per_video, self.size)
            frame_order = torch.randperm(stop - start, generator=generator).tolist()
            yield from (start + offset for offset in frame_order)
