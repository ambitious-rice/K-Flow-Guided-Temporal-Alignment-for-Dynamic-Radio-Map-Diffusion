"""Dense, continuous DynamicRadioMap windows."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Any, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from rmdm.legacy import LegacyFrameReader, LegacyVideoRecord


class WindowReader(Protocol):
    records: Sequence[Any]

    def frame_count(self, record: Any) -> int: ...

    def read_window(self, record: Any, start: int, length: int) -> dict[str, Any]: ...


def _stable_index(modulus: int, *parts: object) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "little", signed=False) % modulus


@lru_cache(maxsize=32)
def _coprime_strides(modulus: int) -> tuple[int, ...]:
    if modulus <= 1:
        return (1,)
    return tuple(value for value in range(1, modulus) if math.gcd(value, modulus) == 1)


def _epoch_permutation_index(modulus: int, epoch: int, *parts: object) -> int:
    """Visit every index once per deterministic epoch cycle."""

    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if modulus == 1:
        return 0
    cycle, position = divmod(int(epoch), modulus)
    offset = _stable_index(modulus, *parts, "cycle", cycle, "offset")
    strides = _coprime_strides(modulus)
    stride = strides[_stable_index(len(strides), *parts, "cycle", cycle, "stride")]
    return (offset + position * stride) % modulus


class WindowDataset(Dataset):
    """One stateless random W-frame window per video, or fixed eval windows."""

    SAMPLER_VERSION = "joint_window_epoch_permutation_v2"

    def __init__(
        self,
        *,
        root: str | None = None,
        split: str = "train",
        split_file: str | None = None,
        window_size: int = 16,
        seed: int = 20260717,
        cache_size: int = 8,
        tx_heatmap_sigma_px: float = 1.5,
        fixed_starts: Sequence[int] | None = None,
        video_ids: Sequence[str] | None = None,
        reader: WindowReader | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if reader is None:
            if root is None or split_file is None:
                raise ValueError("root and split_file are required when reader is not supplied")
            reader = LegacyFrameReader(
                root=root,
                split=split,
                split_file=split_file,
                cache_size=cache_size,
                tx_heatmap_sigma_px=tx_heatmap_sigma_px,
                video_ids=set(video_ids) if video_ids is not None else None,
            )
        self.reader = reader
        self.split = str(split)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.epoch = 0
        self.fixed_starts = tuple(int(value) for value in fixed_starts) if fixed_starts is not None else None
        if not self.reader.records:
            raise ValueError("WindowDataset contains no videos")
        if self.fixed_starts is not None:
            for record in self.reader.records:
                frame_count = self.reader.frame_count(record)
                for start in self.fixed_starts:
                    if start < 0 or start + self.window_size > frame_count:
                        raise ValueError(f"Invalid fixed start={start} for frame_count={frame_count}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        multiplier = len(self.fixed_starts) if self.fixed_starts is not None else 1
        return len(self.reader.records) * multiplier

    def _resolve_item(self, index: int) -> tuple[Any, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.fixed_starts is not None:
            count = len(self.fixed_starts)
            return self.reader.records[index // count], self.fixed_starts[index % count]
        record = self.reader.records[index]
        max_start = self.reader.frame_count(record) - self.window_size
        video_id = getattr(record, "video_id", str(record))
        start = _epoch_permutation_index(
            max_start + 1,
            self.epoch,
            self.SAMPLER_VERSION,
            self.seed,
            self.split,
            video_id,
        )
        return record, start

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, start = self._resolve_item(index)
        arrays = self.reader.read_window(record, start, self.window_size)
        video_id = getattr(record, "video_id", str(record))
        item: dict[str, Any] = {
            key: torch.from_numpy(arrays[key]).unsqueeze(1).to(dtype=torch.float32).contiguous()
            for key in ("building", "tx", "vehicle", "target")
        }
        item.update(
            {
                "video_id": video_id,
                "start": int(start),
                "frame_names": list(arrays["frame_names"]),
            }
        )
        return item
