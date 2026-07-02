"""Clip datasets for temporal RMDM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .loaders import DynamicRadioMapRMDM
except Exception:
    from loaders import DynamicRadioMapRMDM


class DynamicRadioMapRMDMClip(Dataset):
    """Return fixed-length DynamicRadio clips for RMDM temporal training/eval.

    The per-frame preprocessing intentionally matches ``DynamicRadioMapRMDM`` so
    the original RMDM checkpoint remains a valid initialization:
    ``[building, tx_heatmap, traffic_grid_uint8 / 255]`` and RSS target in
    ``[0, 1]``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        split_file: str = "split.json",
        clip_length: int = 32,
        frame_stride: int = 16,
        clip_stride: int = 1,
        cache_size: int = 8,
        tx_heatmap_sigma_px: float = 1.5,
        starts: list[int] | None = None,
        include_k2: bool = True,
        k2_root: str | None = None,
        path_mask_threshold: float = 0.03,
        path_mask_dilate: int = 5,
        scene_ids: list[str] | tuple[str, ...] | str | None = None,
    ):
        self.frame_dataset = DynamicRadioMapRMDM(
            root=root,
            split=split,
            split_file=split_file,
            frame_stride=1,
            cache_size=cache_size,
            tx_heatmap_sigma_px=tx_heatmap_sigma_px,
        )
        self.clip_length = int(clip_length)
        self.frame_stride = max(1, int(frame_stride))
        self.clip_stride = max(1, int(clip_stride))
        self.starts = [int(value) for value in starts] if starts is not None else None
        self.include_k2 = bool(include_k2)
        self.k2_root = Path(k2_root).expanduser().resolve() if k2_root else None
        self.path_mask_threshold = float(path_mask_threshold)
        self.path_mask_dilate = int(path_mask_dilate)
        self.scene_ids = self._normalize_scene_ids(scene_ids)
        if self.scene_ids is not None:
            keep = [
                record
                for record in self.frame_dataset.records
                if str(record.get("scene_id")) in self.scene_ids
            ]
            if not keep:
                raise ValueError(f"No records matched scene_ids={sorted(self.scene_ids)}")
            self.frame_dataset.records = keep
            self.frame_dataset.frame_index = self.frame_dataset._build_frame_index()
        self.clip_index = self._build_clip_index()

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, start = self.clip_index[int(idx)]
        record = self.frame_dataset.records[record_idx]
        inputs = []
        images = []
        k2_frames = []
        names = []
        for offset in range(self.clip_length):
            frame_idx = start + offset * self.clip_stride
            flat_idx = record_idx * self.frame_count + frame_idx
            cond, image, name = self.frame_dataset[flat_idx]
            inputs.append(cond)
            images.append(image)
            if self.include_k2:
                k2_frames.append(self._load_k2_frame(record, frame_idx))
            names.append(name)
        item = {
            "inputs": torch.stack(inputs, dim=0).contiguous(),
            "image": torch.stack(images, dim=0).contiguous(),
            "names": names,
            "clip_name": names[0].replace("frame_", "clip_"),
            "record_idx": int(record_idx),
            "start": int(start),
        }
        if self.include_k2:
            k2 = torch.stack(k2_frames, dim=0).contiguous()
            item["k2"] = k2
            item["path_mask"] = self._make_path_mask(k2)
        return item

    @property
    def frame_count(self) -> int:
        count = int(self.frame_dataset.split_meta.get("frame_count_per_tx_sample", 0) or 0)
        if count > 0:
            return count
        first = self.frame_dataset.records[0]
        return int(len(np.load(first["frame_indices_path"])))

    def _build_clip_index(self) -> list[tuple[int, int]]:
        max_start = self.frame_count - (self.clip_length - 1) * self.clip_stride
        if max_start <= 0:
            raise ValueError(
                f"clip_length={self.clip_length}, clip_stride={self.clip_stride} exceed frame_count={self.frame_count}"
            )
        starts = self.starts if self.starts is not None else list(range(0, max_start, self.frame_stride))
        starts = [start for start in starts if 0 <= start < max_start]
        return [
            (record_idx, start)
            for record_idx in range(len(self.frame_dataset.records))
            for start in starts
        ]

    def _normalize_scene_ids(self, scene_ids: list[str] | tuple[str, ...] | str | None) -> set[str] | None:
        if scene_ids is None:
            return None
        if isinstance(scene_ids, str):
            values = [item.strip() for item in scene_ids.split(",")]
        else:
            values = [str(item).strip() for item in scene_ids]
        values = [item for item in values if item]
        return set(values) if values else None

    def _resolve_k2_path(self, record: dict[str, Any]) -> Path:
        if self.k2_root is None:
            path = Path(record["tx_dir"]) / "k2_neg_norm.npz"
            if path.exists():
                return path
            raise FileNotFoundError(f"K2 file not found for {record['scene_id']}/{record['episode_id']}/{record['tx_id']}: {path}")
        candidates = [
            self.k2_root / "scenes" / record["scene_id"] / "episodes" / record["episode_id"] / record["tx_id"] / "k2_neg_norm.npz",
            self.k2_root / "scenes" / record["scene_id"] / "episodes" / record["episode_id"] / "tx" / record["tx_id"] / "k2_neg_norm.npz",
            self.k2_root / record["scene_id"] / record["episode_id"] / record["tx_id"] / "k2_neg_norm.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"K2 file not found for {record['scene_id']}/{record['episode_id']}/{record['tx_id']}. Tried:\n{candidate_text}")

    def _load_k2_frame(self, record: dict[str, Any], frame_idx: int) -> torch.Tensor:
        path = self._resolve_k2_path(record)
        cached = self.frame_dataset._cache_get("k2_npz", path)
        if cached is None:
            with np.load(path) as data:
                for key in ("k2_uint8", "k2_pred_uint8", "k2_neg_norm_uint8", "arr_0"):
                    if key in data:
                        cached = np.asarray(data[key], dtype=np.uint8)
                        break
                else:
                    raise KeyError(f"No supported K2 key found in {path}")
            self.frame_dataset._cache_put("k2_npz", path, cached)
        frame = np.asarray(cached[int(frame_idx)], dtype=np.float32) / 255.0
        return torch.from_numpy(frame).unsqueeze(0)

    def _make_path_mask(self, k2: torch.Tensor) -> torch.Tensor:
        if k2.shape[0] <= 1:
            return torch.zeros((0, 1, *k2.shape[-2:]), dtype=k2.dtype)
        mask = ((k2[:-1] > self.path_mask_threshold) | (k2[1:] > self.path_mask_threshold)).to(dtype=k2.dtype)
        radius = max(0, self.path_mask_dilate)
        if radius > 0:
            mask = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return mask.contiguous()
