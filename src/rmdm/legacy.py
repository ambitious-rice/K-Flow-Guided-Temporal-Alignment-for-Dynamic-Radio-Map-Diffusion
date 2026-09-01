"""The only import boundary between the new package and legacy RMDM code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LegacyVideoRecord:
    index: int
    scene_id: str
    episode_id: str
    tx_id: str

    @property
    def video_id(self) -> str:
        return f"{self.scene_id}/{self.episode_id}/{self.tx_id}"


class LegacyFrameReader:
    """Expose video-level reads while containing all legacy private API use."""

    def __init__(
        self,
        root: str,
        split: str,
        split_file: str,
        cache_size: int = 8,
        tx_heatmap_sigma_px: float = 1.5,
        video_ids: set[str] | None = None,
    ) -> None:
        from lib.loaders import DynamicRadioMapRMDM

        self.dataset = DynamicRadioMapRMDM(
            root=root,
            split=split,
            split_file=split_file,
            frame_stride=1,
            cache_size=cache_size,
            tx_heatmap_sigma_px=tx_heatmap_sigma_px,
        )
        records = []
        for index, item in enumerate(self.dataset.records):
            record = LegacyVideoRecord(index, str(item["scene_id"]), str(item["episode_id"]), str(item["tx_id"]))
            if video_ids is None or record.video_id in video_ids:
                records.append(record)
        if video_ids is not None:
            found = {record.video_id for record in records}
            missing = video_ids - found
            if missing:
                preview = sorted(missing)[:5]
                raise KeyError(f"Requested videos are absent from split={split}: {preview}")
        self.records = records

    def frame_count(self, record: LegacyVideoRecord) -> int:
        return len(self.dataset.frame_ids_by_record[record.index])

    def read_window(self, record: LegacyVideoRecord, start: int, length: int) -> dict[str, Any]:
        legacy = self.dataset.records[record.index]
        frame_ids = self.dataset.frame_ids_by_record[record.index][start : start + length]
        if len(frame_ids) != length:
            raise IndexError(f"Incomplete window {record.video_id} start={start} length={length}")

        building = np.asarray(
            self.dataset._load_npz_array(legacy["building_mask_path"], "building_mask"), dtype=np.float32
        )
        if building.max(initial=0.0) > 1.0:
            building = building / 255.0
        tx = self.dataset._make_tx_heatmap(legacy, sigma_px=self.dataset.tx_heatmap_sigma_px)
        traffic_all = self.dataset._load_npz_array(legacy["traffic_grid_path"], "traffic_grid_uint8")

        vehicles = []
        targets = []
        names = []
        for frame_id in frame_ids:
            traffic = np.asarray(traffic_all[int(frame_id)], dtype=np.float32)
            vehicle = (traffic > 1.5).astype(np.float32)
            target = self.dataset._read_png_gray(
                Path(legacy["rss_png_dir"]) / f"frame_{int(frame_id):06d}.png"
            ).astype(np.float32) / 255.0
            vehicles.append(vehicle)
            targets.append(target)
            names.append(f"{record.video_id}/frame_{int(frame_id):06d}.png")

        return {
            "building": np.broadcast_to(building, (length, *building.shape)).copy(),
            "tx": np.broadcast_to(tx, (length, *tx.shape)).copy(),
            "vehicle": np.stack(vehicles),
            "target": np.stack(targets),
            "frame_names": names,
        }


def build_stage1_hwm(checkpoint_path: str | Path) -> nn.Module:
    """Build the exact legacy HWM and load only its weights from an SF checkpoint."""
    from unet import Generic_UNet

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    hwm = Generic_UNet(5, 32, 1, 5, anchor_out=True, upscale_logits=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    prefix = "unet.hwm."
    hwm_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not hwm_state:
        raise KeyError(f"No {prefix!r} weights found in {checkpoint_path}")
    hwm.load_state_dict(hwm_state, strict=True)
    return hwm

