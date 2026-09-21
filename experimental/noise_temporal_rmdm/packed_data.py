"""Contiguous uint8 cache for globally shuffled T1 frame training."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rmdm.legacy import LegacyFrameReader, LegacyVideoRecord


PACKED_SCHEMA = "noise_temporal_rmdm_packed_frames_v1"


def _decode_record(payload: tuple[int, dict[str, Any], int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    index, record, frames_per_video = payload
    with np.load(record["building_mask_path"]) as data:
        building = np.asarray(data["building_mask"])
    if building.max(initial=0) <= 1:
        building = building * 255
    with np.load(record["traffic_grid_path"]) as data:
        traffic = np.asarray(data["traffic_grid_uint8"][:frames_per_video])
    vehicle = (traffic > 1.5).astype(np.uint8)
    target = np.empty((frames_per_video, *building.shape), dtype=np.uint8)
    png_root = Path(record["rss_png_dir"])
    for frame in range(frames_per_video):
        with Image.open(png_root / f"frame_{frame:06d}.png") as image:
            target[frame] = np.asarray(image.convert("L"), dtype=np.uint8)
    return index, np.asarray(building, dtype=np.uint8), vehicle, target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackedFrameReader:
    """Read frames by mmap slicing without PNG opens or NPZ decompression."""

    def __init__(self, root: str | Path, *, split_file: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"packed cache is incomplete: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != PACKED_SCHEMA:
            raise ValueError(f"unsupported packed cache schema: {metadata.get('schema')!r}")
        split_path = Path(split_file).expanduser().resolve()
        if metadata.get("split_sha256") != _sha256(split_path):
            raise ValueError("packed cache does not match the configured split file")
        self.frames_per_video = int(metadata["frames_per_video"])
        self.image_size = int(metadata["image_size"])
        self.records = [LegacyVideoRecord(**record) for record in metadata["records"]]
        self._building: np.ndarray | None = None
        self._vehicle: np.ndarray | None = None
        self._target: np.ndarray | None = None

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._target is None:
            self._building = np.load(self.root / "building_uint8.npy", mmap_mode="r")
            self._vehicle = np.load(self.root / "vehicle_uint8.npy", mmap_mode="r")
            self._target = np.load(self.root / "target_uint8.npy", mmap_mode="r")
        assert self._building is not None and self._vehicle is not None and self._target is not None
        return self._building, self._vehicle, self._target

    def frame_count(self, record: LegacyVideoRecord) -> int:
        return self.frames_per_video

    def read_window(self, record: LegacyVideoRecord, start: int, length: int) -> dict[str, Any]:
        stop = int(start) + int(length)
        if start < 0 or stop > self.frames_per_video:
            raise IndexError(f"incomplete packed window start={start} length={length}")
        building, vehicle, target = self._arrays()
        index = int(record.index)
        frame_slice = slice(index * self.frames_per_video + start, index * self.frames_per_video + stop)
        names = [f"{record.video_id}/frame_{frame:06d}.png" for frame in range(start, stop)]
        return {
            "building": np.broadcast_to(
                np.asarray(building[index], dtype=np.float32) / 255.0,
                (length, self.image_size, self.image_size),
            ),
            "vehicle": np.asarray(vehicle[frame_slice], dtype=np.float32),
            "target": np.asarray(target[frame_slice], dtype=np.float32) / 255.0,
            "frame_names": names,
        }


def build_packed_cache(
    *, data_root: str | Path, split_file: str | Path, output: str | Path,
    frames_per_video: int, workers: int = 32,
) -> None:
    """Build an atomic cache. A metadata file is published only after all arrays finish."""

    data_root = Path(data_root).expanduser().resolve()
    split_file = Path(split_file).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"packed cache already complete: {output}")
    reader = LegacyFrameReader(
        root=str(data_root), split="train", split_file=str(split_file), cache_size=2, include_tx=False
    )
    if not reader.records:
        raise ValueError("training split contains no videos")
    first = reader.read_window(reader.records[0], 0, 1)
    image_size = int(first["target"].shape[-1])
    video_count = len(reader.records)
    frame_count = video_count * int(frames_per_video)
    partials = {
        name: output / f".{name}_uint8.partial.npy" for name in ("building", "vehicle", "target")
    }
    for path in partials.values():
        if path.exists():
            path.unlink()
    building = np.lib.format.open_memmap(
        partials["building"], mode="w+", dtype=np.uint8,
        shape=(video_count, image_size, image_size),
    )
    vehicle = np.lib.format.open_memmap(
        partials["vehicle"], mode="w+", dtype=np.uint8,
        shape=(frame_count, image_size, image_size),
    )
    target = np.lib.format.open_memmap(
        partials["target"], mode="w+", dtype=np.uint8,
        shape=(frame_count, image_size, image_size),
    )
    records = reader.dataset.records
    tasks = ((index, records[index], frames_per_video) for index in range(video_count))
    completed = 0
    context = mp.get_context("fork")
    with context.Pool(processes=max(1, int(workers))) as pool:
        for index, building_array, vehicle_array, target_array in pool.imap_unordered(
            _decode_record, tasks, chunksize=1
        ):
            start = index * frames_per_video
            stop = start + frames_per_video
            building[index] = building_array
            vehicle[start:stop] = vehicle_array
            target[start:stop] = target_array
            completed += 1
            if completed % 100 == 0 or completed == video_count:
                print(f"packed {completed}/{video_count} videos", flush=True)
    building.flush()
    vehicle.flush()
    target.flush()
    del building, vehicle, target
    for name, partial in partials.items():
        os.replace(partial, output / f"{name}_uint8.npy")
    metadata = {
        "schema": PACKED_SCHEMA,
        "split": "train",
        "split_file": str(split_file),
        "split_sha256": _sha256(split_file),
        "frames_per_video": int(frames_per_video),
        "image_size": image_size,
        "records": [asdict(record) for record in reader.records],
    }
    temporary = output / ".metadata.partial.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "metadata.json")
