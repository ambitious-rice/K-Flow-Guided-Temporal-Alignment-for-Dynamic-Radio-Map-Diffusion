"""Disposable mmap cache for globally shuffled Tx-blind frame training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from rmdm.legacy import LegacyFrameReader


PACKED_SCHEMA = "noise_temporal_rmdm_packed_frames_v2"
MARKER = ".noise_temporal_packed"


def _has_marker(path: Path) -> bool:
    marker = path / MARKER
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8") == PACKED_SCHEMA + "\n"
    except OSError:
        return False


@dataclass(frozen=True)
class PackedRecord:
    index: int
    scene_id: str
    episode_id: str
    tx_id: str
    scene_index: int
    episode_index: int

    @property
    def video_id(self) -> str:
        return f"{self.scene_id}/{self.episode_id}/{self.tx_id}"


def _decode_record(
    payload: tuple[int, dict[str, Any], list[int], bool],
) -> tuple[int, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    index, record, frame_ids, load_vehicle = payload
    with np.load(record["building_mask_path"]) as data:
        building = np.asarray(data["building_mask"], dtype=np.float32)
    if building.max(initial=0.0) > 1.0:
        building /= 255.0
    encoded_building = np.rint(building * 255.0).astype(np.uint8)
    if not np.array_equal(encoded_building.astype(np.float32) / 255.0, building):
        raise ValueError(f"building is not losslessly uint8-packable: {record['scene_id']}")
    vehicle = None
    if load_vehicle:
        with np.load(record["traffic_grid_path"]) as data:
            traffic = np.asarray(data["traffic_grid_uint8"][frame_ids])
        vehicle = (traffic > 1.5).astype(np.uint8)
    target = np.empty((len(frame_ids), *encoded_building.shape), dtype=np.uint8)
    png_root = Path(record["rss_png_dir"])
    for offset, frame_id in enumerate(frame_ids):
        with Image.open(png_root / f"frame_{frame_id:06d}.png") as image:
            target[offset] = np.asarray(image.convert("L"), dtype=np.uint8)
    return index, encoded_building, vehicle, target, np.asarray(frame_ids, dtype=np.int32)


class PackedFrameReader:
    """Read fully shuffled frames without opening PNG or compressed NPZ files."""

    def __init__(self, root: str | Path, *, source_root: str | Path, split_file: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not _has_marker(self.root):
            raise ValueError(f"packed cache marker is missing: {self.root}")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PACKED_SCHEMA or manifest.get("state") != "complete":
            raise ValueError("packed cache is incomplete or unsupported")
        for key, configured in (("source_root", source_root), ("split_file", split_file)):
            packed = manifest.get(key)
            if not isinstance(packed, str) or Path(packed).resolve() != Path(configured).expanduser().resolve():
                raise ValueError(f"packed {key} mismatch")
        if manifest.get("split") != "train":
            raise ValueError("packed split mismatch")

        records_data = manifest.get("records")
        if not isinstance(records_data, list):
            raise ValueError("packed records must be a list")
        required = {"scene_id", "episode_id", "tx_id", "scene_index", "episode_index"}
        records: list[PackedRecord] = []
        for index, item in enumerate(records_data):
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError(f"invalid packed record at index {index}")
            records.append(PackedRecord(index=index, **item))
        if len({record.video_id for record in records}) != len(records):
            raise ValueError("duplicate packed video record")

        videos = len(records)
        scenes = manifest.get("scenes")
        episodes = manifest.get("episodes")
        frames = manifest.get("frames_per_video")
        height = manifest.get("height")
        width = manifest.get("width")
        counts = (scenes, episodes, frames, height, width)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("invalid packed dimensions")
        scene_map: dict[int, str] = {}
        episode_map: dict[int, tuple[str, str]] = {}
        for record in records:
            if not 0 <= record.scene_index < scenes or not 0 <= record.episode_index < episodes:
                raise ValueError(f"packed index out of range: {record.video_id}")
            scene = scene_map.setdefault(record.scene_index, record.scene_id)
            episode = episode_map.setdefault(record.episode_index, (record.scene_id, record.episode_id))
            if scene != record.scene_id or episode != (record.scene_id, record.episode_id):
                raise ValueError(f"inconsistent packed index: {record.video_id}")
        if set(scene_map) != set(range(scenes)) or set(episode_map) != set(range(episodes)):
            raise ValueError("packed indices do not cover their declared ranges")

        specs = {
            "targets": (np.dtype(np.uint8), (videos, frames, height, width)),
            "vehicles": (np.dtype(np.uint8), (episodes, frames, height, width)),
            "buildings": (np.dtype(np.uint8), (scenes, height, width)),
            "frame_ids": (np.dtype(np.int32), (videos, frames)),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, (dtype, shape) in specs.items():
            try:
                array = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"invalid packed array: {name}") from error
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"packed {name} dtype or shape mismatch")
            arrays[name] = array
        self.records = records
        self.frames_per_video = frames
        self.targets = arrays["targets"]
        self.vehicles = arrays["vehicles"]
        self.buildings = arrays["buildings"]
        self.frame_ids = arrays["frame_ids"]

    def frame_count(self, record: PackedRecord) -> int:
        return self.frames_per_video

    def read_window(self, record: PackedRecord, start: int, length: int) -> dict[str, Any]:
        stop = int(start) + int(length)
        if start < 0 or stop > self.frames_per_video:
            raise IndexError(f"incomplete packed window start={start} length={length}")
        building = self.buildings[record.scene_index].astype(np.float32) / 255.0
        ids = self.frame_ids[record.index, start:stop]
        return {
            "building": np.broadcast_to(building, (length, *building.shape)),
            "vehicle": self.vehicles[record.episode_index, start:stop].astype(np.float32),
            "target": self.targets[record.index, start:stop].astype(np.float32) / 255.0,
            "frame_names": [f"{record.video_id}/frame_{int(frame):06d}.png" for frame in ids],
        }


def build_packed_cache(
    *, data_root: str | Path, split_file: str | Path, output: str | Path,
    frames_per_video: int, workers: int = 32,
) -> None:
    """Build an atomic cache with vehicle maps deduplicated by episode."""

    data_root = Path(data_root).expanduser().resolve()
    split_file = Path(split_file).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    temporary = Path(str(output) + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    reader = LegacyFrameReader(
        root=str(data_root), split="train", split_file=str(split_file), cache_size=2, include_tx=False
    )
    if not reader.records:
        raise ValueError("training split contains no videos")
    records = reader.records
    scenes = list(dict.fromkeys(record.scene_id for record in records))
    episodes = list(dict.fromkeys((record.scene_id, record.episode_id) for record in records))
    scene_indices = {name: index for index, name in enumerate(scenes)}
    episode_indices = {name: index for index, name in enumerate(episodes)}
    frame_ids_by_record = [
        [int(value) for value in reader.dataset.frame_ids_by_record[record.index][:frames_per_video]]
        for record in records
    ]
    if any(len(ids) != frames_per_video for ids in frame_ids_by_record):
        raise ValueError("at least one training video has too few frames")
    first = reader.read_window(records[0], 0, 1)
    height, width = map(int, first["target"].shape[-2:])
    arrays: list[np.ndarray] = []
    try:
        temporary.mkdir(parents=True)
        (temporary / MARKER).write_text(PACKED_SCHEMA + "\n", encoding="utf-8")
        specs = (
            ("targets.npy", np.uint8, (len(records), frames_per_video, height, width)),
            ("vehicles.npy", np.uint8, (len(episodes), frames_per_video, height, width)),
            ("buildings.npy", np.uint8, (len(scenes), height, width)),
            ("frame_ids.npy", np.int32, (len(records), frames_per_video)),
        )
        arrays = [
            np.lib.format.open_memmap(temporary / name, mode="w+", dtype=dtype, shape=shape)
            for name, dtype, shape in specs
        ]
        targets, vehicles, buildings, frame_ids = arrays
        seen_scenes: set[int] = set()
        seen_episodes: set[int] = set()
        raw_records = reader.dataset.records
        first_episode_record: dict[tuple[str, str], int] = {}
        for index, record in enumerate(records):
            first_episode_record.setdefault((record.scene_id, record.episode_id), index)
        tasks = (
            (
                index,
                raw_records[record.index],
                frame_ids_by_record[index],
                first_episode_record[(record.scene_id, record.episode_id)] == index,
            )
            for index, record in enumerate(records)
        )
        context = mp.get_context("fork")
        with context.Pool(processes=max(1, int(workers))) as pool:
            for completed, decoded in enumerate(pool.imap_unordered(_decode_record, tasks, chunksize=1), 1):
                index, building, vehicle, target, ids = decoded
                record = records[index]
                scene_index = scene_indices[record.scene_id]
                episode_index = episode_indices[(record.scene_id, record.episode_id)]
                targets[index] = target
                frame_ids[index] = ids
                if scene_index not in seen_scenes:
                    buildings[scene_index] = building
                    seen_scenes.add(scene_index)
                elif not np.array_equal(buildings[scene_index], building):
                    raise ValueError(f"building mismatch within scene {record.scene_id}")
                if vehicle is not None:
                    vehicles[episode_index] = vehicle
                    seen_episodes.add(episode_index)
                if completed % 100 == 0 or completed == len(records):
                    print(f"packed {completed}/{len(records)} videos", flush=True)
        if seen_scenes != set(range(len(scenes))) or seen_episodes != set(range(len(episodes))):
            raise ValueError("cache construction did not cover every scene and episode")
        for array in arrays:
            array.flush()
        items = [
            {
                "scene_id": record.scene_id,
                "episode_id": record.episode_id,
                "tx_id": record.tx_id,
                "scene_index": scene_indices[record.scene_id],
                "episode_index": episode_indices[(record.scene_id, record.episode_id)],
            }
            for record in records
        ]
        manifest = {
            "schema": PACKED_SCHEMA,
            "state": "complete",
            "source_root": str(data_root),
            "split_file": str(split_file),
            "split": "train",
            "videos": len(records),
            "episodes": len(episodes),
            "scenes": len(scenes),
            "frames_per_video": int(frames_per_video),
            "height": height,
            "width": width,
            "records": items,
        }
        temp_manifest = temporary / "manifest.json.tmp"
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_manifest, temporary / "manifest.json")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_packed_cache(
    root: str | Path, *, data_root: str | Path, split_file: str | Path, sample_videos: int = 64,
) -> dict[str, int | str]:
    """Compare evenly spaced cached videos against the source reader."""

    packed = PackedFrameReader(root, source_root=data_root, split_file=split_file)
    legacy = LegacyFrameReader(
        root=str(Path(data_root).expanduser().resolve()), split="train",
        split_file=str(Path(split_file).expanduser().resolve()), include_tx=False,
    )
    if [record.video_id for record in packed.records] != [record.video_id for record in legacy.records]:
        raise ValueError("packed and source video order differs")
    count = min(int(sample_videos), len(packed.records))
    if count <= 0:
        raise ValueError("sample_videos must be positive")
    indices = np.unique(np.linspace(0, len(packed.records) - 1, num=count, dtype=np.int64))
    compared = 0
    for index in indices:
        packed_record = packed.records[int(index)]
        legacy_record = legacy.records[int(index)]
        last = packed.frame_count(packed_record) - 1
        for start in sorted({0, last // 2, last}):
            cached = packed.read_window(packed_record, start, 1)
            source = legacy.read_window(legacy_record, start, 1)
            for key in ("building", "vehicle", "target"):
                if not np.array_equal(cached[key], source[key]):
                    raise ValueError(f"cache mismatch: {packed_record.video_id} frame={start} key={key}")
            if cached["frame_names"] != source["frame_names"]:
                raise ValueError("cached frame name mismatch")
            compared += 1
    return {
        "schema": PACKED_SCHEMA,
        "videos": len(packed.records),
        "sampled_videos": len(indices),
        "compared_frames": compared,
    }
