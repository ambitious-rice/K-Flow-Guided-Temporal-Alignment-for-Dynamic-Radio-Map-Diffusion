"""Disposable contiguous NVMe backend for TX-prior W1."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rmdm.legacy import LegacyFrameReader

FORMAT = "tx_prior_packed_v2"
MARKER = ".tx_prior_packed"
DEFAULT_ROOT = Path("/home/fzj/.cache/rmdm/tx_prior")
ARRAY_SPECS = {
    "targets": np.dtype(np.uint8),
    "vehicles": np.dtype(np.uint8),
    "tx": np.dtype(np.float32),
    "buildings": np.dtype(np.float32),
    "frame_ids": np.dtype(np.int32),
}


def _has_marker(path: Path) -> bool:
    marker = path / MARKER
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8") == FORMAT + "\n"
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
    def video_id(self):
        return f"{self.scene_id}/{self.episode_id}/{self.tx_id}"


class PackedFrameReader:
    def __init__(
        self,
        root,
        *,
        source_root,
        split_file,
        tx_heatmap_sigma_px,
        split="train",
        video_ids=None,
    ):
        self.root = Path(root).expanduser().resolve()
        if not _has_marker(self.root):
            raise ValueError("packed cache marker is missing")
        path = self.root / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("packed manifest must be an object")
        if manifest.get("format") != FORMAT or manifest.get("state") != "complete":
            raise ValueError("packed cache is incomplete or unsupported")
        for key, value in (("source_root", source_root), ("split_file", split_file)):
            packed_value = manifest.get(key)
            if not isinstance(packed_value, str):
                raise ValueError(f"invalid packed {key}")
            if Path(packed_value).resolve() != Path(value).expanduser().resolve():
                raise ValueError(f"packed {key} mismatch")
        if manifest.get("split") != split:
            raise ValueError("packed split mismatch")
        packed_sigma = manifest.get("tx_heatmap_sigma_px")
        if type(packed_sigma) not in (int, float):
            raise ValueError("invalid packed TX sigma")
        if float(packed_sigma) != float(tx_heatmap_sigma_px):
            raise ValueError("packed TX sigma mismatch")
        records_data = manifest.get("records")
        if not isinstance(records_data, list):
            raise ValueError("packed records must be a list")
        required_record_keys = {
            "scene_id",
            "episode_id",
            "tx_id",
            "scene_index",
            "episode_index",
        }
        records = []
        for index, item in enumerate(records_data):
            if not isinstance(item, dict) or set(item) != required_record_keys:
                raise ValueError(f"invalid packed record at index {index}")
            if not all(isinstance(item[key], str) and item[key] for key in ("scene_id", "episode_id", "tx_id")):
                raise ValueError(f"invalid packed record name at index {index}")
            if type(item["scene_index"]) is not int or type(item["episode_index"]) is not int:
                raise ValueError(f"invalid packed record indices at index {index}")
            records.append(PackedRecord(index=index, **item))
        if type(manifest.get("videos")) is not int or manifest["videos"] != len(records):
            raise ValueError("record count mismatch")
        videos = len(records)
        scenes = manifest.get("scenes")
        episodes = manifest.get("episodes")
        frames = manifest.get("frames_per_video")
        shape = manifest.get("shape")
        if type(scenes) is not int or scenes <= 0:
            raise ValueError("invalid scene count")
        if type(episodes) is not int or episodes <= 0:
            raise ValueError("invalid episode count")
        if type(frames) is not int or frames <= 0:
            raise ValueError("invalid frame count")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] != frames
        ):
            raise ValueError("invalid packed shape")
        if len({record.video_id for record in records}) != videos:
            raise ValueError("duplicate packed video record")
        scene_by_index = {}
        index_by_scene = {}
        episode_by_index = {}
        index_by_episode = {}
        for record in records:
            if not 0 <= record.scene_index < scenes:
                raise ValueError(f"scene index out of range for {record.video_id}")
            if not 0 <= record.episode_index < episodes:
                raise ValueError(f"episode index out of range for {record.video_id}")
            if (
                scene_by_index.setdefault(record.scene_index, record.scene_id)
                != record.scene_id
                or index_by_scene.setdefault(record.scene_id, record.scene_index)
                != record.scene_index
            ):
                raise ValueError(f"inconsistent scene index for {record.video_id}")
            episode_name = (record.scene_id, record.episode_id)
            if (
                episode_by_index.setdefault(record.episode_index, episode_name)
                != episode_name
                or index_by_episode.setdefault(episode_name, record.episode_index)
                != record.episode_index
            ):
                raise ValueError(f"inconsistent episode index for {record.video_id}")
        if set(scene_by_index) != set(range(scenes)):
            raise ValueError("packed scene indices do not cover the declared range")
        if set(episode_by_index) != set(range(episodes)):
            raise ValueError("packed episode indices do not cover the declared range")

        _, height, width = shape
        expected_shapes = {
            "targets": (videos, frames, height, width),
            "vehicles": (episodes, frames, height, width),
            "tx": (videos, height, width),
            "buildings": (scenes, height, width),
            "frame_ids": (videos, frames),
        }
        arrays = {}
        for name, dtype in ARRAY_SPECS.items():
            declared_dtype = manifest.get(f"{name.rstrip('s')}_dtype")
            if name == "frame_ids":
                declared_dtype = manifest.get("frame_id_dtype")
            if declared_dtype != dtype.name:
                raise ValueError(f"packed {name} dtype declaration mismatch")
            try:
                array = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"invalid packed array: {name}") from error
            if array.dtype != dtype:
                raise ValueError(f"packed {name} dtype mismatch")
            if array.shape != expected_shapes[name]:
                raise ValueError(f"packed {name} shape mismatch")
            arrays[name] = array
        if video_ids is not None:
            records = [r for r in records if r.video_id in video_ids]
            missing = set(video_ids) - {r.video_id for r in records}
            if missing:
                raise KeyError(
                    f"videos absent from packed cache: {sorted(missing)[:5]}"
                )
        self.records, self.manifest = records, manifest
        self.targets = arrays["targets"]
        self.tx = arrays["tx"]
        self.vehicles = arrays["vehicles"]
        self.buildings = arrays["buildings"]
        self.frame_ids = arrays["frame_ids"]

    def frame_count(self, record):
        return int(self.manifest["frames_per_video"])

    def read_window(self, record, start, length):
        stop = start + length
        if start < 0 or stop > self.frame_count(record):
            raise IndexError("incomplete packed window")
        building, tx = self.buildings[record.scene_index], self.tx[record.index]
        ids = self.frame_ids[record.index, start:stop]
        return {
            "building": np.broadcast_to(building, (length, *building.shape)).astype(
                np.float32
            ),
            "tx": np.broadcast_to(tx, (length, *tx.shape)).astype(np.float32),
            "vehicle": self.vehicles[record.episode_index, start:stop].astype(
                np.float32
            ),
            "target": self.targets[record.index, start:stop].astype(np.float32) / 255.0,
            "frame_names": [f"{record.video_id}/frame_{int(i):06d}.png" for i in ids],
        }


def build_cache(
    *,
    root,
    source_root,
    split_file,
    split="train",
    tx_heatmap_sigma_px=1.5,
    progress: Callable[[int, int], None] | None = None,
):
    root, temporary = Path(root).expanduser().resolve(), Path(
        str(Path(root).expanduser().resolve()) + ".building"
    )
    if root.exists() or temporary.exists():
        raise FileExistsError(root if root.exists() else temporary)
    reader = LegacyFrameReader(
        source_root, split, split_file, tx_heatmap_sigma_px=tx_heatmap_sigma_px
    )
    records = reader.records
    if not records:
        raise ValueError("no selected videos")
    scenes = list(dict.fromkeys(r.scene_id for r in records))
    episodes = list(dict.fromkeys((r.scene_id, r.episode_id) for r in records))
    si, ei = (
        {v: i for i, v in enumerate(scenes)},
        {v: i for i, v in enumerate(episodes)},
    )
    first = reader.read_window(records[0], 0, reader.frame_count(records[0]))
    frames, h, w = first["target"].shape
    if frames != 100:
        raise ValueError("packed W1 requires 100 frames")
    try:
        temporary.mkdir(parents=True)
        (temporary / MARKER).write_text(FORMAT + "\n", encoding="utf-8")
        specs = (
            ("targets.npy", ARRAY_SPECS["targets"], (len(records), frames, h, w)),
            ("tx.npy", ARRAY_SPECS["tx"], (len(records), h, w)),
            ("vehicles.npy", ARRAY_SPECS["vehicles"], (len(episodes), frames, h, w)),
            ("buildings.npy", ARRAY_SPECS["buildings"], (len(scenes), h, w)),
            ("frame_ids.npy", ARRAY_SPECS["frame_ids"], (len(records), frames)),
        )
        arrays = [
            np.lib.format.open_memmap(temporary / n, mode="w+", dtype=d, shape=s)
            for n, d, s in specs
        ]
        targets, tx, vehicles, buildings, frame_ids = arrays
        seen_s, seen_e = set(), set()
        episode_frame_ids = {}
        items = []
        for i, record in enumerate(records):
            window = (
                first
                if i == 0
                else reader.read_window(record, 0, reader.frame_count(record))
            )
            for name in ("target", "vehicle", "building", "tx"):
                if np.asarray(window[name]).shape != (frames, h, w):
                    raise ValueError(f"{name} video shape mismatch")
            if len(window["frame_names"]) != frames:
                raise ValueError("frame name count mismatch")
            ids = [
                int(Path(name).stem.split("_")[-1]) for name in window["frame_names"]
            ]
            sidx, eidx = si[record.scene_id], ei[(record.scene_id, record.episode_id)]
            encoded_target = np.rint(window["target"] * 255).astype(np.uint8)
            decoded_target = encoded_target.astype(np.float32) / 255.0
            if not np.array_equal(decoded_target, window["target"]):
                raise ValueError("target is not losslessly uint8-packable")
            if not np.array_equal(
                window["tx"], np.broadcast_to(window["tx"][0], (frames, h, w))
            ):
                raise ValueError("TX heatmap changes within a video")
            if not np.array_equal(
                window["building"],
                np.broadcast_to(window["building"][0], (frames, h, w)),
            ):
                raise ValueError("building mask changes within a video")
            targets[i] = encoded_target
            tx[i] = window["tx"][0]
            frame_ids[i] = ids
            if sidx not in seen_s:
                buildings[sidx] = window["building"][0]
                seen_s.add(sidx)
            elif not np.array_equal(buildings[sidx], window["building"][0]):
                raise ValueError(f"building mismatch within scene {record.scene_id}")
            if eidx not in seen_e:
                vehicles[eidx] = window["vehicle"].astype(np.uint8)
                episode_frame_ids[eidx] = ids
                seen_e.add(eidx)
            else:
                if episode_frame_ids[eidx] != ids:
                    raise ValueError(
                        f"frame IDs mismatch within episode {record.scene_id}/{record.episode_id}"
                    )
                if not np.array_equal(vehicles[eidx], window["vehicle"]):
                    raise ValueError(
                        f"vehicle mismatch within episode {record.scene_id}/{record.episode_id}"
                    )
            items.append(
                {
                    "scene_id": record.scene_id,
                    "episode_id": record.episode_id,
                    "tx_id": record.tx_id,
                    "scene_index": sidx,
                    "episode_index": eidx,
                }
            )
            if progress is not None:
                progress(i + 1, len(records))
        for array in arrays:
            array.flush()
        manifest = {
            "format": FORMAT,
            "state": "complete",
            "source_root": str(Path(source_root).resolve()),
            "split_file": str(Path(split_file).resolve()),
            "split": split,
            "tx_heatmap_sigma_px": float(tx_heatmap_sigma_px),
            "videos": len(records),
            "episodes": len(episodes),
            "scenes": len(scenes),
            "frames_per_video": frames,
            "shape": [frames, h, w],
            "target_dtype": "uint8",
            "vehicle_dtype": "uint8",
            "tx_dtype": "float32",
            "building_dtype": "float32",
            "frame_id_dtype": "int32",
            "records": items,
        }
        tmp = temporary / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, temporary / "manifest.json")
        os.replace(temporary, root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def inspect_cache(root):
    return json.loads((Path(root).expanduser().resolve() / "manifest.json").read_text())


def verify_cache(
    root,
    *,
    source_root,
    split_file,
    tx_heatmap_sigma_px=1.5,
    split="train",
    sample_videos=64,
):
    packed = PackedFrameReader(
        root,
        source_root=source_root,
        split_file=split_file,
        tx_heatmap_sigma_px=tx_heatmap_sigma_px,
        split=split,
    )
    legacy = LegacyFrameReader(
        source_root,
        split,
        split_file,
        tx_heatmap_sigma_px=tx_heatmap_sigma_px,
    )
    packed_ids = [record.video_id for record in packed.records]
    legacy_ids = [record.video_id for record in legacy.records]
    if packed_ids != legacy_ids:
        raise ValueError("packed and legacy video order differs")
    count = min(int(sample_videos), len(packed.records))
    if count <= 0:
        raise ValueError("sample_videos must be positive")
    indices = np.linspace(0, len(packed.records) - 1, num=count, dtype=np.int64)
    compared_frames = 0
    for index in np.unique(indices):
        packed_record = packed.records[int(index)]
        legacy_record = legacy.records[int(index)]
        last = packed.frame_count(packed_record) - 1
        for start in sorted({0, last // 2, last}):
            packed_window = packed.read_window(packed_record, start, 1)
            legacy_window = legacy.read_window(legacy_record, start, 1)
            for key in ("building", "tx", "vehicle", "target"):
                if not np.array_equal(packed_window[key], legacy_window[key]):
                    raise ValueError(
                        f"packed verification mismatch: {packed_record.video_id} "
                        f"frame={start} key={key}"
                    )
            if packed_window["frame_names"] != legacy_window["frame_names"]:
                raise ValueError("packed frame name verification mismatch")
            compared_frames += 1
    return {
        "format": FORMAT,
        "state": "verified",
        "videos": len(packed.records),
        "sampled_videos": len(np.unique(indices)),
        "compared_frames": compared_frames,
    }


def remove_cache(root, *, confirm=False):
    root = Path(root).expanduser().resolve()
    if not confirm:
        raise ValueError("remove requires explicit confirmation")
    candidates = (root, Path(str(root) + ".building"))
    for path in candidates:
        if path.is_symlink() or (path.exists() and not _has_marker(path)):
            raise ValueError(f"refusing unmarked path: {path}")
    for path in candidates:
        if path.exists():
            shutil.rmtree(path)
