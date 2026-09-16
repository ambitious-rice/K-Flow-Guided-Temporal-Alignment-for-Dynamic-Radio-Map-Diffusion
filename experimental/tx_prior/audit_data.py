"""CPU-only index and stratified label/condition/cache audit; no data writes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

from .config import load_config
from .evaluate import write_result
from .packed import PackedFrameReader
from .runner import _dataset, resolve_task_root


def array_summary(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), finite=bool(np.isfinite(array).all()),
                min=float(array.min()), max=float(array.max()), mean=float(array.mean()),
                energy=float(np.square(array.astype(np.float64)).mean()),
                zero_fraction=float((array == 0).mean()), one_fraction=float((array == 1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experimental/tx_prior/remote.yaml")
    parser.add_argument("--videos-per-scene", type=int, default=3)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 49, 99])
    parser.add_argument("--packed-root", default="/home/fzj/.cache/rmdm/tx_prior")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, smoke=False)
    output = Path(args.output).resolve()
    allowed = resolve_task_root(Path.cwd(), config.pipeline.output_root) / "train/data_audit"
    if not output.is_relative_to(allowed) or output.exists():
        raise ValueError("output must be new and under train/data_audit")
    if args.videos_per_scene < 1 or any(f < 0 or f >= 100 for f in args.frames):
        raise ValueError("invalid sampling size/frame")
    started = time.perf_counter()
    root = Path(config.data.root)
    index = json.loads((root / "index.json").read_text())
    metadata = json.loads((root / "dataset_meta.json").read_text())
    split = json.loads(Path(config.data.split_file).read_text())
    samples = index["samples"]
    counts = Counter(s["scene_id"] for s in samples)
    paths = [s["sample_meta"] for s in samples]
    issues = []
    if len(paths) != len(set(paths)):
        issues.append("duplicate_sample_meta_paths")
    assigned = [scene for name in ("train", "val", "test") for scene in split[name]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(counts):
        issues.append("scene_split_overlap_or_missing_scene")
    index_by_scene = {}
    for scene in counts:
        q = [s for s in samples if s["scene_id"] == scene]
        index_by_scene[scene] = {k: sum(int(s.get("stats", {}).get(k, 0)) for s in q)
            for k in ("building_nonzero_uint8_count", "vehicle_nonzero_uint8_count", "clip_low_count", "clip_high_count", "fallback_low_count", "vehicle_cell_count")}
    packed = PackedFrameReader(args.packed_root, source_root=config.data.root,
        split_file=config.data.split_file, tx_heatmap_sigma_px=config.data.tx_heatmap_sigma_px)
    packed_records = {r.video_id: r for r in packed.records}
    checked_cache = 0
    frame_rows = []
    video_rows = []
    static_buildings = {}
    split_counts = {}
    for name in ("train", "val", "test"):
        dataset = _dataset(config, split=name, fixed_starts=tuple(range(100)))
        reader = dataset.reader
        groups = defaultdict(list)
        for record in reader.records:
            groups[record.scene_id].append(record)
        split_counts[name] = dict(scenes=len(groups), videos=len(reader.records), frames=len(dataset))
        for scene, records in groups.items():
            selected = np.linspace(0, len(records)-1, min(args.videos_per_scene, len(records)), dtype=int)
            for position in selected:
                record = records[position]
                original = reader.dataset.records[record.index]
                try:
                    sample_meta = json.loads(original["sample_meta_path"].read_text())
                    if any(sample_meta.get(k) != original[k] for k in ("scene_id", "episode_id", "tx_id")):
                        issues.append(f"metadata_identity_mismatch:{record.video_id}")
                    indices = np.load(original["frame_indices_path"])
                    if not np.array_equal(indices, np.arange(100)):
                        issues.append(f"frame_indices_not_0_to_99:{record.video_id}")
                    traffic = reader.dataset._load_npz_array(original["traffic_grid_path"], "traffic_grid_uint8")
                    if traffic.shape != (100, 128, 128) or not set(np.unique(traffic)).issubset({0, 1, 2}):
                        issues.append(f"traffic_shape_or_codes:{record.video_id}")
                    png_count = sum(1 for p in original["rss_png_dir"].iterdir() if p.name.startswith("frame_") and p.suffix == ".png")
                    if png_count != 100:
                        issues.append(f"png_count:{record.video_id}:{png_count}")
                    post = sample_meta.get("rss_postprocess", {})
                    if any(post.get(k) != metadata["rss_postprocess"].get(k)
                           for k in ("rss_min_dbm", "rss_max_dbm", "shadow_denoise")):
                        issues.append(f"label_mapping_or_postprocess_mismatch:{record.video_id}")
                    video_rows.append(dict(split=name, video=record.video_id, png_count=png_count,
                        frame_indices=indices.tolist(), tx_position=sample_meta.get("tx_position"),
                        raw_rss_npz_exists=(original["tx_dir"] / "rss_maps.npz").is_file(),
                        postprocess=post, metadata_stats=sample_meta.get("stats")))
                    previous = None
                    for frame in args.frames:
                        arrays = reader.read_window(record, frame, 1)
                        for key in ("building", "tx", "vehicle", "target"):
                            a = arrays[key]
                            if a.shape != (1, 128, 128) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                                issues.append(f"invalid_array:{record.video_id}:{frame}:{key}")
                        target = arrays["target"][0]
                        building = arrays["building"][0] > .5
                        vehicle = arrays["vehicle"][0] > .5
                        if (target[building] > 0).any() or (target[vehicle] > 0).any():
                            issues.append(f"nonzero_GT_on_obstacle:{record.video_id}:{frame}")
                        png_path = original["rss_png_dir"] / f"frame_{frame:06d}.png"
                        with Image.open(png_path) as image:
                            mode = image.mode
                        if mode not in ("L", "P"):
                            issues.append(f"unexpected_png_mode:{record.video_id}:{frame}:{mode}")
                        peak = np.unravel_index(arrays["tx"][0].argmax(), target.shape)
                        maxima = np.argwhere(target == target.max())
                        distance = float(np.sqrt(((maxima - np.asarray(peak)) ** 2).sum(1)).min())
                        if arrays["tx"].max() < .5 or distance > 3:
                            issues.append(f"TX_peak_GT_alignment:{record.video_id}:{frame}:{distance}")
                        if name == "train":
                            other = packed.read_window(packed_records[record.video_id], frame, 1)
                            if any(not np.array_equal(arrays[k], other[k]) for k in arrays):
                                issues.append(f"packed_mismatch:{record.video_id}:{frame}")
                            checked_cache += 1
                        static_buildings[scene] = arrays["building"][0].copy()
                        frame_rows.append(dict(split=name, scene=scene, video=record.video_id, frame=frame,
                            png_mode=mode, target=array_summary(target),
                            building_fraction=float(building.mean()), vehicle_fraction=float(vehicle.mean()),
                            tx_peak=list(map(int, peak)), tx_max=float(arrays["tx"].max()),
                            nearest_GT_max_distance_to_TX_cells=distance,
                            temporal_mse_vs_previous_sample=None if previous is None else float(((target-previous)**2).mean()),
                            traffic_codes=list(map(int, np.unique(traffic[frame])))))
                        previous = target
                except Exception as error:
                    issues.append(f"read_error:{record.video_id}:{type(error).__name__}:{error}")
            print(f"Checked {name}/{scene}: {len(selected)} videos", flush=True)
    identical_building_scenes = []
    keys = list(static_buildings)
    for i, first in enumerate(keys):
        for second in keys[i+1:]:
            if np.array_equal(static_buildings[first], static_buildings[second]):
                identical_building_scenes.append([first, second])
    distribution = {}
    for name in ("train", "val", "test"):
        rows = [r for r in frame_rows if r["split"] == name]
        distribution[name] = {k: statistics.fmean(r["target"][k] for r in rows)
                              for k in ("mean", "energy", "zero_fraction", "one_fraction")}
    write_result(output, dict(schema="tx_prior_data_audit_v1", args=vars(args),
        index_samples=len(samples), index_scenes=dict(counts), split_counts=split_counts,
        index_scene_stats=index_by_scene, dataset_name=metadata.get("dataset_name"),
        rss_postprocess=metadata["rss_postprocess"], rss_mapping=metadata["rss_uint8_mapping"],
        raw_npz_contract=metadata.get("rss_npz_contract"), sampled_videos=len(video_rows),
        sampled_frames=len(frame_rows), packed_compared_frames=checked_cache,
        issues=issues, identical_building_scenes=identical_building_scenes,
        sampled_distribution=distribution, videos=video_rows, frames=frame_rows,
        elapsed_seconds=time.perf_counter()-started,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print(f"Audit complete: {len(issues)} issues, {len(frame_rows)} sampled frames: {output}", flush=True)


if __name__ == "__main__":
    main()
