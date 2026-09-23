"""Reuse the image cache; keep source supervision separate from model inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experimental.noise_temporal_rmdm.packed_data import PackedFrameReader
from rmdm.data import WindowDataset
from rmdm.legacy import LegacyFrameReader
from .config import load_config


class TrainingDataset(WindowDataset):
    def __init__(self, config):
        reader = PackedFrameReader(config.data.packed_cache_root, source_root=config.data.root,
                                   split_file=config.data.split_file)
        super().__init__(reader=reader, window_size=config.data.window_size, seed=config.sampling.seed,
                         fixed_starts=range(100) if config.data.window_size == 1 else None)
        self.source_masks = np.load(config.source_masks, mmap_mode="r")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        record, _ = self._resolve_item(index)
        source = torch.from_numpy(self.source_masks[record.index].copy()).float()
        item["source_label"] = source[None, None].expand(self.window_size, 1, -1, -1)
        return item


def prepare_source_masks(config):
    packed = PackedFrameReader(config.data.packed_cache_root, source_root=config.data.root,
                               split_file=config.data.split_file)
    reader = LegacyFrameReader(config.data.root, "train", str(Path(config.data.split_file).resolve()), include_tx=False)
    records = {record.video_id: reader.dataset.records[record.index] for record in reader.records}
    destination = Path(config.source_masks)
    temporary = destination.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                      shape=(len(packed.records), 128, 128))
    for index, record in enumerate(packed.records):
        array[index] = reader.dataset._make_tx_heatmap(records[record.video_id], sigma_px=1.5) > 0.5
    array.flush()
    if not np.all(array.sum((1, 2)) > 0):
        raise ValueError("Some training sources have no pixels inside the map")
    temporary.replace(destination)
    print(f"Prepared {len(array)} source masks at {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    prepare_source_masks(load_config(parser.parse_args().config))
