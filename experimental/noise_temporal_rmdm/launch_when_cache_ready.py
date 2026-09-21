"""Wait for the RAM cache, validate it, then replace this process with formal training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from .config import load_config
from .packed_data import PackedFrameReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--gpus", default="0,2,4")
    parser.add_argument("--main-process-port", type=int, default=29629)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = load_config(config_path)
    cache_root = Path(config.data.packed_cache_root).expanduser().resolve()
    if not config.data.packed_cache_root:
        raise ValueError("formal auto-launch requires data.packed_cache_root")
    deadline = time.monotonic() + args.timeout_seconds
    metadata = cache_root / "metadata.json"
    while not metadata.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"packed cache did not become ready: {cache_root}")
        time.sleep(args.poll_seconds)

    split_file = Path(config.data.split_file)
    if not split_file.is_absolute():
        split_file = repository_root / split_file
    reader = PackedFrameReader(cache_root, split_file=split_file)
    reader.read_window(reader.records[0], 0, 1)
    reader.read_window(reader.records[-1], config.data.frames_per_video - 1, 1)
    print(f"validated packed cache with {len(reader.records)} videos; launching formal training", flush=True)

    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    python_path = [str(repository_root / "src"), str(repository_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", str(len(gpu_ids)), "--num_machines", "1",
        "--mixed_precision", config.train.mixed_precision, "--dynamo_backend", "no",
        "--main_process_port", str(args.main_process_port),
        "-m", "experimental.noise_temporal_rmdm.train",
        "--config", str(config_path), "--repository-root", str(repository_root),
    ]
    os.chdir(repository_root)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
