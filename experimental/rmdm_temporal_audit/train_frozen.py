"""Train only zero-initialized temporal refiners from the selected W1 model."""

import argparse
from pathlib import Path

from experimental.noise_temporal_rmdm.config import load_config
from experimental.noise_temporal_rmdm.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    source = Path("experimental/noise_temporal_rmdm/t16_local.yaml")
    config = load_config(source)
    config.runtime.output_root = args.output_root
    config.train.max_steps = args.steps
    config.train.checkpoint_every_steps = args.checkpoint_every
    config.validation.every_steps = args.checkpoint_every
    config.train.per_gpu_batch_size = args.batch_size
    config.train.gradient_accumulation_steps = 3
    config.train.learning_rate = 2e-5
    config.train.min_learning_rate = 2e-6
    config.train.warmup_steps = min(60, args.steps // 5)
    config.model.expected_trainable_parameters_min = 0
    run(config, config_path=source, repository_root=Path.cwd(), freeze_spatial=True,
        initialize_from_t1=config.runtime.initialize_from_t1)


if __name__ == "__main__":
    main()
