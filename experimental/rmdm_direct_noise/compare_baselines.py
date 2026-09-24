"""Evaluate existing baselines with the scratch screen's CPU-paired protocol."""
import argparse
from pathlib import Path

import torch

from experimental.noise_temporal_rmdm.config import load_config
from .scratch import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['original', 'noise_t1'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config('experimental/noise_temporal_rmdm/t1.yaml')
    config.data.root = args.data_root
    config.data.split_file = str(Path(config.data.split_file).resolve())
    config.validation.batch_size = 16
    config.validation.rates = [1., 2., 3.]
    config.validation.noise_standard_deviations = [0., .01, .03, .05, .07, .09]
    if args.kind == 'original':
        from experimental.noise_temporal_rmdm.validate_original_rmdm import _load_model
        model, step = _load_model(args.checkpoint)
    else:
        from experimental.noise_temporal_rmdm.model import build_model
        from experimental.noise_temporal_rmdm.checkpoint import load
        model = build_model(config)
        payload = load(args.checkpoint, model, expected_phase='t1')
        step = int(payload['global_step'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_validation(config, checkpoint_path=args.checkpoint, repository_root=Path.cwd(),
                   output_path=args.output, model=model.cuda(), checkpoint_step=step,
                   evaluated_model=args.kind)


if __name__ == '__main__':
    main()
