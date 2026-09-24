"""Validation-only condition ablation for a checkpoint trained from scratch."""
import argparse
from pathlib import Path

import torch

from experimental.noise_temporal_rmdm.config import load_config
from .scratch import ScratchNoiseRMDM, run_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reported-sigma', type=float, default=0.)
    parser.add_argument('--data-root')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['initialization'] != 'random_from_scratch':
        raise ValueError('Expected this experiment\'s from-scratch checkpoint')
    config = load_config(checkpoint['args']['config'])
    config.data.root = args.data_root or checkpoint['args']['data_root']
    config.data.split_file = str(Path(config.data.split_file).resolve())
    config.validation.batch_size = 16
    config.validation.rates = [1., 2., 3.]
    config.validation.noise_standard_deviations = [0., .01, .03, .05, .07, .09]
    model = ScratchNoiseRMDM(checkpoint['mode'], checkpoint['args']['seed'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.cuda()
    model.reported_sigma = args.reported_sigma
    run_validation(config, checkpoint_path=args.checkpoint, repository_root=Path.cwd(),
        output_path=args.output, model=model, checkpoint_step=checkpoint['step'],
        evaluated_model=f"scratch_{checkpoint['mode']}_reported_sigma_{args.reported_sigma:g}")


if __name__ == '__main__':
    main()
