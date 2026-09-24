"""Evaluate a selected adapted baseline; test split is explicit and never used by training."""
import argparse
import json
from pathlib import Path

import torch

from .data import protocol, evaluation_batches, evaluate
from .models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--sigmas', default='0,0.01,0.03,0.05,0.07,0.09')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    method = checkpoint['args']['method']
    if method == 'vae':
        raise ValueError('VAE reconstruction is not a radio-map prediction baseline')
    model = build_model(method, checkpoint['args']['phase']).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    cfg = protocol(args.data_root)
    batches = evaluation_batches(cfg, split=args.split, batch_size=args.batch_size)
    result = evaluate(model, method, cfg, batches, device=torch.device('cuda:0'), split=args.split,
                      sigmas=tuple(float(x) for x in args.sigmas.split(',')), steps=args.steps)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=checkpoint['step'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
