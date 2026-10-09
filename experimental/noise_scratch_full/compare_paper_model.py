"""Evaluate the frozen paper W16 checkpoint on the scratch run's exact inputs."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from experimental.noise_loss_screen.common import write
from experimental.paired_evaluation.evaluate import load_model
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rank, local, world = (int(os.environ[k]) for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.manual_seed(20261008)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = Path('runs/noise_hvdit_tolerance_20260924/w16_x0/checkpoints/step_016000.pth')
    config = 'experimental/noise_hvdit/w16_tolerance.yaml'
    start = time.time()
    output = root / 'paper_model_comparison'
    if rank == 0:
        write(output / 'status.json', dict(state='loading', started_at=start, checkpoint=str(checkpoint), weights='model', world=world))
    model, cfg = load_model('hvdit', config, str(checkpoint), 'model')
    model = model.cuda().eval()
    if rank == 0:
        write(output / 'status.json', dict(state='evaluating', started_at=start, checkpoint=str(checkpoint), weights='model', world=world))
    rows = evaluate(model, cfg, root / 'inputs', rank, world, full=True)
    if rank == 0:
        assert len(rows) == 1512
        write(output / 'matrix.json', dict(complete=True, checkpoint=str(checkpoint), weights='model', config=config, rows=rows))
        write(output / 'status.json', dict(state='complete', started_at=start, finished_at=time.time(), seconds=time.time()-start, rows=len(rows), checkpoint=str(checkpoint), weights='model', world=world))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
