"""Run independent baseline stages sequentially on one assigned GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import ROOT
from .train import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane', choices=['conv', 'diffusion'], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.lane == 'conv':
        stages = [('radiounet_first', 'radiounet', 'first', '', 20000, 32, 4, 1e-4),
                  ('radiounet_second', 'radiounet', 'second', 'radiounet_first', 20000, 32, 4, 1e-4),
                  ('rmegan_global', 'rmegan', 'first', '', 20000, 32, 4, 1e-4),
                  ('rmegan_local', 'rmegan', 'second', 'rmegan_global', 20000, 32, 4, 5e-5)]
    else:
        stages = [('radiodiff_vae', 'vae', 'first', '', 40000, 8, 16, 5e-5),
                  ('radiodiff', 'radiodiff', 'first', 'radiodiff_vae', 40000, 8, 16, 5e-5)]
    status_path = root / f'{args.lane}_pipeline.json'
    state = dict(state='running', pid=os.getpid(), lane=args.lane,
                 gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 smoke=args.smoke, completed=[])
    write_json(status_path, state)
    for name, method, phase, previous, steps, batch, accumulation, lr in stages:
        output = root / name
        status = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        if status.get('state') == 'complete' and (output / 'best.pt').exists():
            state['completed'].append(name)
            continue
        cmd = [sys.executable, '-m', 'experimental.radio_baselines.train', '--method', method,
               '--phase', phase, '--data-root', args.data_root, '--cache-root', args.cache_root,
               '--output', str(output), '--steps', str(2 if args.smoke else steps),
               '--batch-size', str(2 if args.smoke else batch),
               '--accumulation', str(1 if args.smoke else accumulation), '--lr', str(lr),
               '--workers', '0' if args.smoke else '4', '--warmup', '1' if args.smoke else '500',
               '--val-every', '2' if args.smoke else '4000', '--val-batch-size', '2' if args.smoke else '8',
               '--disc-start', '0' if args.smoke else '20000']
        if previous:
            cmd += ['--init', str(root / previous / 'best.pt')]
        if args.smoke:
            cmd += ['--smoke']
        if output.exists():
            if (output / 'last.pt').exists():
                cmd += ['--resume']
            else:
                raise RuntimeError(f'Incomplete stage without a resumable checkpoint: {output}')
        state.update(current_stage=name, command=cmd)
        write_json(status_path, state)
        with (root / f'{name}.log').open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            write_json(status_path, state)
            code = process.wait()
        if code:
            failed = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
            failed.update(state='failed', exit_code=code, log=str(root / f'{name}.log'))
            write_json(output / 'status.json', failed)
            state.update(state='failed', exit_code=code)
            write_json(status_path, state)
            raise SystemExit(code)
        state['completed'].append(name)
        write_json(status_path, state)
    groups = ({'radiounet': ['radiounet_first', 'radiounet_second'],
               'rmegan': ['rmegan_global', 'rmegan_local']} if args.lane == 'conv'
              else {'radiodiff': ['radiodiff']})
    for method, candidates in groups.items():
        choices = []
        for stage in candidates:
            status = json.loads((root / stage / 'status.json').read_text())
            choices.append(dict(stage=stage, checkpoint=str(root / stage / 'best.pt'),
                                mse=status['best_mse'], step=status['best_step']))
        write_json(root / f'{method}_selected.json', dict(method=method,
                   selection_role='clean_validation_only', candidates=choices,
                   selected=min(choices, key=lambda x: x['mse'])))
    state.update(state='complete', finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(status_path, state)


if __name__ == '__main__':
    main()
