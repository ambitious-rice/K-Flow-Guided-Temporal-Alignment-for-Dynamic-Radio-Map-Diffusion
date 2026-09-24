"""Train clean-observation baselines with validation-selected stage checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rmdm.data import SamplingPolicy
from .data import ROOT, protocol, train_dataset, evaluation_batches, to_device, evaluate
from .models import build_model, conditions
from .losses import RMEObjectives, VAELoss


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['radiounet', 'rmegan', 'vae', 'radiodiff'], required=True)
    p.add_argument('--phase', choices=['first', 'second'], default='first')
    p.add_argument('--data-root', required=True)
    p.add_argument('--cache-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--init', default='')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val-every', type=int, default=4000)
    p.add_argument('--val-batch-size', type=int, default=8)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--disc-start', type=int, default=20000)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.val_every, args.patience) < 1:
        raise ValueError('Invalid training limits')
    if args.phase == 'second' and args.method not in ('radiounet', 'rmegan'):
        raise ValueError('Second phase applies only to RadioUNet/RME-GAN')
    if (args.phase == 'second' or args.method == 'radiodiff') and not args.init and not args.resume:
        raise ValueError('Second-stage training needs the selected first-stage checkpoint')
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device('cuda:0')
    output = Path(args.output).resolve()
    if args.resume:
        if not (output / 'last.pt').exists():
            raise FileNotFoundError(output / 'last.pt')
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(state='initializing', pid=os.getpid(), args=vars(args)))
    cfg = protocol(args.data_root)
    dataset = train_dataset(cfg, args.cache_root)
    val_batches = evaluation_batches(cfg, batch_size=args.val_batch_size, smoke=args.smoke)
    model = build_model(args.method, args.phase).to(device)
    criterion = None
    if args.method == 'rmegan':
        criterion = RMEObjectives().to(device)
        discriminator = model.discriminator
    elif args.method == 'vae':
        criterion = VAELoss().to(device)
        discriminator = criterion.discriminator
    else:
        discriminator = None
    if args.init and not args.resume:
        initial = torch.load(args.init, map_location='cpu', weights_only=False, mmap=True)
        if args.method == 'radiodiff':
            if initial['args']['method'] != 'vae':
                raise ValueError('RadioDiff requires the clean-data VAE checkpoint')
            model.vae.load_state_dict(initial['model'], strict=True)
        else:
            if initial['args']['method'] != args.method or initial['args']['phase'] != 'first':
                raise ValueError('Unexpected first-stage checkpoint')
            model.load_state_dict(initial['model'], strict=True)
        del initial

    parameters = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('discriminator.')]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .99), weight_decay=0)
    d_optimizer = (torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=0)
                   if discriminator is not None else None)
    step = epoch = consumed = stale = 0
    best = float('inf')
    best_step = 0
    checks = {}
    if args.resume:
        payload = torch.load(output / 'last.pt', map_location=device, weights_only=False)
        for key in ('method', 'phase', 'batch_size', 'accumulation', 'seed', 'steps', 'lr', 'disc_start'):
            if vars(args)[key] != payload['args'][key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        if d_optimizer:
            discriminator.load_state_dict(payload['discriminator'])
            d_optimizer.load_state_dict(payload['d_optimizer'])
        step, epoch, consumed = payload['step'], payload['epoch'], payload['consumed_batches']
        best, best_step, stale = payload['best'], payload['best_step'], payload['stale']
        checks = payload['checks']
        torch.set_rng_state(payload['rng']['cpu'].cpu())
        torch.cuda.set_rng_state(payload['rng']['cuda'].cpu())
        np.random.set_state(payload['rng']['numpy'])
        random.setstate(payload['rng']['python'])
        del payload
    metadata = dict(args=vars(args), protocol=cfg.to_dict(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, initialization=args.init or 'random', train_samples=len(dataset),
        condition_layout=['building', 'zero_tx', 'vehicle', 'sparse_rss', 'sampling_mask'],
        observation_sigma=0., global_batch_size=args.batch_size * args.accumulation,
        generator_trainable_parameters=sum(p.numel() for p in parameters),
        checkpoint_selection='clean validation full-image MSE averaged over rates 1/2/3; VAE reconstruction MSE',
        sampling_randomness='CPU; same stateless mask policy as paired evaluation',
        method_label={'radiounet': 'RadioUNet-adapted', 'rmegan': 'RME-GAN-adapted',
                      'vae': 'RadioDiff-VAE', 'radiodiff': 'RadioDiff-adapted'}[args.method])
    write_json(output / 'config.json', metadata)
    sampling = SamplingPolicy(cfg.sampling, split='train')
    started = time.monotonic()
    starting_step = step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if d_optimizer:
        d_optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        dataset.set_epoch(epoch)
        sampling.set_epoch(epoch)
        # Recreate workers per epoch: WindowDataset stores epoch in process-local state.
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch),
                            num_workers=args.workers, pin_memory=True, persistent_workers=False)
        micro = 0
        running = {}
        for index, dense in enumerate(loader):
            if index < consumed:
                continue
            sparse_cpu = sampling(dense)  # No measurement-noise function is used during training.
            if not checks.get('clean_observations'):
                if not torch.equal(sparse_cpu['observed_rss'], sparse_cpu['target'] * sparse_cpu['sampling_mask']):
                    raise AssertionError('Training observations are not clean')
                checks['clean_observations'] = True
            batch = to_device(sparse_cpu, device)
            x, target = conditions(batch), batch['target'][:, 0]
            if micro == 0:
                progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
                scale = min(1., (step + 1) / max(1, args.warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
                for opt in (optimizer, d_optimizer):
                    if opt:
                        for group in opt.param_groups:
                            group['lr'] = args.lr * scale
            use_disc = discriminator is not None and (args.method == 'rmegan' or step >= args.disc_start)
            if discriminator is not None:
                discriminator.requires_grad_(False).eval()
            # Float32 for the VAE adaptive GAN weight/LPIPS and RME losses; mixed backbone precision.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.method == 'radiodiff':
                    loss = model(x, target)
                    logs = dict(diffusion=loss)
                elif args.method == 'vae':
                    pred, posterior = model(target)
                else:
                    pred = model(x)
            if args.method == 'vae':
                loss, logs = criterion.generator_loss(model, pred, target, posterior, use_disc)
            elif args.method == 'rmegan':
                logits = model.discriminate(pred.float(), x)
                adv = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                loss, logs = criterion(pred, target, x, args.phase, adv)
            elif args.method == 'radiounet':
                loss = F.mse_loss(pred.float(), target)
                logs = dict(mse=loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}, microbatch {micro}')
            (loss / args.accumulation).backward()
            if use_disc:
                discriminator.requires_grad_(True).train()
                if args.method == 'rmegan':
                    real = model.discriminate(target, x)
                    fake = model.discriminate(pred.detach().float(), x)
                    d_loss = .5 * (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                                   F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
                else:
                    real, fake = discriminator(target), discriminator(pred.detach().float())
                    d_loss = .5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())
                if not torch.isfinite(d_loss):
                    raise FloatingPointError('Nonfinite discriminator loss')
                (d_loss / args.accumulation).backward()
                logs['discriminator'] = d_loss
            logs['loss'] = loss
            for name, value in logs.items():
                running[name] = running.get(name, 0.) + value.detach().item() / args.accumulation
            micro += 1
            consumed = index + 1
            # Dataset has ample full batches; a rare epoch tail uses the actual microbatch count.
            if micro < args.accumulation and consumed < len(loader):
                continue
            if micro < args.accumulation:
                factor = args.accumulation / micro
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
                if args.method == 'vae' and use_disc:
                    for parameter in discriminator.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(factor)
            if not checks.get('backward'):
                missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
                           and not n.startswith('discriminator.')]
                if missing:
                    raise AssertionError(f'Trainable parameters without gradients: {missing}')
                if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters):
                    raise AssertionError('All generator gradients are zero')
                checks['backward'] = True
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            probe = next(p for p in parameters if p.grad is not None and p.grad.abs().sum() > 0)
            before = probe.detach().clone() if not checks.get('weights_updated') else None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if use_disc:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1., error_if_nonfinite=True)
                d_optimizer.step()
            if d_optimizer:
                d_optimizer.zero_grad(set_to_none=True)
            if before is not None:
                checks['weights_updated'] = bool((probe.detach() != before).any())
                if not checks['weights_updated']:
                    raise AssertionError('Optimizer failed to update weights')
            step += 1
            if step == 1 or step % 50 == 0 or args.smoke:
                row = dict(step=step, epoch=epoch, **running, grad_norm=float(norm),
                           lr=optimizer.param_groups[0]['lr'], observation_sigma=0.,
                           seconds_per_step=(time.monotonic() - started) / max(1, step - starting_step))
                with (output / 'train.jsonl').open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                write_json(output / 'status.json', dict(state='running', pid=os.getpid(), step=step,
                           best_mse=best if math.isfinite(best) else None, best_step=best_step, checks=checks))
            micro, running = 0, {}
            if step % args.val_every == 0 or step == args.steps:
                result = evaluate(model, args.method, cfg, val_batches, device=device)
                result.update(step=step, phase=args.phase)
                write_json(output / f'val_{step:06d}.json', result)
                score = result['score']
                improved = score < best
                if improved:
                    best, best_step, stale = score, step, 0
                else:
                    stale += 1
                # VAE must have a chance to train its adversarial phase before stopping.
                can_stop = args.method != 'vae' or step >= args.disc_start + args.val_every
                stopping = step >= args.steps or (stale >= args.patience and can_stop)
                payload = dict(args=vars(args), model=model.state_dict(), optimizer=optimizer.state_dict(),
                               step=step, epoch=epoch, consumed_batches=consumed, best=best,
                               best_step=best_step, stale=stale, checks=checks,
                               rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                                        numpy=np.random.get_state(), python=random.getstate()))
                if discriminator is not None:
                    payload.update(discriminator=discriminator.state_dict(), d_optimizer=d_optimizer.state_dict())
                save_checkpoint(output / 'last.pt', payload)
                if improved:
                    # Selection checkpoint omits optimizer state; last.pt supports exact resume.
                    save_checkpoint(output / 'best.pt', dict(args=vars(args), model=model.state_dict(), step=step, score=score))
                if stopping and args.smoke:
                    selected = torch.load(output / 'best.pt', map_location=device, weights_only=False)
                    model.load_state_dict(selected['model'], strict=True)
                    replay = evaluate(model, args.method, cfg, val_batches, device=device)
                    if abs(replay['score'] - selected['score']) > 1e-7:
                        raise AssertionError('Checkpoint roundtrip changed deterministic validation')
                    checks['checkpoint_roundtrip'] = True
                    write_json(output / 'roundtrip.json', replay)
                write_json(output / 'status.json', dict(state='complete' if stopping else 'running',
                           step=step, pid=os.getpid(), best_mse=best, best_step=best_step, stale=stale,
                           checks=checks, stop_reason=('max_steps' if step >= args.steps else 'early_stopping') if stopping else None))
                if stopping:
                    return
                model.train()
            if step >= args.steps:
                return
        epoch += 1
        consumed = 0


if __name__ == '__main__':
    main()
