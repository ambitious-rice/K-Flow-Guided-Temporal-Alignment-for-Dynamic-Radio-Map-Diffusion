"""Freeze three matched protocols and reuse existing integer-seeded input bank."""
import json
from pathlib import Path
from experimental.noise_loss_screen.common import write
from .model import VARIANTS

ROOT=Path('runs/noise_paper_injection_20261009')


def main():
    old=Path('runs/noise_scratch_full_20261008').resolve()
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    protocol=dict(goal='Test paper-inspired sigma injection on6GPUs,2pervariant; no old-model diagnostic phase',
                  variants=list(VARIANTS),gpus=['0,1','2,3','4,5'],initialization='All random, identical common backbone seed; no pretrained weights',
                  training='Full9000 videos, W16; same arithmetic RNG and batches across arms; sigma U[0,.09); rates discrete uniform1-10%',
                  objective='Original six terms weight1; clean sampled-point MSE; final x0 after two differentiable steps; HWM/cal auxiliary terms once',
                  reference_inputs=str(old/'inputs'),validation='Reuse exact fixed24videos: tune8,confirmation16, rates1/2/3%; four true/input sigmas plus intermediate truth for8videos',
                  pairing='Same observations, mask, initial diffusion tensor and validation sampler as previous scratch/paper-model comparison',
                  selection='Tune only: best reconstruction, largest positive sigma gain, lowest reconstruction among positive-gain candidates; confirmation never used to select',
                  final_repeats='Chosen best_usable or best_alignment; two extra diffusion seeds; sigma/mask/observations unchanged; seed20261009+repeat*100000+filtered_entry_index',
                  success='Report true-vs-tune-selected-fixed gain on confirmation at each rate and aggregate, row minima, bootstrap over paired videos, reconstruction versus frozen original/scratch models; no forced wrong-sigma loss',
                  limits='Adapted mechanisms, not paper reproductions; HWM remains observation-conditioned; learned proximal modules are not claimed exact Bayesian updates; equal steps/data but measured runtime may differ',
                  time_budget='7h training+validation;8.75h segment hardlimit incl final evaluation, checkpoint/resume on time limit')
    write(ROOT/'protocol.json',protocol)
    for variant,gpus in zip(VARIANTS,protocol['gpus']):
        root=ROOT/variant;root.mkdir(exist_ok=True)
        (root/'inputs').symlink_to(old/'inputs',target_is_directory=True)
        (root/'model_config.yaml').write_text((old/'model_config.yaml').read_text())
        spec=dict(variant=variant,gpus=gpus,seed=20261009,max_steps=40000,microbatch=8,accumulation=2,learning_rate=1e-4,warmup=1000,ema_decay=.999,
                  validation_every=1000,validation_milestones=[500,1500,2500],initial_validation=False,checkpoint_every=500,rollout_gap=20,prox_lambda=7.,train_hours=7.,total_hours=8.75,
                  early_stopping=dict(min_steps=4000,patience=4,mse_relative_delta=.002,alignment_absolute_delta=.002))
        write(root/'config.json',spec)
    record=dict(goal=protocol['goal'],status='prepared',cwd=str(Path.cwd()),environment='/data_p6/fzj/conda/envs/RMDM/bin/python',protocol=protocol,
                output=str(ROOT.resolve()),next='Functional tests and simultaneous2GPU smokes, then matched three-arm training',source=dict(dirty=True))
    write(Path('.agents/runs')/(ROOT.name+'.yaml'),record)


if __name__=='__main__':main()
