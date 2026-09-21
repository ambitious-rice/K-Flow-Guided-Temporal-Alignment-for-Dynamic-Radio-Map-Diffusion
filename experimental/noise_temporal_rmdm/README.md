# Noise-aware Tx-blind RMDM

This experiment trains the single-frame T1 model that will initialize a later
T16 temporal model. The local Git checkout is the source of truth.

## Inputs and objective

The transmitter location is unknown and is never loaded into the experiment
batch or passed to either model branch. The scene input is `[building,
vehicle]`. The observation branch receives only `[observed_rss,
sampling_mask]` plus the known measurement variance `v=sigma^2`.

For normalized clean RSS `X`, mask `M`, and standard Gaussian `epsilon`, the
synthetic measurement is

`Y = M * (X + sigma * epsilon)`

without clipping. One sigma is used per sample/clip and pixel errors are
independent conditional on sigma. The training distribution is:

- probability 0.20: `sigma = 0`;
- probability 0.65: `sigma ~ U(0, 0.05)`;
- probability 0.15: `sigma ~ U(0.05, 0.09)`.

The RSS labels are the dataset PNG values divided by 255. Since the PNG maps
the 100 dB interval `[-124, -24] dBm` linearly to `[0,1]`, normalized sigma
0.01 is equivalent to an RSS standard deviation of 1 dB. Configuration stays
in normalized units to match the paper convention.

The loss is epsilon-prediction DDPM MSE plus clean calibration MSE and the
legacy equation/obstacle physics terms. The Tx source-anchor term is removed.
Legacy gradient checkpointing remains disabled because its backward recompute
does not preserve BF16 autocast; the target 96 GB GPUs do not require it.

## Data gates

`configs/splits/m20_formal075_clean16_scene_split.json` excludes all four
user-verified invalid scenes. The split has 12 train scenes, two validation
scenes, two final-test scenes, and no scene overlap. The tracked validation
manifest deterministically selects 10 videos from each validation scene.

## Commands

Synthetic CPU forward/backward:

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.noise_temporal_rmdm.smoke \
  --config experimental/noise_temporal_rmdm/smoke.yaml --backward
```

One-process real-data smoke (choose an actually idle GPU first):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. accelerate launch --num_processes 1 \
  -m experimental.noise_temporal_rmdm.train \
  --config experimental/noise_temporal_rmdm/smoke.yaml \
  --repository-root . --smoke --smoke-data-limit 256
```

GPU selection is an operational launch decision: check live utilization, then
set `CUDA_VISIBLE_DEVICES` and the Accelerate process count. It is deliberately
not hard-coded or policy-gated in the experiment configuration.

The formal target is global batch 256 for 40,000 optimizer steps: 10.24
million frame presentations, or about 11.38 passes over the 900,000-frame
Clean16 training split. The checked-in config is the currently tested two-GPU
layout (`128 x 2 GPUs x accumulation 1`). If more GPUs are free at launch,
preserve the same global batch with `64 x 4 x 1` or `32 x 8 x 1`; do not change
the 40,000-step horizon merely because world size changes.

Project policy requires formal training to use a tested, committed and pushed
revision. Use `t1.yaml` only after tests, real-data smoke, and a short local
DDP smoke pass. The runner records Git metadata but does not implement policy
as runtime gate code; see the repository-root `AGENTS.md`.
