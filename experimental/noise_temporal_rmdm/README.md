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
manifest deterministically selects 10 videos from each validation scene. A
separate manifest selects 10 videos from each final-test scene; test samples
must not be used for checkpoint selection.

## Commands

GPU selection is an operational launch decision: check live utilization, then
set `CUDA_VISIBLE_DEVICES` and the Accelerate process count. It is deliberately
not hard-coded or policy-gated in the experiment configuration.

The formal loader uses a generated contiguous uint8 mmap cache so the 900,000
frames can remain globally shuffled without reopening individual PNGs or
decompressing whole traffic NPZ files per sample. Targets are stored per video,
vehicle grids once per episode, and building masks once per scene. Tx is not
stored. Build it on the machine-local fast filesystem:

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.noise_temporal_rmdm.build_cache \
  --config experimental/noise_temporal_rmdm/t1.yaml \
  --output /dev/shm/noise_temporal_clean16_train_v2 --workers 32
```

The builder writes to a `.building` directory, atomically publishes the final
directory only after completion, and then compares 64 evenly spaced videos at
their first, middle, and last frames against the source reader. The runner
checks the source and split paths plus every array shape before training. The
original PNG/NPZ layout remains unchanged. If no packed cache is configured,
the fallback loader shuffles video blocks to preserve NPZ cache locality.

To launch automatically as soon as a concurrently built cache becomes ready:

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.noise_temporal_rmdm.launch_when_cache_ready \
  --config experimental/noise_temporal_rmdm/t1.yaml --gpus 0,2,4
```

The v2 cache is disposable and consumes about 15 GB. Keep it while training or
checkpoint resume may still be needed. The local tmpfs cache path is
`/dev/shm/noise_temporal_clean16_train_v2`; the lab-server NVMe cache path is
`/home/fzj/.cache/rmdm/noise_temporal_clean16_v2`. Remove only that exact cache
directory after the corresponding run is safely complete. A reboot clears the
tmpfs copy but not the lab-server NVMe copy.

The checked-in formal target uses the currently tested three-GPU layout:
`128 x 3 GPUs x accumulation 1`, global batch 384, for 27,000 optimizer steps.
This is 10.368 million frame presentations, or about 11.52 passes over the
900,000-frame Clean16 training split. If world size changes, preserve roughly
the same sample exposure: two GPUs use `128 x 2` for 40,000 steps; four, six,
or eight GPUs can use per-GPU batches 96, 64, or 48 for 27,000 steps.

## Evaluation schedule

Every 3,375 steps, retain a milestone checkpoint and run deterministic quick
validation on the two validation scenes: 10 videos per scene, frames 0 and 50,
sampling rates 1% and 3%, sigma values 0/0.03/0.05/0.09, and DDIM20. This is
the only protocol used to select a checkpoint.

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.noise_temporal_rmdm.validate \
  --config experimental/noise_temporal_rmdm/t1.yaml \
  --checkpoint runs/noise_temporal_rmdm/t1/checkpoints/step_003375.pth \
  --output runs/noise_temporal_rmdm/t1/validation/step_003375.json \
  --split val
```

After selection, run the final partial test once: two untouched test scenes,
10 videos per scene, frames 0/25/50/75, rates 1%/2%/3%, all six configured
sigma levels, and DDIM20. Test results are report-only and never feed back
into training or checkpoint choice.

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.noise_temporal_rmdm.validate \
  --config experimental/noise_temporal_rmdm/t1.yaml \
  --checkpoint <validation-selected-checkpoint> \
  --output runs/noise_temporal_rmdm/t1/final_test.json \
  --split test
```

Project policy requires formal training to use a tested, committed and pushed
revision. Development-only smoke programs and tests live on the dedicated
`experimental/noise-temporal-rmdm-smoke` branch. The production runner records
Git metadata but does not implement policy as runtime gate code; see this
experiment's scoped `AGENTS.md`.
