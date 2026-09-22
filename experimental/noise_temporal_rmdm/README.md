# Noise-aware Tx-blind RMDM

This experiment contains the completed single-frame T1 model and its T16
joint spatio-temporal continuation. The local Git checkout is the source of
truth.

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

The v2 cache is disposable and consumes about 17 GB. Keep it while training or
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

## T16 joint model

T16 uses continuous 16-frame windows from one video. The measurement-noise
sampler already draws one sigma per clip and independent pixel/frame errors
conditional on that sigma, so no noise semantics change from T1. Tx remains
absent from the reader, cache, model, and loss.

The complete T1 condition branch and legacy diffusion U-Net are copied from the
validation-selected step-20250 checkpoint. Three temporal refiners are added at
the existing `[B,T,C,H,W]` boundaries:

- the clean calibration prior produced by HWM;
- the four-channel diffusion condition `[building, vehicle, calibration,
  normalized_variance]`;
- the predicted diffusion noise.

Each refiner encodes frames to a 32x32 spatial grid with 192 channels, applies
four pre-normalized six-head temporal-attention/MLP blocks independently at
each spatial location, and decodes a residual at the original resolution. The
spatial encoder also contains a residual convolutional block, allowing each
temporal token to summarize a local image region. Across the three refiners,
T16 adds 7,379,142 trainable parameters: T1 has 63,219,298 parameters and T16
has 70,598,440. All weights remain jointly trainable.

Each temporal residual output projection is zero initialized. The strict
initializer accepts only the T1 checkpoint schema, copies every one of its 548
state tensors, and permits missing keys only below `temporal_hook.stages`. As a
result, the initial T16 function is exactly equal to applying the selected T1
model frame by frame; temporal behavior is learned without an initialization
jump.

The reviewed remote configuration is `t16.yaml`. It points to the persistent
17 GB deduplicated mmap cache and selected T1 checkpoint on `Nice2`. The code is
ready but formal training must not be launched until user approval.

For the local two-GPU launch, `t16_local.yaml` uses the tmpfs v2 cache and the
local selected T1 checkpoint. The 96 GB cards use five clips per GPU with
accumulation 3, preserving the same global 30-clip batch and 21,600-step sample
exposure as the remote configuration.

Proposed two-GPU launch after approval:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src:. \
  /share1/fzj/miniconda3/envs/RMDM/bin/python \
  -m accelerate.commands.launch --multi_gpu --num_processes 2 \
  --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  --main_process_port 29629 \
  -m experimental.noise_temporal_rmdm.train \
  --config experimental/noise_temporal_rmdm/t16.yaml \
  --repository-root /data_16T_137/fzj/RMDM/project
```

The reviewed baseline is three clips per GPU with accumulation 5: 30 clips or
480 frames per optimizer update. Train for 21,600 optimizer steps with a
`5e-5` cosine learning rate, 540-step warmup, and `5e-6` floor. This exposes
exactly 10.368 million frames, matching T1. Save and validate every 2,700
steps. Checkpoint selection uses only the tracked two-scene DDIM20
validation protocol; evaluate at least the final four milestones before
selection. The partial two-scene DDIM20 test is run once after selection and is
report-only. Full 128x128 BF16 forward/backward checks used 10.37 GiB for one
clip and 40.13 GiB for four clips on a 96 GB GPU; three clips leaves necessary
headroom on each remote 48 GB GPU for optimizer and DDP/NCCL state.
