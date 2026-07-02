# RMDM Temporal PINN Pilot: Easy/Hard Two-Scene Gate

## Dataset

Root:

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1
```

Source split:

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1/split.json
```

Pilot split:

```text
/data/fzj/RMDM/configs/splits/rmdm_temporal_pilot_easy_hard_2scene.json
```

Selected train scenes:

| Scene | Role | Rough difficulty proxy | Reason |
|---|---:|---:|---|
| `town05_opt_junction_2086` | easy | 0.323 | Lowest train-scene proxy score |
| `town02_opt_junction_0188` | hard | 0.802 | Highest train-scene proxy score |

The proxy was computed from sampled building occupancy, vehicle occupancy,
RSS spatial variation, and adjacent-frame RSS change. The pilot split uses a
deterministic sorted 80/10/10 split within the two selected scenes:

| Split | Tx samples |
|---|---:|
| train | 1200 |
| val | 150 |
| test | 150 |

## Baseline

Single-frame RMDM checkpoint:

```text
/data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth
```

Baseline pilot eval:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/baseline_val20_ddim20
```

Protocol:

```text
split: val
num clips: 20
clip length: 32
frame stride: 32
DDIM steps: 20
sampler model: TemporalPINNUNet initialized from the 2D checkpoint with zero temporal gates
```

Baseline metrics:

| Metric | Mean |
|---|---:|
| MAE | 0.0126958 |
| MSE | 0.00113361 |
| PSNR | 29.9138 |
| cal MAE | 0.0362648 |
| cal MSE | 0.00689362 |
| cal PINN | 0.00571002 |
| cal k-flow | 0.203784 |
| pred k-flow EPE | 0.627058 |
| pred k-flow L1 | 0.406060 |

Val50/DDIM20 baseline:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/baseline_val50_ddim20
```

| Metric | Mean |
|---|---:|
| MAE | 0.0136432 |
| MSE | 0.00125892 |
| PSNR | 29.8412 |
| pred k-flow EPE | 0.642645 |
| pred k-flow L1 | 0.415665 |

## Temporal PINN Candidate

Training session:

```text
tmux: rmdm_temporal_pilot_stage2_32f_500
```

Training output:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/stage2_32f_pos_kflow_500
```

Log:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/logs/stage2_32f_pos_kflow_500.log
```

Config summary:

```text
stage: stage2
clip_length: 32
prior_injection_mode: uemb
frame positional encoding: enabled
kflow_weight: 0.01
kflow_warmup_steps: 500
batch_size: 1
max_steps: 500
mixed_precision: bf16
GPU: CUDA_VISIBLE_DEVICES=2
```

Gate rule before full-dataset training:

```text
Proceed only if the temporal candidate improves at least one temporal/k-flow
metric without a clear frame-fidelity regression versus the baseline protocol.
```

## First Candidate Result

500-step temporal PINN candidate:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/stage2_32f_pos_kflow_500/model_temporal_pinn.pth
```

Val50/DDIM20:

```text
/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/final_val50_ddim20
```

| Metric | Baseline | Candidate | Direction |
|---|---:|---:|---|
| MAE | 0.0136432 | 0.0133513 | better |
| MSE | 0.00125892 | 0.00125603 | slightly better |
| PSNR | 29.8412 | 29.8621 | slightly better |
| pred k-flow EPE | 0.642645 | 0.642942 | worse |
| pred k-flow L1 | 0.415665 | 0.415901 | worse |

Interpretation: this is not enough to scale to full data. Frame fidelity is
slightly better, but the requested temporal/k-flow behavior did not improve.

## Refined k-flow Mask Probes

The first candidate used only the coarse K2 support mask. The next probes keep
the K2 path mask but further restrict k-flow loss to target-flow-active pixels:

```text
refined mask = K2 path mask filtered by top target-flow magnitude quantile
```

Active probes:

| Probe | GPU | Save dir |
|---|---:|---|
| q70, kflow_weight 0.01 | 2 | `/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/stage2_32f_pos_kflow_q70_w001_500` |
| q70, kflow_weight 0.02 | 3 | `/data/fzj/RMDM/experiments/rmdm_temporal_pilot_easy_hard_2scene/stage2_32f_pos_kflow_q70_w002_500` |

These are the next gate candidates before any full-dataset training.
