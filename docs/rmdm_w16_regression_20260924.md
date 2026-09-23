# RMDM W16 regression audit (2026-09-24)

## Paired validation

All rows use the same 40 W16 clips per condition, two validation scenes, DDIM20,
rates 1 and 3, and measurement sigma 0, 0.03, 0.05, 0.09. T1 is evaluated
framewise on the same W16 clips. MSE is averaged equally across eight conditions.

| Checkpoint / intervention | MSE |
| --- | ---: |
| T1 selected step 20250, framewise | 0.000978 |
| T16 selected step 16200 | 0.001866 |
| T16 step 16200, all temporal stages disabled | 0.002058 |
| T16 step 16200, condition stage disabled | 0.001808 |
| Frozen-T1 temporal-only step 1000 | 0.000971 |
| Frozen-T1 temporal-only step 1200 | **0.000970** |

The joint T16 run degraded the spatial weights: removing all temporal modules
from its selected checkpoint still gives more than twice the T1 error. The
condition adapter is also harmful; it edits the four-channel tensor containing
building mask, vehicle mask, HWM calibration, and known measurement variance.

The regression is strongly rate dependent. At rate 1, T16 and T1 are close;
at rate 3 and sigma 0, T1 MSE is 0.000419 versus T16 0.002157. At rate 3 and
sigma 0.09, they are 0.001093 and 0.003063. Disabling every temporal stage
does not recover rate 3, which implicates changes to the spatial/HWM model.
The old T16 is actually better than T1 at rate 1 and high sigma, so a useful
temporal effect is obscured by its collapse at rate 3.

| Rate | Sigma | T1 | Original T16 | Frozen-T1 T16 step 1200 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.000687 | 0.000728 | 0.000686 |
| 1 | 0.03 | 0.001049 | 0.001021 | 0.001043 |
| 1 | 0.05 | 0.001306 | 0.001265 | 0.001298 |
| 1 | 0.09 | 0.001912 | 0.001699 | 0.001888 |
| 3 | 0 | 0.000419 | 0.002157 | 0.000421 |
| 3 | 0.03 | 0.000582 | 0.002373 | 0.000580 |
| 3 | 0.05 | 0.000772 | 0.002622 | 0.000764 |
| 3 | 0.09 | 0.001093 | 0.003063 | 0.001076 |

As a diagnostic two-checkpoint combination, choosing the original T16
at rate 1 and T1 at rate 3 gives 0.000947 average MSE (3.1% below T1). This
is a derived routing score from the same paired rows, not a trained single
model. It motivates a rate-conditioned temporal residual or separate
rate-specialized models rather than unconditional joint fine-tuning.

An earlier T16 checkpoint (step 5400) scored 0.009894 on a fixed four-condition,
10-clip-per-condition subset. Disabling only the predicted-noise temporal stage
reduced it to 0.001779, while disabling all temporal stages gave 0.001691.
This points to an unstable temporal residual on epsilon that compounds over
DDIM steps. At the selected step 16200, the same stage partly compensates for
spatial-weight drift, so its effect changes during training.

On the fixed four-condition subset, swapping the T1 HWM and variance embedding
into the selected T16 checkpoint (with temporal stages disabled) gave MSE
0.002739; swapping only the T1 diffusion denoiser gave 0.004169. The unmodified
T16 spatial weights gave 0.002050 on that subset. Both isolated swaps worsen
the result, showing that HWM and denoiser co-adapted during joint training;
neither can be repaired in isolation by copying T1 weights. Reinitializing
from the complete T1 checkpoint and freezing it is the controlled route.

## Architectural comparison with historical direct adapters

`archive/old_python/dual_decoder_adapter_student.py` inserted zero-initialized
separable 3D adapters inside the original HWM and diffusion output decoders.
It first trained adapters with the original model frozen, then fine-tuned the
base at 1e-6 while adapters used 5e-5. It also used low-timestep sampling and
an x0 reconstruction term. In the historical paired key-path test, MSE moved
from 0.007217 to 0.006849 (5.1% lower), PSNR from 23.80 to 24.21 dB, and
temporal optical-flow error from 0.676 to 0.309. A simpler frozen RMDM plus
output 3D residual head improved MSE from 0.003931 to 0.003888 (1.1%).

These are within-protocol gains only. The older work used T32, known Tx and
raw RSS/mask in the diffusion condition, and different data. The current
noise-aware T16 is Tx-blind, sends observations through HWM only, and adds
temporal modules at three boundary tensors. Its joint training updated both
the original 63.2M spatial parameters and 7.38M temporal parameters at 5e-5.
With 30 clips per optimizer update versus T1's 384 independent frames, it also
draws far fewer independent diffusion timesteps and observation conditions.

## Repair experiment

The first controlled repair initializes from the selected T1 checkpoint and
freezes all spatial/HWM parameters. Only 7.38M zero-initialized temporal
parameters train on GPUs 3 and 4 at 2e-5, with 30 clips per update. GPUs 0 and
2 continue the main HVDiT W1-epsilon pipeline with its global batch of 256.
Checkpoint 200 on the fixed four-condition subset gives MSE 0.001346 versus
T1's 0.001344; checkpoint 600 gives 0.001335 (0.6% lower). Disabling the
condition stage at step 600 gives 0.001339, disabling the epsilon stage gives
0.001335, and keeping only calibration gives 0.001336. Later full validation
at step 1000 gives 0.000971 across eight conditions versus T1's 0.000978,
about 0.7% lower. Rate 3 at sigma 0.09 is 0.001077 versus T1's 0.001093 and
the old T16's 0.003063. Step 1200 gives 0.000970, about 0.8% below T1. The
severe regression is fixed, but the gain is small.
The next candidate is a calibration/decoder feature adapter with
the base frozen, followed by base learning rate near 1e-6 only if paired
validation improves.

Artifacts are in `runs/rmdm_temporal_audit_20260924/`; the main pipeline is
`noise_hvdit_pipeline_2gpu`, and the repair trial is `rmdm_w16_frozen_trial`.
