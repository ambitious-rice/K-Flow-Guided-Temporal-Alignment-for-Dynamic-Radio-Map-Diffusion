# Noise-aware RSS feature retraining

This exploratory experiment asks whether measurement sigma can improve complete
W16 DDIM-50 reconstruction when trained in the observation encoder. It retains
all diffusion steps. It has no output projection, external sigma remapping, or
loss that deliberately makes incorrect sigma inputs perform poorly.

Four matched two-GPU runs:

| Variant | Measurement sigma access |
| --- | --- |
| `blind` | None; existing conditioning evaluated at sigma zero |
| `film` | New sigma/log-variance embedding and zero-initialized modulation heads |
| `reliability` | Sigma-dependent attenuation of RSS features in both stems |
| `reliability_fixed` | Same trainable stems, always using sigma 0.03 |

For RSS projection weights W and mask M, independent measurement noise induces
feature noise variance `q * sum(W**2 * M)`, where `q = sigma**2`. The reliability
stem uses gain `s / (s + feature_noise_variance)` with learned positive channel
scales s. It attenuates RSS tokens while preserving separately projected mask
tokens. This is a Wiener-like feature gate, not an exact posterior. At q=0 the
original stem is recovered. The fixed variant controls for added parameters and
the effect of constant attenuation. FiLM provides an unrestricted conditioning
comparison; its new heads start at zero.

All variants start from the same pretrained W16 checkpoint at old sigma zero.
All DiT parameters are retrained; HWM and old conditioning stay frozen. This is
warm-start training, not training from scratch. The objective is clean-map MSE
plus 0.1 times temporal-difference error, evaluated through standard diffusion
training states, not single-step inference.

The saved input protocol selects 600 videos (12 training scenes, 50 distinct
episodes per scene, one transmitter per episode). Each base window produces a
quartet with sigma 0 and continuous samples from (0, .03), (.03, .06), (.06, .09).
Masks, Gaussian fields and diffusion states are shared within each quartet;
integer seeds and the video manifest are shared between all variants. Training
rates are 1%, 2%, 3%. With 16 examples per GPU and two GPUs, global batch is 32
(eight base window draws) without gradient accumulation.

Selection uses eight validation videos at 2%, complete DDIM-50 and unseen-point
MSE. Raw and EMA checkpoints, including initialization, compete under the same
rule. Confirmation uses 16 additional validation videos with four standard
noise levels and intermediate levels on eight videos. These are reused
exploratory validation sets; no formal test data is used. Final evaluation
sweeps input sigma and compares true sigma with the best constant selected only
on the selection set, as well as separately trained blind/fixed controls.
Spatial MSE, PSNR and temporal error must all be reported. Merely responding to
sigma is not evidence of better reconstruction. A selected step-zero checkpoint
must not be presented as a training improvement.

Run configuration, integer seeds, dependency patch, selected videos, validation
input references, checkpoints, logs, prediction tensors and final CSV/JSON
results live in `runs/noise_feature_retrain_20261008`. The run record lives in
`.agents/runs/noise_feature_retrain_20261008.yaml`. Local source has preexisting
changes; their necessary dependency patch and model YAML are snapshotted there.

The supervisor runs all four jobs concurrently, limits training to 7.25 hours,
and terminates its process groups at 8.75 hours including evaluation. The normal
target is 12,000 optimizer steps. It does not extend the budget automatically.

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_feature_retrain.launch --root runs/noise_feature_retrain_20261008
```
