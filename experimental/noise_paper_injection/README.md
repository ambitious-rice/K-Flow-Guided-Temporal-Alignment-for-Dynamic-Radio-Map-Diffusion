# Three paper-inspired sigma mechanisms in W16 RMDM

User authorized three concurrent two-GPU training experiments on 2026-10-09,
without a separate diagnosis phase. Keep W16 DiT, trainable HWM, original six
losses (all weight 1), clean sampled-point targets, full DDIM50 reconstruction.
All models start randomly; no pretrained weight is opened. Common backbone
initialization seed is 20261009. Added channels/HyPaNet use an isolated RNG.

## Mechanisms and correspondence

1. `noise_map` (GPUs 0,1): FFDNet / DRUNet noise-level map concatenated to RSS
   and mask in both tubelet stems. Map value is physical normalized-RSS sigma,
   not variance. Added map channel initialized orthogonally; existing common
   RSS/mask columns preserved. Remove old global sigma embedding/modulation.
   Rate modulation stays. This changes the input mechanism, not DiT into CNN.
   FFDNet: https://arxiv.org/pdf/1710.04026 (Sec III-B/D, Eq5).
   DRUNet: https://arxiv.org/pdf/2008.13751 (Sec3).
2. `usrnet_hyper` (GPUs 2,3): USRNet-style three-layer ReLU/Softplus parameter
   network outputs positive alpha,beta. Its inputs are sigma/0.09, rate/10,
   t/999. Sampling rate replaces SR scale factor; diffusion time is added.
   At each step z=x_t/sqrt(alpha_bar_t), then perform the exact masked data
   subproblem x=z+M*(y-z)/(1+alpha), then feed sqrt(alpha_bar_t)*x and beta map
   to DiT. Direct RSS channels into DiT are zeroed to route measurements through
   the data module. We reuse the 50 diffusion iterations rather than reproduce
   USRNet's 8-stage ResUNet. Alpha/beta are learned end to end, no forced ranking.
   https://github.com/cszn/USRNet/blob/master/models/network_usrnet.py
3. `diffpir_prox` (GPUs 4,5): DiffPIR Eq12 data proximal update after every x0
   prediction; rho_t=lambda*sigma_y²/bar_sigma_t²,
   bar_sigma_t²=(1-alpha_bar_t)/alpha_bar_t and lambda=7 fixed before validation.
   x0_updated=x0+M*(y-x0)/(1+rho_t). No measurement sigma is fed to the DiT;
   its direct RSS channels are zero, and it retains mask/scene conditions.
   Unlike the earlier frozen fusion pilot, all weights train through the update,
   which varies with diffusion time. DiffPIR originally uses a pretrained prior
   and sampling hyperparameters; this adaptation retains RMDM's deterministic
   DDIM update (eta=0), conditional HWM and supervised training. Original
   DiffPIR inpainting experiments are noiseless; our noisy sparse masking is an
   extension of its data-subproblem formula, not a reproduced paper result.
   https://arxiv.org/pdf/2305.08995 (Eq12, Eq26, Table3).

HWM continues seeing the original measurements in ALL arms and retains its
detached gate and auxiliary supervision. Its conditioning embedding now uses
rate only. Consequently the prior is not independent of the observations;
B/C are discriminatively trained modules, not exact Bayesian posterior samplers.
The observation-conditioned HWM is a shared departure from those paper priors.

## Matched training and validation

Two differentiable DDIM steps during training for ALL three arms: t sampled as
before, then max(t-20,0). Apply the six original losses to final x0 and HWM cal.
No new ranking loss, no output sigma remapping, no target/source in model input.
This allows an observation update to propagate into unobserved pixels and
differs from the earlier one-step training reference. At inference use50steps.

Full9000video data, sigma uniform[0,.09), rate discrete uniform1..10%; same
arithmetic seeds and samples across arms. TwoGPUs x microbatch8 x accumulation2
= global32 (previous four-GPU run used64). AdamW1e-4, warmup1000, cosine max40000,
EMA.999, gradclip1, BF16 training, FP32 evaluation. Initial smoke can adjust
microbatch/accumulation equally in all arms while preserving global32.

Reuse the exact `noise_scratch_full_20261008/inputs` bank; no new observations
or random masks. Tune8videos x3rates x4truth levels x4input levels, raw+EMA at
0,500,1000,1500,2000,2500,3000,and every1000. Early stop: protect4000 steps;
patience4 thereafter; reset for0.2%relative MSE improvement OR0.002absolute
alignment-gain improvement. Save every500; archive at each validation. Preserve
model,EMA,optimizer,per-rankRNG,stopping state for resume.

Save three selections: minimum correct-sigma MSE, maximum positive sigma gain,
and minimum MSE among positive-gain checkpoints. These are selection rules, not
evidence of success. Confirm all selected candidates on the other16videos with
the full360case bank including intermediate sigmas. Extra repeats1,2 on
confirmation four-sigma cases for best_usable, falling back to best_alignment.
Only diffusion initialization changes:20261009+repeat*100000+filtered_entry_index;
all sigma interventions within a case share initialization. Report gain versus
the constant chosen on tune, paired video bootstrap, row minima per rate,
absolute reconstruction against frozen paper W16 and prior scratch checkpoints.
The confirmation set has historical exploratory reuse; no formal test involved.

Single segment bounded at7h training/validation,8.75h incl final evaluation.
Timeout means resumable, not converged. Independent runs may reach different
step counts; compare common milestones as well as wall time and selected weights.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_paper_injection.launch --root runs/noise_paper_injection_20261009 --smoke
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_paper_injection.launch --root runs/noise_paper_injection_20261009
```
