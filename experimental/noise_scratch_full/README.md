# Full-data random initialization experiment

Train all HWM and DiT weights from random initialization on the full 9,000-video
training manifest. Keep original full-map clean MSE, HWM calibration, equation,
obstacle and source losses. Replace noisy observation tolerance with a
mask-normalized sampled-point clean-target MSE at every sigma. All six weights
are 1. Sampling rates are discrete uniform 1–10%; measurement standard deviation
is continuous uniform [0, 0.09). No pretrained checkpoint is loaded.

The fixed validation bank and integer sampling seeds are saved under the run's
`inputs/`. Full DDIM-50 validation changes only supplied sigma for each fixed
observation/noise realization. Monitor both raw and EMA weights at steps 0, 250,
500, and every 1,000 steps. Only the eight `tune` videos determine checkpoint
selection and stopping. The other 16 validation videos provide confirmation;
these are exploratory validation data, not the formal test set.

The maximum is 40,000 steps with 1,000 warmup steps followed by cosine decay.
Early stopping counts stalled validations only after step 8,000. Either a
0.2% relative improvement in correct-sigma unobserved-point MSE or an increase
of 0.002 in `1 - correct_sigma_MSE / best_fixed_sigma_MSE` resets patience.
The score and gain each use the best raw/EMA result at that validation and
separate meaningful-improvement anchors. Six consecutive stalled validations
stop training (earliest step 14,000). Raw metrics and both independent best
checkpoints are always retained; negligible improvements may still select a
checkpoint without resetting patience. Early stopping is an operational
criterion, not proof that sigma conditioning succeeds or that an optimum was reached.

Run on GPUs 0–3, microbatch 16, accumulation 1, global batch 64. Save optimizer,
EMA, per-rank CPU/CUDA RNG and stopping state in `training/last.pt` every 500
steps and at validation. Restarting the launcher resumes that checkpoint.
The current segment allows 7.25 hours of training/validation and 8.75 hours
total including confirmation. A time limit yields `ready_to_resume`; it does
not report convergence. Check the saved checkpoint step after a hard limit.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_scratch_full.launch --root runs/noise_scratch_full_20261008
```
