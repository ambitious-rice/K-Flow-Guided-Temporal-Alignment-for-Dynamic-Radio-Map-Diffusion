# Noise-conditioned HVDiT, source-supervised x0 / epsilon comparison

Approved 2026-09-23: restore the original HVDiT topology, train W1-x0 then
W1-epsilon on local GPUs 0,2,3,4, compare validation performance, and expand
only the clearly better target to W16. If neither wins clearly, train both
W16 targets sequentially. No remote execution is needed.

## Model and objective

The original dual stems, condition pyramid, local/global attention and pixel
decoder are retained. Widths are 384/768, local depth 2, global depth 11;
W16 uses temporal patches of two frames. A 512-wide variance MLP embeds
`[v/.0081, log1p(v/.0081), v==0]`; sampling rate has a separate continuous
embedding. Zero-initialized projections modulate both observation stems,
local/global blocks, pixel decoding, and HWM encoder/decoder features.
W1 has 140,386,333 parameters. HWM gate remains detached as in the original.

The model receives building, vehicle, RSS, sampling mask, rate and variance.
It creates a zero legacy Tx channel internally. Only the loss reads
`source_label`; validation/test never load it. Source masks reproduce the
legacy Gaussian heatmap threshold `>0.5` with sigma=1.5 pixels. The existing
source loss supervises cal toward 1 at those pixels, rather than regressing
Tx coordinates. The equation, obstacle, source and clean-calibration terms
initially all have weight 1, with equation k=0.2. No observation-alignment,
held-out-point or temporal smoothness loss is added to this first comparison.

Measurement corruption: `Y=M*(X+sigma*epsilon)` without clipping;
P(clean)=.20, .65*U(0,.05), .15*U(.05,.09). One sigma per window, independent
errors by pixel/frame. Source labels never corrupt the clean target.

W1-x0 and W1-epsilon use the same random initialization, data order, masks,
noise, optimizer, schedule and architecture. Only the diffusion prediction
target changes. Native x0/epsilon losses have different implicit timestep
weighting; this experiment compares training recipes, not parameterization
alone. No weights are transferred between the two W1 targets.

## Recipe

Clean16: 12 train / 2 val / 2 test scenes; four invalid scenes excluded.
W1 starts from scratch (old DiT weights have seen current val scenes).

| | W1 | W16 |
|---|---|---|
| Optimizer steps | 40,000 | 40,000 |
| Global batch | 256 frames | 32 clips / 512 frames |
| Four-GPU microbatch / accumulation | 16 / 4 | 2 / 4 |
| LR / final LR | 2e-4 / 2e-5 | 5e-5 / 5e-6 |
| Warmup | 2,000 | 1,000 |

AdamW (.9,.95), eps=1e-8, decay=.01, clip norm=1, BF16, EMA=.999;
sampling rates 1..10 uniformly. Gradient checkpointing is enabled.
W16 initializes from its matching validation-selected W1 state, including
noise conditioning. Inflation preserves weight scales but is not promised
to be functionally identical to independent W1 frames.

## Evaluation and automatic selection

Every 4,000 steps, all four GPUs run fixed DDIM20 validation of raw and EMA
weights on all tracked validation videos, starts=0,48, all 32 frames, rates
1/2/3 and sigma=0/.01/.03/.05/.07/.09. This is **validation**, never test.
W1 predicts every physical frame in those same clips independently. Both models use
the same masks, observations and per-frame initial diffusion noise.
PSNR is averaged per frame for both models. Metrics include full image,
unobserved free space, observed points and error in frame-to-frame changes.

Early stop after three consecutive validations without a lower mean unobserved MSE
(best of raw/EMA). Save best.pth as well as resumable last.pth; the patience
state is restored on resume. Both W1 and W16 have a 40,000-step ceiling.

After training, the top three (step, raw/EMA) candidates by DDIM20-validation
mean unobserved MSE are evaluated with DDIM50 on all tracked val videos and
starts 0,48 (32 physical frames/video). Select lowest mean unobserved MSE
among candidates within 5% of the best candidate's clean unobserved MSE.

A W1 target wins clearly if mean unobserved MSE improves >=5%, the 95%
paired-video bootstrap CI of its error difference is below zero, and neither
clean nor high-noise (sigma>=.05) error regresses >2%. Bootstrap is stratified
by the two validation scenes; it does not establish broad scene-level
generalization. Otherwise both W16 variants are queued sequentially.
Final tests run once, only after all training/selection decisions are fixed.

## Commands

From repository root with `PYTHONPATH=src:.` and the local RMDM environment:

```bash
python -m experimental.noise_hvdit.data --config experimental/noise_hvdit/w1_x0.yaml
python -m pytest experimental/noise_hvdit/test_model.py -q
python -m experimental.noise_hvdit.pipeline --gpus 0,2,3,4
```

The pipeline launches four-process torchrun jobs, resumes each stage from
`last.pth`, and skips completed stages. `--stop-after N --skip-validation`
on the train entrypoint supports bounded real-data checks without changing
the formal LR horizon. Checkpoints contain optimizer/scheduler, EMA, per-rank
RNG and dataset position. Config and Git provenance accompany each stage.
Keep outputs under `runs/noise_hvdit`; important status is also recorded in
`.agents/runs/noise_hvdit_source.yaml`. Source labels are a separate 141 MiB
cache; the existing Tx-free image cache is reused unchanged.
