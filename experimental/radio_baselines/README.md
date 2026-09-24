# Clean-observation radio-map baselines

User decision on 2026-09-24: keep the existing original RMDM checkpoint as a
baseline; do not train further RMDM variants. Adapt RadioDiff, RadioUNet and
RME-GAN to the same Tx-blind sparse-observation task, and train with **sigma=0**.

## Shared protocol

- Input `[building, zero_Tx, current_vehicle, sparse_RSS, sampling_mask]`, output
  one 128x128 RSS frame in the dataset's existing normalized units.
- No Tx coordinates, Tx labels, future frames, noise levels, or dense target enter
  the predictor. Dense clean targets are used only for supervised training and
  VAE training/encoding. No observation overwrite at inference.
- `m20_formal075_clean16_scene_split.json`; same train packed cache as original
  RMDM, 900,000 frames. Uniform choice of sampling rates 1–10%, exact-budget
  sampling without replacement in free space. Observations are exactly target
  times mask. All masks are drawn on CPU for cross-GPU pairing.
- Clean validation uses the existing validation video manifest, frames 0/50,
  rates 1/2/3%, and full-image MSE to select checkpoints. Predictions are clipped
  to [0,1] consistently with RMDM evaluation. Also report MAE, PSNR and
  unobserved-free-space MSE. VAE selection uses clean reconstruction MSE.
- Validation never advances training RNG. Test is not accessed by training.
  `evaluate.py --split test` is an explicit final-report operation only.
  Noise robustness uses the unchanged checkpoint at sigma
  0/.01/.03/.05/.07/.09 with CPU-paired masks and measurement noises.
  RadioDiff's latent noise has a different shape than RMDM's pixel noise: it uses
  the same per-frame seed convention, not an assertion of identical tensors.
- These are **adapted baselines**, not reproductions of original paper scores.
  Historical GPU-generated-mask RMDM scores must be re-evaluated with the CPU
  protocol before direct paired comparison.

## Official sources and adaptations

Pinned author repositories and commits are in `vendor/sources.json`, with each
license retained. No pretrained radio-map or RMDM weights are loaded.

### RadioUNet-adapted

Source: <https://github.com/RonLevie/RadioUNet> (`lib/modules.py`). Retain the
official RadioWNet U-Net widths, kernels, skip connections, and two-stage training:
train first U, then freeze it and train refinement U. The public input becomes
five channels and spatial resolution becomes 128 rather than 256. The wrapper
skips evaluation of the unused U; regression checks compare both stage outputs
exactly against the original forward function. Use normalized RSS MSE rather
than the upstream dataset-specific scaling by 256.

### RME-GAN-adapted

Sources: <https://github.com/achinthaw/RME-GAN> and
<https://arxiv.org/html/2212.12817v1>, Sections IV-B–IV-D. Its generator is the
first U of the same RadioWNet (the two author files differ only in default input
count); preserve the official discriminator convolution/BatchNorm stack.

- With no true Tx input, fit a log-distance template jointly over a fixed 16x16
  candidate source grid and least-squares intercept/nonpositive slope, using
  only observed positions and RSS. Template is training guidance, not an extra
  predictor input. This is an explicit replacement for the known-Tx LDPL fit.
- Retain global then local training of the same generator. Global uses MSE,
  TV and template gradient direction; local uses MSE, TV, MS-SSIM, geometry and
  Fourier losses. Both use adversarial training.
- Correct upstream discriminator's real-target-zero bug and Sigmoid plus
  BCEWithLogits mismatch. Condition D on the public observations for both real
  and fake examples, replacing the upstream real/fake-label input shortcut.
- Local losses follow the paper's differentiable definitions. The supplied
  script detaches the FFT and casts normalized RSS to integers, making its
  frequency term unsuitable for learning. Here compare 100 fixed highest-spatial-
  frequency coefficients with clean targets, using differentiable torch FFT.
- Geometry picks the strongest **observed** point per 10x10 spatial partition,
  ignoring empty cells; regular partitions replace SLIC. MS-SSIM uses five
  downsampled scales and 3x3 box windows for 128x128 inputs. These choices are
  adaptations, not claims of exact upstream loss equivalence.
- On normalized RSS, global weights are adversarial=.01, MSE=10, TV=.01,
  gradient=.01. Local: adversarial=.01, MSE=1, TV=.001, MS-SSIM=.1,
  geometry=1, high-frequency=.01. Every component is logged. No weights were
  chosen using test data or noisy validation.

### RadioDiff-adapted

Source: <https://github.com/UNIC-Lab/RadioDiff>. Retain its KL autoencoder
(base width128, multipliers1/2/4, three latent channels, 4x compression), Swin-B
condition backbone, dual-head attention U-Net (width128, multipliers1/2/4/4),
adaptive FFT blocks, weighted drift/noise loss and constant-SDE reverse update.

- Train the autoencoder on clean training maps first; freeze the best VAE for
  latent diffusion. VAE uses L1+MSE, VGG LPIPS, learned log variance, KL weight
  1e-6 and adaptive-weight hinge GAN (.5). The frozen ImageNet/VGG perceptual
  metric is the only pretrained component; predictors including Swin start
  randomly. The VAE adversarial phase starts at 20k within the 40k budget.
- Replace Swin's RGB patch projection with a five-input-channel projection.
  Condition features and latent maps are sized for 128/32. Keep the upstream
  default latent scale .3. No unrelated PINN or propagation auxiliary heads.
- Fix upstream FFT weights to `[C,H,W//2+1,2]` to match actual padded-convolution
  feature sizes. Execute FFT in float32 under bf16 autocast.
- Use 20 reverse-SDE steps for validation, deterministic per-frame CPU draws
  for initial and transition noise, and never reset global seeds inside sampling.
  This is RadioDiff's constant SDE, **not DDIM**. Its original config defaults
  to one-step sampling; 20 steps are explicitly recorded here.

## Training and execution

`pipeline.py` has two independent GPU lanes. `conv` runs RadioUNet first/refinement,
then RME-GAN global/local; each stage caps at 20k optimizer updates. `diffusion`
runs VAE then RadioDiff, each capped at 40k. All stages use global batch128,
validate every4k and stop after three non-improving validations. VAE stopping
waits until its adversarial phase has been exercised. Best stage checkpoint
initializes the next stage; optimizers reset between stages. After both convolutional
stages finish, choose between their best checkpoints by clean validation MSE
and record the selected stage. Baseline training
budgets are recorded explicitly and are not claimed to match original RMDM's
global batch384/27k exactly.

AdamW with zero weight decay, betas .9/.99, norm clipping1, warmup500, cosine
decay to 10% of initial LR. RadioUNet and global RME-GAN LR1e-4; local RME-GAN,
VAE and latent diffusion LR5e-5. Backbone autocast bf16, loss/FFT float32.

Example (paths supplied by `.agents/config.yaml` and run record):

```bash
PYTHONPATH=src:. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -m experimental.radio_baselines.pipeline \
  --lane conv --data-root DATA --cache-root CACHE --output UNIQUE_RUN_ROOT
PYTHONPATH=src:. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m experimental.radio_baselines.pipeline \
  --lane diffusion --data-root DATA --cache-root CACHE --output UNIQUE_RUN_ROOT
```

Use `--smoke` on the pipeline to exercise all stages with real data, two updates,
backward/weight-update checks, validation, checkpoint roundtrips and stage
inheritance. Formal output must be distinct from smoke output. A stage records
last model/optimizers/RNG/epoch/batch cursor for resume. The pipeline skips
complete stages and resumes only stages with an existing `last.pt`; it never
overwrites an unknown partial run.

Requirements added to the existing RMDM environment: einops==0.8.1,
lpips==0.1.4. Existing torch/torchvision are retained. PyTorch's VGG16 cache is
used for LPIPS; model files are not committed to Git.
