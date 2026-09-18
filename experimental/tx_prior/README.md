# tx_prior

Single task directory for the W1 TX-conditioned scene prior. The model receives
`building`, `vehicle`, and explicit `tx`. Its HWM has three input channels, and
its input and condition stems have no observation projection or fusion;
`observed_rss` and `sampling_mask` do not enter either network. Training uses
the standard epsilon-DDPM objective without samples, observation noise,
observation alignment, heldout loss, or condition dropout.

`smoke.py` is a preflight-only helper. Real two-step smoke training is explicit:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/smoke.yaml \
  --smoke --smoke-data-limit 512
```

The production environment on this machine uses `torch==2.11.0+cu128` and
`natten==0.21.6+torch2110cu128`.

The fixed two-GPU remote smoke uses the verified packed cache explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src:. \
  /home/fzj/.venvs/rmdm_tx_prior/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train \
  --config experimental/tx_prior/remote_smoke.yaml \
  --smoke --smoke-data-limit 512 \
  --packed-root /home/fzj/.cache/rmdm/tx_prior
```

This performs exactly two optimizer steps with batch 64/GPU, accumulation 2,
and global batch 256. It writes only the fixed `runs/tx_prior/smoke` stage.

If smoke stopped at step 0 before writing a checkpoint, restart it in place:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/smoke.yaml \
  --smoke --smoke-data-limit 512 --restart-incomplete
```

This restart is accepted only when `status.json` is readable, its state is
`training` or `failed`, `global_step` is zero, and no `checkpoints/last.pth`
exists. Existing status and history records are retained, and a new history
event is appended.

Formal training is also explicit:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml
```

Resume formal training in the same stage directory:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml \
  --resume-from runs/tx_prior/train/checkpoints/last.pth
```

The full local configuration is `train.yaml`, fixed to GPUs 4-7, batch 64/GPU,
accumulation 1, global batch 256, BF16, no gradient checkpointing, and
`runs/tx_prior`. It runs through at most 80k, validates first at 10k and every
5k thereafter. Non-improvements before 25k do not consume patience; starting
at 25k, two consecutive validations without a lower `full_image.nmse` stop the
run. The local runner calls
`build_scene_prior_system` and never constructs sparse observations.

Outputs remain under one task root: smoke uses `runs/tx_prior/smoke`, and formal
training uses `runs/tx_prior/train`. Before the first epsilon run, the completed
x0 baseline is archived once as the sibling `runs/tx_prior/x0`; epsilon then
starts from step zero in a fresh `train` directory. Later fixes and resumes stay
in that same directory. Fresh runs refuse a non-empty matching stage directory.
Resume with the matching stage's `checkpoints/last.pth`.

`remote.yaml` is the formal machine-specific equivalent for the two GPUs and
paths on `lab_server_137` (batch 64/GPU, accumulation 2, global batch 256). Run
it with repository root `/data_16T_137/fzj/RMDM/project`; deploy only the
task-level `project/runs/tx_prior` symlink so it resolves to the unified task
root `/data_16T_137/fzj/RMDM/runs/tx_prior` (leave other `project/runs`
contents untouched). Resume from the same
`runs/tx_prior/train/checkpoints/last.pth`; do not create another task or retry
directory.

Training code must call `make_prior_training_batch` directly on the dense
dataset batch and must not instantiate `SamplingPolicy`. Prior evaluation uses
`deterministic_prior_noise_like`, whose seed depends only on frame identity and
the experiment seed (there is intentionally no observation-rate argument).

## Standalone sampling comparison

Evaluate one epsilon checkpoint without changing training files:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. python -m experimental.tx_prior.evaluate \
  --config experimental/tx_prior/remote.yaml \
  --checkpoint runs/tx_prior/train/checkpoints/best.pth --batch-size 4 --steps 20 50
```

Add `--smoke --smoke-frames 4` for a one-video end-to-end precheck. Formal
evaluation requires all 3000 manifest frames and calls `runner.validate_prior`.
Both step counts use the same CPU-loaded checkpoint, seed, frame-keyed noise,
manifest and eta=0. Evaluation is FP32: training validation directly calls core
methods and bypasses the BF16 prepared-forward wrapper. Model weights load
strictly; optimizer/RNG states are not restored. JSON results are exclusive-create
under `train/sampling_eval/step_NNNNNN/ddimNN.json`; smoke uses a separate `smoke`
subdirectory. Existing results are never overwritten. `--output-dir` may select
only a descendant of that sampling-evaluation directory. No previews are generated.

`python -m experimental.tx_prior.diagnose --checkpoint ... --output
runs/tx_prior/train/diagnostics/step_NNNNNN.json` runs small paired train/val
fresh-noise checks: forward/cache parity, timestep-binned epsilon/x0 errors,
scene shuffle, copy-cal/zero-x0 baselines, and FP32/BF16 forward comparison.
An optional `--old-checkpoint` loads the original 96-channel UNet strictly and
uses its native traffic encoding on the **same current dataset frames**. These
are domain-transfer diagnostics, not a reproduction on the missing old data.
Default sample size is 30 frames/split, with paired val DDIM20/50 (eta=0).
For smoke use 2 frames/split, timesteps 100/900, and a separate smoke JSON.
Outputs refuse overwrite; training files and model weights are never changed.

Diagnostic and standalone evaluation tools are maintained on the Git branch
`experimental/tx-prior-diagnostics`, separately from production training on main.
For diagnostic two-scene validation, use `--config experimental/tx_prior/remote_two_scene.yaml
--expected-frames 2000 --output-dir runs/tx_prior/train/sampling_eval/two_scene`.
The fixed manifest retains exactly the original 10 videos per remaining scene,
excluding user-reported unusable `town01_opt_junction_0087`. Train/test membership,
noise seeds and old three-scene results are unchanged. Results include per-scene
metrics and additive statistics; do not mix these with three-scene model selection.
`evaluate_legacy` evaluates the old native-condition UNet on all 3000 current
validation frames. `audit_data` is CPU-only and checks the full index plus
stratified PNG/condition/frame alignment and exact packed-reader equivalence;
it does not modify datasets. Results stay under the existing task's `train`.

## Disposable packed cache

Raw data remains the source of truth on the `/data_p6` mechanical array. The
packed cache is a derived, disposable artifact under `/home/fzj` on the root
NVMe filesystem. Its fixed recommended location is
`/home/fzj/.cache/rmdm/tx_prior`; an incomplete build is never accepted by the
reader. `train.yaml` intentionally keeps the legacy backend as its default;
switching always requires an explicit `--packed-root`.

Build and verify only when source-disk contention is acceptable:

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.tx_prior.packed_cli build \
  --source-root /data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/M20_Formal075_RadioMapSeerPack \
  --split-file /data_p6/fzj/resources/RMDM/components/dataset_metadata/multi20_formal_scene_split.json
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.tx_prior.packed_cli verify \
  --source-root /data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/M20_Formal075_RadioMapSeerPack \
  --split-file /data_p6/fzj/resources/RMDM/components/dataset_metadata/multi20_formal_scene_split.json
```

The builder prints a flushed progress line after the first video, every 100
videos, and completion. Verification first checks the complete legacy video
order, then compares 64 evenly selected videos at their first, middle, and last
frames against raw data. It reports `compared_frames` without rereading the
entire split.

At a checkpoint boundary, add
`--packed-root /home/fzj/.cache/rmdm/tx_prior` to the same resume
command. This does not change `WindowDataset` indexing or epoch permutation.
Remove it only through the marker-checked lifecycle command:
`PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.tx_prior.packed_cli remove --confirm`.
It can always be rebuilt from the raw source.
